"""自建向量记忆、进程内 Mem0 OSS 与已部署 Docker Mem0 的统一适配器。

这里没有 Mock：VectorMemory 执行真实的本地向量计算；Mem0OssMemory 使用 mem0ai、
Qdrant local mode 和本地 Hash Embedding；DockerMem0Memory 调用正在运行的 Mem0 REST
服务。Docker 认证密钥只在容器内部由环境变量读取，不进入父进程、日志或报告。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol
from urllib.parse import quote
from uuid import uuid4


@dataclass(frozen=True)
class AuthScope:
    """经过认证后得到的租户/用户作用域，后端不接受模型提供的 identity。"""

    tenant_id: str
    user_id: str

    @property
    def storage_user_id(self) -> str:
        """组合租户与用户，防止不同租户的本地 user_id 冲突。"""

        return f"{self.tenant_id}:{self.user_id}"


@dataclass(frozen=True)
class MemoryHit:
    """归一化后的记忆命中结果。"""

    id: str
    text: str
    score: float


class MemoryBackend(Protocol):
    """两种记忆实现共享的最小生命周期契约。"""

    name: str

    def add(self, scope: AuthScope, text: str, *, memory_id: str | None = None) -> str: ...
    def get(self, scope: AuthScope, memory_id: str) -> str | None: ...
    def search(self, scope: AuthScope, query: str, *, top_k: int = 3) -> list[MemoryHit]: ...
    def update(self, scope: AuthScope, memory_id: str, text: str) -> None: ...
    def delete(self, scope: AuthScope, memory_id: str) -> None: ...
    def close(self) -> None: ...


def hash_embedding(text: str, *, dimensions: int = 256) -> list[float]:
    """生成稳定、无网络依赖的字符 n-gram 向量。

    每个 2/3 字符片段通过 BLAKE2b 映射到固定桶并附带正负符号，最后 L2 归一化。
    它不是高质量语义模型，但足以构成可复现的本地向量基线与 Mem0 OSS Embedder。
    """

    normalized = " ".join(text.lower().split())
    grams = [normalized[i : i + n] for n in (2, 3) for i in range(max(0, len(normalized) - n + 1))]
    if not grams:
        grams = [normalized or "_"]
    vector = [0.0] * dimensions
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """已归一化向量的点积；显式实现便于观察实际计算。"""

    return sum(a * b for a, b in zip(left, right, strict=True))


class VectorMemory:
    """自建向量记忆基线：内存记录、Hash Embedding、强制租户/用户过滤。"""

    name = "self_built_vector"

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def add(self, scope: AuthScope, text: str, *, memory_id: str | None = None) -> str:
        memory_id = memory_id or uuid4().hex
        if memory_id in self._records:
            raise ValueError(f"duplicate memory_id: {memory_id}")
        self._records[memory_id] = {
            "scope": scope,
            "text": text,
            "embedding": hash_embedding(text),
            "deleted": False,
        }
        return memory_id

    def get(self, scope: AuthScope, memory_id: str) -> str | None:
        record = self._records.get(memory_id)
        if not record or record["scope"] != scope or record["deleted"]:
            return None
        return str(record["text"])

    def search(self, scope: AuthScope, query: str, *, top_k: int = 3) -> list[MemoryHit]:
        query_vector = hash_embedding(query)
        hits = []
        for memory_id, record in self._records.items():
            if record["scope"] != scope or record["deleted"]:
                continue
            score = _cosine(query_vector, record["embedding"])
            # 低于零表示哈希碰撞的反相关；不应为了凑满 top_k 返回。
            if score > 0.0:
                hits.append(MemoryHit(memory_id, record["text"], score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def update(self, scope: AuthScope, memory_id: str, text: str) -> None:
        if self.get(scope, memory_id) is None:
            raise KeyError("memory not found in authorized scope")
        self._records[memory_id].update(text=text, embedding=hash_embedding(text))

    def delete(self, scope: AuthScope, memory_id: str) -> None:
        if self.get(scope, memory_id) is None:
            raise KeyError("memory not found in authorized scope")
        # Tombstone 防止旧快照在测试中重新出现；生产系统还需持久化 delete version。
        self._records[memory_id]["deleted"] = True

    def close(self) -> None:
        return None


class DockerMem0Memory:
    """通过 `docker exec` 调用现有 Mem0 REST API，密钥永不离开容器。

    调用使用 stdin 传 JSON，避免把用户文本拼到 shell 命令。容器内的小段 Python 从
    ADMIN_API_KEY 环境变量创建 X-API-Key 头；响应只返回业务 JSON，不回显 Header。
    """

    name = "mem0_docker"

    _CONTAINER_SCRIPT = r"""
import json, os, sys, urllib.error, urllib.request
method, path = sys.argv[1], sys.argv[2]
raw = sys.stdin.buffer.read()
request = urllib.request.Request(
    'http://127.0.0.1:8000' + path,
    data=raw if raw else None,
    method=method,
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': os.environ['ADMIN_API_KEY'],
    },
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        sys.stdout.buffer.write(response.read())
except urllib.error.HTTPError as error:
    sys.stderr.write(error.read().decode('utf-8', 'replace'))
    raise
"""

    def __init__(self, container: str = "mem0-dev-mem0-1") -> None:
        self.container = container
        self._owned: dict[str, AuthScope] = {}

    @classmethod
    def available(cls, container: str = "mem0-dev-mem0-1") -> bool:
        """检查目标容器是否处于 running；不读取任何容器环境变量。"""

        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _request(self, method: str, path: str, payload: dict | None = None) -> object:
        """执行一次容器内 HTTP 请求，结构化返回服务错误。"""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else b""
        result = subprocess.run(
            ["docker", "exec", "-i", self.container, "python", "-c", self._CONTAINER_SCRIPT, method, path],
            input=body,
            capture_output=True,
            timeout=90,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace")[-1200:]
            raise RuntimeError(f"Mem0 Docker API {method} {path} failed: {message}")
        raw = result.stdout.decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def ready(self) -> bool:
        """用无副作用搜索验证认证、Embedding Provider 和向量存储均可用。"""

        probe = AuthScope("week4-health", "probe")
        try:
            self.search(probe, "health probe", top_k=1)
        except RuntimeError:
            return False
        return True

    def add(self, scope: AuthScope, text: str, *, memory_id: str | None = None) -> str:
        # REST API 分配 ID；外部 memory_id 只作为可追踪 metadata，不控制数据库主键。
        response = self._request(
            "POST",
            "/memories",
            {
                "messages": [{"role": "user", "content": text}],
                "user_id": scope.storage_user_id,
                "metadata": {
                    "tenant_id": scope.tenant_id,
                    "application_user_id": scope.user_id,
                    "external_memory_id": memory_id or uuid4().hex,
                },
                "infer": False,
            },
        )
        created = _result_items(response)
        if not created or not created[0].get("id"):
            raise RuntimeError(f"Mem0 add response lacks id: {response!r}")
        created_id = str(created[0]["id"])
        self._owned[created_id] = scope
        return created_id

    def get(self, scope: AuthScope, memory_id: str) -> str | None:
        try:
            response = self._request("GET", f"/memories/{quote(memory_id, safe='')}")
        except RuntimeError:
            return None
        item = response if isinstance(response, dict) else {}
        if not _belongs_to(item, scope):
            return None
        return _memory_text(item)

    def search(self, scope: AuthScope, query: str, *, top_k: int = 3) -> list[MemoryHit]:
        response = self._request(
            "POST",
            "/search",
            {
                "query": query,
                "filters": {
                    "user_id": scope.storage_user_id,
                    "tenant_id": scope.tenant_id,
                    "application_user_id": scope.user_id,
                },
                "top_k": top_k,
                "threshold": 0.0,
            },
        )
        hits = []
        for item in _result_items(response):
            if _belongs_to(item, scope):
                hits.append(MemoryHit(str(item["id"]), _memory_text(item), float(item.get("score", 0.0))))
        return hits[:top_k]

    def update(self, scope: AuthScope, memory_id: str, text: str) -> None:
        response = self._request("GET", f"/memories/{quote(memory_id, safe='')}")
        item = response if isinstance(response, dict) else {}
        if not _belongs_to(item, scope):
            raise KeyError("memory not found in authorized scope")
        # 某些 Mem0 版本在 update 未传 metadata 时会清空自定义字段。必须显式保留
        # tenant/application_user，否则更新后的记录会脱离最终授权边界。
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        self._request("PUT", f"/memories/{quote(memory_id, safe='')}", {"text": text, "metadata": metadata})

    def delete(self, scope: AuthScope, memory_id: str) -> None:
        if self.get(scope, memory_id) is None:
            raise KeyError("memory not found in authorized scope")
        self._request("DELETE", f"/memories/{quote(memory_id, safe='')}")
        self._owned.pop(memory_id, None)

    def close(self) -> None:
        """只删除本实例创建且仍存在的精确 ID，不调用危险的全库 reset。"""

        for memory_id, scope in list(self._owned.items()):
            try:
                self.delete(scope, memory_id)
            except (KeyError, RuntimeError):
                pass


class Mem0CloudMemory:
    """Mem0 Platform 适配器；API Key 只从环境读取，所有 ID 操作重复授权。"""

    name = "mem0_cloud"

    def __init__(self, api_key: str | None = None, *, infer: bool = True) -> None:
        from mem0 import MemoryClient

        key = api_key or os.getenv("MEM0_API_KEY")
        if not key:
            raise ValueError("缺少 MEM0_API_KEY")
        self.client = MemoryClient(api_key=key)
        self.infer = infer
        self._owned: dict[str, AuthScope] = {}

    def add(self, scope: AuthScope, text: str, *, memory_id: str | None = None) -> str:
        response = self.client.add(
            [{"role": "user", "content": text}],
            user_id=scope.storage_user_id,
            metadata={
                "tenant_id": scope.tenant_id,
                "application_user_id": scope.user_id,
                "external_memory_id": memory_id or uuid4().hex,
            },
            infer=self.infer,
            output_format="v1.1",
        )
        items = _result_items(response)
        if not items or not items[0].get("id"):
            raise RuntimeError(f"Mem0 Cloud add response lacks id: {response!r}")
        created_id = str(items[0]["id"])
        self._owned[created_id] = scope
        return created_id

    def get(self, scope: AuthScope, memory_id: str) -> str | None:
        try:
            item = self.client.get(memory_id)
        except Exception:
            return None
        if not item or not _belongs_to(item, scope):
            return None
        return _memory_text(item)

    def search(self, scope: AuthScope, query: str, *, top_k: int = 3) -> list[MemoryHit]:
        response = self.client.search(
            query,
            filters={
                "user_id": scope.storage_user_id,
                "tenant_id": scope.tenant_id,
                "application_user_id": scope.user_id,
            },
            top_k=top_k,
            threshold=0.0,
            output_format="v1.1",
        )
        return [
            MemoryHit(str(item["id"]), _memory_text(item), float(item.get("score", 0.0)))
            for item in _result_items(response)
            if _belongs_to(item, scope)
        ][:top_k]

    def update(self, scope: AuthScope, memory_id: str, text: str) -> None:
        item = self.client.get(memory_id)
        if not item or not _belongs_to(item, scope):
            raise KeyError("memory not found in authorized scope")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        self.client.update(memory_id, text=text, metadata=metadata)

    def delete(self, scope: AuthScope, memory_id: str) -> None:
        item = self.client.get(memory_id)
        if not item or not _belongs_to(item, scope):
            raise KeyError("memory not found in authorized scope")
        self.client.delete(memory_id)
        self._owned.pop(memory_id, None)

    def close(self) -> None:
        """删除本次实验精确创建的 ID，不影响账号中的其他记忆。"""

        for memory_id in list(self._owned):
            try:
                self.client.delete(memory_id)
            except Exception:
                pass
            self._owned.pop(memory_id, None)


class LocalHashEmbedding:
    """注入 Mem0 OSS 的真实本地 Embedder，执行确定性向量计算。"""

    def __init__(self, config=None) -> None:
        self.config = config

    def embed(self, text: str, memory_action=None) -> list[float]:
        return hash_embedding(text)


class DisabledInferenceLLM:
    """离线 `infer=False` 模式的安全哨兵；若意外调用会明确失败而非联网。"""

    def __init__(self, config=None) -> None:
        self.config = config

    def generate_response(self, *args, **kwargs):
        raise RuntimeError("offline Mem0 is configured with infer=False; LLM extraction is disabled")


class Mem0OssMemory:
    """真实进程内 Mem0 OSS：Qdrant local mode + 本地 Hash Embedding。"""

    name = "mem0_oss_local"

    def __init__(self, root: Path | None = None) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        from mem0 import Memory
        from mem0.configs.llms.openai import OpenAIConfig
        from mem0.utils.factory import EmbedderFactory, LlmFactory

        # Mem0 1.x 的配置枚举只接受内置 provider 名，因此保留名称、替换工厂实现。
        # 这不是 Mock：Embedding 真实计算并由 Qdrant 持久化；LLM 因 infer=False 不参与。
        EmbedderFactory.provider_to_class["huggingface"] = "memory_backends.LocalHashEmbedding"
        LlmFactory.provider_to_class["openai"] = ("memory_backends.DisabledInferenceLLM", OpenAIConfig)

        self._temp = None
        if root is None:
            self._temp = tempfile.TemporaryDirectory(prefix="week4-mem0-")
            root = Path(self._temp.name)
        root.mkdir(parents=True, exist_ok=True)
        collection = f"week4_{uuid4().hex[:12]}"
        self.memory = Memory.from_config(
            {
                "version": "v1.1",
                "history_db_path": str(root / "history.db"),
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": collection,
                        "path": str(root / "qdrant"),
                        "embedding_model_dims": 256,
                        "on_disk": True,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": "local-hash-256", "embedding_dims": 256},
                },
                "llm": {"provider": "openai", "config": {"model": "disabled-offline"}},
            }
        )
        self._owned: dict[str, AuthScope] = {}

    def add(self, scope: AuthScope, text: str, *, memory_id: str | None = None) -> str:
        result = self.memory.add(
            [{"role": "user", "content": text}],
            user_id=scope.storage_user_id,
            metadata={
                "tenant_id": scope.tenant_id,
                "application_user_id": scope.user_id,
                "external_memory_id": memory_id or uuid4().hex,
            },
            infer=False,
        )
        items = _result_items(result)
        if not items:
            raise RuntimeError(f"Mem0 OSS add response lacks item: {result!r}")
        created_id = str(items[0]["id"])
        self._owned[created_id] = scope
        return created_id

    def get(self, scope: AuthScope, memory_id: str) -> str | None:
        item = self.memory.get(memory_id)
        if not item or not _belongs_to(item, scope):
            return None
        return _memory_text(item)

    def search(self, scope: AuthScope, query: str, *, top_k: int = 3) -> list[MemoryHit]:
        filters = {
            "user_id": scope.storage_user_id,
            "tenant_id": scope.tenant_id,
            "application_user_id": scope.user_id,
        }
        try:
            # Mem0 v3 将 identity 放进 filters。
            response = self.memory.search(query, filters=filters, limit=top_k, threshold=0.0, rerank=False)
        except Exception as exc:
            # PyPI 1.0.x 仍要求顶层 user_id；适配差异但保留额外 tenant 过滤。
            if "At least one of 'user_id'" not in str(exc) and "Top-level entity" not in str(exc):
                raise
            response = self.memory.search(
                query,
                user_id=scope.storage_user_id,
                filters={"tenant_id": scope.tenant_id, "application_user_id": scope.user_id},
                limit=top_k,
                threshold=0.0,
                rerank=False,
            )
        return [
            MemoryHit(str(item["id"]), _memory_text(item), float(item.get("score", 0.0)))
            for item in _result_items(response)
            if _belongs_to(item, scope)
        ][:top_k]

    def update(self, scope: AuthScope, memory_id: str, text: str) -> None:
        item = self.memory.get(memory_id)
        if not item or not _belongs_to(item, scope):
            raise KeyError("memory not found in authorized scope")
        # Mem0 1.0.x 的 update(data) 若省略 metadata 会清空自定义 metadata；显式
        # 回传原值，保证 tenant/application_user 过滤在更新后仍然成立。
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        self.memory.update(memory_id, text, metadata=metadata)

    def delete(self, scope: AuthScope, memory_id: str) -> None:
        if self.get(scope, memory_id) is None:
            raise KeyError("memory not found in authorized scope")
        self.memory.delete(memory_id)
        self._owned.pop(memory_id, None)

    def close(self) -> None:
        for memory_id, scope in list(self._owned.items()):
            try:
                self.delete(scope, memory_id)
            except (KeyError, RuntimeError):
                # ID 来自本实例成功 add 的返回值；即便记录因第三方版本缺陷丢了
                # metadata，也只清理这个精确 ID，绝不调用 reset/delete_all。
                try:
                    self.memory.delete(memory_id)
                except Exception:
                    pass
        if hasattr(self.memory, "close"):
            self.memory.close()
        if self._temp:
            self._temp.cleanup()


def choose_mem0_backend(*, prefer_cloud: bool = True, prefer_docker: bool = True) -> MemoryBackend:
    """优先 Cloud，其次健康 Docker，最后回退进程内 OSS。"""

    if prefer_cloud and os.getenv("MEM0_API_KEY"):
        try:
            return Mem0CloudMemory()
        except ValueError:
            pass
    if prefer_docker and DockerMem0Memory.available():
        docker_backend = DockerMem0Memory()
        if docker_backend.ready():
            return docker_backend
    return Mem0OssMemory()


def _result_items(response: object) -> list[dict]:
    """兼容 Mem0 OSS/REST 不同版本的 list、results 和 result 包装。"""

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        for key in ("results", "result", "memories"):
            value = response.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
        if "id" in response:
            return [response]
    return []


def _memory_text(item: dict) -> str:
    """兼容 `memory`、`text`、`data` 三种返回字段。"""

    return str(item.get("memory") or item.get("text") or item.get("data") or "")


def _belongs_to(item: dict, scope: AuthScope) -> bool:
    """按服务返回的实际 identity 和 metadata 做最终授权，不仅相信 search filter。"""

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    stored_user = item.get("user_id") or metadata.get("user_id")
    tenant = item.get("tenant_id") or metadata.get("tenant_id")
    application_user = item.get("application_user_id") or metadata.get("application_user_id")
    return (
        stored_user == scope.storage_user_id
        and tenant == scope.tenant_id
        and application_user == scope.user_id
    )
