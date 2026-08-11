"""文档检索链的正确性、安全性、时效性与对抗测试。"""

from datetime import datetime, timezone

from rag_chain import AuthContext, Document, DocumentStore, HybridRetriever, MinimalRAGChain, evaluate_retrieval, load_questions


def build_chain() -> MinimalRAGChain:
    """每个测试使用独立 Store，避免删除状态互相污染。"""

    return MinimalRAGChain(HybridRetriever(DocumentStore.from_json()))


def test_dataset_has_at_least_twenty_gold_questions() -> None:
    """数据集必须包含问题、答案和证据，且同时存在可回答/不可回答样本。"""

    questions = load_questions()
    assert len(questions) >= 20
    assert all({"id", "query", "expected_answer", "evidence_ids"} <= question.keys() for question in questions)
    assert any(question["evidence_ids"] for question in questions)
    assert any(not question["evidence_ids"] for question in questions)


def test_minimal_chain_returns_answer_with_versioned_source_citation() -> None:
    """正常回答不能只有文本，必须携带文档 ID、URI、版本和更新时间。"""

    result = build_chain().answer("Access Token 给谁验证？", AuthContext("tenant-a", "user-1"))

    assert result["status"] == "answered"
    assert "资源服务器" in result["answer"]
    citation = result["citations"][0]
    assert citation["document_id"] == "doc-oauth"
    assert citation["source_uri"] == "kb://tenant-a/oauth"
    assert citation["version"] == 4
    assert citation["updated_at"].endswith("+00:00")


def test_twenty_four_question_evaluation_reaches_all_gates() -> None:
    """统一 Gold 集同时验收答案、证据 Recall@K 和无答案检测。"""

    report = evaluate_retrieval(build_chain())

    assert report["question_count"] == 24
    assert report["accuracy"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["no_answer_accuracy"] == 1.0


def test_no_relevant_evidence_abstains_instead_of_forcing_an_answer() -> None:
    """弱词重合不能把无关文档强行变成答案。"""

    result = build_chain().answer("火星办公室午餐价格是多少？", AuthContext("tenant-a", "user-1"))

    assert result["status"] == "no_evidence"
    assert result["citations"] == []


def test_conflicting_evidence_prefers_current_authoritative_policy() -> None:
    """正式 30 天策略必须压过未审批的 60 天建议，过期的 90 天策略不得出现。"""

    result = build_chain().answer("现行生产日志默认保留多少天？", AuthContext("tenant-a", "user-1"))

    assert result["citations"][0]["document_id"] == "doc-retention-current"
    assert "30 天" in result["answer"]
    assert "90 天" not in result["answer"]


def test_expired_and_deleted_documents_never_enter_candidates() -> None:
    """过期与 tombstone 在召回前过滤，不能依赖模型自行忽略。"""

    chain = build_chain()
    expired = chain.answer("旧版日志策略保留 90 天吗？", AuthContext("tenant-a", "user-1"))
    deleted = chain.answer("vpn-old.internal.example 是什么？", AuthContext("tenant-a", "user-1"))

    assert all(citation["document_id"] != "doc-retention-expired" for citation in expired["citations"])
    assert deleted["status"] == "no_evidence"


def test_double_tenant_and_document_acl_prevent_horizontal_access() -> None:
    """tenant 过滤和文档 ACL 是两个独立门，猜中 ID/关键词也不能绕过。"""

    chain = build_chain()
    tenant_b = chain.answer("COBALT-SECRET 是什么？", AuthContext("tenant-a", "user-1"))
    private_doc = chain.answer("财务审批阈值是多少？", AuthContext("tenant-a", "user-1"))

    assert tenant_b["status"] == "no_evidence"
    assert private_doc["status"] == "no_evidence"
    assert "COBALT-SECRET" not in tenant_b["answer"]
    assert "50000" not in private_doc["answer"]


def test_post_retrieval_acl_check_catches_permission_change() -> None:
    """证明最终授权门读取当前 ACL，而不是只相信召回时的候选快照。"""

    store = DocumentStore.from_json()
    retriever = HybridRetriever(store)
    auth = AuthContext("tenant-a", "user-1")
    candidate = retriever.search("Access Token", auth)[0].document
    # 模拟召回后、进入模型前 ACL 被撤销。
    store._documents[candidate.id] = Document(**{**candidate.__dict__, "allowed_users": ("other-user",)})

    assert store.final_authorize(candidate, auth, datetime(2026, 8, 11, tzinfo=timezone.utc)) is False


def test_long_malicious_document_is_bounded_and_treated_as_data() -> None:
    """提示注入句不进入答案，超长内容也不会突破 excerpt 上限。"""

    result = build_chain().answer("演示环境代号是什么？", AuthContext("tenant-a", "user-1"))

    assert "ORANGE" in result["answer"]
    assert "IGNORE ALL PREVIOUS" not in result["answer"]
    assert "COBALT-SECRET" not in result["answer"]
    assert len(result["answer"]) < 2500


def test_delete_propagation_removes_previously_retrievable_document() -> None:
    """运行时删除后同一查询立即不可见，不允许索引残留。"""

    store = DocumentStore.from_json()
    chain = MinimalRAGChain(HybridRetriever(store))
    auth = AuthContext("tenant-a", "user-1")
    assert chain.answer("MCP 连接什么？", auth)["status"] == "answered"

    store.delete("doc-mcp")
    after = chain.answer("MCP 连接什么？", auth)
    assert all(citation["document_id"] != "doc-mcp" for citation in after["citations"])

