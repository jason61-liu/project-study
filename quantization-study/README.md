# Qwen3 模型量化工具

本目录包含两种 Qwen3 模型量化方案：

| 方案 | 文件 | 平台支持 | 量化类型 | 推理引擎 |
|-----|------|---------|---------|---------|
| **BitsAndBytes** | [bitsAndBytes.py](bitsAndBytes.py) | macOS / Linux | 动态量化 | Transformers |
| **AWQ** | [awq_quantize.py](awq_quantize.py) | Linux | 静态量化 | vLLM |

---

## 方案一: BitsAndBytes 量化

### 特点

- ✅ **跨平台**: macOS 和 Linux 都支持
- ✅ **安装简单**: 仅需 `bitsandbytes` 和 `transformers`
- ✅ **动态量化**: 不需要保存单独的量化模型文件
- ✅ **内存高效**: 4-bit 量化可大幅减少内存占用

### 环境要求

```bash
pip install torch transformers bitsandbytes datasets
```

### 使用方法

```bash
python bitsAndBytes.py
```

### 量化配置

**4-bit 量化 (NF4)** - 推荐用于内存受限场景:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # 双重量化
    bnb_4bit_quant_type="nf4",            # NF4 量化类型
    bnb_4bit_compute_dtype=torch.float16
)
```

**8-bit 量化** - 更好的精度:
```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```

### 内存占用对比

| 量化类型 | 内存占用 (Qwen3-4B) | 精度损失 |
|---------|-------------------|---------|
| FP16    | ~8 GB             | 无      |
| 8-bit   | ~4 GB             | 很小    |
| 4-bit   | ~2.5 GB           | 小      |

### 加载量化模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Qwen3-4B",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
```

---

## 方案二: AWQ 量化

### 特点

- ✅ **静态量化**: 生成独立的量化模型文件
- ✅ **推理高效**: 使用 vLLM 推理，性能更好
- ✅ **精度优秀**: AWQ 算法在保持精度的同时大幅压缩模型
- ❌ **平台限制**: 仅支持 Linux + CUDA

### 环境要求

```bash
# Python 3.8 - 3.11 (注意: Python 3.12 暂不支持)
pip install llmcompressor transformers datasets torch
```

### 使用方法

```bash
# 1. 修改脚本中的模型路径
MODEL_PATH = "/path/to/Qwen3-4B"
QUANT_PATH = "/path/to/Qwen3-4B-awq"

# 2. 运行量化
python awq_quantize.py
```

### AWQ 量化配置

```python
AWQModifier(
    targets=["Linear"],
    ignore=["lm_head"],
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,           # 4-bit 权重量化
                "type": "int",
                "symmetric": False,
                "strategy": "group",
                "group_size": 128,
            },
        },
    },
)
```

### 内存占用

| 模型 | FP16 | AWQ 4-bit | 压缩比 |
|-----|------|----------|-------|
| Qwen3-0.6B | ~1.2 GB | ~0.4 GB | 3x |
| Qwen3-4B | ~8 GB | ~2.3 GB | 3.5x |

### 加载 AWQ 量化模型 (使用 vLLM)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-4B-awq",
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

outputs = llm.generate("你好，请介绍一下你自己。", sampling_params)
print(outputs[0].text)
```

---

## 两种方案对比

| 特性 | BitsAndBytes | AWQ |
|-----|-------------|-----|
| **量化类型** | 动态量化 | 静态量化 |
| **模型文件** | 无需保存，加载时量化 | 生成独立的量化文件 |
| **推理引擎** | Transformers | vLLM |
| **macOS 支持** | ✅ | ❌ |
| **Linux 支持** | ✅ | ✅ |
| **推理速度** | 中等 | 快 |
| **内存占用** | 低 | 更低 |
| **精度损失** | 小 | 很小 |
| **部署复杂度** | 简单 | 中等 |

---

## 如何选择？

### 选择 BitsAndBytes 如果你：

- 使用 macOS 进行开发
- 需要快速实验不同量化配置
- 不想保存额外的量化模型文件
- 使用 Transformers 进行推理

### 选择 AWQ 如果你：

- 使用 Linux + NVIDIA GPU
- 需要最佳推理性能 (使用 vLLM)
- 需要部署量化模型到生产环境
- 对模型精度有更高要求

---

## 依赖安装

### BitsAndBytes

```bash
pip install -r requirements.txt
```

内容:
```
torch
transformers
bitsandbytes
datasets
```

### AWQ

```bash
pip install -r requirements-awq.txt
```

内容:
```
llmcompressor
transformers
datasets
torch
vllm  # 用于推理
```

---

## 注意事项

1. **BitsAndBytes** 是动态量化，每次加载模型时都会进行量化，无需提前保存
2. **AWQ** 是静态量化，需要运行一次量化过程生成量化模型文件
3. AWQ 量化过程需要较大 GPU 显存，建议 >= 16GB
4. 在 macOS 上，BitsAndBytes 会使用 MPS (Metal Performance Shaders) 加速

---

## 相关文档

- [BitsAndBytes 官方文档](https://huggingface.co/docs/bitsandbytes/)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [llm-compressor 文档](https://docs.vllm.ai/projects/llm-compressor/)
- [vLLM 文档](https://docs.vllm.ai/)
