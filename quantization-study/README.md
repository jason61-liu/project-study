# Qwen3-4B 模型量化工具

使用 BitsAndBytes 对 Qwen3-4B 模型进行量化。

## 为什么选择 BitsAndBytes？

- ✅ **macOS 兼容性好**: 不需要 CUDA，可以在 macOS 上正常运行
- ✅ **安装简单**: 通过 pip/uv 直接安装，无需编译
- ✅ **动态量化**: 不需要保存单独的量化模型文件
- ✅ **内存高效**: 4-bit 量化可大幅减少内存占用

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.35+

## 安装依赖

```bash
# 激活虚拟环境
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate

# 使用 uv 安装依赖
uv pip install -r requirements.txt
```

## 使用方法

运行量化脚本：

```bash
python main.py
```

然后选择量化类型：
- **1**: 4-bit 量化 (NF4，推荐用于内存受限场景)
- **2**: 8-bit 量化 (更好的精度)

## 量化配置说明

### 4-bit 量化
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # 双重量化，进一步压缩
    bnb_4bit_quant_type="nf4",            # NF4 量化类型
    bnb_4bit_compute_dtype=torch.float16 # 计算精度
)
```

### 8-bit 量化
```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0  # 异常值阈值
)
```

## 模型路径配置

- **源模型**: `/Users/shiyiliu/workspace/githubproject/github-study/models/Qwen/Qwen3-4B`
- **量化配置保存**: `/Users/shiyiliu/workspace/githubproject/github-study/models/Qwen/Qwen3-4B-quan`

## 注意事项

1. BitsAndBytes 是**动态量化**，不会生成独立的量化模型文件
2. 每次加载模型时需要指定量化配置
3. 量化配置信息会保存到 `Qwen3-4B-quan/quantization_config.txt`
4. 在 macOS 上，模型会使用 MPS (Metal Performance Shaders) 加速

## 内存占用对比

| 量化类型 | 内存占用 (约) | 精度损失 |
|---------|-------------|---------|
| FP16    | ~8 GB       | 无      |
| 8-bit   | ~4 GB       | 很小    |
| 4-bit   | ~2.5 GB     | 小      |

## 加载量化模型示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 定义量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Qwen3-4B",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/Qwen3-4B",
    trust_remote_code=True
)
```
