# AWQ量化说明 (Linux环境)

这个脚本用于在Linux环境下使用AWQ对Qwen3-4B模型进行量化。

## 环境要求

- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Python**: 3.8 - 3.11 (注意：Python 3.12暂不支持)
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 11.8 or 12.x
- **显存**: 建议 >= 16GB

## 安装依赖

```bash
# 创建虚拟环境
python3 -m venv awq_env
source awq_env/bin/activate

# 安装依赖
pip install -r requirements-awq.txt
```

或者使用uv：

```bash
source /path/to/venv/bin/activate
uv pip install -r requirements-awq.txt
```

## 使用方法

### 1. 修改路径

编辑 `awq_quantize.py`，修改模型路径：

```python
MODEL_PATH = "/path/to/your/Qwen3-4B"
QUANT_PATH = "/path/to/your/Qwen3-4B-awq"
```

### 2. 运行量化

```bash
python awq_quantize.py
```

### 3. 按提示操作

脚本会询问：
- 是否显示原始模型信息
- 确认开始量化
- 是否测试量化后的模型

## AWQ量化配置

默认配置：

```python
QUANT_CONFIG = {
    "zero_point": True,      # 使用零点量化
    "q_group_size": 128,     # 分组大小
    "w_bit": 4,              # 4-bit量化
    "version": "GEMM"        # GEMM版本（推理更快）
}
```

## 校准数据

脚本使用10条中文文本进行校准，你可以根据需要修改 `CALIBRATION_DATASET` 列表。

## 内存占用

- 原始模型 (FP16): ~8 GB
- AWQ量化后: ~2-3 GB
- 显存需求: ~16 GB (量化过程)

## 加载量化模型

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "/path/to/Qwen3-4B-awq",
    device_map="auto",
    safetensors=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/Qwen3-4B-awq",
    trust_remote_code=True
)

# 推理
prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 常见问题

### 1. ImportError: No module named 'awq'

确保已安装autoawq：
```bash
pip install autoawq
```

### 2. CUDA out of memory

- 减小校准数据集大小
- 使用更小的batch_size
- 确保有足够的GPU显存

### 3. Python 3.12不兼容

AWQ目前不支持Python 3.12，请使用Python 3.8-3.11。

## AWQ vs 其他量化方法

| 方法 | 精度 | 速度 | macOS支持 | Linux支持 |
|-----|-----|-----|----------|----------|
| AWQ | 高 | 快 | ❌ | ✅ |
| GPTQ | 高 | 快 | ❌ | ✅ |
| BitsAndBytes | 中 | 中 | ✅ | ✅ |

## 输出文件

量化完成后，目标目录将包含：

- `model.safetensors` - 量化后的模型权重
- `config.json` - 模型配置
- `quantization_config.json` - 量化配置
- `tokenizer_*` - tokenizer文件
- `quantization_info.txt` - 量化信息文档
