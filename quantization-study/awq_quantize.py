#!/usr/bin/env python3
"""
AWQ量化脚本 - 使用 llm-compressor 对 Qwen3-4B 模型进行 AWQ 量化
适用于 Linux 环境（需安装 llmcompressor 和 vLLM）
"""

import os
import json
import tempfile
import torch
from transformers import AutoTokenizer, AutoConfig
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot

# 模型路径配置
MODEL_PATH = "/mnt/hgfs/vm-share/models/Qwen/Qwen3-4B"
QUANT_PATH = "/mnt/hgfs/vm-share/models/Qwen/Qwen3-4B-awq"

# AWQ量化配置
W_BIT = 4
GROUP_SIZE = 128

# 校准数据集
CALIBRATION_DATASET = [
    "你好，请介绍一下你自己。",
    "今天天气怎么样？",
    "如何学习编程？",
    "请解释什么是人工智能。",
    "写一个Python函数来计算斐波那契数列。",
    "什么是机器学习？",
    "介绍一下量子计算。",
    "如何优化神经网络性能？",
    "请推荐几本学习Python的书籍。",
    "什么是深度学习？",
]


def print_system_info():
    print("\n" + "=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)


def print_original_model_info():
    print("\n" + "=" * 60)
    print("原始模型信息")
    print("=" * 60)

    # 只读取配置文件而不加载整个模型
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 估算参数量 (Qwen3-4B 约 4B 参数)
    total_params = config.num_hidden_layers * config.hidden_size * config.num_attention_heads * 4 + config.vocab_size * config.hidden_size

    size_fp16_mb = total_params * 2 / 1024 / 1024
    size_int4_mb = total_params * 0.5 / 1024 / 1024

    print(f"\n模型路径: {MODEL_PATH}")
    print(f"模型架构: {config.architectures}")
    print(f"估计参数量: {total_params:,}")
    print(f"FP16模型大小: {size_fp16_mb:.2f} MB ({size_fp16_mb/1024:.2f} GB)")
    print(f"AWQ 4-bit后大小: {size_int4_mb:.2f} MB ({size_int4_mb/1024:.2f} GB)")
    print(f"预期压缩比: {size_fp16_mb/size_int4_mb:.2f}x")

    print("=" * 60)


def quantize_model():
    print(f"\n开始 AWQ 量化 (使用 llm-compressor)...")
    print(f"源模型路径: {MODEL_PATH}")
    print(f"量化模型保存路径: {QUANT_PATH}")
    print(f"量化配置: W{W_BIT}A16, group_size={GROUP_SIZE}")

    # 新版 llmcompressor 使用 config_groups 配置量化参数
    awq_modifier = AWQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": W_BIT,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": GROUP_SIZE,
                },
            },
        },
    )

    # 将校准数据集保存为临时 JSON 文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for text in CALIBRATION_DATASET:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        temp_dataset_path = f.name

    print("\n正在执行 AWQ 量化（可能需要几分钟）...")
    try:
        oneshot(
            model=MODEL_PATH,
            dataset_path=temp_dataset_path,
            recipe=[awq_modifier],
            output_dir=QUANT_PATH,
            num_calibration_samples=len(CALIBRATION_DATASET),
            max_seq_length=512,
            precision="float16",
            trust_remote_code_model=True,
        )
    finally:
        # 删除临时文件
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(QUANT_PATH)

    # 保存说明文件
    config_info = f"""# AWQ量化配置信息 (使用 llm-compressor)

## 源模型
{MODEL_PATH}

## 量化模型路径
{QUANT_PATH}

## 量化配置
- 权重量化位数: {W_BIT}-bit
- 激活: 16-bit (FP16)
- 分组大小: {GROUP_SIZE}
- 忽略层: lm_head

## 使用方法（必须使用 vLLM）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{QUANT_PATH}", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate("你好，请介绍一下你自己。", sampling_params)
print(outputs[0].texts)
```
"""
    with open(os.path.join(QUANT_PATH, "README_QUANT.md"), "w", encoding="utf-8") as f:
        f.write(config_info)

    print("\n" + "=" * 60)
    print("AWQ 量化完成!")
    print(f"量化模型已保存至: {QUANT_PATH}")
    print("=" * 60)


def main():
    try:
        print_system_info()
        print_original_model_info()
        quantize_model()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()