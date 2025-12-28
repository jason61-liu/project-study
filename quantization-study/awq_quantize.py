#!/usr/bin/env python3
"""
AWQ量化脚本 - 对Qwen3-4B模型进行AWQ量化
适用于Linux环境
"""

import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 模型路径配置
MODEL_PATH = "/Users/shiyiliu/workspace/githubproject/github-study/models/Qwen/Qwen3-4B"
QUANT_PATH = "/Users/shiyiliu/workspace/githubproject/github-study/models/Qwen/Qwen3-4B-awq"

# AWQ量化配置
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 用于量化的校准数据
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
    "什么是深度学习？"
]


def print_system_info():
    """
    打印系统信息
    """
    print("\n" + "="*60)
    print("系统信息")
    print("="*60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*60)


def print_original_model_info():
    """
    打印原始模型信息
    """
    print("\n" + "="*60)
    print("原始模型信息")
    print("="*60)

    from transformers import AutoModelForCausalLM

    print("\n正在加载原始模型以获取信息...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    size_fp16_mb = total_params * 2 / 1024 / 1024
    size_int4_mb = total_params * 0.5 / 1024 / 1024

    print(f"\n模型路径: {MODEL_PATH}")
    print(f"总参数量: {total_params:,}")
    print(f"FP16模型大小: {size_fp16_mb:.2f} MB ({size_fp16_mb/1024:.2f} GB)")
    print(f"AWQ 4-bit后大小: {size_int4_mb:.2f} MB ({size_int4_mb/1024:.2f} GB)")
    print(f"预期压缩比: {size_fp16_mb/size_int4_mb:.2f}x")

    if torch.cuda.is_available():
        print(f"\nGPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("="*60)

    del model
    torch.cuda.empty_cache()


def quantize_model():
    """
    使用AWQ算法量化模型
    """
    print(f"\n开始AWQ量化...")
    print(f"源模型路径: {MODEL_PATH}")
    print(f"量化模型保存路径: {QUANT_PATH}")
    print(f"量化配置: {QUANT_CONFIG}")

    # 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # 创建量化模型
    print("\n正在创建量化模型...")
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        safetensors=True,
        trust_remote_code=True
    )

    # 量化模型
    print("\n正在执行AWQ量化...")
    print(f"使用 {len(CALIBRATION_DATASET)} 条校准数据...")

    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=CALIBRATION_DATASET
    )

    # 保存量化模型
    print(f"\n正在保存量化模型到: {QUANT_PATH}")
    model.save_quantized(QUANT_PATH)
    tokenizer.save_pretrained(QUANT_PATH)

    # 保存量化配置信息
    config_info = f"""
# AWQ量化配置信息

## 模型源路径
{MODEL_PATH}

## 量化模型保存路径
{QUANT_PATH}

## 量化配置
- zero_point: {QUANT_CONFIG['zero_point']}
- q_group_size: {QUANT_CONFIG['q_group_size']}
- w_bit: {QUANT_CONFIG['w_bit']}
- version: {QUANT_CONFIG['version']}

## 使用方法

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "{QUANT_PATH}",
    device_map="auto",
    safetensors=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "{QUANT_PATH}",
    trust_remote_code=True
)
```
"""

    with open(os.path.join(QUANT_PATH, "quantization_info.txt"), "w", encoding="utf-8") as f:
        f.write(config_info)

    print(f"\n✓ 量化完成！量化模型已保存到: {QUANT_PATH}")

    # 计算实际模型大小
    model_size = sum(
        os.path.getsize(os.path.join(QUANT_PATH, f))
        for f in os.listdir(QUANT_PATH)
        if os.path.isfile(os.path.join(QUANT_PATH, f))
    ) / 1024 / 1024

    print(f"量化模型文件大小: {model_size:.2f} MB")


def test_quantized_model():
    """
    测试量化后的模型
    """
    print("\n" + "="*60)
    print("测试量化模型")
    print("="*60)

    print("\n正在加载量化模型...")
    model = AutoAWQForCausalLM.from_quantized(
        QUANT_PATH,
        device_map="auto",
        safetensors=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        QUANT_PATH,
        trust_remote_code=True
    )

    # 测试推理
    prompt = "你好，请介绍一下你自己。"
    print(f"\n测试提示词: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n模型回复: {response}")
    print("="*60)


if __name__ == "__main__":
    # 检查源模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 源模型路径不存在: {MODEL_PATH}")
        exit(1)

    print("="*60)
    print("AWQ量化工具 - Qwen3-4B")
    print("="*60)

    # 打印系统信息
    print_system_info()

    # 打印原始模型信息
    show_info = input("\n是否显示原始模型信息? (y/n): ").strip().lower()
    if show_info == 'y':
        print_original_model_info()

    # 确认开始量化
    print("\n" + "="*60)
    print("量化配置")
    print("="*60)
    print(f"源模型: {MODEL_PATH}")
    print(f"目标路径: {QUANT_PATH}")
    print(f"量化位数: {QUANT_CONFIG['w_bit']}-bit")
    print(f"分组大小: {QUANT_CONFIG['q_group_size']}")
    print("="*60)

    confirm = input("\n确认开始量化? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消量化。")
        exit(0)

    # 执行量化
    quantize_model()

    # 询问是否测试
    test_input = input("\n是否测试量化后的模型? (y/n): ").strip().lower()
    if test_input == 'y':
        test_quantized_model()

    print("\n" + "="*60)
    print("量化流程完成！")
    print("="*60)
