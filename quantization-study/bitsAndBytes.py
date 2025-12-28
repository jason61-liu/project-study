#!/usr/bin/env python3
"""
BitsAndBytes量化脚本 - 对Qwen3-4B模型进行BitsAndBytes量化
注意: BitsAndBytes量化是动态的，不需要单独保存量化模型文件
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model_size_info(model):
    """
    获取模型大小信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算不同精度下的模型大小
    size_fp32_mb = total_params * 4 / 1024 / 1024
    size_fp16_mb = total_params * 2 / 1024 / 1024
    size_int8_mb = total_params * 1 / 1024 / 1024
    size_int4_mb = total_params * 0.5 / 1024 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_fp32_mb": size_fp32_mb,
        "size_fp16_mb": size_fp16_mb,
        "size_int8_mb": size_int8_mb,
        "size_int4_mb": size_int4_mb,
    }


def print_original_model_info():
    """
    打印原始模型信息（量化前）
    """
    print("\n" + "=" * 60)
    print("原始模型信息 (量化前)")
    print("=" * 60)

    print("\n正在加载原始模型以获取信息...")

    # 加载原始模型（FP16）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    info = get_model_size_info(model)

    print(f"\n模型路径: {MODEL_PATH}")
    print(f"总参数量: {info['total_params']:,}")
    print(f"可训练参数量: {info['trainable_params']:,}")
    print(f"\n不同精度下的模型大小:")
    print(
        f"  - FP32: {info['size_fp32_mb']:.2f} MB ({info['size_fp32_mb']/1024:.2f} GB)"
    )
    print(
        f"  - FP16: {info['size_fp16_mb']:.2f} MB ({info['size_fp16_mb']/1024:.2f} GB)"
    )
    print(
        f"  - INT8: {info['size_int8_mb']:.2f} MB ({info['size_int8_mb']/1024:.2f} GB)"
    )
    print(
        f"  - INT4: {info['size_int4_mb']:.2f} MB ({info['size_int4_mb']/1024:.2f} GB)"
    )

    print(f"\n内存节省:")
    print(f"  - 8-bit vs FP16: {info['size_fp16_mb']/info['size_int8_mb']:.2f}x 压缩")
    print(f"  - 4-bit vs FP16: {info['size_fp16_mb']/info['size_int4_mb']:.2f}x 压缩")

    # 内存使用情况
    if torch.cuda.is_available():
        print(f"\nGPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"\n使用 MPS (Apple Silicon) 加速")

    print("=" * 60)

    # 清理模型以释放内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return info


# 模型路径配置
MODEL_PATH = "/Users/shiyiliu/workspace/vm-share/models/Qwen/Qwen3-4B"
QUANT_CONFIG_PATH = "//Users/shiyiliu/workspace/vm-share/models/Qwen/Qwen3-4B-quan"

# BitsAndBytes量化配置
BNB_4BIT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

BNB_8BIT_CONFIG = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)


def load_and_quantize_model_4bit():
    """
    加载模型并使用4-bit量化
    """
    print(f"正在加载模型并进行4-bit量化...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"量化配置: NF4, double_quant=True")

    # 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载模型并应用4-bit量化
    print("\n正在加载模型并应用4-bit BitsAndBytes量化...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BNB_4BIT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    print("\n✓ 4-bit量化模型加载完成！")
    print(f"模型设备: {model.device}")

    return model, tokenizer


def load_and_quantize_model_8bit():
    """
    加载模型并使用8-bit量化
    """
    print(f"正在加载模型并进行8-bit量化...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"量化配置: 8-bit, threshold=6.0")

    # 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载模型并应用8-bit量化
    print("\n正在加载模型并应用8-bit BitsAndBytes量化...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BNB_8BIT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    print("\n✓ 8-bit量化模型加载完成！")
    print(f"模型设备: {model.device}")

    return model, tokenizer


def test_quantized_model(model, tokenizer, quant_type="4-bit"):
    """
    测试量化后的模型
    """
    print(f"\n正在测试{quant_type}量化模型...")

    # 测试推理
    prompt = "你好，请介绍一下你自己。"
    print(f"\n测试提示词: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n模型回复: {response}")


def save_quant_config(quant_type="4bit"):
    """
    保存量化配置信息（BitsAndBytes是动态量化，不需要保存模型文件）
    """
    os.makedirs(QUANT_CONFIG_PATH, exist_ok=True)

    config_info = f"""
# BitsAndBytes量化配置信息

## 模型源路径
{MODEL_PATH}

## 量化类型
{quant_type}

## 量化配置
"""

    if quant_type == "4bit":
        config_info += f"""
- load_in_4bit: True
- bnb_4bit_use_double_quant: True
- bnb_4bit_quant_type: nf4
- bnb_4bit_compute_dtype: float16
"""
    else:
        config_info += f"""
- load_in_8bit: True
- llm_int8_threshold: 6.0
"""

    config_info += """
## 使用方法

BitsAndBytes是动态量化，每次加载模型时需要指定量化配置：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "原始模型路径",
    quantization_config=quantization_config,
    device_map="auto"
)
```
"""

    with open(
        os.path.join(QUANT_CONFIG_PATH, "quantization_config.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(config_info)

    print(f"\n✓ 量化配置已保存到: {QUANT_CONFIG_PATH}")


def print_model_info(model, quant_type="4-bit"):
    """
    打印量化后的模型信息
    """
    print("\n" + "=" * 60)
    print(f"量化后模型信息 ({quant_type})")
    print("=" * 60)

    info = get_model_size_info(model)

    print(f"\n模型路径: {MODEL_PATH}")
    print(f"总参数量: {info['total_params']:,}")
    print(f"可训练参数量: {info['trainable_params']:,}")

    # 根据量化类型显示不同的信息
    if "4" in quant_type:
        actual_size = info["size_int4_mb"]
        baseline_size = info["size_fp16_mb"]
        compression = baseline_size / actual_size
        print(f"\n当前配置: 4-bit NF4 量化")
        print(f"理论模型大小: {actual_size:.2f} MB ({actual_size/1024:.2f} GB)")
        print(f"相比 FP16 压缩比: {compression:.2f}x")
        print(f"内存节省: {(1 - 1/compression)*100:.1f}%")
    else:  # 8-bit
        actual_size = info["size_int8_mb"]
        baseline_size = info["size_fp16_mb"]
        compression = baseline_size / actual_size
        print(f"\n当前配置: 8-bit 量化")
        print(f"理论模型大小: {actual_size:.2f} MB ({actual_size/1024:.2f} GB)")
        print(f"相比 FP16 压缩比: {compression:.2f}x")
        print(f"内存节省: {(1 - 1/compression)*100:.1f}%")

    # 设备信息
    print(f"\n模型设备: {model.device if hasattr(model, 'device') else 'auto'}")

    # 内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU内存使用:")
        print(f"  - 已分配: {allocated:.2f} GB")
        print(f"  - 已保留: {reserved:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"\n使用 MPS (Apple Silicon) 加速")

    print("=" * 60)


if __name__ == "__main__":
    # 检查源模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 源模型路径不存在: {MODEL_PATH}")
        exit(1)

    print("=" * 60)
    print("BitsAndBytes量化工具 - Qwen3-4B")
    print("=" * 60)

    # 先显示原始模型信息
    show_original = input("\n是否显示原始模型信息? (y/n): ").strip().lower()
    if show_original == "y":
        original_info = print_original_model_info()
        print("\n")

    print("请选择量化类型:")
    print("1. 4-bit量化 (NF4, 推荐用于内存受限场景)")
    print("2. 8-bit量化 (更好的精度)")

    choice = input("\n请输入选择 (1/2): ").strip()

    if choice == "1":
        # 4-bit量化
        model, tokenizer = load_and_quantize_model_4bit()
        print_model_info(model, "4-bit")
        save_quant_config("4bit")

        test_input = input("\n是否测试量化后的模型? (y/n): ").strip().lower()
        if test_input == "y":
            test_quantized_model(model, tokenizer, "4-bit")

    elif choice == "2":
        # 8-bit量化
        model, tokenizer = load_and_quantize_model_8bit()
        print_model_info(model, "8-bit")
        save_quant_config("8bit")

        test_input = input("\n是否测试量化后的模型? (y/n): ").strip().lower()
        if test_input == "y":
            test_quantized_model(model, tokenizer, "8-bit")

    else:
        print("无效的选择！")
        exit(1)
