"""
Author: liushiyi liushiyi2013@163.com
Date: 2025-10-01 10:50:48
LastEditors: liushiyi liushiyi2013@163.com
LastEditTime: 2025-10-01 10:58:10
FilePath: /pyproject/vllmtest/offline.py
Description:

Copyright (c) 2025 , All Rights Reserved.
"""

from vllm import LLM, SamplingParams


def main():
    # 使用聊天格式输入（适用于 Qwen 等支持 chat template 的模型）
    # 将聊天格式转换为字符串
    chat_messages = [
        {"role": "system", "content": "你是个友善的AI助手。"},
        {"role": "user", "content": "模仿李白风格写一首窗外的古诗"},
    ]

    # 使用LLM的chat template将消息转换为字符串
    llm = LLM(
        model="../../githubproject/models/Qwen3-0.6B",
        max_model_len=30720,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=65535,  # 可选：限制批处理 token 总数
    )

    # 将聊天消息转换为字符串格式
    prompt = llm.get_tokenizer().apply_chat_template(chat_messages, tokenize=False)

    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    # 生成输出
    outputs = llm.generate(prompt, sampling_params)

    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
