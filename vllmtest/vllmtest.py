import requests
import json


def call_vllm_completion():
    """
    调用本地vLLM服务的completion API
    相当于curl命令的Python实现：
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "../../models/Qwen3-0.6B",
            "prompt": "San Francisco is a",
            "max_tokens": 7,
            "temperature": 0
        }'
    """
    url = "http://localhost:8000/v1/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": "../../models/Qwen3-0.6B",
        "prompt": "你是谁，请用中文回答",
        "max_tokens": 7,
        "temperature": 0,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常

        # 返回JSON响应
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None


if __name__ == "__main__":
    result = call_vllm_completion()
    if result:
        print("API调用成功:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("API调用失败")
