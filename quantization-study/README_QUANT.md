# AWQ量化配置信息 (使用 llm-compressor)

## 源模型
/mnt/hgfs/vm-share/models/Qwen/Qwen3-0.6B

## 量化模型路径
/mnt/hgfs/vm-share/models/Qwen/Qwen3-0.6B-awq

## 量化配置
- 权重量化位数: 4-bit
- 激活: 16-bit (FP16)
- 分组大小: 128
- 忽略层: lm_head

## 使用方法（必须使用 vLLM）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/mnt/hgfs/vm-share/models/Qwen/Qwen3-0.6B-awq", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate("你好，请介绍一下你自己。", sampling_params)
print(outputs[0].texts)
```
