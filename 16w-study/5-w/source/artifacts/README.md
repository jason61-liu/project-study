# 实验产物状态

正式 `deepseek-v4-pro` 对比尚未在当前进程运行，因为没有检测到 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。

为避免把测试替身结果冒充真实模型指标，本目录暂不生成 `architecture-comparison.json/.md`。按照上级 README 注入环境变量并执行：

```bash
python run_experiment.py --repeats 3
```

成功后会在本目录生成架构对比表、分场景结果、推荐决策树和 36 条逐轨迹数据。

