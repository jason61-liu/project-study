# E2B Python SDK 测试

本目录使用官方通用沙箱 SDK `e2b`。

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/e2b-demo
pip install -r requirements.txt
pytest -v
```

不设置 `E2B_API_KEY` 时，安装烟雾测试会运行，真实沙箱测试会跳过。
如需运行全部测试：

```bash
export E2B_API_KEY=e2b_your_api_key
pytest -v
```

真实测试会按用例创建 `code-interpreter-v1` 或 `opencode` 沙箱，并在测试结束后销毁沙箱。

## 官方模板实测

以下版本于 2026-07-28 从真实沙箱读取。官方模板别名可能指向更新后的构建，
因此后续运行时版本可能变化。

### `code-interpreter-v1`

| 组件 | 实测版本 |
| --- | --- |
| Git | 2.47.3 |
| Python | 3.13.14 |
| NumPy | 2.3.5 |
| Pandas | 2.2.3 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.9 |
| scikit-learn | 1.6.1 |
| SymPy | 1.14.0 |
| Jupyter Server | 2.20.0 |
| IPython Kernel | 6.31.0 |

该模板已预装 Git 和完整的常用科学计算栈，适合数据分析、模型实验和 Git
仓库操作。沙箱使用 Python 3.13，而本地 `.venv` 使用 Python 3.12，安装额外的
二进制扩展包时需要确认其支持 Python 3.13 和 NumPy 2。

### `opencode`

| 组件 | 实测结果 |
| --- | --- |
| OpenCode | 1.17.13 |
| Git | 2.39.5 |
| Python | 3.11.6 |
| Node.js | 20.9.0 |
| npm | 10.1.0 |
| NumPy | 未安装 |
| Pandas | 未安装 |
| SciPy | 未安装 |
| Matplotlib | 未安装 |
| scikit-learn | 未安装 |
| SymPy | 未安装 |

该模板适合使用 OpenCode 操作代码仓库，已预装 Git、Python 和 Node.js，
但不是科学计算环境。如需同时使用 OpenCode 和科学计算包，应基于 `opencode`
构建自定义模板并预装依赖。

可分别运行模板检查：

```bash
pytest -v tests/test_e2b.py::test_code_interpreter_has_scientific_packages_and_git
pytest -v tests/test_e2b.py::test_opencode_template_tools_and_scientific_packages
```

## 创建临时沙箱

下面的命令使用官方 `base` 模板创建一个保留 15 分钟的沙箱：

```bash
export E2B_API_KEY=e2b_your_api_key
python create_sandbox.py
```

脚本不会把 API Key 传入沙箱，也不会把它写入文件。
