"""pytest 路径配置：让测试直接导入相邻 source 模块，而无需先安装包。"""

import sys
from pathlib import Path


# parents[1] 指向 3-w/source。插入首位可确保测试使用工作区当前代码，而不是环境中
# 可能存在的同名已安装模块。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
