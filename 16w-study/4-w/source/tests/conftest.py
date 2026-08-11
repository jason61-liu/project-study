"""让 pytest 直接导入相邻 source 模块，无需先构建 wheel。"""

from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

