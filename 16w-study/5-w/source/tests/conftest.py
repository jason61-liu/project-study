"""让测试直接导入 source 下的教学模块。"""

import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

