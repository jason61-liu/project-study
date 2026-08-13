"""让测试可从 7-w/source 目录外直接运行。"""

from pathlib import Path
import sys


SOURCE = Path(__file__).parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

