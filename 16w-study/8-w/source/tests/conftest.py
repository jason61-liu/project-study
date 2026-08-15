from pathlib import Path
import sys


SOURCE = Path(__file__).parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
