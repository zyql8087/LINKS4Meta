from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference.inference_inverse import *  # noqa: F401,F403,E402
from scripts.inference.inference_inverse import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
