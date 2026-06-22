from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inverse_il.train_inverse_bio import *  # noqa: F401,F403,E402
from scripts.inverse_il.train_inverse_bio import _load_initial_il_checkpoint_if_configured  # noqa: F401,E402
from scripts.inverse_il.train_inverse_bio import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
