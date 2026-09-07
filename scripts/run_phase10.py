from __future__ import annotations

import sys

from mabe.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "experiment-temporal")
    main()
