"""Run the Phase 6 real-video feature benchmark and schema acceptance gate."""

import sys

from mabe.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "build-features")
    main()
