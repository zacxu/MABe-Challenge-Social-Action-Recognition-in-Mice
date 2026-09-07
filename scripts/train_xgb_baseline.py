"""Train and evaluate the complete Phase 7 XGBoost baseline."""

import sys

from mabe.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train-xgb-baseline")
    main()
