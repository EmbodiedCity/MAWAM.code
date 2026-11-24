"""CLI entry point for `python3 -m ma_task_harness`."""

import sys
from ma_task_harness.orchestrator.cli import main

if __name__ == "__main__":
    sys.exit(main())
