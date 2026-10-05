"""Top-level entry point so the project can be run as `python main.py <file>`."""
from __future__ import annotations

import sys
from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
