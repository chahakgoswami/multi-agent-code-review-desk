"""Minimal CLI entry point for the Multi-Agent Code Review Desk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-review",
        description="Multi-Agent Code Review Desk – submit a Python file for review.",
    )
    parser.add_argument(
        "file",
        metavar="FILE",
        type=Path,
        help="Path to the Python source file to review.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extra debug information.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path: Path = args.file
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 1
    if not path.is_file():
        print(f"[ERROR] Path is not a file: {path}", file=sys.stderr)
        return 1

    try:
        code = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Could not read file: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Loaded {path} ({len(code.splitlines())} lines).")

    if args.verbose:
        print("[DEBUG] File contents:")
        for i, line in enumerate(code.splitlines(), 1):
            print(f"  {i:4d} | {line}")

    # ------------------------------------------------------------------
    # Demonstrate the mock LLM backend
    # ------------------------------------------------------------------
    from src.agents.mock_llm import MockLLMBackend  # local import keeps CLI fast

    llm = MockLLMBackend()
    response = llm.complete(code)
    print("\n[MockLLM Response]")
    print(response)
    print(f"[MockLLM Hash] {llm.last_hash()}")

    print("\n[INFO] Agents will be wired in on subsequent days.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
