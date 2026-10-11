"""CLI entry point for the Multi-Agent Code Review Desk."""
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
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="OUTPUT",
        help="Path for the JSON report output file (default: <file>.review.json).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all human-in-the-loop CRITICAL escalations (non-interactive).",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Suppress generated test code from terminal output.",
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
    # Run the orchestrator
    # ------------------------------------------------------------------
    from src.orchestrator import Orchestrator
    from src.conflict_resolver import ConflictResolver, persist_report
    from src.report_renderer import ReportRenderer

    orchestrator = Orchestrator()
    print("[INFO] Running agents in parallel …")
    report = orchestrator.run(code)

    # ------------------------------------------------------------------
    # Resolve conflicts (with human-in-the-loop if needed)
    # ------------------------------------------------------------------
    resolver = ConflictResolver(auto_approve=args.auto_approve)
    report = resolver.resolve(report)

    if resolver.resolution_records:
        print(f"[INFO] Resolved {len(resolver.resolution_records)} conflict(s).")
        for rec in resolver.resolution_records:
            print(
                f"  {rec.finding_key}: {rec.original_severity} → "
                f"{rec.resolved_severity}  ({rec.note})"
            )

    # ------------------------------------------------------------------
    # Pretty-print the report
    # ------------------------------------------------------------------
    renderer = ReportRenderer(show_tests=not args.no_tests)
    renderer.render(report)

    # ------------------------------------------------------------------
    # Persist JSON report
    # ------------------------------------------------------------------
    output_path: Path = args.output or path.with_suffix(".review.json")
    persist_report(report, output_path)
    print(f"[INFO] JSON report written to: {output_path}")

    # Exit code: 0=PASS, 1=WARN/FAIL
    from src.orchestrator import Verdict
    return 0 if report.verdict == Verdict.PASS else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
