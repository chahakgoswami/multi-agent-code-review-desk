"""ReportRenderer: pretty-prints a CodeReviewReport to the terminal using colorama."""
from __future__ import annotations

import sys
from typing import IO, List, Optional

try:
    from colorama import Back, Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLORAMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _COLORAMA_AVAILABLE = False

    class _Stub:  # type: ignore[no-redef]
        """No-op stub when colorama is not installed."""
        def __getattr__(self, _: str) -> str:
            return ""

    Fore = Back = Style = _Stub()  # type: ignore[assignment]

from src.orchestrator import (
    CodeReviewReport,
    FindingVote,
    UnifiedSeverity,
    Verdict,
)


# ---------------------------------------------------------------------------
# Colour mappings
# ---------------------------------------------------------------------------

_SEVERITY_COLOUR = {
    UnifiedSeverity.INFO: Fore.CYAN,
    UnifiedSeverity.LOW: Fore.GREEN,
    UnifiedSeverity.MEDIUM: Fore.YELLOW,
    UnifiedSeverity.WARNING: Fore.YELLOW,
    UnifiedSeverity.HIGH: Fore.RED,
    UnifiedSeverity.ERROR: Fore.RED,
    UnifiedSeverity.CRITICAL: Fore.MAGENTA,
}

_VERDICT_COLOUR = {
    Verdict.PASS: Fore.GREEN,
    Verdict.WARN: Fore.YELLOW,
    Verdict.FAIL: Fore.RED,
}

_VERDICT_ICON = {
    Verdict.PASS: "✓ PASS",
    Verdict.WARN: "⚠ WARN",
    Verdict.FAIL: "✗ FAIL",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bold(text: str) -> str:
    return f"{Style.BRIGHT}{text}{Style.RESET_ALL}"


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{Style.RESET_ALL}"


def _severity_label(severity: UnifiedSeverity) -> str:
    colour = _SEVERITY_COLOUR.get(severity, "")
    label = severity.value.upper().ljust(8)
    return _colour(_bold(label), colour)


# ---------------------------------------------------------------------------
# ReportRenderer
# ---------------------------------------------------------------------------


class ReportRenderer:
    """Render a ``CodeReviewReport`` as coloured terminal output.

    Parameters
    ----------
    stream:
        Output stream; defaults to ``sys.stdout``.
    width:
        Column width for dividers; defaults to 70.
    show_tests:
        Whether to print the generated test code at the end.
    """

    def __init__(
        self,
        stream: Optional[IO[str]] = None,
        width: int = 70,
        show_tests: bool = True,
    ) -> None:
        self._stream = stream or sys.stdout
        self._width = width
        self._show_tests = show_tests

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def render(self, report: CodeReviewReport) -> None:
        """Print the full report to *stream*."""
        self._divider("=")
        self._render_header(report)
        self._divider("-")
        self._render_statistics(report)
        self._divider("-")
        self._render_agent_confidence(report)
        self._divider("-")
        self._render_votes(report)
        if report.has_conflicts:
            self._divider("-")
            self._render_conflicts(report)
        if self._show_tests and report.test_result is not None:
            self._divider("-")
            self._render_tests(report)
        self._divider("=")

    # ------------------------------------------------------------------ #
    # Section renderers
    # ------------------------------------------------------------------ #

    def _render_header(self, report: CodeReviewReport) -> None:
        verdict_colour = _VERDICT_COLOUR.get(report.verdict, "")
        verdict_text = _VERDICT_ICON.get(report.verdict, report.verdict.value)
        title = _bold("Multi-Agent Code Review Desk — Final Report")
        self._print(title)
        self._print(
            f"  Verdict : {_colour(_bold(verdict_text), verdict_colour)}"
        )
        self._print(f"  Elapsed : {report.elapsed_seconds:.2f}s")
        self._print(f"  Summary : {report.summary}")

    def _render_statistics(self, report: CodeReviewReport) -> None:
        self._print(_bold("Statistics"))
        self._print(f"  Total raw findings  : {len(report.normalised_findings)}")
        self._print(f"  Merged vote groups  : {len(report.votes)}")
        self._print(
            f"  Critical findings   : "
            + _colour(str(report.critical_count), Fore.MAGENTA)
        )
        self._print(
            f"  High / Error        : "
            + _colour(str(report.high_or_error_count), Fore.RED)
        )
        self._print(
            f"  Conflicts detected  : "
            + (
                _colour(str(len(report.conflict_votes)), Fore.YELLOW)
                if report.has_conflicts
                else _colour("0", Fore.GREEN)
            )
        )

    def _render_agent_confidence(self, report: CodeReviewReport) -> None:
        self._print(_bold("Agent Confidence Scores"))
        agents = [
            ("ReviewerAgent", report.reviewer_confidence),
            ("SecurityScannerAgent", report.security_confidence),
            ("TestWriterAgent", report.test_writer_confidence),
        ]
        for name, conf in agents:
            bar = self._confidence_bar(conf)
            self._print(f"  {name:<22} {bar} {conf:.2%}")

    def _render_votes(self, report: CodeReviewReport) -> None:
        self._print(_bold(f"Merged Findings ({len(report.votes)})"))
        if not report.votes:
            self._print(_colour("  No findings.", Fore.GREEN))
            return

        for vote in report.votes:
            self._render_single_vote(vote)

    def _render_single_vote(self, vote: FindingVote) -> None:
        sev_label = _severity_label(vote.agreed_severity)
        conflict_tag = (
            _colour(" [CONFLICT]", Fore.YELLOW) if vote.severity_conflict else ""
        )
        agents_str = ", ".join(
            _colour(c.source_agent, Fore.CYAN) for c in vote.contributors
        )
        self._print(
            f"  {sev_label}  {_bold(vote.finding_key)}{conflict_tag}"
        )
        # Title from first contributor
        if vote.contributors:
            title = vote.contributors[0].title[:80]
            self._print(f"    {title}")
        self._print(
            f"    votes={vote.vote_count}  "
            f"confidence={vote.max_confidence:.2f}  "
            f"agents=[{agents_str}]"
        )
        # Show line numbers if available
        lines = [c.line for c in vote.contributors if c.line is not None]
        if lines:
            self._print(f"    lines: {', '.join(str(ln) for ln in sorted(set(lines)))}")
        self._print("")

    def _render_conflicts(self, report: CodeReviewReport) -> None:
        self._print(
            _bold(_colour(f"Conflicts Requiring Attention ({len(report.conflict_votes)})", Fore.YELLOW))
        )
        for vote in report.conflict_votes:
            severities = sorted(
                set(c.severity.value for c in vote.contributors)
            )
            agents = [(c.source_agent, c.severity.value) for c in vote.contributors]
            self._print(f"  {_bold(vote.finding_key)}")
            self._print(f"    Severity disagreement: {', '.join(severities)}")
            for agent, sev in agents:
                self._print(
                    f"      - {_colour(agent, Fore.CYAN)}: {sev}"
                )
            self._print("")

    def _render_tests(self, report: CodeReviewReport) -> None:
        assert report.test_result is not None
        tr = report.test_result
        self._print(_bold("Generated Test Stubs"))
        self._print(f"  Functions found : {tr.function_count}")
        self._print(f"  Tests generated : {tr.test_count}")
        self._print(f"  Coverage est.   : {tr.coverage_estimate:.1%}")
        if tr.parse_error:
            self._print(
                _colour(f"  Parse error: {tr.parse_error}", Fore.RED)
            )
        else:
            self._print("")
            # Print first 30 lines of generated code
            code_lines = tr.generated_code.splitlines()
            max_lines = 30
            for line in code_lines[:max_lines]:
                self._print(f"  {_colour(line, Fore.CYAN)}")
            if len(code_lines) > max_lines:
                remaining = len(code_lines) - max_lines
                self._print(
                    _colour(f"  ... ({remaining} more lines omitted)", Fore.YELLOW)
                )
        self._print("")

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _print(self, text: str = "") -> None:
        print(text, file=self._stream)

    def _divider(self, char: str = "-") -> None:
        self._print(_colour(char * self._width, Fore.BLUE))

    @staticmethod
    def _confidence_bar(value: float, width: int = 20) -> str:
        """Render a simple ASCII progress bar for a 0.0-1.0 value."""
        filled = int(round(value * width))
        empty = width - filled
        bar = "█" * filled + "░" * empty
        if value >= 0.8:
            colour = Fore.GREEN
        elif value >= 0.5:
            colour = Fore.YELLOW
        else:
            colour = Fore.RED
        return _colour(f"[{bar}]", colour)
