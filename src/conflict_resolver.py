"""ConflictResolver: weighted voting, human-in-the-loop approval, and JSON persistence."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.orchestrator import (
    CodeReviewReport,
    FindingVote,
    UnifiedSeverity,
    Verdict,
)


# ---------------------------------------------------------------------------
# Agent weight registry
# ---------------------------------------------------------------------------

# Weights must sum to ≤ 1.0 (they are used as multipliers, not normalised).
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ReviewerAgent": 0.35,
    "SecurityScannerAgent": 0.50,
    "TestWriterAgent": 0.15,
}


# ---------------------------------------------------------------------------
# Resolution outcome
# ---------------------------------------------------------------------------


@dataclass
class ResolutionRecord:
    """Records the resolution decision for a single conflicted finding."""

    finding_key: str
    original_severity: str          # severity before resolution
    resolved_severity: str          # severity after weighted vote
    weighted_scores: Dict[str, float]  # per-agent weighted scores
    human_approved: Optional[bool] = None  # None = not required
    escalated_to_critical: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Applies weighted voting to conflicted findings and handles human approval.

    Workflow for each conflicted FindingVote:
    1. Compute a weighted severity score across all contributing agents.
    2. Map the score back to a UnifiedSeverity.
    3. If the resolved severity is CRITICAL, require explicit human approval
       (interactive CLI prompt) before finalising the finding as CRITICAL.
    4. If the human declines, downgrade to HIGH.
    5. Unresolved conflicts (no clear winner after weighting) are flagged.

    Parameters
    ----------
    weights:
        Per-agent weight dict.  Defaults to ``_DEFAULT_WEIGHTS``.
    auto_approve:
        When *True* all human-in-the-loop prompts are auto-approved (useful
        for non-interactive / test environments).  Defaults to *False*.
    prompt_fn:
        Callable used to obtain human input.  Defaults to ``input()``.
        Signature: ``(message: str) -> str``.
    """

    _SEVERITY_RANK_TO_ENUM: Dict[int, UnifiedSeverity] = {
        0: UnifiedSeverity.INFO,
        1: UnifiedSeverity.LOW,
        2: UnifiedSeverity.MEDIUM,
        3: UnifiedSeverity.HIGH,
        4: UnifiedSeverity.CRITICAL,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        auto_approve: bool = False,
        prompt_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        self._auto_approve = auto_approve
        self._prompt_fn = prompt_fn or input
        self._resolution_records: List[ResolutionRecord] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve(self, report: CodeReviewReport) -> CodeReviewReport:
        """Resolve conflicts in *report* in-place and return the updated report.

        Only ``FindingVote`` entries with ``severity_conflict=True`` are
        processed.  Non-conflicted votes are left untouched.
        """
        self._resolution_records.clear()

        for vote in report.votes:
            if not vote.severity_conflict:
                continue
            record = self._resolve_vote(vote)
            self._resolution_records.append(record)

        # Re-derive the report verdict after resolution
        report.verdict = self._recompute_verdict(report)
        report.summary = self._rebuild_summary(report)
        return report

    @property
    def resolution_records(self) -> List[ResolutionRecord]:
        """Return resolution records from the most recent ``resolve()`` call."""
        return list(self._resolution_records)

    # ------------------------------------------------------------------ #
    # Core resolution
    # ------------------------------------------------------------------ #

    def _resolve_vote(self, vote: FindingVote) -> ResolutionRecord:
        """Compute weighted severity for *vote* and apply human gating if needed."""
        # --- Weighted score computation --------------------------------
        weighted_scores: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for contributor in vote.contributors:
            agent = contributor.source_agent
            weight = self._weights.get(agent, 0.10)
            rank = contributor.severity.rank
            score = weight * rank * contributor.confidence
            weighted_scores[agent] = round(score, 4)
            weighted_sum += score
            total_weight += weight

        # Normalise
        normalised = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Map back to enum: round to nearest integer rank (0-4)
        resolved_rank = max(0, min(4, round(normalised)))
        resolved_severity = self._SEVERITY_RANK_TO_ENUM[resolved_rank]

        record = ResolutionRecord(
            finding_key=vote.finding_key,
            original_severity=vote.agreed_severity.value,
            resolved_severity=resolved_severity.value,
            weighted_scores=weighted_scores,
        )

        # --- Human approval gate for CRITICAL --------------------------
        if resolved_severity == UnifiedSeverity.CRITICAL:
            approved = self._request_human_approval(vote, record)
            record.human_approved = approved
            if not approved:
                resolved_severity = UnifiedSeverity.HIGH
                record.resolved_severity = UnifiedSeverity.HIGH.value
                record.note = "Downgraded from CRITICAL to HIGH: human declined approval."
            else:
                record.escalated_to_critical = True
                record.note = "Confirmed CRITICAL by human reviewer."
        elif resolved_severity != vote.agreed_severity:
            record.note = (
                f"Severity adjusted from {vote.agreed_severity.value} "
                f"to {resolved_severity.value} via weighted voting."
            )
        else:
            record.note = "Severity unchanged after weighted voting."

        # Apply resolved severity back to the vote
        vote.agreed_severity = resolved_severity
        # Mark conflict as resolved
        vote.severity_conflict = False

        return record

    # ------------------------------------------------------------------ #
    # Human-in-the-loop
    # ------------------------------------------------------------------ #

    def _request_human_approval(self, vote: FindingVote, record: ResolutionRecord) -> bool:
        """Prompt the human to approve escalating *vote* to CRITICAL.

        Returns *True* when approved, *False* otherwise.
        In ``auto_approve`` mode always returns *True* without prompting.
        """
        if self._auto_approve:
            return True

        agents = ", ".join(c.source_agent for c in vote.contributors)
        print()
        print("=" * 60)
        print("  ⚠  HUMAN REVIEW REQUIRED  ⚠")
        print("=" * 60)
        print(f"  Finding   : {vote.finding_key}")
        print(f"  Agents    : {agents}")
        print(f"  Votes     : {vote.vote_count}")
        print(f"  Max conf. : {vote.max_confidence:.2f}")
        print(f"  Weighted scores: {record.weighted_scores}")
        print()
        print("  Weighted voting resolved this finding as CRITICAL.")
        print("  Please approve or decline marking it as CRITICAL.")
        print()

        while True:
            try:
                answer = self._prompt_fn(
                    "  Approve CRITICAL marking? [yes/no]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [WARN] Input interrupted; defaulting to 'no'.")
                return False

            if answer in ("yes", "y"):
                print("  ✓ Approved.")
                return True
            elif answer in ("no", "n"):
                print("  ✗ Declined — downgraded to HIGH.")
                return False
            else:
                print("  Please enter 'yes' or 'no'.")

    # ------------------------------------------------------------------ #
    # Verdict / summary helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recompute_verdict(report: CodeReviewReport) -> Verdict:
        """Recompute the verdict after conflict resolution."""
        if report.critical_count > 0:
            return Verdict.FAIL
        if report.high_or_error_count > 0:
            return Verdict.FAIL
        if report.has_conflicts:
            return Verdict.WARN
        medium_count = sum(
            1
            for v in report.votes
            if v.agreed_severity in (UnifiedSeverity.MEDIUM, UnifiedSeverity.WARNING)
        )
        if medium_count > 0:
            return Verdict.WARN
        return Verdict.PASS

    @staticmethod
    def _rebuild_summary(report: CodeReviewReport) -> str:
        parts = [
            f"{len(report.votes)} merged finding(s)",
            f"{report.critical_count} critical",
            f"{report.high_or_error_count} high/error",
            f"{len(report.conflict_votes)} unresolved conflict(s)",
            f"verdict={report.verdict.value}",
        ]
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def _vote_to_dict(vote: FindingVote) -> dict:
    """Serialise a FindingVote to a plain dict."""
    return {
        "finding_key": vote.finding_key,
        "agreed_severity": vote.agreed_severity.value,
        "severity_conflict": vote.severity_conflict,
        "vote_count": vote.vote_count,
        "max_confidence": round(vote.max_confidence, 4),
        "contributors": [
            {
                "source_agent": c.source_agent,
                "rule_id": c.rule_id,
                "title": c.title,
                "description": c.description,
                "severity": c.severity.value,
                "confidence": round(c.confidence, 4),
                "line": c.line,
                "context": c.context,
            }
            for c in vote.contributors
        ],
    }


def persist_report(report: CodeReviewReport, output_path: Path) -> None:
    """Serialise *report* to a JSON file at *output_path*.

    Creates parent directories as needed.  Overwrites any existing file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "verdict": report.verdict.value,
        "summary": report.summary,
        "elapsed_seconds": round(report.elapsed_seconds, 4),
        "agent_confidence": {
            "reviewer": round(report.reviewer_confidence, 4),
            "security_scanner": round(report.security_confidence, 4),
            "test_writer": round(report.test_writer_confidence, 4),
        },
        "statistics": {
            "total_findings": len(report.normalised_findings),
            "merged_votes": len(report.votes),
            "critical_count": report.critical_count,
            "high_or_error_count": report.high_or_error_count,
            "conflict_count": len(report.conflict_votes),
        },
        "votes": [_vote_to_dict(v) for v in report.votes],
        "generated_tests": (
            report.test_result.generated_code
            if report.test_result is not None
            else None
        ),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
