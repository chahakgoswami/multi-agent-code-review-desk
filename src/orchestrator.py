"""Orchestrator and ConsensusEngine for the Multi-Agent Code Review Desk."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.agents.reviewer import Finding, ReviewResult, ReviewerAgent, Severity
from src.agents.security_scanner import (
    SecurityResult,
    SecurityScannerAgent,
    VulnFinding,
    VulnSeverity,
)
from src.agents.test_writer import TestResult, TestWriterAgent


# ---------------------------------------------------------------------------
# Unified severity scale used across all agents
# ---------------------------------------------------------------------------


class UnifiedSeverity(str, Enum):
    """Severity levels that span both reviewer and security agent outputs."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    WARNING = "warning"
    HIGH = "high"
    ERROR = "error"
    CRITICAL = "critical"

    # Ordinal for comparisons
    _order: Dict[str, int]

    @property
    def rank(self) -> int:  # type: ignore[override]
        _ranks = {
            "info": 0,
            "low": 1,
            "medium": 2,
            "warning": 2,
            "high": 3,
            "error": 3,
            "critical": 4,
        }
        return _ranks[self.value]


# ---------------------------------------------------------------------------
# Normalised finding — common currency across all agents
# ---------------------------------------------------------------------------


@dataclass
class NormalisedFinding:
    """A finding expressed in a common format regardless of originating agent."""

    source_agent: str  # e.g. "ReviewerAgent"
    rule_id: str
    title: str
    description: str
    severity: UnifiedSeverity
    confidence: float
    line: Optional[int] = None
    context: Optional[str] = None

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line is not None else ""
        return (
            f"[{self.severity.value.upper()}] {self.rule_id} ({self.source_agent}){loc}: "
            f"{self.title}"
        )


# ---------------------------------------------------------------------------
# Per-finding vote record
# ---------------------------------------------------------------------------


@dataclass
class FindingVote:
    """Aggregated vote for a merged finding."""

    finding_key: str  # canonical key used for deduplication
    contributors: List[NormalisedFinding] = field(default_factory=list)
    agreed_severity: UnifiedSeverity = UnifiedSeverity.INFO
    severity_conflict: bool = False
    vote_count: int = 0
    max_confidence: float = 0.0

    def __str__(self) -> str:
        conflict_marker = " [CONFLICT]" if self.severity_conflict else ""
        agents = ", ".join(c.source_agent for c in self.contributors)
        return (
            f"  {self.finding_key}{conflict_marker} — "
            f"severity={self.agreed_severity.value}, "
            f"votes={self.vote_count}, "
            f"confidence={self.max_confidence:.2f}, "
            f"agents=[{agents}]"
        )


# ---------------------------------------------------------------------------
# Final unified report
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CodeReviewReport:
    """Unified report produced by the ConsensusEngine."""

    review_result: Optional[ReviewResult] = None
    security_result: Optional[SecurityResult] = None
    test_result: Optional[TestResult] = None

    normalised_findings: List[NormalisedFinding] = field(default_factory=list)
    votes: List[FindingVote] = field(default_factory=list)

    verdict: Verdict = Verdict.PASS
    summary: str = ""
    elapsed_seconds: float = 0.0

    # Agent confidence scores
    reviewer_confidence: float = 0.0
    security_confidence: float = 0.0
    test_writer_confidence: float = 0.0

    @property
    def has_conflicts(self) -> bool:
        return any(v.severity_conflict for v in self.votes)

    @property
    def conflict_votes(self) -> List[FindingVote]:
        return [v for v in self.votes if v.severity_conflict]

    @property
    def critical_count(self) -> int:
        return sum(
            1 for v in self.votes if v.agreed_severity == UnifiedSeverity.CRITICAL
        )

    @property
    def high_or_error_count(self) -> int:
        return sum(
            1
            for v in self.votes
            if v.agreed_severity in (UnifiedSeverity.HIGH, UnifiedSeverity.ERROR)
        )

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"CodeReviewReport — Verdict: {self.verdict.value}",
            f"  Elapsed     : {self.elapsed_seconds:.2f}s",
            f"  Findings    : {len(self.votes)} merged",
            f"  Conflicts   : {len(self.conflict_votes)}",
            f"  Critical    : {self.critical_count}",
            f"  High/Error  : {self.high_or_error_count}",
            f"  Summary     : {self.summary}",
            "  Votes:",
        ]
        for v in self.votes:
            lines.append(str(v))
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_SEVERITY_MAP_REVIEWER: Dict[str, UnifiedSeverity] = {
    Severity.INFO.value: UnifiedSeverity.INFO,
    Severity.WARNING.value: UnifiedSeverity.WARNING,
    Severity.ERROR.value: UnifiedSeverity.ERROR,
}

_SEVERITY_MAP_SECURITY: Dict[str, UnifiedSeverity] = {
    VulnSeverity.LOW.value: UnifiedSeverity.LOW,
    VulnSeverity.MEDIUM.value: UnifiedSeverity.MEDIUM,
    VulnSeverity.HIGH.value: UnifiedSeverity.HIGH,
    VulnSeverity.CRITICAL.value: UnifiedSeverity.CRITICAL,
}


def _normalise_review_finding(f: Finding) -> NormalisedFinding:
    return NormalisedFinding(
        source_agent="ReviewerAgent",
        rule_id=f.rule_id,
        title=f.message,
        description=f.message,
        severity=_SEVERITY_MAP_REVIEWER.get(f.severity.value, UnifiedSeverity.INFO),
        confidence=0.80,  # ReviewerAgent does not store per-finding confidence
        line=f.line,
        context=f.context,
    )


def _normalise_security_finding(f: VulnFinding) -> NormalisedFinding:
    return NormalisedFinding(
        source_agent="SecurityScannerAgent",
        rule_id=f.rule_id,
        title=f.title,
        description=f.description,
        severity=_SEVERITY_MAP_SECURITY.get(f.severity.value, UnifiedSeverity.INFO),
        confidence=f.confidence,
        line=f.line,
        context=f.context,
    )


def _normalise_test_finding(test_result: TestResult) -> List[NormalisedFinding]:
    """Convert TestResult into a single synthetic NormalisedFinding for coverage."""
    findings: List[NormalisedFinding] = []
    if test_result.parse_error:
        findings.append(
            NormalisedFinding(
                source_agent="TestWriterAgent",
                rule_id="T001",
                title="Syntax error prevented test generation",
                description=test_result.parse_error,
                severity=UnifiedSeverity.ERROR,
                confidence=test_result.confidence,
            )
        )
    if test_result.coverage_estimate < 0.5 and not test_result.parse_error:
        findings.append(
            NormalisedFinding(
                source_agent="TestWriterAgent",
                rule_id="T002",
                title="Low estimated test coverage",
                description=(
                    f"Estimated coverage is {test_result.coverage_estimate:.1%}; "
                    "consider adding more tests."
                ),
                severity=UnifiedSeverity.WARNING,
                confidence=test_result.confidence,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# ConsensusEngine
# ---------------------------------------------------------------------------


class ConsensusEngine:
    """Merges findings from multiple agents, detects conflicts, and assigns verdicts.

    Deduplication key: ``rule_id`` (exact match).  Findings with the same
    ``rule_id`` from different agents are merged into a single ``FindingVote``.
    When merged findings disagree on severity by more than one rank, the
    conflict flag is set and the *escalated* (higher) severity is used.
    """

    # How many rank levels difference constitutes a conflict
    CONFLICT_RANK_THRESHOLD = 1

    def build_report(
        self,
        review_result: Optional[ReviewResult],
        security_result: Optional[SecurityResult],
        test_result: Optional[TestResult],
        elapsed: float = 0.0,
    ) -> CodeReviewReport:
        report = CodeReviewReport(
            review_result=review_result,
            security_result=security_result,
            test_result=test_result,
            elapsed_seconds=elapsed,
        )

        # Capture per-agent confidence
        if review_result is not None:
            report.reviewer_confidence = review_result.confidence
        if security_result is not None:
            report.security_confidence = security_result.confidence
        if test_result is not None:
            report.test_writer_confidence = test_result.confidence

        # --- Normalise all findings ------------------------------------
        all_findings: List[NormalisedFinding] = []

        if review_result is not None:
            for f in review_result.findings:
                all_findings.append(_normalise_review_finding(f))

        if security_result is not None:
            for f in security_result.findings:
                all_findings.append(_normalise_security_finding(f))

        if test_result is not None:
            all_findings.extend(_normalise_test_finding(test_result))

        report.normalised_findings = all_findings

        # --- Merge / vote -------------------------------------------
        report.votes = self._merge(all_findings)

        # --- Verdict --------------------------------------------------
        report.verdict = self._determine_verdict(report)
        report.summary = self._build_summary(report)

        return report

    # ------------------------------------------------------------------ #
    # Merging logic
    # ------------------------------------------------------------------ #

    def _merge(self, findings: List[NormalisedFinding]) -> List[FindingVote]:
        """Group findings by rule_id, detect conflicts, produce votes."""
        groups: Dict[str, List[NormalisedFinding]] = {}
        for f in findings:
            groups.setdefault(f.rule_id, []).append(f)

        votes: List[FindingVote] = []
        for rule_id, group in groups.items():
            vote = self._build_vote(rule_id, group)
            votes.append(vote)

        # Sort by agreed severity rank descending
        votes.sort(key=lambda v: v.agreed_severity.rank, reverse=True)
        return votes

    def _build_vote(self, rule_id: str, group: List[NormalisedFinding]) -> FindingVote:
        """Produce a single FindingVote from a group of same-rule findings."""
        severities = [f.severity for f in group]
        ranks = [s.rank for s in severities]

        min_rank = min(ranks)
        max_rank = max(ranks)
        conflict = (max_rank - min_rank) > self.CONFLICT_RANK_THRESHOLD

        # Use the highest severity when there is a conflict (escalation)
        agreed_severity = max(severities, key=lambda s: s.rank)

        max_confidence = max(f.confidence for f in group)

        return FindingVote(
            finding_key=rule_id,
            contributors=list(group),
            agreed_severity=agreed_severity,
            severity_conflict=conflict,
            vote_count=len(group),
            max_confidence=max_confidence,
        )

    # ------------------------------------------------------------------ #
    # Verdict / summary
    # ------------------------------------------------------------------ #

    @staticmethod
    def _determine_verdict(report: CodeReviewReport) -> Verdict:
        if report.critical_count > 0:
            return Verdict.FAIL
        if report.high_or_error_count > 0:
            return Verdict.FAIL
        # Any conflict regardless of severity → at least WARN
        if report.has_conflicts:
            return Verdict.WARN
        # Count medium/warning findings
        medium_count = sum(
            1
            for v in report.votes
            if v.agreed_severity in (UnifiedSeverity.MEDIUM, UnifiedSeverity.WARNING)
        )
        if medium_count > 0:
            return Verdict.WARN
        return Verdict.PASS

    @staticmethod
    def _build_summary(report: CodeReviewReport) -> str:
        parts = [
            f"{len(report.votes)} merged finding(s)",
            f"{report.critical_count} critical",
            f"{report.high_or_error_count} high/error",
            f"{len(report.conflict_votes)} conflict(s)",
            f"verdict={report.verdict.value}",
        ]
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runs all three agents in parallel and feeds results to the ConsensusEngine."""

    def __init__(
        self,
        reviewer: Optional[ReviewerAgent] = None,
        security_scanner: Optional[SecurityScannerAgent] = None,
        test_writer: Optional[TestWriterAgent] = None,
        consensus_engine: Optional[ConsensusEngine] = None,
        max_workers: int = 3,
    ) -> None:
        self._reviewer = reviewer or ReviewerAgent()
        self._security_scanner = security_scanner or SecurityScannerAgent()
        self._test_writer = test_writer or TestWriterAgent()
        self._consensus = consensus_engine or ConsensusEngine()
        self._max_workers = max_workers

    def run(self, code: str) -> CodeReviewReport:
        """Analyse *code* with all agents in parallel and return a unified report."""
        start = time.monotonic()

        review_result: Optional[ReviewResult] = None
        security_result: Optional[SecurityResult] = None
        test_result: Optional[TestResult] = None
        errors: Dict[str, Exception] = {}

        tasks = {
            "reviewer": (self._reviewer.analyze, code),
            "security": (self._security_scanner.analyze, code),
            "test_writer": (self._test_writer.analyze, code),
        }

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_key = {
                executor.submit(fn, arg): key for key, (fn, arg) in tasks.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    if key == "reviewer":
                        review_result = result
                    elif key == "security":
                        security_result = result
                    elif key == "test_writer":
                        test_result = result
                except Exception as exc:  # pragma: no cover
                    errors[key] = exc

        elapsed = time.monotonic() - start

        report = self._consensus.build_report(
            review_result=review_result,
            security_result=security_result,
            test_result=test_result,
            elapsed=elapsed,
        )

        return report
