"""ReviewerAgent: rule-based + mock-LLM code reviewer."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .base import AgentBase
from .mock_llm import MockLLMBackend


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Finding:
    """A single review finding."""

    rule_id: str
    message: str
    severity: Severity
    line: Optional[int] = None  # 1-based line number, or None if file-level
    context: Optional[str] = None  # snippet of relevant code

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line is not None else ""
        ctx = f"\n    > {self.context}" if self.context else ""
        return f"[{self.severity.value.upper()}] {self.rule_id}{loc}: {self.message}{ctx}"


@dataclass
class ReviewResult:
    """Aggregated result returned by ReviewerAgent.analyze()."""

    findings: List[Finding] = field(default_factory=list)
    confidence: float = 1.0
    llm_note: str = ""
    parse_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    def findings_by_severity(self, severity: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def error_count(self) -> int:
        return len(self.findings_by_severity(Severity.ERROR))

    @property
    def warning_count(self) -> int:
        return len(self.findings_by_severity(Severity.WARNING))

    @property
    def info_count(self) -> int:
        return len(self.findings_by_severity(Severity.INFO))


# ---------------------------------------------------------------------------
# Rule-based heuristics
# ---------------------------------------------------------------------------


class _RuleEngine:
    """Applies a catalogue of deterministic heuristic rules to an AST and raw source."""

    # Maximum cyclomatic-complexity proxy: number of branching nodes in a function
    COMPLEXITY_THRESHOLD = 7
    # Maximum number of lines for a single function before we warn
    LONG_FUNCTION_THRESHOLD = 40
    # Maximum nesting depth before we warn
    NESTING_DEPTH_THRESHOLD = 4

    def run(self, code: str, tree: ast.Module) -> List[Finding]:
        findings: List[Finding] = []
        lines = code.splitlines()

        findings.extend(self._check_style(code, lines))
        findings.extend(self._check_ast(tree, lines))
        return findings

    # ------------------------------------------------------------------ #
    # Style checks (regex / text-level)
    # ------------------------------------------------------------------ #

    def _check_style(self, code: str, lines: List[str]) -> List[Finding]]:
        findings: List[Finding] = []

        for i, line in enumerate(lines, 1):
            stripped = line.rstrip()

            # Trailing whitespace
            if line != stripped and line.endswith((" ", "\t")):
                findings.append(
                    Finding(
                        rule_id="R001",
                        message="Trailing whitespace detected.",
                        severity=Severity.INFO,
                        line=i,
                        context=repr(line),
                    )
                )

            # Lines that are overly long (>120 chars)
            if len(stripped) > 120:
                findings.append(
                    Finding(
                        rule_id="R002",
                        message=f"Line exceeds 120 characters ({len(stripped)} chars).",
                        severity=Severity.INFO,
                        line=i,
                        context=stripped[:80] + "...",
                    )
                )

            # Use of print() in non-test code
            if re.search(r"\bprint\s*\(", stripped):
                findings.append(
                    Finding(
                        rule_id="R003",
                        message="print() call found; consider using logging instead.",
                        severity=Severity.INFO,
                        line=i,
                        context=stripped.strip(),
                    )
                )

            # TODO / FIXME / HACK comments
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX)\b", stripped, re.IGNORECASE):
                findings.append(
                    Finding(
                        rule_id="R004",
                        message="Unresolved TODO/FIXME/HACK comment.",
                        severity=Severity.INFO,
                        line=i,
                        context=stripped.strip(),
                    )
                )

            # Bare except
            if re.match(r"\s*except\s*:", stripped):
                findings.append(
                    Finding(
                        rule_id="R005",
                        message="Bare 'except:' clause silences all exceptions.",
                        severity=Severity.WARNING,
                        line=i,
                        context=stripped.strip(),
                    )
                )

            # Use of eval
            if re.search(r"\beval\s*\(", stripped):
                findings.append(
                    Finding(
                        rule_id="R006",
                        message="Use of eval() is a potential code-injection risk.",
                        severity=Severity.ERROR,
                        line=i,
                        context=stripped.strip(),
                    )
                )

            # Use of exec
            if re.search(r"\bexec\s*\(", stripped):
                findings.append(
                    Finding(
                        rule_id="R007",
                        message="Use of exec() is dangerous; avoid dynamic code execution.",
                        severity=Severity.ERROR,
                        line=i,
                        context=stripped.strip(),
                    )
                )

            # Mutable default argument (simple regex version; AST version is more accurate)
            if re.search(r"def\s+\w+\s*\(.*=\s*(\[|\{)", stripped):
                findings.append(
                    Finding(
                        rule_id="R008",
                        message="Possible mutable default argument (list or dict literal).",
                        severity=Severity.WARNING,
                        line=i,
                        context=stripped.strip(),
                    )
                )

        return findings

    # ------------------------------------------------------------------ #
    # AST-level checks
    # ------------------------------------------------------------------ #

    def _check_ast(self, tree: ast.Module, lines: List[str]) -> List[Finding]:
        findings: List[Finding] = []
        findings.extend(self._check_functions(tree, lines))
        findings.extend(self._check_globals(tree, lines))
        findings.extend(self._check_imports(tree, lines))
        return findings

    def _check_functions(self, tree: ast.Module, lines: List[str]) -> List[Finding]:
        findings: List[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            fn_name = node.name
            fn_line = node.lineno

            # Missing docstring
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                findings.append(
                    Finding(
                        rule_id="R009",
                        message=f"Function '{fn_name}' is missing a docstring.",
                        severity=Severity.INFO,
                        line=fn_line,
                    )
                )

            # Missing return type annotation
            if node.returns is None:
                findings.append(
                    Finding(
                        rule_id="R010",
                        message=f"Function '{fn_name}' has no return type annotation.",
                        severity=Severity.INFO,
                        line=fn_line,
                    )
                )

            # Missing parameter annotations
            for arg in node.args.args:
                if arg.annotation is None and arg.arg not in ("self", "cls"):
                    findings.append(
                        Finding(
                            rule_id="R011",
                            message=f"Parameter '{arg.arg}' in '{fn_name}' lacks a type annotation.",
                            severity=Severity.INFO,
                            line=fn_line,
                        )
                    )

            # Complexity proxy: count branching nodes
            complexity = self._cyclomatic_proxy(node)
            if complexity > self.COMPLEXITY_THRESHOLD:
                findings.append(
                    Finding(
                        rule_id="R012",
                        message=(
                            f"Function '{fn_name}' has high complexity "
                            f"(branching score={complexity}); consider decomposition."
                        ),
                        severity=Severity.WARNING,
                        line=fn_line,
                    )
                )

            # Long function
            end_line = getattr(node, "end_lineno", None)
            if end_line is not None:
                fn_len = end_line - fn_line + 1
                if fn_len > self.LONG_FUNCTION_THRESHOLD:
                    findings.append(
                        Finding(
                            rule_id="R013",
                            message=(
                                f"Function '{fn_name}' is {fn_len} lines long "
                                f"(threshold={self.LONG_FUNCTION_THRESHOLD})."
                            ),
                            severity=Severity.WARNING,
                            line=fn_line,
                        )
                    )

            # Deep nesting
            max_depth = self._max_nesting_depth(node)
            if max_depth > self.NESTING_DEPTH_THRESHOLD:
                findings.append(
                    Finding(
                        rule_id="R014",
                        message=(
                            f"Function '{fn_name}' has deep nesting "
                            f"(depth={max_depth}); flatten control flow."
                        ),
                        severity=Severity.WARNING,
                        line=fn_line,
                    )
                )

            # Unreachable code after return/raise
            findings.extend(self._check_unreachable(node))

        return findings

    def _check_globals(self, tree: ast.Module, lines: List[str]) -> List[Finding]:
        findings: List[Finding] = []
        # Detect global variable *mutations* inside functions via 'global' statement
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings.append(
                    Finding(
                        rule_id="R015",
                        message=f"Global variable mutation: 'global {', '.join(node.names)}' inside function.",
                        severity=Severity.WARNING,
                        line=node.lineno,
                    )
                )
        return findings

    def _check_imports(self, tree: ast.Module, lines: List[str]) -> List[Finding]:
        findings: List[Finding] = []
        # Warn about wildcard imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        findings.append(
                            Finding(
                                rule_id="R016",
                                message=f"Wildcard import 'from {node.module} import *' pollutes namespace.",
                                severity=Severity.WARNING,
                                line=node.lineno,
                            )
                        )
        return findings

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cyclomatic_proxy(func_node: ast.AST) -> int:
        """Count branching nodes as a cyclomatic complexity proxy."""
        branching_types = (
            ast.If,
            ast.For,
            ast.While,
            ast.ExceptHandler,
            ast.With,
            ast.Assert,
            ast.comprehension,
        )
        count = 1  # base complexity
        for node in ast.walk(func_node):
            if isinstance(node, branching_types):
                count += 1
            elif isinstance(node, ast.BoolOp):  # 'and' / 'or'
                count += len(node.values) - 1
        return count

    @staticmethod
    def _max_nesting_depth(func_node: ast.AST) -> int:
        """Return the maximum block-nesting depth inside func_node."""
        block_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

        def _depth(node: ast.AST, current: int) -> int:
            max_d = current
            for child in ast.iter_child_nodes(node):
                if isinstance(child, block_types):
                    max_d = max(max_d, _depth(child, current + 1))
                else:
                    max_d = max(max_d, _depth(child, current))
            return max_d

        return _depth(func_node, 0)

    @staticmethod
    def _check_unreachable(func_node: ast.FunctionDef) -> List[Finding]:
        """Detect statements after return/raise/break/continue at the same level."""
        findings: List[Finding] = []
        terminator_types = (ast.Return, ast.Raise, ast.Break, ast.Continue)

        def _scan_body(stmts: list) -> None:
            for idx, stmt in enumerate(stmts):
                if isinstance(stmt, terminator_types):
                    remaining = stmts[idx + 1 :]
                    # Filter out lone Expr nodes that are just docstrings / ellipsis
                    real_remaining = [
                        s for s in remaining
                        if not (isinstance(s, ast.Expr) and isinstance(s.value, (ast.Constant, ast.Ellipsis)))
                    ]
                    if real_remaining:
                        findings.append(
                            Finding(
                                rule_id="R017",
                                message="Unreachable code detected after terminating statement.",
                                severity=Severity.ERROR,
                                line=real_remaining[0].lineno,
                            )
                        )
                    break  # no point scanning further siblings
                # Recurse into sub-bodies
                for attr in ("body", "orelse", "handlers", "finalbody"):
                    sub = getattr(stmt, attr, [])
                    if isinstance(sub, list):
                        _scan_body(sub)

        _scan_body(func_node.body)
        return findings


# ---------------------------------------------------------------------------
# ReviewerAgent
# ---------------------------------------------------------------------------


class ReviewerAgent(AgentBase):
    """Agent that performs code review using heuristics and the mock LLM."""

    def __init__(self, llm: Optional[MockLLMBackend] = None) -> None:
        super().__init__(name="ReviewerAgent")
        self._llm = llm or MockLLMBackend()
        self._rule_engine = _RuleEngine()
        self._last_result: Optional[ReviewResult] = None

    # ------------------------------------------------------------------ #
    # AgentBase interface
    # ------------------------------------------------------------------ #

    def analyze(self, code: str) -> ReviewResult:
        """Parse *code*, apply heuristic rules, query mock LLM, return ReviewResult."""
        result = ReviewResult()

        # --- Parse AST ------------------------------------------------
        tree: Optional[ast.Module] = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            result.parse_error = str(exc)
            result.confidence = 0.4
            result.findings.append(
                Finding(
                    rule_id="R000",
                    message=f"Syntax error: {exc}",
                    severity=Severity.ERROR,
                    line=exc.lineno,
                )
            )
            self._last_result = result
            return result

        # --- Heuristic rules ------------------------------------------
        rule_findings = self._rule_engine.run(code, tree)
        result.findings.extend(rule_findings)

        # --- Mock LLM -------------------------------------------------
        llm_response = self._llm.complete(code)
        result.llm_note = llm_response

        # Map LLM response keywords → additional findings
        llm_findings = self._parse_llm_response(llm_response)
        result.findings.extend(llm_findings)

        # --- Confidence heuristic -------------------------------------
        result.confidence = self._compute_confidence(code, result)

        self._last_result = result
        return result

    def report(self) -> str:
        """Return a human-readable summary of the most recent analysis."""
        if self._last_result is None:
            return f"[{self.name}] No analysis has been run yet."
        res = self._last_result
        lines = [f"[{self.name}] Review Report"]
        lines.append(f"  Findings : {len(res.findings)} total")
        lines.append(f"    errors   : {res.error_count}")
        lines.append(f"    warnings : {res.warning_count}")
        lines.append(f"    info     : {res.info_count}")
        lines.append(f"  Confidence: {res.confidence:.2f}")
        if res.parse_error:
            lines.append(f"  Parse error: {res.parse_error}")
        lines.append("  LLM Note  : " + res.llm_note.splitlines()[0])
        if res.findings:
            lines.append("  Findings detail:")
            for f in res.findings:
                lines.append(f"    {f}")
        return "\n".join(lines)

    def confidence_score(self) -> float:
        """Return confidence of the most recent analysis (0.0–1.0)."""
        if self._last_result is None:
            return 0.0
        return self._last_result.confidence

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_llm_response(response: str) -> List[Finding]:
        """Convert well-known LLM canned phrases into structured findings."""
        findings: List[Finding] = []
        mappings = [
            ("off-by-one", "R018", "LLM flagged potential off-by-one error.", Severity.WARNING),
            ("not descriptive", "R019", "LLM: variable names are not descriptive enough.", Severity.INFO),
            ("complexity is high", "R020", "LLM: function complexity is high; consider decomposition.", Severity.WARNING),
            ("missing docstring", "R021", "LLM: missing docstring on public function.", Severity.INFO),
            ("mutable default", "R022", "LLM: mutable default argument in function signature.", Severity.WARNING),
            ("global variable", "R023", "LLM: global variable mutation inside function.", Severity.WARNING),
            ("unreachable code", "R024", "LLM: unreachable code after return statement.", Severity.ERROR),
            ("shadowing", "R025", "LLM: shadowing of built-in name detected.", Severity.WARNING),
            ("circular import", "R026", "LLM: circular import risk identified.", Severity.WARNING),
            ("type annotation", "R027", "LLM: missing type annotations on function parameters.", Severity.INFO),
        ]
        lower = response.lower()
        for keyword, rule_id, message, severity in mappings:
            if keyword in lower:
                findings.append(Finding(rule_id=rule_id, message=message, severity=severity))
        return findings

    @staticmethod
    def _compute_confidence(code: str, result: ReviewResult) -> float:
        """Derive a confidence score from code size and finding mix."""
        line_count = len(code.splitlines())
        # More lines → harder to be certain, but cap at 0.95
        base = min(0.95, 0.5 + 0.45 * min(line_count, 200) / 200)
        # Penalise a little for each ERROR finding
        penalty = 0.02 * result.error_count
        return max(0.1, round(base - penalty, 4))
