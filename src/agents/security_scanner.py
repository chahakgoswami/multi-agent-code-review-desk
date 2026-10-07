"""SecurityScannerAgent: pattern-based security vulnerability scanner."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .base import AgentBase
from .mock_llm import MockLLMBackend


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class VulnSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VulnFinding:
    """A single security vulnerability finding."""

    cve_id: str          # Mock CVE-style ID, e.g. "MOCK-2024-0001"
    rule_id: str         # Internal rule identifier
    title: str           # Short human-readable title
    description: str     # Detailed description
    severity: VulnSeverity
    confidence: float    # 0.0-1.0
    line: Optional[int] = None
    context: Optional[str] = None

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line is not None else ""
        ctx = f"\n    > {self.context}" if self.context else ""
        return (
            f"[{self.severity.value.upper()}] {self.cve_id} / {self.rule_id}{loc}: "
            f"{self.title}{ctx}"
        )


@dataclass
class SecurityResult:
    """Aggregated result returned by SecurityScannerAgent.analyze()."""

    findings: List[VulnFinding] = field(default_factory=list)
    confidence: float = 1.0
    llm_note: str = ""
    scanned_lines: int = 0

    def findings_by_severity(self, severity: VulnSeverity) -> List[VulnFinding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def critical_count(self) -> int:
        return len(self.findings_by_severity(VulnSeverity.CRITICAL))

    @property
    def high_count(self) -> int:
        return len(self.findings_by_severity(VulnSeverity.HIGH))

    @property
    def medium_count(self) -> int:
        return len(self.findings_by_severity(VulnSeverity.MEDIUM))

    @property
    def low_count(self) -> int:
        return len(self.findings_by_severity(VulnSeverity.LOW))


# ---------------------------------------------------------------------------
# Vulnerability pattern catalogue
# ---------------------------------------------------------------------------

# Each entry: (rule_id, cve_id, title, pattern, severity, confidence, description)
# pattern is a compiled regex applied line-by-line.
_LINE_PATTERNS: List[Tuple[str, str, str, re.Pattern, VulnSeverity, float, str]] = [
    # ------------------------------------------------------------------ #
    # Hardcoded secrets
    # ------------------------------------------------------------------ #
    (
        "S001",
        "MOCK-2024-0001",
        "Hardcoded password assignment",
        re.compile(
            r"(?i)\b(password|passwd|pwd)\s*=\s*[\'\"][^\'\"]\S+[\'\"]",
            re.IGNORECASE,
        ),
        VulnSeverity.CRITICAL,
        0.90,
        "A password appears to be hardcoded in the source. Rotate and use environment variables or a secrets manager.",
    ),
    (
        "S002",
        "MOCK-2024-0002",
        "Hardcoded API key or token",
        re.compile(
            r"(?i)\b(api_key|apikey|secret_key|auth_token|access_token)\s*=\s*[\'\"][^\'\"]\S+[\'\"]",
            re.IGNORECASE,
        ),
        VulnSeverity.CRITICAL,
        0.90,
        "An API key or token appears to be hardcoded. Use environment variables or a vault.",
    ),
    (
        "S003",
        "MOCK-2024-0003",
        "Hardcoded AWS credential",
        re.compile(
            r"(?i)(AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID)\s*=\s*[\'\"][A-Za-z0-9+/]{16,}[\'\"]",
            re.IGNORECASE,
        ),
        VulnSeverity.CRITICAL,
        0.95,
        "An AWS credential appears to be hardcoded. Rotate immediately and use IAM roles or environment variables.",
    ),
    (
        "S004",
        "MOCK-2024-0004",
        "Hardcoded private key marker",
        re.compile(
            r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        ),
        VulnSeverity.CRITICAL,
        0.98,
        "A PEM-encoded private key literal is present in the source code.",
    ),
    # ------------------------------------------------------------------ #
    # SQL injection
    # ------------------------------------------------------------------ #
    (
        "S005",
        "MOCK-2024-0005",
        "SQL injection via string concatenation",
        re.compile(
            r'(execute|cursor\.execute|db\.execute)\s*\(\s*[f\"\'].*[\+%].*\)',
            re.IGNORECASE,
        ),
        VulnSeverity.HIGH,
        0.85,
        "SQL query constructed via string concatenation or formatting. Use parameterised queries.",
    ),
    (
        "S006",
        "MOCK-2024-0006",
        "SQL injection via f-string query",
        re.compile(
            r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE).*\{',
        ),
        VulnSeverity.HIGH,
        0.80,
        "SQL keyword found in f-string with variable interpolation — possible SQL injection.",
    ),
    (
        "S007",
        "MOCK-2024-0007",
        "Raw SQL string with user-supplied variable",
        re.compile(
            r'(?i)\b(query|sql|stmt)\s*=\s*[f\"\'].*%s.*[\'\"f]',
        ),
        VulnSeverity.MEDIUM,
        0.75,
        "SQL string with %s placeholder assembled in a potentially unsafe way.",
    ),
    # ------------------------------------------------------------------ #
    # Unsafe deserialization
    # ------------------------------------------------------------------ #
    (
        "S008",
        "MOCK-2024-0008",
        "Unsafe pickle deserialization",
        re.compile(
            r"\bpickle\.loads?\s*\(",
        ),
        VulnSeverity.HIGH,
        0.92,
        "pickle.load(s) deserializes arbitrary Python objects and can execute code. Avoid on untrusted data.",
    ),
    (
        "S009",
        "MOCK-2024-0009",
        "Unsafe marshal deserialization",
        re.compile(
            r"\bmarshal\.loads?\s*\(",
        ),
        VulnSeverity.HIGH,
        0.88,
        "marshal.load(s) can execute arbitrary code when processing untrusted input.",
    ),
    (
        "S010",
        "MOCK-2024-0010",
        "Unsafe YAML load",
        re.compile(
            r"\byaml\.load\s*\([^,)]*\)(?!\s*#.*safe)",
        ),
        VulnSeverity.HIGH,
        0.87,
        "yaml.load() without Loader=yaml.SafeLoader can deserialize arbitrary Python objects.",
    ),
    (
        "S011",
        "MOCK-2024-0011",
        "Unsafe shelve or dbm usage",
        re.compile(
            r"\bshelve\.open\s*\(",
        ),
        VulnSeverity.MEDIUM,
        0.70,
        "shelve uses pickle internally; untrusted shelf data can execute arbitrary code.",
    ),
    # ------------------------------------------------------------------ #
    # Command injection / code execution
    # ------------------------------------------------------------------ #
    (
        "S012",
        "MOCK-2024-0012",
        "Shell injection via os.system",
        re.compile(
            r"\bos\.system\s*\(",
        ),
        VulnSeverity.HIGH,
        0.88,
        "os.system() passes commands to the shell; prefer subprocess with a list argument and shell=False.",
    ),
    (
        "S013",
        "MOCK-2024-0013",
        "Shell injection via subprocess shell=True",
        re.compile(
            r"\bsubprocess\.(run|Popen|call|check_output)\s*\(.*shell\s*=\s*True",
        ),
        VulnSeverity.HIGH,
        0.85,
        "subprocess with shell=True allows shell metacharacter injection. Use shell=False with a list.",
    ),
    (
        "S014",
        "MOCK-2024-0014",
        "Dynamic code execution via eval",
        re.compile(
            r"\beval\s*\(",
        ),
        VulnSeverity.CRITICAL,
        0.93,
        "eval() executes arbitrary Python code. This is a critical injection risk.",
    ),
    (
        "S015",
        "MOCK-2024-0015",
        "Dynamic code execution via exec",
        re.compile(
            r"\bexec\s*\(",
        ),
        VulnSeverity.CRITICAL,
        0.93,
        "exec() executes arbitrary Python code. This is a critical injection risk.",
    ),
    (
        "S016",
        "MOCK-2024-0016",
        "Dynamic code execution via compile+exec",
        re.compile(
            r"\bcompile\s*\(.*\bexec\b",
        ),
        VulnSeverity.HIGH,
        0.80,
        "compile() with exec mode can run attacker-controlled code.",
    ),
    # ------------------------------------------------------------------ #
    # Path traversal
    # ------------------------------------------------------------------ #
    (
        "S017",
        "MOCK-2024-0017",
        "Potential path traversal",
        re.compile(
            r'open\s*\(.*\+.*\)',
        ),
        VulnSeverity.MEDIUM,
        0.70,
        "File path constructed via concatenation inside open() — possible path traversal.",
    ),
    # ------------------------------------------------------------------ #
    # Cryptographic weaknesses
    # ------------------------------------------------------------------ #
    (
        "S018",
        "MOCK-2024-0018",
        "Use of weak hash algorithm MD5",
        re.compile(
            r"\bhashlib\.md5\s*\(",
        ),
        VulnSeverity.MEDIUM,
        0.85,
        "MD5 is cryptographically broken. Use SHA-256 or stronger for security-sensitive hashing.",
    ),
    (
        "S019",
        "MOCK-2024-0019",
        "Use of weak hash algorithm SHA1",
        re.compile(
            r"\bhashlib\.sha1\s*\(",
        ),
        VulnSeverity.MEDIUM,
        0.82,
        "SHA-1 is deprecated for security use. Use SHA-256 or stronger.",
    ),
    (
        "S020",
        "MOCK-2024-0020",
        "Use of DES or RC4 cipher",
        re.compile(
            r"(?i)\b(DES|RC4|Blowfish)\b",
        ),
        VulnSeverity.HIGH,
        0.78,
        "Weak or broken cipher in use. Replace with AES-256-GCM or ChaCha20-Poly1305.",
    ),
    # ------------------------------------------------------------------ #
    # Insecure network / TLS
    # ------------------------------------------------------------------ #
    (
        "S021",
        "MOCK-2024-0021",
        "SSL certificate verification disabled",
        re.compile(
            r"verify\s*=\s*False",
        ),
        VulnSeverity.HIGH,
        0.90,
        "TLS certificate verification is disabled. This makes the connection vulnerable to MITM attacks.",
    ),
    (
        "S022",
        "MOCK-2024-0022",
        "Use of http:// URL (plaintext)",
        re.compile(
            r'[\"\']http://(?!localhost|127\.0\.0\.1)',
        ),
        VulnSeverity.LOW,
        0.65,
        "Non-localhost HTTP URL detected. Prefer HTTPS to protect data in transit.",
    ),
    # ------------------------------------------------------------------ #
    # Debug / development artefacts
    # ------------------------------------------------------------------ #
    (
        "S023",
        "MOCK-2024-0023",
        "Debug mode enabled",
        re.compile(
            r"(?i)\bdebug\s*=\s*True",
        ),
        VulnSeverity.MEDIUM,
        0.80,
        "Debug mode is enabled. This may expose stack traces and internal state to users in production.",
    ),
    (
        "S024",
        "MOCK-2024-0024",
        "Flask/Django secret key hardcoded",
        re.compile(
            r"(?i)SECRET_KEY\s*=\s*[\'\"][^\'\"]\S+[\'\"]",
        ),
        VulnSeverity.CRITICAL,
        0.92,
        "Framework SECRET_KEY is hardcoded. Rotate and load from environment variables.",
    ),
    # ------------------------------------------------------------------ #
    # XML / template injection
    # ------------------------------------------------------------------ #
    (
        "S025",
        "MOCK-2024-0025",
        "XML external entity (XXE) risk",
        re.compile(
            r"\bxml\.etree\.ElementTree\.fromstring\s*\(",
        ),
        VulnSeverity.MEDIUM,
        0.72,
        "ElementTree.fromstring() may be vulnerable to XML bombs; consider defusedxml.",
    ),
]

# AST-level checks  --------------------------------------------------------
# Each entry: (rule_id, cve_id, title, severity, confidence, description, checker_fn)
# checker_fn(node) -> bool  returns True if the node is vulnerable


def _ast_random_not_secrets(node: ast.AST) -> bool:
    """Detect random.random() / random.randint() used for secret generation (AST)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return (
            isinstance(func.value, ast.Name)
            and func.value.id == "random"
            and func.attr in ("random", "randint", "choice", "shuffle")
        )
    return False


# ---------------------------------------------------------------------------
# SecurityScannerAgent
# ---------------------------------------------------------------------------


class SecurityScannerAgent(AgentBase):
    """Agent that performs security-focused static analysis without external tools."""

    def __init__(self, llm: Optional[MockLLMBackend] = None) -> None:
        super().__init__(name="SecurityScannerAgent")
        self._llm = llm or MockLLMBackend()
        self._last_result: Optional[SecurityResult] = None

    # ------------------------------------------------------------------ #
    # AgentBase interface
    # ------------------------------------------------------------------ #

    def analyze(self, code: str) -> SecurityResult:
        """Scan *code* for vulnerability patterns and return a SecurityResult."""
        result = SecurityResult()
        lines = code.splitlines()
        result.scanned_lines = len(lines)

        # --- Line-level pattern scanning -------------------------------
        for lineno, line in enumerate(lines, 1):
            for rule_id, cve_id, title, pattern, severity, conf, description in _LINE_PATTERNS:
                if pattern.search(line):
                    result.findings.append(
                        VulnFinding(
                            cve_id=cve_id,
                            rule_id=rule_id,
                            title=title,
                            description=description,
                            severity=severity,
                            confidence=conf,
                            line=lineno,
                            context=line.strip()[:120],
                        )
                    )

        # --- AST-level scanning ----------------------------------------
        ast_findings = self._ast_scan(code)
        result.findings.extend(ast_findings)

        # --- Mock LLM additional signal --------------------------------
        llm_response = self._llm.complete(code)
        result.llm_note = llm_response
        llm_findings = self._parse_llm_response(llm_response)
        result.findings.extend(llm_findings)

        # --- Confidence ------------------------------------------------
        result.confidence = self._compute_confidence(result)

        self._last_result = result
        return result

    def report(self) -> str:
        """Return a human-readable summary of the most recent security scan."""
        if self._last_result is None:
            return f"[{self.name}] No scan has been run yet."
        res = self._last_result
        lines_out = [f"[{self.name}] Security Scan Report"]
        lines_out.append(f"  Scanned  : {res.scanned_lines} line(s)")
        lines_out.append(f"  Findings : {len(res.findings)} total")
        lines_out.append(f"    critical : {res.critical_count}")
        lines_out.append(f"    high     : {res.high_count}")
        lines_out.append(f"    medium   : {res.medium_count}")
        lines_out.append(f"    low      : {res.low_count}")
        lines_out.append(f"  Confidence: {res.confidence:.4f}")
        if res.findings:
            lines_out.append("  Findings detail:")
            for f in res.findings:
                lines_out.append(f"    {f}")
        return "\n".join(lines_out)

    def confidence_score(self) -> float:
        """Return confidence of the most recent scan (0.0-1.0)."""
        if self._last_result is None:
            return 0.0
        return self._last_result.confidence

    # ------------------------------------------------------------------ #
    # AST scanning
    # ------------------------------------------------------------------ #

    def _ast_scan(self, code: str) -> List[VulnFinding]:
        """Parse code and apply AST-level security checks."""
        findings: List[VulnFinding] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings  # Syntax errors handled at caller level if needed

        for node in ast.walk(tree):
            findings.extend(self._check_assert_used_for_auth(node))
            findings.extend(self._check_random_for_security(node))
            findings.extend(self._check_try_except_pass(node))

        return findings

    @staticmethod
    def _check_assert_used_for_auth(node: ast.AST) -> List[VulnFinding]:
        """Flag assert statements that look like authentication checks."""
        findings: List[VulnFinding] = []
        if not isinstance(node, ast.Assert):
            return findings
        # Heuristic: assert with a comparison or function call at top level
        test = node.test
        if isinstance(test, (ast.Compare, ast.Call)):
            # Check if there is any name that smells like auth
            src = ast.dump(test)
            auth_keywords = ("user", "admin", "auth", "login", "password", "perm", "role")
            if any(kw in src.lower() for kw in auth_keywords):
                findings.append(
                    VulnFinding(
                        cve_id="MOCK-2024-0026",
                        rule_id="S026",
                        title="assert used for authentication or authorization check",
                        description=(
                            "assert statements are stripped when Python runs with -O (optimise). "
                            "Never use assert for security checks."
                        ),
                        severity=VulnSeverity.HIGH,
                        confidence=0.75,
                        line=node.lineno,
                        context=ast.unparse(node) if hasattr(ast, "unparse") else None,
                    )
                )
        return findings

    @staticmethod
    def _check_random_for_security(node: ast.AST) -> List[VulnFinding]:
        """Flag use of random module for token/secret generation."""
        findings: List[VulnFinding] = []
        if not isinstance(node, ast.Assign):
            return findings
        # Look for targets with security-sounding names
        target_names = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                target_names.append(t.id.lower())
            elif isinstance(t, ast.Attribute):
                target_names.append(t.attr.lower())

        sec_names = ("token", "secret", "key", "nonce", "salt", "otp", "password")
        if any(s in name for s in sec_names for name in target_names):
            if _ast_random_not_secrets(node.value):
                findings.append(
                    VulnFinding(
                        cve_id="MOCK-2024-0027",
                        rule_id="S027",
                        title="Insecure random used for security-sensitive value",
                        description=(
                            "random module is not cryptographically secure. "
                            "Use secrets.token_bytes() or os.urandom() for security tokens."
                        ),
                        severity=VulnSeverity.HIGH,
                        confidence=0.85,
                        line=node.lineno,
                        context=ast.unparse(node) if hasattr(ast, "unparse") else None,
                    )
                )
        return findings

    @staticmethod
    def _check_try_except_pass(node: ast.AST) -> List[VulnFinding]:
        """Flag bare except…pass patterns that swallow security exceptions."""
        findings: List[VulnFinding] = []
        if not isinstance(node, ast.ExceptHandler):
            return findings
        # bare handler (except:) or broad (except Exception:)
        is_bare = node.type is None
        is_broad = (
            node.type is not None
            and isinstance(node.type, ast.Name)
            and node.type.id in ("Exception", "BaseException")
        )
        if is_bare or is_broad:
            # body is only Pass or empty
            has_only_pass = all(isinstance(s, ast.Pass) for s in node.body)
            if has_only_pass:
                findings.append(
                    VulnFinding(
                        cve_id="MOCK-2024-0028",
                        rule_id="S028",
                        title="Silent broad exception handler",
                        description=(
                            "A bare or broad except clause with only pass silently swallows "
                            "errors that may indicate security violations."
                        ),
                        severity=VulnSeverity.MEDIUM,
                        confidence=0.78,
                        line=node.lineno,
                    )
                )
        return findings

    # ------------------------------------------------------------------ #
    # LLM response parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_llm_response(response: str) -> List[VulnFinding]:
        """Convert well-known LLM canned phrases into structured security findings."""
        findings: List[VulnFinding] = []
        mappings = [
            (
                "hardcoded credential",
                "MOCK-2024-0101",
                "S101",
                "LLM: Hardcoded credential detected",
                VulnSeverity.CRITICAL,
                0.70,
            ),
            (
                "sql query constructed",
                "MOCK-2024-0102",
                "S102",
                "LLM: SQL injection risk from string-built query",
                VulnSeverity.HIGH,
                0.70,
            ),
            (
                "unsafe deserialization",
                "MOCK-2024-0103",
                "S103",
                "LLM: Unsafe deserialization pattern",
                VulnSeverity.HIGH,
                0.70,
            ),
            (
                "eval()",
                "MOCK-2024-0104",
                "S104",
                "LLM: Dynamic code execution via eval()",
                VulnSeverity.CRITICAL,
                0.70,
            ),
            (
                "code injection",
                "MOCK-2024-0105",
                "S105",
                "LLM: Code injection risk",
                VulnSeverity.CRITICAL,
                0.70,
            ),
        ]
        lower = response.lower()
        for keyword, cve_id, rule_id, title, severity, conf in mappings:
            if keyword in lower:
                findings.append(
                    VulnFinding(
                        cve_id=cve_id,
                        rule_id=rule_id,
                        title=title,
                        description=f"Mock LLM flagged: '{keyword}' pattern in submitted code.",
                        severity=severity,
                        confidence=conf,
                    )
                )
        return findings

    # ------------------------------------------------------------------ #
    # Confidence computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_confidence(result: SecurityResult) -> float:
        """Derive overall scan confidence from finding counts and severities."""
        if result.scanned_lines == 0:
            return 0.0
        # Start high; scale back for very short files (less context)
        base = min(0.95, 0.60 + 0.35 * min(result.scanned_lines, 100) / 100)
        # Average per-finding confidence if there are findings
        if result.findings:
            avg_finding_conf = sum(f.confidence for f in result.findings) / len(result.findings)
            return round((base + avg_finding_conf) / 2, 4)
        return round(base, 4)
