"""TestWriterAgent: infers function signatures and generates stub pytest tests."""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import AgentBase
from .mock_llm import MockLLMBackend


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FunctionSignature:
    """Parsed representation of a function found in the submitted code."""

    name: str
    args: List[str]          # parameter names
    arg_types: List[str]     # type annotation strings ("unknown" if absent)
    return_type: str         # return annotation string or "unknown"
    is_async: bool
    lineno: int
    docstring: Optional[str] = None

    def __str__(self) -> str:
        params = ", ".join(
            f"{n}: {t}" for n, t in zip(self.args, self.arg_types)
        )
        async_pfx = "async " if self.is_async else ""
        return f"{async_pfx}def {self.name}({params}) -> {self.return_type}"


@dataclass
class TestResult:
    """Aggregated result returned by TestWriterAgent.analyze()."""

    generated_code: str = ""
    signatures: List[FunctionSignature] = field(default_factory=list)
    coverage_estimate: float = 0.0   # 0.0-1.0, fraction of functions with tests
    confidence: float = 1.0
    llm_note: str = ""
    parse_error: Optional[str] = None

    @property
    def function_count(self) -> int:
        return len(self.signatures)

    @property
    def test_count(self) -> int:
        """Count top-level def test_* in generated_code."""
        if not self.generated_code:
            return 0
        return len(re.findall(r"^def test_", self.generated_code, re.MULTILINE))


# ---------------------------------------------------------------------------
# Signature extractor
# ---------------------------------------------------------------------------


class _SignatureExtractor(ast.NodeVisitor):
    """Walk the AST and collect all function definitions (including nested)."""

    def __init__(self) -> None:
        self.signatures: List[FunctionSignature] = []

    def _resolve_annotation(self, annotation: Optional[ast.expr]) -> str:
        if annotation is None:
            return "unknown"
        try:
            return ast.unparse(annotation)
        except Exception:  # pragma: no cover
            return "unknown"

    def _extract_docstring(self, node: ast.FunctionDef) -> Optional[str]:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value.strip()
        return None

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args
        param_names: List[str] = []
        param_types: List[str] = []
        for arg in args.args:
            param_names.append(arg.arg)
            param_types.append(self._resolve_annotation(arg.annotation))
        # vararg
        if args.vararg:
            param_names.append("*" + args.vararg.arg)
            param_types.append(self._resolve_annotation(args.vararg.annotation))
        # kwarg
        if args.kwarg:
            param_names.append("**" + args.kwarg.arg)
            param_types.append(self._resolve_annotation(args.kwarg.annotation))

        sig = FunctionSignature(
            name=node.name,
            args=param_names,
            arg_types=param_types,
            return_type=self._resolve_annotation(node.returns),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            lineno=node.lineno,
            docstring=self._extract_docstring(node),
        )
        self.signatures.append(sig)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_func(node)


# ---------------------------------------------------------------------------
# Edge-case inferrer
# ---------------------------------------------------------------------------


class _EdgeCaseInferrer:
    """Generates human-readable edge-case descriptions from a FunctionSignature."""

    _NUMERIC_TYPES = {"int", "float", "complex"}
    _SEQ_TYPES = {"list", "tuple", "set", "frozenset", "List", "Tuple", "Set"}
    _DICT_TYPES = {"dict", "Dict"}
    _STR_TYPES = {"str"}
    _BOOL_TYPES = {"bool"}
    _OPT_RE = re.compile(r"Optional\[(.+)\]")

    def infer(self, sig: FunctionSignature) -> List[str]:
        """Return a list of edge-case descriptions for the signature."""
        cases: List[str] = []
        # Generic cases that always apply
        cases.append("normal / happy-path invocation")
        for name, typ in zip(sig.args, sig.arg_types):
            if name in ("self", "cls"):
                continue
            cases.extend(self._cases_for_param(name, typ))
        if not cases:
            cases.append("no-arg edge case")
        return cases

    def _cases_for_param(self, name: str, typ: str) -> List[str]:
        """Generate cases based on the parameter type hint."""
        cases: List[str] = []
        base = self._unwrap_optional(typ)

        if base in self._NUMERIC_TYPES:
            cases += [
                f"{name}=0 (zero value)",
                f"{name} negative value",
                f"{name} very large value",
            ]
        elif base in self._SEQ_TYPES or base.startswith(("List[", "list[", "Tuple", "Set[")):
            cases += [
                f"{name}=[] (empty sequence)",
                f"{name} single-element sequence",
                f"{name} large sequence",
            ]
        elif base in self._DICT_TYPES or base.startswith(("Dict[", "dict[")):
            cases += [
                f"{name}={{}} (empty dict)",
                f"{name} with unexpected keys",
            ]
        elif base in self._STR_TYPES:
            cases += [
                f"{name}='' (empty string)",
                f"{name} containing special characters",
                f"{name} very long string",
            ]
        elif base in self._BOOL_TYPES:
            cases += [
                f"{name}=True",
                f"{name}=False",
            ]
        elif self._OPT_RE.match(typ):
            cases.append(f"{name}=None (optional parameter is None)")
        else:
            cases.append(f"{name} unexpected type / None")
        return cases

    def _unwrap_optional(self, typ: str) -> str:
        m = self._OPT_RE.match(typ)
        if m:
            return m.group(1).strip()
        return typ


# ---------------------------------------------------------------------------
# Test code generator
# ---------------------------------------------------------------------------


class _TestCodeGenerator:
    """Produces pytest stub source code from parsed signatures."""

    _INDENT = "    "

    def generate(self, signatures: List[FunctionSignature], llm_note: str) -> str:
        """
        Build a full pytest module string from *signatures*.
        Includes a module-level docstring, imports, and one test function per
        (function, edge-case) pair plus an extra LLM-inspired test when applicable.
        """
        inferrer = _EdgeCaseInferrer()
        blocks: List[str] = []

        # --- Module header ---
        blocks.append(
            '"""Auto-generated stub tests – produced by TestWriterAgent.\n'
            'Fill in the assertions; these are scaffolding only.\n'
            '"""'
        )
        blocks.append("import pytest")
        blocks.append("")

        if not signatures:
            blocks.append("# No functions found in submitted code.")
            blocks.append("def test_placeholder() -> None:")
            blocks.append(f"{self._INDENT}# No functions to test.")
            blocks.append(f"{self._INDENT}pass")
            return "\n".join(blocks) + "\n"

        for sig in signatures:
            edge_cases = inferrer.infer(sig)
            for case_idx, case_desc in enumerate(edge_cases):
                test_name = self._test_name(sig.name, case_idx, case_desc)
                func_block = self._render_test(
                    test_name=test_name,
                    sig=sig,
                    case_desc=case_desc,
                    case_idx=case_idx,
                )
                blocks.append(func_block)

        # --- LLM-inspired extra test when there's a matched canned note ---
        llm_extra = self._llm_extra_test(signatures, llm_note)
        if llm_extra:
            blocks.append(llm_extra)

        return "\n".join(blocks) + "\n"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _test_name(self, func_name: str, idx: int, case_desc: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", case_desc.lower()).strip("_")
        slug = slug[:40]
        return f"test_{func_name}_{slug}"

    def _render_test(
        self,
        test_name: str,
        sig: FunctionSignature,
        case_desc: str,
        case_idx: int,
    ) -> str:
        i = self._INDENT
        lines: List[str] = []
        lines.append(f"def {test_name}() -> None:")
        lines.append(f'{i}"""Edge case: {case_desc}.')
        if sig.docstring:
            # Include first line of original docstring as context
            first = sig.docstring.splitlines()[0][:80]
            lines.append(f"{i}Source docstring: {first}")
        lines.append(f'{i}"""')

        # Build a placeholder call
        call_args = self._placeholder_args(sig, case_idx)
        if sig.is_async:
            lines.append(f"{i}import asyncio")
            lines.append(f"{i}# TODO: provide test fixtures")
            lines.append(f"{i}result = asyncio.run({sig.name}({call_args}))")
        else:
            lines.append(f"{i}# TODO: provide test fixtures")
            lines.append(f"{i}result = {sig.name}({call_args})")

        if sig.return_type not in ("None", "unknown", "NoReturn"):
            lines.append(f"{i}assert result is not None  # TODO: sharpen assertion")
        else:
            lines.append(f"{i}# void function – assert side-effects or no exception")
            lines.append(f"{i}assert True")
        lines.append("")
        return "\n".join(lines)

    def _placeholder_args(self, sig: FunctionSignature, case_idx: int) -> str:
        """Generate placeholder argument values for a call expression."""
        _type_defaults = {
            "int": "0",
            "float": "0.0",
            "str": "''",
            "bool": "True",
            "list": "[]",
            "dict": "{}",
            "tuple": "()",
            "set": "set()",
            "bytes": "b''",
            "None": "None",
        }
        parts: List[str] = []
        for name, typ in zip(sig.args, sig.arg_types):
            if name in ("self", "cls"):
                continue
            if name.startswith("*"):
                continue
            base = re.sub(r"Optional\[(.+)\]", r"\1", typ).strip()
            base = re.sub(r"List\[.*\]", "list", base)
            base = re.sub(r"Dict\[.*\]", "dict", base)
            default = _type_defaults.get(base, "None")
            parts.append(default)
        return ", ".join(parts)

    def _llm_extra_test(self, signatures: List[FunctionSignature], llm_note: str) -> str:
        """Produce one extra test stub suggested by the LLM canned response."""
        lower = llm_note.lower()
        keywords_to_test: List[Tuple[str, str]] = [
            ("off-by-one", "boundary / off-by-one"),
            ("complexity is high", "complex branching path"),
            ("missing docstring", "public API contract"),
            ("mutable default", "mutable default argument mutation"),
            ("global variable", "global state side-effect"),
            ("unreachable code", "control-flow path"),
            ("type annotation", "type coercion / wrong type input"),
        ]
        for keyword, scenario in keywords_to_test:
            if keyword in lower:
                fn = signatures[0].name if signatures else "module"
                i = self._INDENT
                lines = [
                    f"def test_{fn}_llm_suggested_{re.sub(r'[^a-z0-9]+', '_', scenario.lower())[:30]}() -> None:",
                    f'{i}"""LLM-suggested scenario: {scenario}.',
                    f'{i}Auto-generated from mock-LLM canned response.',
                    f'{i}"""',
                    f"{i}# TODO: implement based on scenario '{scenario}'",
                    f"{i}pytest.skip('LLM-suggested stub – not yet implemented')",
                    "",
                ]
                return "\n".join(lines)
        return ""


# ---------------------------------------------------------------------------
# TestWriterAgent
# ---------------------------------------------------------------------------


class TestWriterAgent(AgentBase):
    """Agent that generates stub pytest tests from submitted Python source."""

    def __init__(self, llm: Optional[MockLLMBackend] = None) -> None:
        super().__init__(name="TestWriterAgent")
        self._llm = llm or MockLLMBackend()
        self._generator = _TestCodeGenerator()
        self._last_result: Optional[TestResult] = None

    # ------------------------------------------------------------------ #
    # AgentBase interface
    # ------------------------------------------------------------------ #

    def analyze(self, code: str) -> TestResult:
        """Parse *code*, extract signatures, generate stub tests, return TestResult."""
        result = TestResult()

        # --- Parse AST -------------------------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            result.parse_error = str(exc)
            result.confidence = 0.3
            result.generated_code = (
                '"""Could not generate tests: syntax error in submitted code."""\n'
                f"# SyntaxError: {exc}\n"
            )
            self._last_result = result
            return result

        # --- Extract signatures ----------------------------------------
        extractor = _SignatureExtractor()
        extractor.visit(tree)
        result.signatures = extractor.signatures

        # --- Mock LLM call --------------------------------------------
        llm_response = self._llm.complete(code)
        result.llm_note = llm_response

        # --- Generate test code ----------------------------------------
        result.generated_code = self._generator.generate(result.signatures, llm_response)

        # --- Coverage estimate -----------------------------------------
        result.coverage_estimate = self._estimate_coverage(result)

        # --- Confidence ------------------------------------------------
        result.confidence = self._compute_confidence(code, result)

        self._last_result = result
        return result

    def report(self) -> str:
        """Return a human-readable summary of the most recent test generation."""
        if self._last_result is None:
            return f"[{self.name}] No analysis has been run yet."
        res = self._last_result
        lines = [f"[{self.name}] Test Generation Report"]
        lines.append(f"  Functions found : {res.function_count}")
        lines.append(f"  Tests generated : {res.test_count}")
        lines.append(f"  Coverage estimate: {res.coverage_estimate:.1%}")
        lines.append(f"  Confidence      : {res.confidence:.4f}")
        if res.parse_error:
            lines.append(f"  Parse error: {res.parse_error}")
        if res.llm_note:
            lines.append("  LLM Note  : " + res.llm_note.splitlines()[0])
        lines.append("")
        lines.append("--- Generated Test Code ---")
        lines.append(res.generated_code)
        return "\n".join(lines)

    def confidence_score(self) -> float:
        """Return confidence of the most recent analysis (0.0-1.0)."""
        if self._last_result is None:
            return 0.0
        return self._last_result.confidence

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_coverage(result: TestResult) -> float:
        """Estimate what fraction of discovered functions have at least one test stub."""
        if result.function_count == 0:
            return 0.0
        # Each function gets at least one test stub, so coverage is 1.0 when
        # there are signatures; degrade slightly if only placeholder was generated.
        if result.test_count == 0:
            return 0.0
        # Assume one happy-path + N edge-case tests per function.
        # Coverage estimate: min(1.0, tests/functions * 0.2) with a floor.
        ratio = result.test_count / max(1, result.function_count)
        # Normalise: each function should have ~5 tests to count as fully covered.
        return round(min(1.0, ratio / 5.0), 4)

    @staticmethod
    def _compute_confidence(code: str, result: TestResult) -> float:
        """Confidence based on code size and whether signatures were found."""
        line_count = len(code.splitlines())
        base = min(0.92, 0.50 + 0.42 * min(line_count, 200) / 200)
        if result.function_count == 0:
            base *= 0.6  # Much less confident if we found no functions
        return round(base, 4)
