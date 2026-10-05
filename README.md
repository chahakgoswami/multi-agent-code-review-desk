# Multi-Agent Code Review Desk

Multi-agent code review system with reviewer, security scanner, and test-writer agents plus consensu

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You orchestrate teams of agents, not toys.

## 7-day build plan

- [ ] Day 1: Scaffold the project structure with a src/agents/ package, a shared AgentBase class defining the common interface (analyze, report, confidence_score), a mock LLM backend that returns deterministic canned responses based on input hashes, and a minimal CLI entry point that accepts a code snippet file path.
- [ ] Day 2: Implement the ReviewerAgent that parses submitted code, identifies style issues, complexity warnings, and logic smells using rule-based heuristics plus the mock LLM, returning a structured ReviewResult dataclass with findings, severity levels, and a confidence score.
- [ ] Day 3: Implement the SecurityScannerAgent that checks for a catalogue of simulated vulnerability patterns (hardcoded secrets, SQL injection strings, unsafe deserialization markers, etc.) producing SecurityResult objects with CVE-style mock IDs, severity, and confidence, all without any real scanner dependency.
- [ ] Day 4: Implement the TestWriterAgent that reads the submitted code, infers function signatures and edge cases, and generates stub pytest test functions as a string artifact, with the mock LLM producing deterministic test templates, returning a TestResult with generated code and coverage estimate.
- [ ] Day 5: Build the Orchestrator class that runs all three agents in parallel using concurrent.futures, then implements a ConsensusEngine that merges overlapping findings, escalates conflicts where agents disagree on severity, and produces a unified CodeReviewReport with per-finding votes and a final verdict.
- [ ] Day 6: Add a ConflictResolver that applies weighted voting rules, flags unresolved conflicts for human-in-the-loop review via an interactive CLI prompt requiring explicit approval before marking any finding as critical, and persist the final report as a JSON file; also add a ReportRenderer that pretty-prints the report to the terminal with color via colorama.
- [ ] Day 7: Write a comprehensive pytest suite covering AgentBase, each agent, the ConsensusEngine, ConflictResolver, and Orchestrator with mocked inputs; add a pyproject.toml with all dependencies and entry-point scripts; and include a sample_code/ directory with three realistic Python files that exercise all agent paths end-to-end.

_A comprehensive README with an architecture diagram is generated on Day 7._
