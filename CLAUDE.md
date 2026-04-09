# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run python main.py demo    # Extract + audit a single receipt with rich output
uv run python main.py eval    # Run full eval suite across 20 ground truth receipts
```

Dependencies are managed with `uv` (lockfile: `uv.lock`). Install with `uv sync`.

## Architecture

This is an eval-driven receipt inspection system using the Anthropic Claude SDK. It processes receipt images through two stages, then evaluates accuracy against labeled ground truth.

**Pipeline: Image -> Extraction -> Audit -> Evaluation**

- `models.py` — Pydantic schemas (`ReceiptDetails`, `AuditDecision`, `EvaluationRecord`) shared across all modules
- `extraction.py` — Sends receipt images to Claude via `messages.parse(output_format=ReceiptDetails)` with vision (base64 image content blocks) to extract structured data
- `audit.py` — Takes extracted `ReceiptDetails`, sends to Claude via `messages.parse(output_format=AuditDecision)` to evaluate 4 business rules (not travel-related, over $50, math error, handwritten X)
- `evals.py` — Local eval framework with 3 grader types: `string_check` (exact match), `text_similarity` (difflib SequenceMatcher), `llm_grade` (Claude-judged scoring). Runs extraction+audit on test images and compares against ground truth.
- `main.py` — CLI entry point with `demo` and `eval` modes, uses `rich` for formatted output

**Key SDK pattern:** `client.messages.parse(output_format=PydanticModel)` returns a `ParsedMessage` where `.parsed_output` is a typed Pydantic instance. This is the structured output mechanism used throughout.

## Data Layout

- `data/data/test/*.jpg` — 21 test receipt images (20 have ground truth)
- `data/data/ground_truth/extraction/*.json` — Expected `ReceiptDetails` for each test image
- `data/data/ground_truth/audit_results/*.json` — Expected `AuditDecision` for each test image
- Ground truth filenames share the same stem as their corresponding `.jpg` images

## Environment

Requires `.env` with `ANTHROPIC_API_KEY`. Optional `CLAUDE_MODEL` (defaults to `claude-haiku-4-5`).
