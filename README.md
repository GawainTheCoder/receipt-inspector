# Receipt Inspector

AI-powered receipt validation and expense auditing system built with the Anthropic Claude SDK. Inspired by the [OpenAI Eval-Driven System Design cookbook](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection), recreated using Claude's structured output via `messages.parse()`.

## How It Works

Receipt images are processed through a two-stage pipeline:

1. **Extraction** — A receipt image is sent to Claude with vision. The model returns structured data: merchant, location, line items, totals, and handwritten notes.
2. **Audit** — The extracted data is evaluated against four business rules to determine if the receipt needs manual review.

### Audit Rules

A receipt is flagged for audit if **any** of these are true:

| Rule | Trigger |
|------|---------|
| Not travel-related | Expense isn't for gas, hotel, airfare, or car rental |
| Amount over limit | Total exceeds $50 |
| Math error | Line items + tax don't match the total |
| Handwritten X | An "X" appears in handwritten notes |

## Eval Framework

A local evaluation system compares model predictions against 20 labeled ground truth receipts using three grader types:

- **String check** — Exact match for fields like city, state, zipcode, totals, and audit booleans
- **Text similarity** — Fuzzy matching (via `difflib.SequenceMatcher`) for merchant names and handwritten notes
- **LLM-judged** — Claude grades line item accuracy (missed/extra/mistakes) and audit reasoning quality

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-haiku-4-5
```

## Usage

```bash
# Extract and audit a single receipt
uv run python main.py demo

# Run the full eval suite across all 20 ground truth receipts
uv run python main.py eval
```

## Project Structure

```
models.py       — Pydantic schemas (ReceiptDetails, AuditDecision, etc.)
extraction.py   — Receipt image → structured data via Claude vision
audit.py        — Business rule evaluation via Claude
evals.py        — Local graders and evaluation pipeline
main.py         — CLI entry point (demo / eval modes)
data/data/
  test/         — 21 receipt images
  train/        — 146 receipt images
  valid/        — 47 receipt images
  ground_truth/
    extraction/     — 20 labeled extraction JSONs
    audit_results/  — 20 labeled audit decision JSONs
```

## Tech Stack

- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API with `messages.parse()` for structured output
- [Pydantic](https://docs.pydantic.dev/) — Data validation and schema definitions
- [Rich](https://github.com/Textualize/rich) — Terminal output formatting
