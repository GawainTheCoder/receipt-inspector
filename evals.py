from __future__ import annotations

import asyncio
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import AsyncAnthropic

from audit import audit_receipt
from extraction import extract_receipt
from models import AuditDecision, LineItem, ReceiptDetails


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    name: str
    score: float
    max_score: float
    passed: bool


@dataclass
class ReceiptEvalResult:
    image_path: str
    extraction_grades: list[GraderResult] = field(default_factory=list)
    audit_grades: list[GraderResult] = field(default_factory=list)

    @property
    def extraction_pass_rate(self) -> float:
        if not self.extraction_grades:
            return 0.0
        return sum(g.passed for g in self.extraction_grades) / len(self.extraction_grades)

    @property
    def audit_pass_rate(self) -> float:
        if not self.audit_grades:
            return 0.0
        return sum(g.passed for g in self.audit_grades) / len(self.audit_grades)


# ---------------------------------------------------------------------------
# Grader implementations
# ---------------------------------------------------------------------------

def string_check(predicted: str | None, expected: str | None) -> bool:
    """Exact string equality, case-insensitive, with None handling."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False
    return predicted.strip().lower() == expected.strip().lower()


def bool_check(predicted: bool, expected: bool) -> bool:
    return predicted == expected


def text_similarity(
    predicted: str | None, expected: str | None, threshold: float = 0.8
) -> tuple[float, bool]:
    """Fuzzy text similarity via SequenceMatcher. Returns (score, passed)."""
    if predicted is None and expected is None:
        return 1.0, True
    if predicted is None or expected is None:
        return 0.0, False
    score = difflib.SequenceMatcher(
        None, predicted.strip().lower(), expected.strip().lower()
    ).ratio()
    return score, score >= threshold


def notes_similarity(
    predicted: list[str], expected: list[str], threshold: float = 0.8
) -> tuple[float, bool]:
    """Compare two lists of handwritten notes using best-match pairing."""
    if not expected and not predicted:
        return 1.0, True
    if not expected or not predicted:
        return 0.0, False

    scores: list[float] = []
    used: set[int] = set()
    for exp_note in expected:
        best_score = 0.0
        best_idx = -1
        for i, pred_note in enumerate(predicted):
            if i in used:
                continue
            s = difflib.SequenceMatcher(
                None, pred_note.strip().lower(), exp_note.strip().lower()
            ).ratio()
            if s > best_score:
                best_score = s
                best_idx = i
        scores.append(best_score)
        if best_idx >= 0:
            used.add(best_idx)

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, avg >= threshold


# ---------------------------------------------------------------------------
# LLM-judged graders
# ---------------------------------------------------------------------------

MISSED_ITEMS_PROMPT = """\
Your task is to evaluate the correctness of a receipt extraction model.

The following items are the actual (correct) line items from a specific receipt.

{expected_items}

The following items are the line items extracted by the model.

{predicted_items}

Score 0 if the model missed any items from the receipt; otherwise score 1.

Line items are permitted to have small differences or extraction mistakes, but \
each item from the actual receipt must be present in some form in the model's \
output. Only evaluate whether there are MISSED items; ignore other mistakes or \
extra items.

Respond with ONLY a JSON object: {{"score": <number>}}
"""

EXTRA_ITEMS_PROMPT = """\
Your task is to evaluate the correctness of a receipt extraction model.

The following items are the actual (correct) line items from a specific receipt.

{expected_items}

The following items are the line items extracted by the model.

{predicted_items}

Score 0 if the model extracted any extra items not on the receipt; otherwise score 1.

Line items are permitted to have small differences or extraction mistakes, but \
each item in the model's output must correspond to a real item. Only evaluate \
whether there are EXTRA items; ignore other mistakes or missed items.

Respond with ONLY a JSON object: {{"score": <number>}}
"""

ITEM_MISTAKES_PROMPT = """\
Your task is to evaluate the correctness of a receipt extraction model.

The following items are the actual (correct) line items from a specific receipt.

{expected_items}

The following items are the line items extracted by the model.

{predicted_items}

Score 0 to 10 based on number and severity of mistakes in line items.

Score of 10 means the two lists are perfectly identical.

Remove 1 point for each minor mistake (typos, capitalization, category name \
differences), and up to 3 points for significant mistakes (incorrect quantity, \
price, or total, or categories that are not at all similar).

Respond with ONLY a JSON object: {{"score": <number>}}
"""

REASONING_PROMPT = """\
Evaluate the quality of reasoning for an audit decision on a receipt.

Rules for audit decisions:
1. Expenses must be travel-related
2. Expenses must not exceed $50
3. All math should be correct; line items plus tax should equal total
4. There must not be an "X" in handwritten notes

If ANY criterion is violated, the expense should be audited.

Receipt details:
{receipt_json}

GROUND TRUTH (correct decision):
{expected_json}

MODEL OUTPUT (decision to evaluate):
{predicted_json}

Evaluate:
1. For each of 4 criteria, did the model correctly score it as TRUE or FALSE?
2. Based on the model's scoring, did it reason appropriately about each criterion?
3. Is the model's reasoning logically sound, sufficient, and comprehensible?
4. Is the model's reasoning concise, without extraneous details?
5. Is the model's final decision to audit or not audit correct?

Rubric:
- 1 point for each of 4 criteria scored correctly (max 4)
- 3 points for sound reasoning that correctly applies the rules
- 3 points for the correct final audit decision

Total score between 0 and 10 inclusive.

Respond with ONLY a JSON object: {{"score": <number>}}
"""


async def llm_grade(
    client: AsyncAnthropic,
    prompt: str,
    model: str,
    max_score: float,
    pass_threshold: float,
) -> tuple[float, bool]:
    """Use Claude to grade a comparison. Returns (score, passed)."""
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    match = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if match:
        score = min(float(match.group(1)), max_score)
    else:
        score = 0.0
    return score, score >= pass_threshold


def _items_to_str(items: list[LineItem]) -> str:
    return json.dumps([item.model_dump() for item in items], indent=2)


# ---------------------------------------------------------------------------
# Single-receipt evaluation
# ---------------------------------------------------------------------------

async def evaluate_single(
    client: AsyncAnthropic,
    model: str,
    image_path: Path,
    extraction_gt_path: Path,
    audit_gt_path: Path,
) -> ReceiptEvalResult:
    """Run extraction + audit on one image, compare to ground truth."""
    correct_details = ReceiptDetails.model_validate_json(
        extraction_gt_path.read_text()
    )
    correct_audit = AuditDecision.model_validate_json(
        audit_gt_path.read_text()
    )

    predicted_details = await extract_receipt(client, image_path, model)
    predicted_audit = await audit_receipt(client, predicted_details, model)

    result = ReceiptEvalResult(image_path=image_path.name)

    # --- Extraction graders ---

    # String checks
    for name, pred, exp in [
        ("City", predicted_details.location.city, correct_details.location.city),
        ("State", predicted_details.location.state, correct_details.location.state),
        ("Zipcode", predicted_details.location.zipcode, correct_details.location.zipcode),
        ("Time", predicted_details.time, correct_details.time),
        ("Subtotal", predicted_details.subtotal, correct_details.subtotal),
        ("Tax", predicted_details.tax, correct_details.tax),
        ("Total", predicted_details.total, correct_details.total),
    ]:
        passed = string_check(pred, exp)
        result.extraction_grades.append(
            GraderResult(name=name, score=1.0 if passed else 0.0, max_score=1.0, passed=passed)
        )

    # Text similarity
    merchant_score, merchant_passed = text_similarity(
        predicted_details.merchant, correct_details.merchant
    )
    result.extraction_grades.append(
        GraderResult(name="Merchant", score=merchant_score, max_score=1.0, passed=merchant_passed)
    )

    notes_score, notes_passed = notes_similarity(
        predicted_details.handwritten_notes, correct_details.handwritten_notes
    )
    result.extraction_grades.append(
        GraderResult(name="Handwritten Notes", score=notes_score, max_score=1.0, passed=notes_passed)
    )

    # LLM-judged line item graders
    expected_str = _items_to_str(correct_details.items)
    predicted_str = _items_to_str(predicted_details.items)

    missed_score, missed_passed = await llm_grade(
        client,
        MISSED_ITEMS_PROMPT.format(expected_items=expected_str, predicted_items=predicted_str),
        model, max_score=1.0, pass_threshold=1.0,
    )
    result.extraction_grades.append(
        GraderResult(name="Missed Items", score=missed_score, max_score=1.0, passed=missed_passed)
    )

    extra_score, extra_passed = await llm_grade(
        client,
        EXTRA_ITEMS_PROMPT.format(expected_items=expected_str, predicted_items=predicted_str),
        model, max_score=1.0, pass_threshold=1.0,
    )
    result.extraction_grades.append(
        GraderResult(name="Extra Items", score=extra_score, max_score=1.0, passed=extra_passed)
    )

    item_score, item_passed = await llm_grade(
        client,
        ITEM_MISTAKES_PROMPT.format(expected_items=expected_str, predicted_items=predicted_str),
        model, max_score=10.0, pass_threshold=8.0,
    )
    result.extraction_grades.append(
        GraderResult(name="Item Accuracy", score=item_score, max_score=10.0, passed=item_passed)
    )

    # --- Audit graders ---

    for name, pred, exp in [
        ("Not Travel Related", predicted_audit.not_travel_related, correct_audit.not_travel_related),
        ("Amount Over Limit", predicted_audit.amount_over_limit, correct_audit.amount_over_limit),
        ("Math Error", predicted_audit.math_error, correct_audit.math_error),
        ("Handwritten X", predicted_audit.handwritten_x, correct_audit.handwritten_x),
        ("Needs Audit", predicted_audit.needs_audit, correct_audit.needs_audit),
    ]:
        passed = bool_check(pred, exp)
        result.audit_grades.append(
            GraderResult(name=name, score=1.0 if passed else 0.0, max_score=1.0, passed=passed)
        )

    # LLM-judged reasoning quality
    reasoning_score, reasoning_passed = await llm_grade(
        client,
        REASONING_PROMPT.format(
            receipt_json=predicted_details.model_dump_json(indent=2),
            expected_json=correct_audit.model_dump_json(indent=2),
            predicted_json=predicted_audit.model_dump_json(indent=2),
        ),
        model, max_score=10.0, pass_threshold=8.0,
    )
    result.audit_grades.append(
        GraderResult(name="Reasoning Quality", score=reasoning_score, max_score=10.0, passed=reasoning_passed)
    )

    return result


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

async def run_evaluation(
    client: AsyncAnthropic,
    model: str,
    test_dir: Path,
    extraction_gt_dir: Path,
    audit_gt_dir: Path,
    max_concurrent: int = 5,
    on_complete: callable = None,
) -> list[ReceiptEvalResult]:
    """Run evaluation across all ground truth pairs."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval_one(gt_json: Path) -> ReceiptEvalResult:
        async with semaphore:
            image_path = test_dir / f"{gt_json.stem}.jpg"
            audit_gt_path = audit_gt_dir / gt_json.name
            result = await evaluate_single(
                client, model, image_path, gt_json, audit_gt_path
            )
            if on_complete:
                on_complete(result)
            return result

    gt_files = sorted(extraction_gt_dir.glob("*.json"))
    tasks = [_eval_one(f) for f in gt_files]
    return await asyncio.gather(*tasks)
