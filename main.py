from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audit import audit_receipt
from evals import GraderResult, run_evaluation
from extraction import extract_receipt

console = Console()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "data"
TEST_DIR = DATA_DIR / "test"
VALID_DIR = DATA_DIR / "valid"
EXTRACTION_GT_DIR = DATA_DIR / "ground_truth" / "extraction"
AUDIT_GT_DIR = DATA_DIR / "ground_truth" / "audit_results"


# ---------------------------------------------------------------------------
# Demo mode: extract + audit a single receipt
# ---------------------------------------------------------------------------

async def demo_single(client: AsyncAnthropic, model: str) -> None:
    """Extract and audit a single receipt, displaying results."""
    image = sorted(TEST_DIR.glob("*.jpg"))[0]
    console.print(f"\n[bold]Processing:[/bold] {image.name}\n")

    with console.status("Extracting receipt details..."):
        receipt = await extract_receipt(client, image, model)

    console.print(Panel(
        f"[bold]Merchant:[/bold] {receipt.merchant}\n"
        f"[bold]Location:[/bold] {receipt.location.city}, {receipt.location.state} {receipt.location.zipcode}\n"
        f"[bold]Time:[/bold] {receipt.time}\n"
        f"[bold]Subtotal:[/bold] {receipt.subtotal}  [bold]Tax:[/bold] {receipt.tax}  [bold]Total:[/bold] {receipt.total}\n"
        f"[bold]Notes:[/bold] {receipt.handwritten_notes}",
        title="Receipt Details",
    ))

    if receipt.items:
        items_table = Table(title="Line Items")
        items_table.add_column("Description")
        items_table.add_column("Category")
        items_table.add_column("Qty")
        items_table.add_column("Price")
        items_table.add_column("Total")
        for item in receipt.items:
            items_table.add_row(
                item.description or "-",
                item.category or "-",
                item.quantity or "-",
                item.item_price or "-",
                item.total or "-",
            )
        console.print(items_table)

    with console.status("Running audit..."):
        audit = await audit_receipt(client, receipt, model)

    def flag(val: bool) -> str:
        return "[red]TRUE[/red]" if val else "[green]FALSE[/green]"

    console.print(Panel(
        f"[bold]Not Travel Related:[/bold] {flag(audit.not_travel_related)}\n"
        f"[bold]Amount Over Limit:[/bold]  {flag(audit.amount_over_limit)}\n"
        f"[bold]Math Error:[/bold]          {flag(audit.math_error)}\n"
        f"[bold]Handwritten X:[/bold]       {flag(audit.handwritten_x)}\n"
        f"[bold]Needs Audit:[/bold]         {flag(audit.needs_audit)}\n\n"
        f"[bold]Reasoning:[/bold] {audit.reasoning}",
        title="Audit Decision",
        border_style="red" if audit.needs_audit else "green",
    ))


# ---------------------------------------------------------------------------
# Eval mode: run full evaluation suite
# ---------------------------------------------------------------------------

async def run_evals(client: AsyncAnthropic, model: str) -> None:
    """Run the full evaluation suite and display results."""
    completed = 0
    total = len(list(EXTRACTION_GT_DIR.glob("*.json")))

    def on_complete(result):
        nonlocal completed
        completed += 1
        console.print(
            f"  [{completed}/{total}] {result.image_path} — "
            f"extraction {result.extraction_pass_rate:.0%}, "
            f"audit {result.audit_pass_rate:.0%}"
        )

    console.print(f"\n[bold]Running evaluation on {total} receipts...[/bold]\n")

    results = await run_evaluation(
        client, model, TEST_DIR, EXTRACTION_GT_DIR, AUDIT_GT_DIR,
        on_complete=on_complete,
    )

    # Aggregate scores by grader name
    extraction_agg: dict[str, list[GraderResult]] = defaultdict(list)
    audit_agg: dict[str, list[GraderResult]] = defaultdict(list)

    for r in results:
        for g in r.extraction_grades:
            extraction_agg[g.name].append(g)
        for g in r.audit_grades:
            audit_agg[g.name].append(g)

    # Extraction summary table
    ext_table = Table(title="Extraction Results")
    ext_table.add_column("Grader", style="bold")
    ext_table.add_column("Passed", justify="right")
    ext_table.add_column("Total", justify="right")
    ext_table.add_column("Pass Rate", justify="right")
    ext_table.add_column("Avg Score", justify="right")

    for name, grades in extraction_agg.items():
        passed = sum(g.passed for g in grades)
        avg_score = sum(g.score for g in grades) / len(grades)
        max_score = grades[0].max_score
        ext_table.add_row(
            name,
            str(passed),
            str(len(grades)),
            f"{passed / len(grades):.0%}",
            f"{avg_score:.2f}/{max_score:.0f}",
        )

    console.print(ext_table)

    # Audit summary table
    aud_table = Table(title="Audit Results")
    aud_table.add_column("Grader", style="bold")
    aud_table.add_column("Passed", justify="right")
    aud_table.add_column("Total", justify="right")
    aud_table.add_column("Pass Rate", justify="right")
    aud_table.add_column("Avg Score", justify="right")

    for name, grades in audit_agg.items():
        passed = sum(g.passed for g in grades)
        avg_score = sum(g.score for g in grades) / len(grades)
        max_score = grades[0].max_score
        aud_table.add_row(
            name,
            str(passed),
            str(len(grades)),
            f"{passed / len(grades):.0%}",
            f"{avg_score:.2f}/{max_score:.0f}",
        )

    console.print(aud_table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main() -> None:
    load_dotenv()
    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    client = AsyncAnthropic()

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "demo":
        await demo_single(client, model)
    elif mode == "eval":
        await run_evals(client, model)
    else:
        console.print(f"[red]Unknown mode:[/red] {mode}")
        console.print("Usage: python main.py [demo|eval]")
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
