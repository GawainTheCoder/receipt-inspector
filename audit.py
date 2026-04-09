from __future__ import annotations

from anthropic import AsyncAnthropic

from models import AuditDecision, ReceiptDetails

AUDIT_SYSTEM_PROMPT = """\
Evaluate this receipt data to determine if it needs auditing based on \
the following criteria:

1. NOT_TRAVEL_RELATED:
   - IMPORTANT:Travel-related expenses include: gas, fuel, hotel, airfare, car rental, \
tolls, parking at airports/travel destinations
   - If receipt IS for a travel expense, set FALSE
   - If receipt is NOT for a travel expense, set TRUE
   - Example: FUEL/GAS = FALSE (because gas IS travel-related)

2. AMOUNT_OVER_LIMIT: Total amount exceeds $50

3. MATH_ERROR: Line items + tax do not equal the total. Check that items sum \
to subtotal, and subtotal + tax = total (where available).

4. HANDWRITTEN_X: An "X" appears in the handwritten notes (standalone, not \
as part of a word)

For each criterion, determine if violated (true) or not (false). Provide \
clear reasoning for each decision. A receipt needs auditing if ANY criterion \
is violated (true).
"""


async def audit_receipt(
    client: AsyncAnthropic,
    receipt: ReceiptDetails,
    model: str = "claude-haiku-4-5",
) -> AuditDecision:
    """Determine if a receipt needs auditing based on defined criteria."""
    receipt_json = receipt.model_dump_json(indent=2)

    response = await client.messages.parse(
        model=model,
        max_tokens=2048,
        system=AUDIT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Evaluate this receipt:\n\n{receipt_json}",
            }
        ],
        output_format=AuditDecision,
    )

    if response.parsed_output is None:
        raise ValueError("Failed to parse audit decision")

    return response.parsed_output
