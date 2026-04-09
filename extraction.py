from __future__ import annotations

import base64
from pathlib import Path

from anthropic import AsyncAnthropic

from models import ReceiptDetails

EXTRACTION_SYSTEM_PROMPT = """\
Given an image of a retail receipt, extract all relevant information and format \
it as a structured response.

# Task Description

Carefully examine the receipt image and identify:
1. Merchant name and store identification
2. Location information (city, state, ZIP code)
3. Date and time of purchase
4. All purchased items with:
   - Item description/name
   - Item code/SKU (if present)
   - Category (inferred from context if not explicit)
   - Regular price per item (if available)
   - Sale price per item (if discounted)
   - Quantity purchased
   - Total price for line item
5. Financial summary:
   - Subtotal before tax
   - Tax amount
   - Final total
6. Any handwritten notes or annotations (list each separately)

## Important Guidelines

- If information is unclear or missing, return null for that field
- Format dates as ISO format (YYYY-MM-DDTHH:MM:SS)
- Format all monetary values as decimal numbers without currency symbols
- Distinguish between printed text and handwritten notes
- Be precise with amounts and totals
- For ambiguous items, use best judgment based on context
- For gas/fuel receipts, item_price is the per-gallon price and quantity is gallons
"""


async def extract_receipt(
    client: AsyncAnthropic,
    image_path: str | Path,
    model: str = "claude-haiku-4-5",
) -> ReceiptDetails:
    """Extract structured details from a receipt image."""
    image_data = base64.standard_b64encode(
        Path(image_path).read_bytes()
    ).decode("utf-8")

    response = await client.messages.parse(
        model=model,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all information from this receipt image.",
                    },
                ],
            }
        ],
        output_format=ReceiptDetails,
    )

    if response.parsed_output is None:
        raise ValueError(f"Failed to parse extraction for {image_path}")

    return response.parsed_output
