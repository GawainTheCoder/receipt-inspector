from __future__ import annotations

from pydantic import BaseModel, Field


class Location(BaseModel):
    city: str | None = None
    state: str | None = None
    zipcode: str | None = None


class LineItem(BaseModel):
    description: str | None = None
    product_code: str | None = None
    category: str | None = None
    item_price: str | None = None
    sale_price: str | None = None
    quantity: str | None = None
    total: str | None = None


class ReceiptDetails(BaseModel):
    merchant: str | None = None
    location: Location = Field(default_factory=Location)
    time: str | None = None
    items: list[LineItem] = Field(default_factory=list)
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    handwritten_notes: list[str] = Field(default_factory=list)


class AuditDecision(BaseModel):
    not_travel_related: bool = Field(
        default=False, description="True if receipt is not travel-related"
    )
    amount_over_limit: bool = Field(
        default=False, description="True if total exceeds $50"
    )
    math_error: bool = Field(
        default=False, description="True if math errors exist in receipt"
    )
    handwritten_x: bool = Field(
        default=False, description="True if 'X' appears in handwritten notes"
    )
    reasoning: str = Field(
        default="", description="Explanation for audit decision"
    )
    needs_audit: bool = Field(
        default=False, description="Final audit determination"
    )


class EvaluationRecord(BaseModel):
    receipt_image_path: str
    correct_receipt_details: ReceiptDetails
    predicted_receipt_details: ReceiptDetails
    correct_audit_decision: AuditDecision
    predicted_audit_decision: AuditDecision
