from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


IntentType = Literal[
    "return_request",
    "refund_request",
    "complaint",
    "inquiry",
    "product_inquiry",
    "other",
    "exchange_request",
    "order_status",
    "cancel_request",
]

UrgencyType = Literal["low", "medium", "high"]

CategoryType = Literal[
    "wrong_item",
    "damaged_product",
    "late_delivery",
    "missing_item",
    "return_policy",
    "refund_status",
    "payment_issue",
    "order_change",
    "general_question",
    "general_issue",
]


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Raw customer message")
    force_llm: bool = Field(default=False, description="Force LLM mode instead of heuristic")

    @model_validator(mode="after")
    def validate_message(self) -> "AnalyzeRequest":
        if not self.message.strip():
            raise ValueError("message cannot be blank")
        return self


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentType | None = Field(
        default=None,
        description="Detected customer intent or null for irrelevant/out-of-scope input",
    )
    urgency: UrgencyType | None = Field(
        default=None,
        description="Priority level or null for irrelevant/out-of-scope input",
    )
    category: CategoryType | None = Field(
        default=None,
        description="Fine-grained support category or null for irrelevant/out-of-scope input",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_reason: str = Field(
        ...,
        description="Short explanation of why the confidence score was assigned",
    )
    suggested_reply_en: str | None = Field(
        default=None,
        description="Helpful English reply, or null for refusal/out-of-scope cases",
    )
    suggested_reply_ar: str | None = Field(
        default=None,
        description="Helpful Arabic reply, or null for refusal/out-of-scope cases",
    )
    needs_human: bool
    escalation_reason: str | None = Field(
        default=None,
        description="Required when needs_human is true",
    )
    explanation: str | None = Field(
        default=None,
        description="Required when the message is irrelevant, nonsense, or out of scope",
    )
    heuristic_result: dict | None = Field(
        default=None,
        description="Heuristic triage result for comparison",
    )
    llm_result: dict | None = Field(
        default=None,
        description="LLM triage result for comparison",
    )

    @model_validator(mode="after")
    def validate_business_rules(self) -> "TriageResponse":
        text_fields = (
            "intent",
            "urgency",
            "category",
            "confidence_reason",
            "suggested_reply_en",
            "suggested_reply_ar",
            "escalation_reason",
            "explanation",
        )
        for field_name in text_fields:
            value = getattr(self, field_name)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} cannot be empty or whitespace")

        triage_fields = (
            self.intent,
            self.urgency,
            self.category,
            self.suggested_reply_en,
            self.suggested_reply_ar,
        )
        has_null_triage = any(value is None for value in triage_fields)
        has_any_triage = any(value is not None for value in triage_fields)

        if has_null_triage and has_any_triage:
            raise ValueError(
                "Structured triage fields must either all be populated or all be null"
            )

        if not has_any_triage and not self.explanation:
            raise ValueError("explanation is required when triage fields are null")

        if self.confidence < 0.4 and not self.needs_human:
            raise ValueError("needs_human must be true when confidence is below 0.4")

        if self.explanation and has_any_triage:
            raise ValueError("explanation must be null when triage fields are populated")

        if self.needs_human and not self.escalation_reason:
            raise ValueError("escalation_reason is required when needs_human is true")

        if not self.needs_human and self.escalation_reason is not None:
            raise ValueError("escalation_reason must be null when needs_human is false")

        return self
