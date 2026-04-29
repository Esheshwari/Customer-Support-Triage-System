from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import re


@dataclass(frozen=True)
class FAQEntry:
    id: str
    category: str
    title: str
    content: str


FAQ_DATA = [
    FAQEntry(
        id="faq-returns-14-days",
        category="return_policy",
        title="Returns within 14 days",
        content=(
            "Customers can request a return within 14 days of delivery if the item is "
            "unused and in its original packaging."
        ),
    ),
    FAQEntry(
        id="faq-damaged-photo",
        category="damaged_product",
        title="Damaged item evidence",
        content=(
            "For damaged items, support should ask for clear photos of the product and "
            "the outer packaging so the team can validate the issue quickly."
        ),
    ),
    FAQEntry(
        id="faq-late-delivery",
        category="late_delivery",
        title="Late delivery checks",
        content=(
            "If an order is delayed, support should apologize, confirm the order number, "
            "and offer to check the courier status."
        ),
    ),
    FAQEntry(
        id="faq-wrong-item",
        category="wrong_item",
        title="Wrong item received",
        content=(
            "If a customer receives the wrong item, support should apologize and request "
            "the order number plus a photo of the received item label."
        ),
    ),
    FAQEntry(
        id="faq-refund-timing",
        category="refund_status",
        title="Refund timing",
        content=(
            "Approved refunds are typically returned to the original payment method within "
            "5 to 10 business days depending on the bank."
        ),
    ),
    FAQEntry(
        id="faq-order-change",
        category="order_change",
        title="Order changes",
        content=(
            "Order changes or cancellations are only possible before dispatch. Support "
            "should avoid promising changes until order status is confirmed."
        ),
    ),
]


TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    length = max(len(tokens), 1)
    return {token: count / length for token, count in counts.items()}


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve_faq_context(message: str, top_k: int = 2) -> list[FAQEntry]:
    query_tf = _tf(tokenize(message))
    scored: list[tuple[float, FAQEntry]] = []
    for entry in FAQ_DATA:
        entry_tf = _tf(tokenize(f"{entry.title} {entry.content} {entry.category}"))
        score = _cosine_similarity(query_tf, entry_tf)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]
