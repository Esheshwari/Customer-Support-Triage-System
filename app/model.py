from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.rag import FAQEntry, retrieve_faq_context
from app.schema import TriageResponse


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-32b:free"
LOW_CONFIDENCE_HUMAN_THRESHOLD = 0.4
STRONG_MATCH_THRESHOLD = 0.75
CRITICAL_CATEGORIES = {
    "damaged_product",
    "wrong_item",
    "payment_issue",
}
STRONG_SIGNAL_CATEGORIES = {
    "missing_item",
    "damaged_product",
    "wrong_item",
}

TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[a-zA-Z0-9']+")


SYSTEM_PROMPT = """
You are a production customer-support triage engine for a Mumzworld-like e-commerce platform.

Return EXACTLY one JSON object and nothing else.

Hard output contract:
- Output must be valid JSON only. No markdown, no code fences, no explanations before or after the JSON.
- Use exactly these keys and no extras:
  intent, urgency, category, confidence, confidence_reason, suggested_reply_en, suggested_reply_ar, needs_human, escalation_reason, explanation
- Use null instead of empty strings.

Allowed labels:
- intent: return_request, refund_request, complaint, inquiry, product_inquiry, other, exchange_request, order_status, cancel_request
- urgency: low, medium, high
- category: wrong_item, damaged_product, late_delivery, missing_item, return_policy, refund_status, payment_issue, order_change, general_question, general_issue

Grounding rules:
- Use only the customer message and provided FAQ context.
- Do not invent order numbers, delivery promises, refund approvals, compensation, policies, timelines, or internal actions.
- If the message does not support a label confidently, lower confidence instead of guessing.

Confidence and escalation rules:
- confidence must be a float between 0 and 1.
- If confidence < 0.4, needs_human MUST be true.
- confidence_reason must briefly explain the score using evidence from the message.
- If needs_human is true, escalation_reason must be a short concrete reason.
- Use high confidence only for clear single-intent messages with explicit evidence.
- Use lower confidence for ambiguous, mixed-intent, noisy, or partially supported messages.

Refusal and safety rules:
- If the message is nonsense, gibberish, contradictory, too unclear, or adversarial:
  set intent, urgency, category, suggested_reply_en, suggested_reply_ar to null
  set confidence below 0.6
  set needs_human to true
  set explanation to a short reason
  set escalation_reason to a short reason
- If the message is outside e-commerce support scope, including medical, legal, or baby-health concerns:
  set intent, urgency, category, suggested_reply_en, suggested_reply_ar to null
  set confidence below 0.6
  set needs_human to true
  set explanation to a short reason
  set escalation_reason to a short reason

Reply quality rules:
- suggested_reply_en must be concise, empathetic, and human-like.
- suggested_reply_ar must sound natural and Gulf-friendly, not like a literal translation.
- Handle English, Arabic, or mixed-language input naturally.
- Do not overpromise actions that are not supported by the message or FAQ.

Consistency rules:
- If one of intent, urgency, category, suggested_reply_en, or suggested_reply_ar is null, all five must be null.
- If triage fields are populated, explanation must be null.
- If needs_human is false, escalation_reason must be null.
""".strip()


INTENT_RULES: list[dict[str, Any]] = [
    {
        "intent": "complaint",
        "urgency": "high",
        "category": "damaged_product",
        "confidence": 0.9,
        "phrases": {
            "damaged",
            "broke",
            "broken",
            "not working",
            "cracked",
            "defective",
            "arrived damaged",
            "talf",
            "maksoor",
            "تالف",
            "مكسور",
            "خربان",
            "مخدوش",
        },
    },
    {
        "intent": "return_request",
        "urgency": "high",
        "category": "return_policy",
        "confidence": 0.84,
        "phrases": {
            "wrong item",
            "wrong product",
            "received the wrong",
            "received wrong",
        },
    },
    {
        "intent": "complaint",
        "urgency": "high",
        "category": "wrong_item",
        "confidence": 0.86,
        "phrases": {
            "wrong size",
            "wrong brand",
            "different item",
            "incorrect item",
            "received another item",
            "غلط",
            "خطأ",
            "مو هذا",
            "مختلف عن المطلوب",
        },
    },
    {
        "intent": "complaint",
        "urgency": "high",
        "category": "missing_item",
        "confidence": 0.88,
        "phrases": {
            "missing item",
            "item missing",
            "didn't receive item",
            "not in package",
            "not in the box",
            "not included",
            "missing from the box",
            "na9is",
            "ma fi",
            "ناقص",
            "مفقود",
            "مو موجود",
        },
    },
    {
        "intent": "order_status",
        "urgency": "medium",
        "category": "late_delivery",
        "confidence": 0.81,
        "phrases": {
            "shipped",
            "tracking",
            "late",
            "delayed",
            "delivery",
            "not arrived",
            "where is my order",
            "delivery delay",
            "طلب",
            "طلبي",
            "شحن",
            "تأخر",
            "متأخر",
            "تأخير",
            "وين الشحنة",
            "ما وصل",
        },
    },
    {
        "intent": "cancel_request",
        "urgency": "medium",
        "category": "order_change",
        "confidence": 0.84,
        "phrases": {
            "cancel",
            "cancel my order",
            "cancellation",
            "الغاء",
            "إلغاء",
            "ألغي الطلب",
        },
    },
    {
        "intent": "refund_request",
        "urgency": "medium",
        "category": "refund_status",
        "confidence": 0.82,
        "phrases": {
            "refund",
            "money back",
            "refund status",
            "refund not received",
            "استرداد",
            "استرجاع المبلغ",
            "المبلغ ما رجع",
        },
    },
    {
        "intent": "return_request",
        "urgency": "medium",
        "category": "return_policy",
        "confidence": 0.78,
        "phrases": {
            "return",
            "send back",
            "replace",
            "exchange",
            "استرجاع",
            "ارجاع",
            "إرجاع",
            "استبدال",
        },
    },
    {
        "intent": "complaint",
        "urgency": "high",
        "category": "payment_issue",
        "confidence": 0.8,
        "phrases": {
            "payment",
            "charged twice",
            "card charged",
            "declined",
            "visa",
            "mastercard",
            "بطاقة",
            "دفع",
            "خصم مرتين",
            "انسحب المبلغ",
        },
    },
    {
        "intent": "inquiry",
        "urgency": "medium",
        "category": "order_change",
        "confidence": 0.69,
        "phrases": {
            "change address",
            "edit order",
            "modify order",
            "update address",
            "تعديل الطلب",
            "أغير العنوان",
        },
    },
]

GENERAL_SUPPORT_TERMS = {
    "order",
    "delivery",
    "refund",
    "return",
    "item",
    "package",
    "payment",
    "product",
    "shipment",
    "account",
    "support",
    "طلب",
    "طلبي",
    "شحنة",
    "الشحنة",
    "توصيل",
    "استرجاع",
    "ارجاع",
    "إرجاع",
    "استرداد",
    "منتج",
    "الطلب",
    "حساب",
}

OUT_OF_SCOPE_TERMS = {
    "weather",
    "recipe",
    "football",
    "stock",
    "poem",
    "joke",
    "قانوني",
    "محامي",
    "lawsuit",
    "legal",
    "طقس",
    "وصفة",
    "مباراة",
    "نكتة",
}

MEDICAL_TERMS = {
    "medical",
    "doctor",
    "fever",
    "rash",
    "dose",
    "medicine",
    "baby health",
    "pregnant",
    "poison",
    "allergy",
    "seizure",
    "طبي",
    "حرارة",
    "حمى",
    "طفح",
    "جرعة",
    "دواء",
    "صحة الطفل",
    "حامل",
    "حساسية",
}

AMBIGUOUS_TERMS = {
    "issue",
    "problem",
    "help",
    "not happy",
    "something wrong",
    "problem with my order",
    "مشكلة",
    "مو راضي",
    "احتاج مساعدة",
    "ساعدوني",
}

PRODUCT_INQUIRY_TERMS = {
    "organic",
    "available",
    "availability",
    "have",
    "carry",
    "baby food",
    "food",
    "product",
    "منتج",
    "متوفر",
    "توفر",
    "عضوي",
    "طعام",
}

POLITE_TERMS = {
    "can you help",
    "could you help",
    "please assist",
    "help me",
}

QUESTION_TERMS = {
    "how",
    "what",
    "when",
    "where",
    "can i",
    "could you",
    "please help",
    "هل",
    "كيف",
    "متى",
    "وين",
    "لو سمحت",
    "ممكن",
}


@dataclass
class ModelConfig:
    api_key: str | None
    model_name: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class HeuristicDecision:
    intent: str | None
    urgency: str | None
    category: str | None
    confidence: float
    confidence_reason: str
    needs_human: bool
    escalation_reason: str | None
    explanation: str | None
    faq_context: list[FAQEntry]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(_normalize_text(text))]


def _contains_any_phrase(text: str, phrases: set[str]) -> list[str]:
    normalized_text = _normalize_text(text)
    matches: list[str] = []
    for phrase in phrases:
        if _normalize_text(phrase) in normalized_text:
            matches.append(phrase)
    return matches


def _find_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for rule in INTENT_RULES:
        matched_phrases = _contains_any_phrase(text, rule["phrases"])
        if matched_phrases:
            matches.append({**rule, "matched_phrases": matched_phrases})
    return matches


def _is_repetitive_gibberish(tokens: list[str]) -> bool:
    if not tokens:
        return True
    unique_tokens = set(tokens)
    if len(tokens) >= 4 and len(unique_tokens) <= 2:
        return True
    if len(tokens) >= 4 and all(len(token) >= 3 for token in tokens):
        vowelish = "aeiouاوي"
        if sum(1 for token in tokens if not any(char in vowelish for char in token)) >= len(tokens) - 1:
            return True
    return False


def _looks_noisy(tokens: list[str]) -> bool:
    gibberish_like = 0
    for token in tokens:
        if len(token) < 3 or token.isdigit():
            continue
        if token in GENERAL_SUPPORT_TERMS or token in MEDICAL_TERMS:
            continue
        if not any(char in "aeiouاوي" for char in token):
            gibberish_like += 1
    contradiction = {"maybe", "yes", "no"}.issubset(set(tokens))
    return gibberish_like >= 2 or contradiction


def _is_gibberish(text: str) -> bool:
    return (
        len(text) > 5
        and sum(char.isalpha() for char in text) / len(text) < 0.5
    )


def _is_low_quality_text(text: str) -> bool:
    letters = sum(char.isalpha() for char in text)
    ratio = letters / max(len(text), 1)
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    contradiction = {"maybe", "yes", "no"}.issubset(set(tokens))
    gibberish_like = sum(
        1
        for token in tokens
        if len(token) >= 3 and not any(char in "aeiouاوي" for char in token)
    )
    return (
        len(text) > 8
        and (ratio < 0.5 or " " not in text or contradiction or gibberish_like >= 2)
    )


def _make_refusal(
    confidence: float,
    confidence_reason: str,
    explanation: str,
    escalation_reason: str,
) -> HeuristicDecision:
    return HeuristicDecision(
        intent=None,
        urgency=None,
        category=None,
        confidence=confidence,
        confidence_reason=confidence_reason,
        needs_human=True,
        escalation_reason=escalation_reason,
        explanation=explanation,
        faq_context=[],
    )


def _reason_about_message(message: str) -> HeuristicDecision:
    normalized = _normalize_text(message.strip())
    tokens = _tokenize(message)
    faq_context = retrieve_faq_context(message)
    matches = _find_matches(normalized)
    support_signal = _contains_any_phrase(normalized, GENERAL_SUPPORT_TERMS)
    support_like_tokens = {
        "order",
        "delivery",
        "refund",
        "return",
        "item",
        "account",
        "طلب",
        "طلبي",
        "الطلب",
        "شحن",
        "توصيل",
    }
    support_signal = support_signal or any(token in support_like_tokens for token in tokens)
    ambiguous = _contains_any_phrase(normalized, AMBIGUOUS_TERMS)
    has_question = _contains_any_phrase(normalized, QUESTION_TERMS) or "?" in message
    polite_signal = _contains_any_phrase(normalized, POLITE_TERMS)
    product_signal = _contains_any_phrase(normalized, PRODUCT_INQUIRY_TERMS)
    has_strong_match = any(match["confidence"] >= STRONG_MATCH_THRESHOLD for match in matches)
    clear_strong_issue = (
        len([match for match in matches if match["category"] in STRONG_SIGNAL_CATEGORIES]) == 1
        and all(
            match["category"] in STRONG_SIGNAL_CATEGORIES
            or match["category"] in {"late_delivery", "general_question"}
            for match in matches
        )
    )

    if len(tokens) < 2:
        return _make_refusal(
            confidence=0.12,
            confidence_reason="The message is too short to classify safely.",
            explanation="The message is too short to determine a support action.",
            escalation_reason="Human review is needed because the request is too unclear.",
        )

    if _contains_any_phrase(normalized, MEDICAL_TERMS):
        return _make_refusal(
            confidence=0.1,
            confidence_reason="The message contains a medical or baby-health concern.",
            explanation="Medical or baby-health concerns are outside e-commerce support scope.",
            escalation_reason="Human review is required because medical concerns must not be automated here.",
        )

    if _contains_any_phrase(normalized, OUT_OF_SCOPE_TERMS):
        return _make_refusal(
            confidence=0.18,
            confidence_reason="The request is outside the supported e-commerce domain.",
            explanation="The message is out of scope for e-commerce customer support.",
            escalation_reason="Human review is needed because the request is outside support scope.",
        )

    if _is_repetitive_gibberish(tokens) and not matches:
        return _make_refusal(
            confidence=0.1,
            confidence_reason="The text looks repetitive or nonsensical.",
            explanation="The message appears to be nonsense or corrupted text.",
            escalation_reason="Human review is needed because the message cannot be interpreted safely.",
        )

    if _is_low_quality_text(message) and _looks_noisy(tokens) and not clear_strong_issue:
        return _make_refusal(
            confidence=0.12,
            confidence_reason="The message appears too noisy or malformed to interpret safely.",
            explanation="The message looks like low-quality or contradictory text.",
            escalation_reason="Human review is needed because the message cannot be interpreted safely.",
        )

    if _is_low_quality_text(message) and not matches:
        return _make_refusal(
            confidence=0.12,
            confidence_reason="The message appears too noisy or malformed to interpret safely.",
            explanation="The message looks like low-quality or corrupted text.",
            escalation_reason="Human review is needed because the message cannot be interpreted safely.",
        )

    if _is_gibberish(message) and not has_strong_match:
        return _make_refusal(
            confidence=0.12,
            confidence_reason="The message appears too noisy or malformed to interpret safely.",
            explanation="The message looks like gibberish or heavily corrupted text.",
            escalation_reason="Human review is needed because the message cannot be interpreted safely.",
        )

    if _looks_noisy(tokens) and len(tokens) <= 12 and not has_strong_match:
        return _make_refusal(
            confidence=0.22,
            confidence_reason="The message mixes support terms with noisy or contradictory text.",
            explanation="The message is too noisy or contradictory to classify safely.",
            escalation_reason="Human review is needed because the message may be adversarial or corrupted.",
        )

    if len(matches) >= 2 and not any(
        match["category"] in STRONG_SIGNAL_CATEGORIES for match in matches
    ):
        return HeuristicDecision(
            intent="complaint",
            urgency="high",
            category="general_issue",
            confidence=0.39,
            confidence_reason="Multiple competing issues detected.",
            needs_human=True,
            escalation_reason="Multiple issues require human handling.",
            explanation=None,
            faq_context=faq_context,
        )

    if has_question and polite_signal:
        return HeuristicDecision(
            intent="inquiry",
            urgency="low",
            category="general_question",
            confidence=0.55,
            confidence_reason="The customer is politely asking for assistance without a specific product issue.",
            needs_human=True,
            escalation_reason="Request is unclear.",
            explanation=None,
            faq_context=faq_context,
        )

    if ambiguous and support_signal:
        return HeuristicDecision(
            intent="complaint",
            urgency="low",
            category="general_issue",
            confidence=0.45,
            confidence_reason="Mixed or unclear support issue.",
            needs_human=True,
            escalation_reason="Ambiguous issue requires human clarification.",
            explanation=None,
            faq_context=faq_context,
        )

    strong_matches = [
        match for match in matches if match["category"] in STRONG_SIGNAL_CATEGORIES
    ]
    if len(strong_matches) == 1:
        match = strong_matches[0]
        confidence = match["confidence"]
        confidence_reason = "Strong single issue detected despite additional noise."
        if len(matches) > 1:
            if match["category"] == "missing_item" and all(
                other["category"] in {"late_delivery", "general_question"}
                for other in matches
                if other is not match
            ):
                confidence = max(confidence, 0.85)
                confidence_reason = (
                    "Missing item is clearly the main issue, and generic support noise is ignored."
                )
            else:
                confidence = min(confidence, 0.59)
                confidence_reason = (
                    "A strong issue matches the message, but competing evidence lowers confidence."
                )
        return HeuristicDecision(
            intent=match["intent"],
            urgency=match["urgency"],
            category=match["category"],
            confidence=confidence,
            confidence_reason=confidence_reason,
            needs_human=True,
            escalation_reason="Sensitive issue requires human handling.",
            explanation=None,
            faq_context=faq_context,
        )
    if len(strong_matches) > 1:
        return HeuristicDecision(
            intent="complaint",
            urgency="high",
            category="general_issue",
            confidence=0.39,
            confidence_reason="Multiple strong issues are competing in the message.",
            needs_human=True,
            escalation_reason="Multiple issues require human handling.",
            explanation=None,
            faq_context=faq_context,
        )

    if matches:
        match = matches[0]
        category = match["category"]
        confidence = match["confidence"]
        if category == "missing_item" and len(matches) == 1:
            confidence = max(confidence, 0.85)
        if (
            len(matches) == 1
            and len(tokens) < 5
            and category not in STRONG_SIGNAL_CATEGORIES
        ):
            confidence = min(confidence, 0.55)
        force_escalation = category in CRITICAL_CATEGORIES
        needs_human = (
            confidence < LOW_CONFIDENCE_HUMAN_THRESHOLD
            or force_escalation
        )
        escalation_reason = (
            "Sensitive customer issue requires human handling."
            if force_escalation
            else (
                "Low confidence requires human review."
                if confidence < LOW_CONFIDENCE_HUMAN_THRESHOLD
                else None
            )
        )
        return HeuristicDecision(
            intent=match["intent"],
            urgency=match["urgency"],
            category=category,
            confidence=confidence,
            confidence_reason=(
                f"The message explicitly mentions {', '.join(match['matched_phrases'][:2])}, "
                f"which strongly supports {category}."
            ),
            needs_human=needs_human,
            escalation_reason=escalation_reason,
            explanation=None,
            faq_context=faq_context,
        )

    # Product questions should map to a concrete inquiry label instead of being treated as vague support.
    if product_signal and has_question:
        return HeuristicDecision(
            intent="product_inquiry",
            urgency="low",
            category="general_question",
            confidence=0.72,
            confidence_reason="The message asks about product availability or attributes in a clear way.",
            needs_human=False,
            escalation_reason=None,
            explanation=None,
            faq_context=faq_context,
        )

    if has_question:
        return HeuristicDecision(
            intent="other",
            urgency="low",
            category="general_question",
            confidence=0.35,
            confidence_reason="The customer is asking for help, but the topic is too broad to route confidently.",
            needs_human=True,
            escalation_reason="The request is not specific enough for confident automation.",
            explanation=None,
            faq_context=faq_context,
        )

    if support_signal:
        return HeuristicDecision(
            intent="other",
            urgency="low",
            category="general_question",
            confidence=0.36,
            confidence_reason="The message appears support-related, but it lacks enough detail for a precise label.",
            needs_human=True,
            escalation_reason="A human should review because the request needs clarification.",
            explanation=None,
            faq_context=faq_context,
        )

    return _make_refusal(
        confidence=0.2,
        confidence_reason="The message does not contain enough relevant e-commerce support context.",
        explanation="The message is unclear or outside the support categories handled here.",
        escalation_reason="Human review is needed because the request cannot be routed safely.",
    )


def _reply_templates(category: str, urgency: str, faq_context: list[FAQEntry]) -> tuple[str, str]:
    context_hint = faq_context[0].content if faq_context else None

    english_base = {
        "damaged_product": (
            "I'm sorry your item arrived damaged. Please share your order number and clear photos "
            "of the product and outer packaging so we can review this properly."
        ),
        "wrong_item": (
            "I'm sorry you received the wrong item. Please send your order number and a photo of "
            "the item label so we can check what happened and guide you on the next step."
        ),
        "late_delivery": (
            "I'm sorry for the delay. Please share your order number so we can check the latest "
            "delivery status and update you accurately."
        ),
        "missing_item": (
            "I'm sorry an item seems to be missing. Please send your order number and tell us which "
            "item was missing so we can investigate."
        ),
        "refund_status": (
            "I can help with that. Please share your order number so we can review the refund status "
            "and explain the next step."
        ),
        "return_policy": (
            "I can help with your return request. Please share your order number and confirm whether "
            "the item is unused so we can guide you correctly."
        ),
        "payment_issue": (
            "I'm sorry about the payment issue. Please share your order number and a short summary of "
            "what happened so we can review it safely."
        ),
        "order_change": (
            "Please share your order number and the change you need, and we'll check whether the order "
            "can still be updated before dispatch."
        ),
        "general_issue": (
            "I'm sorry there seems to be an issue with your order. Please share your order number and a bit more detail so we can help properly."
        ),
        "general_question": (
            "Thanks for reaching out. Please share a bit more detail, and include your order number if "
            "this is about a specific order, so we can help accurately."
        ),
    }[category]

    arabic_base = {
        "damaged_product": (
            "نعتذر لأن المنتج وصل بحالة غير سليمة. أرسلي رقم الطلب مع صور واضحة للمنتج والتغليف الخارجي، "
            "وبنراجع الحالة بدقة ونفيدك بالخطوة المناسبة."
        ),
        "wrong_item": (
            "نعتذر لأن المنتج اللي وصلك مختلف عن المطلوب. أرسلي رقم الطلب وصورة لملصق المنتج، "
            "وبنتابع معك الحل الأنسب بأسرع وقت."
        ),
        "late_delivery": (
            "نعتذر عن تأخر التوصيل. أرسلي رقم الطلب، وبنتحقق من حالة الشحنة ونرجع لك بتحديث واضح."
        ),
        "missing_item": (
            "نعتذر لأن جزءًا من الطلب غير موجود. أرسلي رقم الطلب واسم القطعة الناقصة، "
            "وبنراجع الموضوع معك بشكل سريع."
        ),
        "refund_status": (
            "أكيد نقدر نساعدك. أرسلي رقم الطلب حتى نتحقق من حالة الاسترداد ونوضح لك الخطوة التالية."
        ),
        "return_policy": (
            "أكيد نقدر نساعدك في الإرجاع. أرسلي رقم الطلب وأكدي لنا إذا كان المنتج غير مستخدم، "
            "وبنوجهك حسب السياسة المتاحة."
        ),
        "payment_issue": (
            "نعتذر عن مشكلة الدفع. أرسلي رقم الطلب ووصفًا مختصرًا لما صار، "
            "وبنراجع الحالة بطريقة آمنة."
        ),
        "order_change": (
            "أرسلي رقم الطلب والتعديل المطلوب، وبنتأكد لك إذا كان ما زال ممكنًا قبل الشحن."
        ),
        "general_issue": (
            "نعتذر عن الإزعاج. أرسلي رقم الطلب وتفاصيل أكثر عن المشكلة حتى نقدر نعالجها بشكل صحيح."
        ),
        "general_question": (
            "شكرًا لتواصلك معنا. شاركينا تفاصيل أكثر، ومعها رقم الطلب إذا كان الموضوع مرتبطًا بطلب معيّن، "
            "حتى نقدر نخدمك بشكل أدق."
        ),
    }[category]

    if context_hint and category in {"return_policy", "refund_status", "order_change"}:
        english_base = f"{english_base} Based on our FAQ, {context_hint.lower()}"
        arabic_base += " وبناءً على المعلومات المتاحة لدينا، بنأكد لك التفاصيل بعد مراجعة الطلب."

    if urgency == "high" and category in {"damaged_product", "wrong_item", "missing_item"}:
        english_base += " We'll prioritize the review once we receive these details."
        arabic_base += " وبمجرد وصول التفاصيل بنعطي الحالة أولوية في المتابعة."

    return english_base, arabic_base


def _heuristic_response(message: str) -> TriageResponse:
    decision = _reason_about_message(message)
    if decision.intent is None:
        return TriageResponse(
            intent=None,
            urgency=None,
            category=None,
            confidence=decision.confidence,
            confidence_reason=decision.confidence_reason,
            suggested_reply_en=None,
            suggested_reply_ar=None,
            needs_human=True,
            escalation_reason=decision.escalation_reason,
            explanation=decision.explanation,
        )

    suggested_reply_en, suggested_reply_ar = _reply_templates(
        category=decision.category,
        urgency=decision.urgency,
        faq_context=decision.faq_context,
    )
    return TriageResponse(
        intent=decision.intent,
        urgency=decision.urgency,
        category=decision.category,
        confidence=decision.confidence,
        confidence_reason=decision.confidence_reason,
        suggested_reply_en=suggested_reply_en,
        suggested_reply_ar=suggested_reply_ar,
        needs_human=decision.needs_human,
        escalation_reason=decision.escalation_reason,
        explanation=None,
    )


def _repair_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(payload)
    for key, value in list(repaired.items()):
        if isinstance(value, str) and not value.strip():
            repaired[key] = None

    confidence = repaired.get("confidence")
    needs_human = repaired.get("needs_human")
    explanation = repaired.get("explanation")

    if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_HUMAN_THRESHOLD:
        repaired["needs_human"] = True
        if not repaired.get("escalation_reason"):
            repaired["escalation_reason"] = "Low-confidence output requires human review."

    if repaired.get("needs_human") and not repaired.get("escalation_reason"):
        repaired["escalation_reason"] = "Human review is required for this case."

    if not repaired.get("confidence_reason"):
        repaired["confidence_reason"] = "The model did not provide a detailed confidence explanation."

    triage_keys = ("intent", "urgency", "category", "suggested_reply_en", "suggested_reply_ar")
    if any(repaired.get(key) is None for key in triage_keys):
        for key in triage_keys:
            repaired[key] = None
        if explanation is None:
            repaired["explanation"] = "The message could not be handled safely by the model."
        repaired["needs_human"] = True
        if not repaired.get("escalation_reason"):
            repaired["escalation_reason"] = "Human review is needed because the model could not complete a safe triage."

    if repaired.get("needs_human") is False:
        repaired["escalation_reason"] = None

    return repaired


def _build_user_prompt(message: str, faq_context: list[FAQEntry]) -> str:
    faq_lines = [
        f"- {entry.title} ({entry.category}): {entry.content}" for entry in faq_context
    ]
    faq_block = "\n".join(faq_lines) if faq_lines else "- No relevant FAQ context found."
    return (
        "Customer message:\n"
        f"{message}\n\n"
        "Relevant FAQ context:\n"
        f"{faq_block}\n\n"
        "Return the JSON object now."
    )


def _call_openrouter(message: str, config: ModelConfig) -> dict[str, Any]:
    faq_context = retrieve_faq_context(message)
    payload = {
        "model": config.model_name,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(message, faq_context)},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "mumzassist-triage-prototype",
    }
    with httpx.Client(timeout=config.timeout_seconds) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _extract_json(content)


def analyze_message(
    message: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> TriageResponse:
    config = ModelConfig(
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        model_name=model_name or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
    )
    fallback = _heuristic_response(message)
    if not config.api_key:
        return fallback

    try:
        parsed = _call_openrouter(message=message, config=config)
        repaired = _repair_llm_payload(parsed)
        response = TriageResponse.model_validate(repaired)
        if response.confidence < LOW_CONFIDENCE_HUMAN_THRESHOLD and not response.needs_human:
            raise ValueError("LLM output violated low-confidence escalation rule")
        return response
    except Exception:
        return fallback