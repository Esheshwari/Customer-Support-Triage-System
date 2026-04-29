from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from app.model import analyze_message
from app.schema import TriageResponse


@dataclass(frozen=True)
class EvalCase:
    name: str
    message: str
    expected_intent: str | None
    expected_refusal: bool
    expect_human: bool
    min_confidence: float | None = None
    max_confidence: float | None = None
    notes: str = ""


TEST_CASES = [
    EvalCase(
        name="return_english",
        message="I want to return the stroller I received yesterday because it is unopened.",
        expected_intent="return_request",
        expected_refusal=False,
        expect_human=False,
        min_confidence=0.6,
        notes="Straightforward return request.",
    ),
    EvalCase(
        name="refund_english",
        message="My refund still has not reached my card. Can you check the status?",
        expected_intent="refund_request",
        expected_refusal=False,
        expect_human=False,
        min_confidence=0.6,
        notes="Clear refund follow-up.",
    ),
    EvalCase(
        name="damaged_arabic",
        message="المنتج وصل مكسور وأحتاج حل بسرعة.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        min_confidence=0.6,
        notes="Arabic damaged-item complaint should escalate.",
    ),
    EvalCase(
        name="late_delivery_arabic",
        message="طلبي متأخر من أسبوع، وين الشحنة؟",
        expected_intent="order_status",
        expected_refusal=False,
        expect_human=False,
        min_confidence=0.6,
        notes="Arabic order-status request.",
    ),
    EvalCase(
        name="mixed_wrong_item",
        message="I ordered diapers but وصلني wrong size and wrong brand.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        min_confidence=0.6,
        notes="Mixed-language wrong-item complaint.",
    ),
    EvalCase(
        name="cancel_request",
        message="Please cancel my order before it ships.",
        expected_intent="cancel_request",
        expected_refusal=False,
        expect_human=False,
        min_confidence=0.6,
        notes="Simple cancellation.",
    ),
    EvalCase(
        name="payment_issue",
        message="I think my card was charged twice for the same baby monitor order.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        min_confidence=0.6,
        notes="Sensitive payment issue should escalate.",
    ),
    EvalCase(
        name="ambiguous_low_confidence",
        message="I have an issue with my recent order and I am not happy.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        max_confidence=0.59,
        notes="Ambiguous support issue must stay low-confidence.",
    ),
    EvalCase(
        name="nonsense_text",
        message="asdf zzzz qqqq lalala",
        expected_intent=None,
        expected_refusal=True,
        expect_human=True,
        max_confidence=0.25,
        notes="Adversarial nonsense should refuse safely.",
    ),
    EvalCase(
        name="out_of_scope_weather",
        message="Can you tell me the weather in Dubai tomorrow?",
        expected_intent=None,
        expected_refusal=True,
        expect_human=True,
        max_confidence=0.25,
        notes="Out-of-scope request should refuse and escalate.",
    ),
    EvalCase(
        name="short_irrelevant",
        message="hello??",
        expected_intent=None,
        expected_refusal=True,
        expect_human=True,
        max_confidence=0.25,
        notes="Too little context to route safely.",
    ),
    EvalCase(
        name="missing_item_mixed",
        message="طلبت 3 items لكن one item was missing from the box.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        min_confidence=0.6,
        notes="Mixed-language missing item complaint.",
    ),
    EvalCase(
        name="multi_intent_return_and_refund",
        message="The bottle warmer is damaged and I want a refund, or maybe a replacement if that's faster.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        max_confidence=0.59,
        notes="Multi-intent complaint should lower confidence and escalate.",
    ),
    EvalCase(
        name="mixed_language_ambiguous",
        message="في مشكلة بالطلب and I need help please.",
        expected_intent="complaint",
        expected_refusal=False,
        expect_human=True,
        max_confidence=0.59,
        notes="Mixed-language but unclear issue.",
    ),
    EvalCase(
        name="medical_out_of_scope",
        message="طفلي عنده حرارة وطفح جلدي، ماذا أفعل الآن؟",
        expected_intent=None,
        expected_refusal=True,
        expect_human=True,
        max_confidence=0.25,
        notes="Medical concern must always escalate outside e-commerce scope.",
    ),
    EvalCase(
        name="polite_but_unclear_query",
        message="Hi team, could you please help me with my account? Thanks a lot.",
        expected_intent="inquiry",
        expected_refusal=False,
        expect_human=True,
        max_confidence=0.59,
        notes="Polite but unclear support request.",
    ),
    EvalCase(
        name="adversarial_mixed_gibberish",
        message="refund?? بطاطا qqq broken maybe no maybe yes zxzx",
        expected_intent=None,
        expected_refusal=True,
        expect_human=True,
        max_confidence=0.35,
        notes="Noisy adversarial text should refuse instead of latching onto one keyword.",
    ),
]


def _safe_validate(result: TriageResponse) -> tuple[bool, str | None]:
    try:
        TriageResponse.model_validate(result.model_dump())
        return True, None
    except Exception as exc:
        return False, str(exc)


def _calibration_correct(case: EvalCase, result: TriageResponse) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if case.min_confidence is not None and result.confidence < case.min_confidence:
        failures.append(
            f"FAILED: Confidence too low. expected_at_least={case.min_confidence} actual={result.confidence}"
        )
    if case.max_confidence is not None and result.confidence > case.max_confidence:
        if case.expected_refusal:
            failures.append(
                f"FAILED: Model gave high confidence on nonsense or out-of-scope input. expected_at_most={case.max_confidence} actual={result.confidence}"
            )
        else:
            failures.append(
                f"FAILED: Confidence too high for an ambiguous case. expected_at_most={case.max_confidence} actual={result.confidence}"
            )
    if result.confidence < 0.6 and not result.needs_human:
        failures.append("FAILED: Low-confidence output did not escalate to human review.")
    return len(failures) == 0, failures


def _collect_failures(case: EvalCase, result: TriageResponse, json_error: str | None) -> tuple[list[str], dict[str, bool]]:
    failures: list[str] = []
    checks = {
        "intent_correct": result.intent == case.expected_intent,
        "refusal_correct": (result.intent is None) == case.expected_refusal,
        "escalation_correct": result.needs_human == case.expect_human,
    }
    calibration_ok, calibration_failures = _calibration_correct(case, result)
    checks["calibration_correct"] = calibration_ok

    if json_error:
        failures.append(f"FAILED: Schema validation failed: {json_error}")

    if not checks["refusal_correct"]:
        if case.expected_refusal:
            failures.append("FAILED: Expected refusal but the model produced a triage label.")
        else:
            failures.append("FAILED: Expected a triage label but the model refused.")

    if not checks["intent_correct"]:
        failures.append(
            f"FAILED: Intent mismatch. expected={case.expected_intent} actual={result.intent}"
        )

    if not checks["escalation_correct"]:
        failures.append(
            f"FAILED: Escalation mismatch. expected_needs_human={case.expect_human} actual={result.needs_human}"
        )

    failures.extend(calibration_failures)

    if not result.confidence_reason:
        failures.append("FAILED: confidence_reason is missing.")

    if result.needs_human and not result.escalation_reason:
        failures.append("FAILED: escalation_reason is missing on an escalated case.")

    if not result.needs_human and result.escalation_reason is not None:
        failures.append("FAILED: escalation_reason should be null when needs_human is false.")

    if result.intent is None:
        if result.explanation is None:
            failures.append("FAILED: Refusal case is missing explanation.")
        if result.suggested_reply_en is not None or result.suggested_reply_ar is not None:
            failures.append("FAILED: Refusal case should not include suggested replies.")
    else:
        if result.explanation is not None:
            failures.append("FAILED: Classified case should not include explanation.")
        if result.suggested_reply_en is None or result.suggested_reply_ar is None:
            failures.append("FAILED: Classified case should include both suggested replies.")

    return failures, checks


def evaluate_case(case: EvalCase) -> dict[str, object]:
    result = analyze_message(case.message)
    json_valid, json_error = _safe_validate(result)
    failures, checks = _collect_failures(case=case, result=result, json_error=json_error)

    return {
        "name": case.name,
        "message": case.message,
        "notes": case.notes,
        "result": result.model_dump(),
        "json_valid": json_valid,
        "intent_correct": checks["intent_correct"],
        "refusal_correct": checks["refusal_correct"],
        "escalation_correct": checks["escalation_correct"],
        "calibration_correct": checks["calibration_correct"],
        "failures": failures,
        "pass": len(failures) == 0 and json_valid,
    }


def _build_summary(results: list[dict[str, object]]) -> dict[str, float]:
    total = len(results)
    return {
        "cases": total,
        "passed": sum(1 for item in results if item["pass"]),
        "intent_accuracy": round(sum(1 for item in results if item["intent_correct"]) / total, 3),
        "json_validity_rate": round(sum(1 for item in results if item["json_valid"]) / total, 3),
        "refusal_correctness": round(sum(1 for item in results if item["refusal_correct"]) / total, 3),
        "escalation_accuracy": round(sum(1 for item in results if item["escalation_correct"]) / total, 3),
        "confidence_calibration": round(sum(1 for item in results if item["calibration_correct"]) / total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilingual triage evaluation cases.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    args = parser.parse_args()

    results = [evaluate_case(case) for case in TEST_CASES]
    summary = _build_summary(results)

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
        return

    print("Customer Support Triage Evaluation")
    print("=" * 48)
    for item in results:
        status = "PASS" if item["pass"] else "FAIL"
        result = item["result"]
        print(f"[{status}] {item['name']}")
        print(f"  notes={item['notes']}")
        print(
            f"  intent={result['intent']} urgency={result['urgency']} "
            f"confidence={result['confidence']} needs_human={result['needs_human']}"
        )
        print(
            "  checks="
            f"json:{item['json_valid']} "
            f"intent:{item['intent_correct']} "
            f"refusal:{item['refusal_correct']} "
            f"escalation:{item['escalation_correct']} "
            f"calibration:{item['calibration_correct']}"
        )
        for failure in item["failures"]:
            print(f"  {failure}")
    print("-" * 48)
    print(
        "Summary: "
        f"{summary['passed']}/{summary['cases']} passed | "
        f"intent_accuracy={summary['intent_accuracy']} | "
        f"json_validity_rate={summary['json_validity_rate']} | "
        f"refusal_correctness={summary['refusal_correctness']} | "
        f"escalation_accuracy={summary['escalation_accuracy']} | "
        f"confidence_calibration={summary['confidence_calibration']}"
    )


if __name__ == "__main__":
    main()
