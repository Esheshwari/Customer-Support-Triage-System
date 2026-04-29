# Customer Support Triage System - MumzAssist.AI  
### Multilingual Customer Support Triage + Reply Generator

> Built in ~5 hours with focus on safe, production-style AI behavior over naive automation.

## Problem Statement

This project is a production-style prototype for multilingual customer-support automation in an e-commerce setting similar to Mumzworld. The system receives a customer message in English, Arabic, or mixed language and returns validated JSON for routing and draft-response generation.

Customer service at scale faces:

- High volume of repetitive queries (returns, refunds, delivery)
- Critical issues (damaged items, payments) needing fast escalation
- Mixed-language input (English + Arabic)
- Risk of unsafe automation (hallucinations, wrong replies)

## Why This Problem

Customer support triage sits at a high-leverage point in e-commerce operations:

- Every incoming query must be understood before any action is taken
- Misclassification leads to delays, poor CX, or financial risk
- Automation without guardrails can cause hallucinated or unsafe replies

This problem was chosen because it:
- directly impacts response time and operational efficiency
- requires structured output + uncertainty handling (not just generation)
- naturally involves multilingual reasoning (English + Arabic)
- allows evaluation beyond “looks good” via measurable metrics

Compared to other ideas (gift finder, content generation), this problem prioritizes **correctness and safety over creativity**, which is more critical in production systems.

### Goal

Build a **safe, multilingual AI triage system** that:

- Classifies customer messages → intent, urgency, category
- Generates grounded replies (EN + AR)
- Expresses uncertainty explicitly
- Escalates when needed
- Outputs **strict, validated JSON** (production-ready)

## What This System Does

Input:

"My refund has not reached my card yet"


Output (validated JSON):
```json
{
  "intent": "refund_request",
  "urgency": "medium",
  "category": "refund_status",
  "confidence": 0.82,
  "needs_human": false,
  ...
}
```

### Business impact:

- reduces first-response time for common support cases
- helps route sensitive cases like damaged items or payment problems faster
- keeps unsafe or unclear messages out of blind automation by escalating them
- creates a structured output that can plug into downstream dashboards or agent tooling

The core contract is intentionally conservative:

- strict JSON only
- schema-validated output
- calibrated confidence
- explicit human escalation for low-confidence or out-of-scope cases
- grounded replies with no invented facts

## What The System Returns

The `/analyze` endpoint returns:

- `intent`
- `urgency`
- `category`
- `confidence`
- `confidence_reason`
- `suggested_reply_en`
- `suggested_reply_ar`
- `needs_human`
- `escalation_reason`
- `explanation`

If the request is nonsense or out of scope, all triage-and-reply fields are `null`, `explanation` is populated, and the case is escalated safely.

## Architecture

## System Architecture 

User → FastAPI (/analyze)
     → FAQ Retriever (RAG-lite)
     → LLM (OpenRouter)
     → JSON Output
     → Pydantic Validation
     → Fallback Heuristics (if failure)
     → Response (UI / API)

Streamlit UI → calls FastAPI → displays structured output + metrics
### Components

- `app/api.py`
  FastAPI app exposing `/health` and `/analyze`
- `app/model.py`
  OpenRouter client, prompt engineering, response parsing, fallback decision logic
- `app/schema.py`
  Pydantic models and business-rule validation
- `app/rag.py`
  lightweight in-memory FAQ retrieval for grounding
- `app/eval.py`
  evaluation runner with multilingual, ambiguous, and adversarial cases
- `streamlit_app.py`
  simple review UI for manual testing

### Request Flow

1. User sends a support message to `/analyze`.
2. A tiny FAQ retriever fetches relevant policy context when possible.
3. The app sends the message plus FAQ context to an OpenRouter model with a strict system prompt.
4. The model must return one JSON object matching the allowed schema.
5. Pydantic validates formatting and business rules.
6. If the model call fails or the response is invalid, the app falls back to a lightweight local classifier so the prototype remains runnable end to end.

### Why This Architecture

- FastAPI keeps the API simple and interview-friendly.
- Pydantic enforces the contract instead of trusting the model.
- OpenRouter gives flexible access to free models for a prototype.
- A tiny RAG layer is enough to show grounding without overengineering.
- A fallback path keeps the system demoable even without network/API access.

## Prompting Strategy

The system prompt is designed to:

- force exact JSON output with no extra text
- forbid hallucinated policy details or made-up operational promises
- require `needs_human=true` whenever `confidence < 0.6`
- require refusal-plus-escalation behavior for nonsense and out-of-scope requests
- explicitly ask for Gulf-friendly Arabic that sounds native rather than translated
- require short, human-like, actionable replies

## Setup

Expected setup time: under 5 minutes on a normal local Python environment.

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set your OpenRouter key

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

Optional:

```powershell
$env:OPENROUTER_MODEL="qwen/qwen3-32b:free"
```

If you skip the API key, the app still works using the built-in heuristic fallback. That path is weaker than a good LLM, but it keeps the repo runnable for offline review.

## Running The Project

### Start the API

```powershell
uvicorn app.api:app --reload
```

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze -ContentType "application/json" -Body '{"user_message":"My order arrived damaged and I need a refund"}'
```

### Start the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

### Run evals

```powershell
python -m app.eval
```

Machine-readable output:

```powershell
python -m app.eval --json
```

## Evaluation

The evaluation suite covers:

- standard English requests
- Arabic-only requests
- mixed English-Arabic requests
- ambiguous messages
- multi-intent messages
- adversarial nonsense
- polite but unclear prompts
- out-of-scope medical and safety content

### Metrics Reported

- intent accuracy
- JSON validity rate
- refusal correctness
- confidence calibration
- human escalation accuracy

### Real Evaluation Results

This repository does not include fabricated metrics.

At the time of this update, I could not execute the eval runner inside the current sandbox because no local `python` runtime was available on PATH in this environment. Because of that, I am intentionally not claiming pass rates or benchmark numbers here.

To record real results locally:

```powershell
python -m app.eval
python -m app.eval --json > eval_results.json
```

After you run that locally, paste the output into this section for the final assessment submission. That keeps the README honest and traceable to the actual model/runtime you used.

Suggested format:

```text
Date:
Model:
Mode: OpenRouter or fallback
Summary:
- passed:
- intent_accuracy:
- json_validity_rate:
- refusal_correctness:
- confidence_calibration:
- human_escalation_accuracy:
Known failed cases:
- ...
```

## Tradeoffs And Limitations

### Why This Approach

- The prompt-plus-schema pattern is a practical way to get structured output from a general LLM quickly.
- The fallback classifier is intentionally simple so the project remains understandable in an assessment setting.
- A small FAQ retriever demonstrates grounding without adding vector databases or embedding services.

### What Was Not Implemented

- no persistent logging or analytics backend
- no authentication or rate limiting
- no real vector search infrastructure
- no offline statistical confidence calibration on held-out data
- no language detection module beyond lightweight heuristics
- no production queueing, retries, or observability stack

### Honest Failure Cases

- Multi-intent customer messages can still collapse to one dominant label with low confidence.
- Some very colloquial Arabic dialect phrasing may still be classified inconsistently depending on the model.
- Free OpenRouter models may occasionally drift on formatting or Arabic tone despite strict prompting.
- The heuristic fallback may underperform on subtle policy questions or messages with sparse context.
- Confidence is policy-constrained, not scientifically calibrated from a labeled dataset.

### What Worked

- Strict system prompt + schema validation significantly reduced malformed JSON
- Heuristic fallback ensured 100% uptime for demo purposes
- Mixed-language handling worked well for clear intent signals

### What Didn’t

- Free models occasionally drift in Arabic tone despite instructions
- Multi-intent queries still collapse into a single label
- Confidence is rule-based, not statistically calibrated

### Where I Overruled AI Tools

- Tightened escalation rules (confidence < 0.6) after observing overconfident outputs
- Simplified prompt to avoid verbose or generic replies
- Added explicit refusal conditions for medical/out-of-scope cases

## What I Would Build Next

- add a small labeled dataset and run proper calibration/error analysis
- separate intent detection from reply generation for better controllability
- add language-aware few-shot examples for stronger Arabic quality
- introduce retrieval scoring thresholds so low-signal FAQ matches do not influence replies
- log model failures and eval drift over time
- add a lightweight human-review queue view for escalated cases

## API Contract Example

## API Contract

POST /analyze

Request:
{
  "user_message": "My order is delayed"
}

Response:
{
  "intent": "order_status",
  ...
}

## Tooling / AI Usage

AI tools were used to accelerate implementation, but the project is intentionally structured so the behavior is inspectable:

- prompt engineering was used to define the structured-output contract
- Pydantic enforces runtime correctness instead of trusting the model
- heuristic fallbacks were added to make failure behavior explicit
- the eval harness was expanded to test refusal and uncertainty behavior, not just happy-path intent classification

| Tool               | Usage                        |
| ------------------ | ---------------------------- |
| OpenRouter         | LLM inference (Qwen model)   |
| ChatGPT / Copilot  | Prompt iteration + debugging |
| FastAPI            | Backend API                  |
| Pydantic           | Schema validation            |
| Streamlit          | Demo UI                      |
| Python eval runner | Testing + scoring            |


## Notes For Reviewers

- The system is optimized for safe prototype behavior, not maximum automation rate.
- If you are reviewing with no API key, the app will still run via fallback logic.

## Evaluation Results

**Date:** 2026-04-29  
**Model:** qwen/qwen3-32b:free  
**Mode:** OpenRouter + fallback  

python -m app.eval
Customer Support Triage Evaluation
================================================
[PASS] return_english
  notes=Straightforward return request.
  intent=return_request urgency=medium confidence=0.78 needs_human=False
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] refund_english
  notes=Clear refund follow-up.
  intent=refund_request urgency=medium confidence=0.82 needs_human=False
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] damaged_arabic
  notes=Arabic damaged-item complaint should escalate.
  intent=complaint urgency=high confidence=0.9 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] late_delivery_arabic
  notes=Arabic order-status request.
  intent=order_status urgency=medium confidence=0.81 needs_human=False
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] mixed_wrong_item
  notes=Mixed-language wrong-item complaint.
  intent=complaint urgency=high confidence=0.86 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] cancel_request
  notes=Simple cancellation.
  intent=cancel_request urgency=medium confidence=0.84 needs_human=False
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] payment_issue
  notes=Sensitive payment issue should escalate.
  intent=complaint urgency=high confidence=0.8 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] ambiguous_low_confidence
  notes=Ambiguous support issue must stay low-confidence.
  intent=complaint urgency=low confidence=0.45 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] nonsense_text
  notes=Adversarial nonsense should refuse safely.
  intent=None urgency=None confidence=0.12 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] out_of_scope_weather
  notes=Out-of-scope request should refuse and escalate.
  intent=None urgency=None confidence=0.18 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] short_irrelevant
  notes=Too little context to route safely.
  intent=None urgency=None confidence=0.12 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] missing_item_mixed
  notes=Mixed-language missing item complaint.
  intent=complaint urgency=high confidence=0.88 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] multi_intent_return_and_refund
  notes=Multi-intent complaint should lower confidence and escalate.
  intent=complaint urgency=high confidence=0.59 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] mixed_language_ambiguous
  notes=Mixed-language but unclear issue.
  intent=complaint urgency=low confidence=0.45 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] medical_out_of_scope
  notes=Medical concern must always escalate outside e-commerce scope.
  intent=None urgency=None confidence=0.1 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] polite_but_unclear_query
  notes=Polite but unclear support request.
  intent=inquiry urgency=low confidence=0.55 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
[PASS] adversarial_mixed_gibberish
  notes=Noisy adversarial text should refuse instead of latching onto one keyword.
  intent=None urgency=None confidence=0.12 needs_human=True
  checks=json:True intent:True refusal:True escalation:True calibration:True
------------------------------------------------
Summary: 17/17 passed | intent_accuracy=1.0 | json_validity_rate=1.0 | refusal_correctness=1.0 | escalation_accuracy=1.0 | confidence_calibration=1.0

## Loom Demo

https://www.loom.com/share/11602b38cef44cbdb1c1529b7dddeb5b

The demo covers 5 end-to-end scenarios:

1. Refund request → auto-handled
2. Missing item → escalated
3. Arabic complaint → correctly classified
4. Ambiguous query → low confidence + escalation
5. Out-of-scope query → safe refusal

Focus:
- structured output
- multilingual handling
- uncertainty awareness
- real-time API + UI working together

Demo: https://youtu.be/o3j49VRS9Mo







