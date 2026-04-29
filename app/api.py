from __future__ import annotations

from fastapi import FastAPI

from app.model import analyze_message
from app.schema import AnalyzeRequest, TriageResponse


app = FastAPI(
    title="Customer Support AI Triage",
    version="0.1.0",
    description="Prototype API for multilingual customer support triage and reply generation.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=TriageResponse)
def analyze(payload: AnalyzeRequest) -> TriageResponse:
    return analyze_message(payload.message)
