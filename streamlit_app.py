from __future__ import annotations

import json

import streamlit as st

from app.model import analyze_message


st.set_page_config(page_title="Support Triage Demo", page_icon="🧾", layout="centered")

st.title("Customer Support AI Triage")
st.caption("Multilingual prototype for e-commerce support messages.")

default_message = (
    "Hi, I received the wrong item today and the box was slightly damaged. "
    "Can you help me with a return?"
)

message = st.text_area(
    "Customer message",
    value=default_message,
    height=180,
    help="English, Arabic, or mixed-language messages are supported.",
)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.error("Please enter a customer message.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_message(message)
        st.subheader("Structured output")
        st.code(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), language="json")

        st.subheader("Quick view")
        col1, col2, col3 = st.columns(3)
        col1.metric("Intent", result.intent or "null")
        col2.metric("Urgency", result.urgency or "null")
        col3.metric("Confidence", f"{result.confidence:.2f}")

        st.write("Needs human:", result.needs_human)
        st.write("Confidence reason:", result.confidence_reason)
        if result.escalation_reason:
            st.warning(f"Escalation reason: {result.escalation_reason}")
        if result.explanation:
            st.info(result.explanation)
