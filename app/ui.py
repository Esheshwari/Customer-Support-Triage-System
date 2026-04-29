import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Assuming the FastAPI app is running on localhost:8000
API_URL = "http://localhost:8000/analyze"

# Sample test messages
SAMPLE_MESSAGES = [
    "I want to return the stroller I received yesterday because it is unopened.",
    "My refund still has not reached my card. Can you check the status?",
    "المنتج وصل مكسور وأحتاج حل بسرعة.",
    "طلبي متأخر من أسبوع، وين الشحنة؟",
    "I ordered diapers but وصلني wrong size and wrong brand.",
    "Please cancel my order before it ships.",
    "I think my card was charged twice for the same baby monitor order.",
    "I have an issue with my recent order and I am not happy.",
    "asdf zzzz qqqq lalala",
    "Can you tell me the weather in Dubai tomorrow?",
    "hello??",
    "طلبت 3 items لكن one item was missing from the box.",
    "The bottle warmer is damaged and I want a refund, or maybe a replacement if that's faster.",
    "في مشكلة بالطلب and I need help please.",
    "طفلي عنده حرارة وطفح جلدي، ماذا أفعل الآن؟",
    "Hi team, could you please help me with my account? Thanks a lot.",
    "refund?? بطاطا qqq broken maybe no maybe yes zxzx",
]

def get_color(confidence):
    if confidence > 0.75:
        return "green"
    elif confidence > 0.4:
        return "orange"
    else:
        return "red"

def analyze_message(message, force_llm=False):
    payload = {"message": message, "force_llm": force_llm}
    response = requests.post(API_URL, json=payload)
    if response.status_code != 200:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None
    try:
        return response.json()
    except ValueError:
        st.error("Invalid JSON response from API")
        return None

def main():
    st.title("Customer Support Triage System")

    # Initialize session state for metrics
    if 'results' not in st.session_state:
        st.session_state.results = []

    # Input section
    st.header("Input")
    
    # Demo buttons
    st.subheader("Quick Demo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Missing Item"):
            st.session_state.message = "I received my order but one item is missing from the package."
    with col2:
        if st.button("Wrong Item"):
            st.session_state.message = "I ordered diapers but received the wrong size and wrong brand."
    with col3:
        if st.button("Refund Request"):
            st.session_state.message = "My refund still has not reached my card. Can you check the status?"
    with col4:
        if st.button("Arabic Query"):
            st.session_state.message = "المنتج وصل مكسور وأحتاج حل بسرعة."
    
    message = st.text_area("Customer Message", value=st.session_state.get('message', ''), height=100)
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("Analyze")
    with col2:
        force_llm = st.checkbox("Force LLM Mode")

    if analyze_button and message:
        result = analyze_message(message, force_llm)
        if result:
            # Store result for metrics
            st.session_state.results.append(result)

            # Output sections
            st.header("Triage Result")
            st.write("Intent:", result.get("intent", "N/A"))
            st.write("Category:", result.get("category", "N/A"))
            st.write("Urgency:", result.get("urgency", "N/A"))
            st.write("Confidence:", f"{result.get('confidence', 0):.2f}")

            needs_human = result.get("needs_human", False)
            if needs_human:
                st.error("Needs Human Review")
            else:
                st.success("Can be handled automatically")

            st.header("Suggested Replies")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("English")
                st.text_area("", value=result.get("suggested_reply_en", "N/A"), height=100, disabled=True)
            with col2:
                st.subheader("Arabic")
                st.text_area("", value=result.get("suggested_reply_ar", "N/A"), height=100, disabled=True)

            st.header("Why this decision?")
            st.write("**Confidence Reason:**", result.get("confidence_reason", "N/A"))
            if result.get("escalation_reason"):
                st.write("**Escalation Reason:**", result.get("escalation_reason"))
            if result.get("explanation"):
                st.write("**Explanation:**", result.get("explanation"))

            # Mode Comparison
            st.header("Mode Comparison")
            heuristic_result = result.get("heuristic_result")
            llm_result = result.get("llm_result")
            if heuristic_result:
                st.subheader("Heuristic Result")
                st.json(heuristic_result)
            if llm_result:
                st.subheader("LLM Result")
                st.json(llm_result)
                if heuristic_result and llm_result:
                    if heuristic_result != llm_result:
                        st.warning("Results differ between modes")

    # Metrics Dashboard
    st.header("Metrics Dashboard")
    if st.session_state.results:
        results = st.session_state.results
        total = len(results)
        escalations = sum(1 for r in results if r.get("needs_human", False))
        escalation_rate = escalations / total * 100

        st.metric("Total Analyses", total)
        st.metric("% Escalations", f"{escalation_rate:.1f}%")

        # Intent distribution
        intents = [r.get("intent") for r in results if r.get("intent")]
        if intents:
            intent_counts = Counter(intents)
            st.bar_chart(pd.DataFrame.from_dict(intent_counts, orient='index', columns=['Count']))

        # Confidence histogram
        confidences = [r.get("confidence", 0) for r in results]
        fig, ax = plt.subplots()
        ax.hist(confidences, bins=10, edgecolor='black')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        st.pyplot(fig)
    else:
        st.write("No analyses yet.")

if __name__ == "__main__":
    main()