import streamlit as st
import joblib
import re

# Load saved model artifacts
model = joblib.load('ticket_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_priority(text: str) -> str:
    urgent_keywords = ['down', 'crash', 'urgent', 'emergency', 'broken', 'error 500', 'not working']
    if any(word in text.lower() for word in urgent_keywords):
        return "HIGH (Urgent)"
    return "NORMAL"

st.set_page_config(page_title="SmartTicket Classifier", page_icon="🎫")
st.title("🎫 SmartTicket Auto Router")
st.write("Type a customer support ticket below to auto-categorize and assign priority.")

user_input = st.text_area("Support Ticket Description:", placeholder="e.g., app crashes every time I try to login urgently")

if st.button("Classify Ticket"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        
        raw_proba = float(model.predict_proba(vec).max())
        proba = round(raw_proba, 4)
        priority = get_priority(user_input)
        
        # Lower threshold for HIGH priority to ensure fast routing for critical tickets
        threshold = 0.35 if "HIGH" in priority else 0.45
        
        if proba >= threshold:
            assigned_dept = str(pred)
            st.success("Ticket auto-routed to **" + assigned_dept + "**")
        else:
            assigned_dept = "HUMAN REVIEW (Suggested: " + str(pred) + ")"
            st.warning("⚠️ Confidence (" + str(round(proba * 100, 1)) + "%) is below threshold. Flagged for manual review.")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Assigned Department", assigned_dept)
        col2.metric("Confidence Score", str(round(proba * 100, 1)) + "%")
        col3.metric("Priority Level", priority)
    else:
        st.warning("Please enter some text to test.")
