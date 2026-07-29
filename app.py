import streamlit as st
import joblib
import re

# Load saved model artifacts
model = joblib.load('ticket_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def get_priority(text):
    urgent_keywords = ['down', 'crash', 'urgent', 'emergency', 'broken', 'error 500']
    if any(word in text.lower() for word in urgent_keywords):
        return "HIGH (Urgent)"
    return "NORMAL"

st.set_page_config(page_title="SmartTicket Classifier", page_icon="🎫")
st.title("🎫 SmartTicket Auto Router")
st.write("Type a customer support ticket below to auto-categorize and assign priority.")

# Interactive Input
user_input = st.text_area("Support Ticket Description:", placeholder="e.g., My invoice was charged twice this month")

if st.button("Classify Ticket"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec).max()
        priority = get_priority(user_input)
        
        # Threshold Check
        assigned_dept = pred if proba >= 0.40 else "HUMAN REVIEW (Low Confidence)"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Assigned Department", assigned_dept)
        col2.metric("Confidence Score", f"{round(proba * 100, 1)}%")
        col3.metric("Priority Level", priority)
    else:
        st.warning("Please enter some text to test.")