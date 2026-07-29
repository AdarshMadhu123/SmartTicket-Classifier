import re
import pandas as pd
import matplotlib.pyplot as plt
import joblib


df=pd.read_csv("tickets.csv")
print(df.shape)
print(df.dtypes)


def clean_text(text: str)-> str:
    text=text.lower()
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    text=re.sub(r"\s+", " ", text).strip()  
    return text


df["clean_text"] = df["ticket_text"].apply(clean_text)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df['clean_text'],df['department'],test_size=0.2,random_state=42,stratify=df["department"])


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english",ngram_range=(1, 2),max_features=3000)
X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)



from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train_vec,y_train)
y_pred=model.predict(X_test_vec)



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
As=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score: ", As, "Confusion Matrix: ", cm, "Classification Report: ", cr)


joblib.dump(model, "ticket_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved to disk.")



def get_priority(text: str) -> str:
    urgent_terms = ["urgent", "down", "crash", "not working", "emergency"]
    for term in urgent_terms:
        if term in text.lower():
            return "HIGH (Urgent)"
    return "NORMAL"



def route_ticket(ticket_id: str, text: str, threshold: float = 0.40):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    priority = get_priority(text)
    
    # Fallback if confidence is under 40%
    if proba < threshold:
        assigned_dept = "HUMAN REVIEW (Low Confidence)"
    else:
        assigned_dept = pred

    print(ticket_id, ": ", text, "->", assigned_dept, "(Confidence:", round(proba, 2), ", Priority:", priority, ")")
    return assigned_dept




if __name__ == "__main__":
    print("\n--- 1. Testing 5 Custom Unseen Sample Tickets ---")
    sample_tickets = [
        ("TCK-3001", "API returns 500 server error on login"),          
        ("TCK-3002", "My invoice was charged twice this month"),         
        ("TCK-3003", "When will my annual leave request be approved?"),  
        ("TCK-3004", "What are your business operating hours?"),         
        ("TCK-3005", "The app is completely down and broken urgently")  
    ]
    
    for tid, txt in sample_tickets:
        route_ticket(tid, txt)

    print("\n--- 2. Live Interactive Input ---")
    user_text = input("Enter a support ticket description to test: ")
    if user_text.strip():
        route_ticket("TCK-USER", user_text)