🎫 SmartTicket Classifier

A lightweight, end-to-end NLP system built to automatically classify and route incoming customer support tickets to the appropriate team. Instead of manually triaging every request, this pipeline cleans raw text, converts it into TF-IDF vectors, runs a multi-class Logistic Regression model, and routes tickets to **BILLING**, **HR**, **TECHNICAL**, or **GENERAL** support.

It includes production-ready operational guardrails: a **45% calibrated confidence threshold** for automated routing, a **HUMAN REVIEW** safety fallback, and a **rule-based priority engine** that tags critical issues for immediate attention.

---

## ⚡ Key Highlights

* **Text Normalization Pipeline:** Uses regex to strip non-alphanumeric noise, normalize casing, and collapse whitespace for consistent feature extraction.
* **TF-IDF Vectorization:** Converts raw text into numerical feature vectors using unigrams and bigrams (`ngram_range=(1, 2)`) to capture context-heavy phrases (e.g., *"not working"*, *"server error"*).
* **Calibrated Multi-Class Classifier:** Leverages Logistic Regression to compute output probability distributions across all 4 departments using `predict_proba`.
* **45% Calibrated Confidence Threshold:** Designed for a 4-class problem (where random chance is 25%). If prediction confidence falls below 45%, the system routes the ticket to a **HUMAN REVIEW** queue to prevent misclassification.
* **Priority Rule Engine:** Scans incoming descriptions for critical terms like *"down"*, *"crash"*, *"broken"*, or *"urgent"* and marks priority as `HIGH (Urgent)` regardless of ML prediction confidence.
* **Interactive Interfaces:** Includes both a **Terminal CLI** and a fully responsive **Streamlit Web UI** (`app.py`) for live ticket triage.

---

## 📊 Performance & Evaluation

The model was trained and evaluated on a stratified split to ensure proportional representation across all departments.

* **Overall Accuracy:** **84.6%**
* **Test Set Support:** 26 tickets

### Department Breakdown

| Department | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **BILLING** | 1.00 | 0.71 | 0.83 | 7 |
| **GENERAL** | 1.00 | 0.83 | 0.91 | 6 |
| **HR** | 0.83 | 0.83 | 0.83 | 6 |
| **TECHNICAL** | 0.70 | 1.00 | 0.82 | 7 |

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Install Required Packages

Run the following command in your terminal:

pip install pandas scikit-learn joblib streamlit

### 3. Run the CLI Engine

Ensure `tickets.csv` is in your working directory, then execute:

python SmartTickect_classifier.py

### 4. Launch the Streamlit Web Application

To run the web interface, execute:

streamlit run app.py

---

## 💻 Sample CLI Output

--- 1. Testing 5 Custom Unseen Sample Tickets ---
TCK-3001 : API returns 500 server error on login -> TECHNICAL (Confidence: 50.8% , Priority: NORMAL)
TCK-3002 : My invoice was charged twice this month -> BILLING (Confidence: 60.0% , Priority: NORMAL)
TCK-3003 : When will my annual leave request be approved? -> HR (Confidence: 45.4% , Priority: NORMAL)
TCK-3004 : What are your business operating hours? -> HUMAN REVIEW (Low Confidence - Top Guess: GENERAL) (Confidence: 38.2% , Priority: NORMAL)
TCK-3005 : The app is completely down and broken urgently -> TECHNICAL (Confidence: 51.1% , Priority: HIGH (Urgent))

--- 2. Live Interactive Input ---
Enter a support ticket description to test: refund delayed for my last order invoice not received
TCK-USER : refund delayed for my last order invoice not received -> BILLING (Confidence: 57.4% , Priority: NORMAL)

---

## 📂 Project Directory Structure

SmartTicket-Classifier/
│
├── tickets.csv                   # Synthetic dataset (ticket_text, department)
├── SmartTickect_classifier.py    # Main ML pipeline, training script & CLI
├── app.py                        # Streamlit web interface
├── ticket_classifier_model.pkl   # Serialized Logistic Regression model
├── tfidf_vectorizer.pkl          # Serialized TF-IDF vectorizer
└── README.md                     # Project documentation

---

## 🧠 Design Choices & Future Roadmap

* **Why Logistic Regression?**
Unlike tree-based models, Logistic Regression yields well-calibrated class probabilities via sigmoid/softmax functions. This calibration is essential for establishing reliable confidence thresholds and driving fallback workflows.
* **Why a 45% Threshold?**
In a 4-class classification problem, random baseline accuracy is 25%. A prediction score >= 45% indicates that the top class holds nearly double the weight of pure chance and dominates the remaining three classes combined.
* **Future Enhancements:**
1. **Transformer Backbones:** Upgrade from TF-IDF to a lightweight transformer (e.g., `DistilBERT` or `MiniLM`) to capture deeper semantic intent and handle informal typing/slang.
2. **Human-in-the-Loop Feedback:** Log tickets routed to `HUMAN REVIEW` and integrate agent corrections into a retrain loop to improve edge-case accuracy over time.
