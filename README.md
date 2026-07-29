# 🎫 SmartTicket Classifier

A quick and lightweight NLP tool built to automatically route incoming customer support tickets to the right team. Instead of manually sorting through every single email or ticket, this pipeline reads the incoming message, cleans the text, runs it through a machine learning classifier, and assigns it to **Billing**, **HR**, **Technical**, or **General** support.

It also includes smart operational features like confidence scoring (flagging uncertain predictions for human review) and priority keyword tagging for urgent issues.

---

## ⚡ Key Highlights

- **Smart Text Cleaning:** Uses regex to strip out unwanted special characters, normalize casing, and keep only meaningful words.
- **TF-IDF Vectorization:** Converts raw text into numerical feature vectors using unigrams and bigrams so the model catches phrase context (e.g., "not working").
- **Logistic Regression Model:** Handles high-dimensional text data well and gives clean probability scores for every classification.
- **Human-in-the-Loop Fallback:** If the model's prediction confidence drops below 40%, it routes the ticket to a **HUMAN REVIEW** queue instead of guessing blindly.
- **Urgent Priority Tagging:** Automatically scans for critical words like *"down"*, *"crash"*, or *"urgent"* and tags the ticket as `HIGH (Urgent)`.
- **Interactive CLI Demo:** Test individual sample tickets right in your terminal or type custom messages live.

---

## 📊 How It Performed

I evaluated the model using a stratified train-test split to make sure each department category was fairly represented.

- **Overall Test Accuracy:** ~84.6%

### Department Performance Breakdown

| Department | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **BILLING** | 1.00 | 0.71 | 0.83 | 7 |
| **GENERAL** | 1.00 | 0.83 | 0.91 | 6 |
| **HR** | 0.83 | 0.83 | 0.83 | 6 |
| **TECHNICAL** | 0.70 | 1.00 | 0.82 | 7 |

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed on your machine.

### 2. Install Required Packages
Run this in your terminal to grab the necessary libraries:

```bash
pip install pandas scikit-learn joblib
```

### 3. Run the Script
Make sure `tickets.csv` is in the same folder as your script, then execute:

```bash
python SmartTickect_classifier.py
```

---

## 💻 Sample Output

```text
--- 1. Testing 5 Custom Unseen Sample Tickets ---
TCK-3001 :  API returns 500 server error on login -> TECHNICAL (Confidence: 0.51 , Priority: NORMAL )
TCK-3002 :  My invoice was charged twice this month -> BILLING (Confidence: 0.6 , Priority: NORMAL )
TCK-3003 :  When will my annual leave request be approved? -> HR (Confidence: 0.45 , Priority: NORMAL )
TCK-3004 :  What are your business operating hours? -> HUMAN REVIEW (Low Confidence) (Confidence: 0.38 , Priority: NORMAL )
TCK-3005 :  The app is completely down and broken urgently -> TECHNICAL (Confidence: 0.51 , Priority: HIGH (Urgent) )

--- 2. Live Interactive Input ---
Enter a support ticket description to test: autopay charged me twice this cycle
TCK-USER :  autopay charged me twice this cycle -> BILLING (Confidence: 0.51 , Priority: NORMAL )
```

---

## 🧠 Design Choices & Future Improvements

* **Why Logistic Regression?**  
  I chose Logistic Regression over Naive Bayes because it produces well-calibrated probability distributions (`predict_proba`). This is essential for building a threshold-based fallback system. It also naturally handles feature overlaps in TF-IDF bigrams without assuming total feature independence.

* **What I'd Improve with More Time/Data:**  
  1. **Transformer Upgrade:** I'd swap the TF-IDF setup for a lightweight fine-tuned transformer like `DistilBERT` or `MiniLM` to better capture deeper semantic context and informal phrasing.  
  2. **Active Learning Loop:** Set up a feedback loop where tickets sent to the manual review queue are logged, corrected by human agents, and fed back into training to continuously improve the model over time.
