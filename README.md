# 🗣️ AI Communication Coach – Self-Introduction Scorer

A web-based tool to evaluate students’ spoken self-introductions using a rubric-driven scoring system.  
The app analyzes transcript text and generates a final score (0–100) with detailed feedback across 5 key communication skills.

Deployed App:  
🔗 https://ai-self-intro-scorer-czcevve3fbarn78mhe3wv7.streamlit.app/

GitHub Repository:  
🔗 https://github.com/Subodhkhanduri/ai-self-intro-scorer

---

## 🎯 Objective

Support students in improving spoken communication by providing **immediate, AI-based feedback** based on a structured rubric.

This project is built for  
**Nirmaan AI Intern Case Study — Communication Program**.

---

## 🧩 Key Features

• Paste or upload transcripts (TXT, PDF, DOCX)  
• Real-time scoring (0–100)  
• Detailed rubric-based evaluation  
• Per-criterion score breakdown  
• Visual performance charts  
• Grammar, vocabulary & filler analysis  
• Speech-rate estimation (Words Per Minute)

---

## 🧠 Product Thinking

### 🌟 Why this matters
- Many students struggle with self-expression and public speaking
- Teachers have limited time for individualized feedback
- Automated feedback enables **continuous practice & improvement**

### 👥 Who are the users?
- Students (primary focus — self-introductions)
- Trainers/educators monitoring progress
- Schools delivering communication programs

### 💡 Business Value
- Scalable evaluation of spoken tasks
- Standardized assessment aligned to rubric
- Track progress over time → measurable outcomes

---

## 🏗️ System Architecture
+-------------+ +------------------+ +------------------------+
| Web Frontend| -----> | Scoring Backend | -----> | Visualization & Output |
| (Streamlit) | | (Python + NLP) | | Score + Feedback |
+-------------+ +------------------+ +------------------------+
|
v
Sentence Transformers + NLP

---

## 📊 Rubric Scoring Breakdown

| Criterion | Max Points | Method |
|----------|------------|--------|
| Content & Structure | 40 | Keyword detection + flow scoring |
| Speech Rate | 10 | WPM calculation |
| Language & Grammar | 20 | Grammar + vocabulary richness |
| Clarity | 15 | Filler word frequency |
| Engagement | 15 | Sentiment positivity (VADER) |

Final Score = Weighted sum mapped to **0–100**

---

## 🧪 How it Works — Scoring Flow

Upload/Paste Transcript
↓
Preprocessing (tokenize, normalize)
↓
Rule-based checks (keywords, order, filler words)
↓
NLP checks (semantic similarity, sentiment)
↓
Rubric-weighted aggregation (Content, Speed…)
↓
Dashboard & Feedback


---

## 🚀 Try it Locally

```bash
git clone https://github.com/Subodhkhanduri/ai-self-intro-scorer.git
cd ai-self-intro-scorer

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt

streamlit run app.py

