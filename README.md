# 🧠 MindBot — RAG-Based Mental Health Support Chatbot
### NLP Final Project 2026

A production-ready, end-to-end mental health chatbot integrating **4 NLP modules** into a single pipeline with an expert Streamlit UI and a FastAPI REST backend.

---

## Pipeline Architecture

```
User Message
     │
     ▼
[Crisis Check] ← Bonus: overrides everything for immediate safety routing
     │
     ▼
[Module 1: Language Detection]   TF-IDF char n-grams + LinearSVC (20 languages)
     │
     ▼
[Module 3: Intent Classification]   Few-shot LLM prompting via Groq (gpt-oss-20b)
     │
     ├── greeting / goodbye / gratitude / out_of_scope
     │        └─ Direct LLM reply (no RAG)
     │
     └── asking_mental_health_question
              │
              ▼
     [Module 2: Emotion Classification]   DistilBERT fine-tuned (6 emotions)
              │
              ▼
     [Module 4: RAG Pipeline]   Qdrant + all-MiniLM-L6-v2 + Groq gpt-oss-120b
              │
              ▼
     Empathy-tuned, language-aware response
```

---

## Modules

| # | Module | Approach | Dataset | Output |
|---|--------|----------|---------|--------|
| 1 | **Language Detection** | TF-IDF char n-grams (2–4) + LinearSVC | `papluca/language-identification` (70k, 20 langs) | Language code + confidence |
| 2 | **Emotion Classifier** | DistilBERT fine-tuned | `dair-ai/emotion` (16k, 6 classes) | Emotion + empathy tone |
| 3 | **Intent Classifier** | Few-shot LLM prompting (gpt-oss-20b) | None — prompt engineering | Intent → routing decision |
| 4 | **Q&A RAG** | Qdrant + sentence-transformers + Groq | `heliosbrahma/mental_health_counseling_conversations` | Grounded response |

---

## Bonus Features

- 🆘 **Crisis Detection** — Keyword scan overrides routing and delivers emergency resources
- 💬 **Conversation History** — Last 10 turns used for contextual continuity
- 📊 **Analytics Dashboard** — Real-time emotion distribution, intent breakdown, emotion-over-time chart
- 🌍 **Multi-language Support** — Detects and responds in the user's language
- 💾 **Chat Export** — Download session as JSON
- ⚙️ **Configurable** — Toggle model strength (120B vs 20B) and retrieval depth from the UI

---

## Required API Keys

> Set these in a `.env` file at the project root before running anything.

| Key | Where to Get It | Free? |
|-----|----------------|-------|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | ✅ Yes |
| `QDRANT_URL` | [cloud.qdrant.io](https://cloud.qdrant.io) → Create cluster → Copy endpoint | ✅ Free tier |
| `QDRANT_API_KEY` | Same dashboard → API Keys tab | ✅ Free tier |

```bash
# Copy the template and fill in your keys
cp .env.example .env
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the language detection model (Module 1)
```bash
python scripts/train_language_model.py
# → saves models/language_detector.pkl
```

### 3. Fine-tune the emotion classifier (Module 2)
```bash
python scripts/train_emotion_model.py
# → saves models/emotion_distilbert/
# Note: takes ~30 min on CPU, ~5 min on GPU
```

### 4. Ingest knowledge base into Qdrant (Module 4)
```bash
python scripts/ingest_rag.py
# Requires QDRANT_URL and QDRANT_API_KEY in .env
# Run only once — re-running creates duplicates
```

### 5. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
# → opens http://localhost:8501
```

### 6. (Optional) Launch the FastAPI backend
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
# → API docs at http://localhost:8000/docs
```

---

## Project Structure

```
NLP_Final_Project/
│
├── notebooks/
│   ├── 01_language_detection.ipynb    ← Module 1: TF-IDF + LinearSVC
│   ├── 02_emotion_classifier.ipynb    ← Module 2: DistilBERT fine-tuning
│   ├── 03_intent_classifier.ipynb     ← Module 3: Few-shot prompting
│   └── 04_rag_pipeline.ipynb          ← Module 4: Qdrant RAG
│
├── src/
│   ├── modules/
│   │   ├── language_detector.py       ← Module 1 inference
│   │   ├── emotion_classifier.py      ← Module 2 inference
│   │   ├── intent_classifier.py       ← Module 3 inference
│   │   └── rag_pipeline.py            ← Module 4 inference + ingestion
│   ├── pipeline/
│   │   └── chat_engine.py             ← Full pipeline orchestration
│   └── utils/
│       ├── preprocessing.py           ← Shared text cleaning
│       └── crisis_detector.py         ← Bonus: crisis keyword detection
│
├── app/
│   └── streamlit_app.py               ← Expert Streamlit UI
│
├── api/
│   └── server.py                      ← FastAPI REST backend
│
├── scripts/
│   ├── train_language_model.py        ← Train & save Module 1 model
│   ├── train_emotion_model.py         ← Train & save Module 2 model
│   └── ingest_rag.py                  ← One-time Qdrant ingestion
│
├── models/                            ← Saved model files (not in git)
├── data/                              ← Datasets (not in git)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Notebooks Guide

Run notebooks in order during the assessment:

```
01 → trains language model → saves models/language_detector.pkl
02 → trains emotion model  → saves models/emotion_distilbert/
03 → prompt engineering     → no model to save (LLM-based)
04 → ingests RAG data       → populates Qdrant collection
```

---

## Tech Stack

| Component | Library / Service |
|-----------|------------------|
| Traditional ML | scikit-learn (TF-IDF, LinearSVC) |
| Deep Learning | PyTorch + HuggingFace Transformers (DistilBERT) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Database | Qdrant Cloud (free tier) |
| LLM | Groq API (`gpt-oss-120b` / `gpt-oss-20b`) |
| UI | Streamlit |
| API | FastAPI + uvicorn |
| Data | HuggingFace Datasets |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Module status |
| `POST` | `/chat` | Send message, get response |
| `DELETE` | `/session/{id}` | Clear session history |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Assessment Notes

- All modules are independently testable via their notebooks
- The app degrades gracefully — if a model isn't trained yet, it shows a clear error
- API keys are loaded from `.env` — never hardcoded
- Module 3 (Intent) requires no training, just a valid `GROQ_API_KEY`
- Module 4 (RAG) requires Qdrant to be populated before use (run `scripts/ingest_rag.py`)
