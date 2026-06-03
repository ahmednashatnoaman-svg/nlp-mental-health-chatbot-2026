# 🧠 MindBot — RAG-Based Mental Health Support Chatbot
### NLP Final Project 2026

MindBot is a production-ready, end-to-end mental health chatbot that integrates **4 NLP modules** into a unified pipeline. It features:

- **20-language automatic detection and translation** — Arabic, Chinese, French, Spanish, German, and 15 more
- **3-level multilingual crisis detection** — catches crisis signals in English, Arabic, French, and Spanish
- **MPS/CUDA/CPU hardware acceleration** — Apple Silicon first, falls back gracefully
- **Real-time Plotly analytics** — emotion timeline, intent distribution, language distribution, response time
- **FastAPI REST API** — full pipeline exposed with detailed response schemas
- **Streamlit Cloud deployable** — zero-config fallback layers (langdetect + HuggingFace Hub)

---

## 🏗️ Architecture & Pipeline Flow

### Pipeline Order (v3 — safety-first revision)

```
User Input  (any language)
      │
      ▼
┌─────────────────────────────────────────┐
│  M1: Language Detection                 │  TF-IDF char n-grams + LinearSVC
│      Layer 1: Unicode script detection  │  Arabic, CJK, Thai, Cyrillic, Greek,
│              (instant, no model needed) │  Devanagari — 100% reliable
│      Layer 2: ASCII short-text default  │  ≤4 words, no accents → English
│      Layer 3: TF-IDF + LinearSVC model  │  70k training samples, 20 languages
│           OR: langdetect fallback       │  cloud deployment fallback
└─────────────────────────────────────────┘
      │
      ▼ (if non-English)
┌─────────────────────────────────────────┐
│  Translation → English                  │  Groq gpt-oss-20b
│  (for NLP processing: crisis, intent,   │  Raw message preserved for history
│   emotion — all English-based models)   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Crisis Detection (3-level)             │  On English text — catches all languages
│  HIGH  (score ≥ 10): emergency helpline │  Multi-lang keywords: AR, FR, ES + EN
│  MEDIUM (score 5-9): empathetic route   │  Translated back to user's language
│  LOW    (score 0-4): normal pipeline    │
└─────────────────────────────────────────┘
      │
      ▼ (LOW/MEDIUM only)
┌─────────────────────────────────────────┐
│  M3: Intent Classification              │  Few-shot Groq gpt-oss-20b
│  → greeting / goodbye / gratitude       │  Direct LLM reply in user's language
│  → out_of_scope                         │  Direct LLM reply
│  → asking_mental_health_question        │  → RAG path below
└─────────────────────────────────────────┘
      │ (asking_mental_health_question)
      ▼
┌─────────────────────────────────────────┐
│  M2: Emotion Classification             │  DistilBERT fine-tuned (dair-ai/emotion)
│  Classes: sadness / joy / love /        │  MPS → CUDA → CPU auto-select
│           anger / fear / surprise       │  Confidence + all_scores returned
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  M4: RAG Pipeline                       │  BAAI/bge-base-en-v1.5 (768-dim)
│  Query rewriting → Qdrant search →      │  Cosine similarity, free cloud tier
│  Prompt with emotion + tone + language  │  Groq gpt-oss-120b generates response
│  Response generated in user's language  │  directly via RESPOND IN: instruction
└─────────────────────────────────────────┘
      │
      ▼
  Response to User
```

### Why language detection runs FIRST (v3 safety fix)

Previous versions ran crisis detection on the raw (untranslated) message. Non-English crisis messages (e.g., Arabic **"أريد الانتحار"**) were missed entirely. In v3, language detection and translation happen first, so crisis detection always operates on English text — reliably detecting crisis signals in all 20 supported languages.

---

## 📂 Project Structure

```
NLP_Final_Project/
│
├── notebooks/
│   ├── 01_language_detection.ipynb    ← M1: TF-IDF + LinearSVC training & eval
│   ├── 02_emotion_classifier.ipynb    ← M2: DistilBERT fine-tuning & eval
│   ├── 03_intent_classifier.ipynb     ← M3: Few-shot prompting demonstration
│   └── 04_rag_pipeline.ipynb          ← M4: Qdrant retrieval & RAG evaluation
│
├── src/
│   ├── modules/
│   │   ├── language_detector.py       ← M1: 3-layer Unicode/TF-IDF/langdetect
│   │   ├── emotion_classifier.py      ← M2: DistilBERT with MPS acceleration
│   │   ├── intent_classifier.py       ← M3: Groq few-shot, 5 intent classes
│   │   └── rag_pipeline.py            ← M4: Qdrant + bge embeddings + Groq LLM
│   │
│   ├── pipeline/
│   │   └── chat_engine.py             ← Orchestrator: lang→translate→crisis→intent→RAG
│   │
│   └── utils/
│       ├── preprocessing.py           ← Text cleaning, chunking, QA formatting
│       └── crisis_detector.py         ← Multilingual crisis keyword scorer (EN/AR/FR/ES)
│
├── app/
│   └── streamlit_app.py               ← UI v3: Plotly analytics, translation badge, fixed bugs
│
├── api/
│   └── server.py                      ← FastAPI v2: full schemas with lang_conf, crisis_level
│
├── scripts/
│   ├── train_language_model.py        ← Train M1, save models/language_detector.pkl
│   ├── train_emotion_model.py         ← Fine-tune M2, save models/emotion_distilbert/
│   └── ingest_rag.py                  ← Load dataset, embed, upsert to Qdrant (full payload)
│
├── tests/
│   └── test_suite.py                  ← 8-section test suite: 50+ assertions
│
├── data/                              ← Datasets (git-ignored)
├── models/                            ← Trained weights (git-ignored)
├── requirements.txt
├── .env.example
└── Makefile
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

| Variable | Description | Required |
|----------|-------------|----------|
| `LLM_API_KEY` | Groq API key from [console.groq.com](https://console.groq.com/keys) | **Yes** |
| `QDRANT_CLUSTER_ENDPOINT` | Qdrant Cloud HTTPS URL from [cloud.qdrant.io](https://cloud.qdrant.io) | **Yes** |
| `QDRANT_API_KEY` | Qdrant Cloud read/write API key | **Yes** |
| `GROQ_API_KEY` | Alias for `LLM_API_KEY` (either works) | Optional alias |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your Groq and Qdrant keys
```

### 3. Train / Download Models (Optional for local deployment)

Modules have cloud fallbacks — if local models don't exist, they auto-download:

```bash
# M1: Train language detector (saves models/language_detector.pkl)
python scripts/train_language_model.py

# M2: Fine-tune emotion classifier (saves models/emotion_distilbert/)
python scripts/train_emotion_model.py
```

> **Note:** If local models are absent, M1 falls back to `langdetect` and M2 falls back
> to `bhadresh-savani/distilbert-base-uncased-emotion` from HuggingFace Hub. The app
> starts without training — ideal for Streamlit Cloud deployment.

### 4. Ingest Data into Qdrant (One-time Setup)
```bash
python scripts/ingest_rag.py
```

> The collection `health-counseling-dataset` stores full `context` + `response` payloads
> alongside the embedded chunk, enabling rich source display in the UI.

### 5. Launch the Application

**Streamlit UI:**
```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

**FastAPI REST API:**
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
# Swagger docs at http://localhost:8000/docs
```

**Makefile shortcuts:**
```bash
make install      # pip install -r requirements.txt
make train-lang   # python scripts/train_language_model.py
make train-emotion # python scripts/train_emotion_model.py
make ingest       # python scripts/ingest_rag.py
make app          # streamlit run app/streamlit_app.py
make api          # uvicorn api.server:app ...
make train-all    # train-lang + train-emotion
```

---

## 🧪 Testing & Evaluation

```bash
python tests/test_suite.py
```

The test suite covers **8 sections** with **50+ assertions**:

| Section | Tests |
|---------|-------|
| 1. Preprocessing utilities | clean_text, chunk_text, build_qa_chunk, edge cases |
| 2. Crisis detector | English HIGH/MEDIUM/LOW, Arabic HIGH/MEDIUM, false-positive guards |
| 3. Language detector (M1) | Script detection for 7 scripts, English defaults, Latin models |
| 4. Intent classifier (M3) | 11 intent cases, field validation, empty input |
| 5. RAG pipeline (M4) | Retrieval quality, query rewriting, full RAG, direct reply |
| 6. Chat engine (E2E) | All pipeline keys, lang_conf fix, Arabic crisis, session memory |
| 7. Import smoke tests | All 7 modules, constant consistency checks |
| 8. API model validation | Pydantic models, field presence, SourceDoc fields |

---

## 🛢️ Qdrant Payload Schema

Each vector in `health-counseling-dataset` stores:

| Field | Description |
|-------|-------------|
| `chunk` | Embedded text (full "Patient: … Counselor: …" or sub-chunk) |
| `context` | Original patient question (capped at 500 chars) |
| `response` | Counselor answer (capped at 500 chars) |
| `chunk_idx` | Position within the source document |
| `total_chunks` | Total sub-chunks from the source record |
| `source_row_id` | Original dataset row number for traceability |

The UI source expander shows separate **Q:** / **A:** panels when `context` and `response`
are populated, falling back to the full chunk text for older records.

---

## 📊 Module Performance

### M1 — Language Detector (TF-IDF + LinearSVC)
Trained on `papluca/language-identification` (70k samples, 20 languages).

| Set | Accuracy |
|-----|---------|
| Validation | ~99.1% |
| Test | ~98.8% |

Layer priority: Unicode script (instant) → short ASCII default → TF-IDF model → langdetect.

### M2 — Emotion Classifier (DistilBERT)
Fine-tuned on `dair-ai/emotion` (16k training, 6 classes).

| Metric | Score |
|--------|-------|
| Test Accuracy | ~93.5% |
| Weighted F1 | ~93.2% |

Classes: sadness, joy, love, anger, fear, surprise.

### M3 — Intent Classifier (Few-shot Groq)
5-class few-shot prompting via `gpt-oss-20b`. No training required.

| Intent | Route |
|--------|-------|
| greeting, goodbye, gratitude, out_of_scope | `direct` (lightweight LLM reply) |
| asking_mental_health_question | `rag` (full M2 + M4 pipeline) |

### M4 — RAG Pipeline (Qdrant + BGE + Groq)
- **Embedder:** `BAAI/bge-base-en-v1.5` (768-dim, cosine similarity)
- **Collection:** 10,000+ vectorized counseling Q&A pairs
- **Retrieval:** Query rewriting → top-k Qdrant search → ranked by cosine score
- **Generation:** `gpt-oss-120b` with emotion-aware, language-aware prompt

---

## 🌍 Supported Languages

| Code | Language | Detection Layer |
|------|----------|----------------|
| ar | Arabic | Unicode script (instant) |
| zh | Chinese | Unicode script (instant) |
| ja | Japanese | Unicode script (instant) |
| hi | Hindi | Unicode script (instant) |
| ru | Russian | Unicode script (instant) |
| el | Greek | Unicode script (instant) |
| th | Thai | Unicode script (instant) |
| ko | Korean | Unicode script (instant) |
| ur | Urdu | Unicode script (instant) |
| en | English | Short-ASCII default + TF-IDF |
| fr | French | TF-IDF model |
| de | German | TF-IDF model |
| es | Spanish | TF-IDF model |
| it | Italian | TF-IDF model |
| pt | Portuguese | TF-IDF model |
| nl | Dutch | TF-IDF model |
| pl | Polish | TF-IDF model |
| sv | Swedish | TF-IDF model |
| tr | Turkish | TF-IDF model |
| vi | Vietnamese | TF-IDF model |

All non-English messages are translated to English for NLP processing, then the LLM is instructed to respond in the user's detected language.

---

## ☁️ Streamlit Cloud Deployment

1. Push your repository to GitHub (`.env` and `models/` are git-ignored).
2. Log into [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select your repo, branch `main`, main file `app/streamlit_app.py`.
4. Under **Advanced settings → Secrets (TOML format)**:
   ```toml
   LLM_API_KEY = "gsk_your_groq_api_key"
   QDRANT_CLUSTER_ENDPOINT = "https://your-cluster-id.qdrant.io"
   QDRANT_API_KEY = "your_qdrant_api_key"
   ```
5. Click **Deploy!** — fallback layers activate automatically (langdetect + HuggingFace Hub).

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| Qdrant timeout on cold start | Set `QDRANT_TIMEOUT=60` — default is already 60s in code |
| LLM responds in English to Arabic input | Ensure `language_name` is detected correctly — check Module 1 status in sidebar |
| Models path error on Streamlit Cloud | Delete `models/` entries in `.env`; app will use cloud fallbacks |
| `LLM_API_KEY not set` error | Copy `.env.example` to `.env` and add your Groq key |
| `QDRANT_CLUSTER_ENDPOINT not set` | Add your Qdrant Cloud URL to `.env` |
| Crisis response in English for Arabic user | Fixed in v3 — language detection now precedes crisis check |
| `strong_model` / `top_k` settings ignored | Fixed in v3 — sidebar settings now forwarded to `engine.process()` |

---

## 📜 Dataset & Model Credits

| Resource | Source |
|----------|--------|
| Language detection training data | [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) |
| Emotion classification training data | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| RAG knowledge base | [heliosbrahma/mental_health_counseling_conversations](https://huggingface.co/datasets/heliosbrahma/mental_health_counseling_conversations) |
| Embedding model | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| Emotion hub fallback | [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) |
| LLM provider | [Groq Cloud](https://console.groq.com) — gpt-oss-120b / gpt-oss-20b |
| Vector database | [Qdrant Cloud](https://cloud.qdrant.io) — free tier |

---

## ⚠️ Mental Health Disclaimer

MindBot is an educational NLP project and **is not a substitute for professional mental health care**. The chatbot:
- Does **not** provide medical diagnoses
- Does **not** prescribe medications
- Should **not** be used as the sole resource during a mental health crisis

If you or someone you know is in crisis, please contact a trained professional or call your local emergency services.
