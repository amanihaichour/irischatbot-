# IRIS Conversational Assistant

A customer-support chatbot built as a complete, free RAG (Retrieval-Augmented Generation) pipeline. It answers product questions in **French and English**, classifies each incoming message, captures customer details for complaints, and connects to real messaging channels.

> This project started as an internship at IRIS. Since no internal company data was available, it is built and demonstrated on a public Kaggle dataset. The pipeline itself is data-agnostic and can be pointed at a real catalog.

## Features

- **RAG architecture**: answers are generated only from information in the knowledge base, so the bot cannot invent products or prices.
- **Hybrid search**: semantic similarity (ChromaDB) combined with exact filters on price and brand.
- **Bilingual (FR/EN)**: French queries are translated to English before retrieval, since the embedding model is mostly English.
- **Typo and slang tolerance**: an extensible domain glossary (`data/glossary.json`) corrects informal terms (e.g. "frigo" to "refrigerateur").
- **Multi-turn memory**: short follow-ups like "et le prix ?" are re-contextualized with the conversation history.
- **Automatic intent classification**: `remerciement`, `reclamation`, `demande_info`, `demande_prix`, `autre`, using few-shot prompting.
- **Complaint flow**: apologizes, asks for name and phone number, extracts them (regex for the phone, LLM for the name) and logs them for a support team.
- **Three-level confidence logic**: answers directly, asks at most one clarification question, or honestly says no matching product exists. It never loops and never presents irrelevant products.
- **Structured logging** in SQLite plus an analysis script.
- **Channels**: web widget, WhatsApp Business Cloud API (working), Instagram Direct (code ready, blocked on the Meta platform side).

## How it works

```
Customer message
      |
      v
Classification (LLM, few-shot)
      |
      +-- reclamation / remerciement --> dedicated flow --> log
      |
      +-- other --> glossary --> FR->EN translation --> constraint extraction
                    (price, brand) --> hybrid search (ChromaDB)
                    --> confidence check --> LLM answer --> log
```

## Tech stack

| Component | Technology |
|---|---|
| Vector database | ChromaDB (local persistent client) |
| Embeddings | all-MiniLM-L6-v2 (Chroma default) |
| Language model | Llama 3.2, run locally with Ollama |
| Translation FR to EN | MyMemory via `deep-translator` |
| Backend | FastAPI (Python) |
| Web interface | HTML / JavaScript chat widget |
| Logging | SQLite |
| Dev tunnel | ngrok (free tier) |

Everything is free to use: no paid API is required.

## Project structure

```
iris/
├── src/
│   ├── api.py                        # FastAPI backend: pipeline, web chat, webhooks
│   ├── clean_data.py                 # Clean and deduplicate the raw Kaggle data
│   ├── build_knowledge_base_kaggle.py# Build the ChromaDB product collection
│   ├── loadFAQs.py                   # Load FAQ entries into the same collection
│   ├── analyze_logs.py               # Analytics on the conversation logs
│   ├── query_kb_no_llm.py            # Test retrieval without the LLM
│   ├── explore_dataset.py            # Dataset exploration
│   └── test_categories.py            # Classification tests
├── frontend/
│   └── index.html                    # Standalone chat widget
├── data/
│   ├── FAQs.json                     # 10 original FAQ entries
│   ├── glossary.json                 # Slang / typo corrections
│   ├── products_clean.csv            # Cleaned product data
│   └── electronics.csv               # Raw Kaggle data (not in the repo, see below)
└── chroma_db/                        # Generated vector store (not in the repo)
```

## Getting started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- (Optional) ngrok, only for testing WhatsApp/Instagram webhooks

### 1. Install

```bash
git clone https://github.com/amanihaichour/irischatbot-.git
cd irischatbot-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
```

### 2. Get the dataset

Download **Datafiniti Electronic Products and Pricing** from Kaggle and save the CSV as `data/electronics.csv`. It is about 180 MB, so it is excluded from Git.

### 3. Build the knowledge base

Run everything from the project root:

```bash
python src/clean_data.py                     # 14,592 rows -> 1,245 unique products
python src/build_knowledge_base_kaggle.py    # embed products into ChromaDB
python src/loadFAQs.py                       # add FAQ entries
```

### 4. Run the server

```bash
uvicorn src.api:app --reload
```

Then open `frontend/index.html` in your browser to chat with the bot.

## Configuration

Secrets are read from environment variables and must never be committed. Create a `.env` file in the project root (it is git-ignored):

```
# Replace with the variable names used in src/api.py
WHATSAPP_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=...
META_PAGE_ACCESS_TOKEN=...
```

## Messaging channels

### WhatsApp Business Cloud API (working)

The backend exposes a webhook that receives messages, runs the full pipeline and replies in the same WhatsApp conversation.

Setup notes:
1. Expose your local server with `ngrok http 8000` and register the public URL as the webhook in the Meta developer console.
2. Configuring the webhook is **not enough**. You must also subscribe the app to the WhatsApp Business Account with an explicit API call: `POST /{WABA_ID}/subscribed_apps`. Without it, no messages arrive.

### Instagram Direct (pending)

The code is implemented and tested at the endpoint level, but real message delivery is blocked by a Meta-side issue: tester invitations stay stuck in a non-active state. The code is kept and should work once this is resolved.

### LinkedIn (out of scope)

The LinkedIn Conversations API requires an approved partner status, which is an administrative process independent of development.

## Logging and analytics

Every exchange is stored in `data/chat_logs.db` (SQLite): session and channel, message category, customer message and reply, the search query actually used, matched products, filters, confidence score, whether clarification was requested, and name/phone/product for complaints and thanks.

```bash
python src/analyze_logs.py
```

The script reports total exchanges and sessions, distribution by category, clarification rate, lowest-confidence matches (which reveal gaps in the catalog), most frequent customer words, and exports a CSV.

> The log database can contain customer names and phone numbers. It is excluded from Git on purpose.

## Known limitations

- Uses a public consumer-electronics dataset, not IRIS's real catalog. It has no home appliances, so those queries correctly return "no match".
- The local LLM is free and unlimited but reasons less well than a top cloud model, which is why the prompts include few-shot examples.
- Translation depends on a free third-party service with no availability guarantee.
- Instagram delivery is blocked by a Meta platform issue.
- Conversation memory for WhatsApp/Instagram is kept in RAM and is lost when the server restarts.

## Recommendations for production

1. Replace the Kaggle data with the real IRIS product catalog.
2. Consider a cloud LLM if real usage exceeds what a local model can handle.
3. Deploy on a permanent server instead of a local ngrok tunnel.
4. Persist conversation memory for external channels (e.g. in SQLite or Redis).
5. Re-evaluate the Instagram integration once the platform issue is resolved.

## Dataset credit

[Datafiniti Electronic Products and Pricing Data](https://www.kaggle.com/datasets/datafiniti/electronic-products-and-pricing-data) (Kaggle).