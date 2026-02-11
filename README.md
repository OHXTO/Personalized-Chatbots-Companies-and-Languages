## Status
🚧 Work in progress (WIP). A local demo is available.


# Personalized Company Chatbot (Catholic Health – Long Island)
This project is a simple company Q&A chatbot demo built during Anote AI Academy.  
It answers questions **only from provided company materials** (local files) using a lightweight **RAG (Retrieval-Augmented Generation)** pipeline and a **local LLM** via Ollama.

**Demo knowledge base:** Catholic Health – Long Island (Mission / Locations / FAQ / etc.)

## What it does
- Chat UI in the browser
- Backend `/api/chat` retrieves relevant text chunks from `server/data/docs/`
- Generates a concise answer using a local model (Ollama + Mistral)
- Returns citations (source file + excerpt) for transparency
- If the question is ambiguous / not supported by the materials, it asks for clarification or says it doesn’t know

## Quick Start

### Prerequisites
- Node.js + npm
- Python 3.10+
- Ollama installed with a model `mistral`

### 1) Start Ollama
Make sure Ollama is running and a model is available:
```bash
ollama list
# optional:
ollama serve

### Backend (Flask)
```bash
cd server
pip install -r requirements.txt
python app.py
# server runs on http://127.0.0.1:5000
```

### Frontend (React)
```
cd frontend
npm install
npm start
# frontend runs on http://localhost:3000
```

### Add / Update Company Materials
Put text files into: *server/data/docs/*

Examples:
- `mission.txt`
- `location.txt`
- `faq.txt`
Restart the backend after adding new files so the index is refreshed.

## Tech Stack
- Frontend: React
- Backend: Flask
- Retrieval (RAG): TF-IDF + cosine similarity over local text chunks
- LLM: Ollama (local Mistral/Llama2)


##Notes

- This is a learning/demo project. The chatbot is configured to answer using only the provided materials.

- Current scope is intentionally small to keep the demo reliable and easy to evaluate.


