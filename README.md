# RAG Document Chat Agent

An agent that lets you add your own PDF documents and chat with them. Instead of searching the web, it searches within your documents using vector embeddings and gives answers only from what you provided.

## What I Learned Building This

The hardest and most surprising part was understanding chunking and embeddings — both were completely new to me. The idea that text can be converted into numbers where similar meanings get similar numbers was a new concept. Once that clicked, everything else followed the same agent pattern I already knew: config → tools → brain.

## How It Works

1. Drop your PDFs into the `documents/` folder
2. Run the agent — it automatically loads and chunks your PDFs
3. Each chunk gets converted to a vector (embedding) and stored in ChromaDB
4. When you ask a question, your question also becomes a vector
5. The agent finds the closest matching chunks and sends them to the LLM
6. The LLM answers using ONLY the retrieved chunks — no hallucination

## Project Structure
```
config.py       → All settings in one place
loader.py       → PDF loading and chunking with overlap
embeddings.py   → Vector storage and search (ChromaDB)
retriever.py    → Search tool for the agent
agent.py        → ReACT loop brain + interactive chat
ingest.py       → Standalone script to load PDFs
documents/      → Drop your PDF files here
```

## Setup
```bash
git clone https://github.com/familyguyfg/rag-document-chat.git
cd rag-document-chat
python3 -m venv venv
source venv/bin/activate
pip install pymupdf chromadb anthropic python-dotenv

cp .env.example .env
# Add your Anthropic API key to .env
```

## Usage
```bash
# 1. Drop PDFs into the documents folder
cp your-paper.pdf documents/

# 2. Run the agent
python3 agent.py
```

The agent will find your PDFs, load them, and start chatting.

## Key Concepts

- **RAG** — Retrieval Augmented Generation: search your docs first, then let the LLM answer
- **Embeddings** — text converted to numbers, similar meaning = similar numbers
- **Chunking** — splitting documents into small overlapping pieces so search is precise
- **Vector Database** — stores embeddings and finds the closest matches by meaning, not keywords
- **Grounded Answers** — LLM only answers from retrieved context, prevents hallucination