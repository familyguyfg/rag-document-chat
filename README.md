# RAG Document Chat Agent

A RAG agent that answers your questions from a set of documents you provide. Instead of searching the web, it searches within your documents using vector embeddings.

## What I Learned Building This

The big new concept for me was **embeddings and vector databases**. I learned that embedding models already exist — you don't build them. You just store your data through them, and similar content gets stored as similar numbers. Then when you ask a question, it finds the closest matching content by meaning, not just keywords.

The architecture is the same as any agent: config → tools → brain. The only difference from my research agent is the tools. Instead of web_search, this agent uses retrieve_context to search a vector database.

## How It Works

1. **Load a PDF** — extract text and split into overlapping chunks
2. **Embed chunks** — convert each chunk into vectors using ChromaDB's built-in model
3. **Store in vector DB** — similar content gets stored as similar numbers
4. **Ask questions** — your question gets embedded too, finds the closest chunks
5. **LLM answers** — only using the retrieved chunks, no hallucination

## Project Structure
```
config.py       → All settings in one place
loader.py       → PDF loading and chunking
embeddings.py   → Vector storage and search (ChromaDB)
retriever.py    → Search tool for the agent
agent.py        → ReACT loop (the brain)
ingest.py       → One-time script to load PDFs
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install pymupdf chromadb anthropic python-dotenv

cp .env.example .env
# Add your Anthropic API key to .env
```

## Usage
```bash
# Step 1: Load a PDF (run once per document)
python3 ingest.py your-document.pdf

# Step 2: Chat with it
python3 agent.py
```

## Key Concepts

- **RAG** = Retrieval Augmented Generation — search your docs first, then let the LLM answer
- **Embeddings** = text converted to numbers, similar meaning = similar numbers
- **Chunking** = splitting documents into small overlapping pieces for precise search
- **Grounded answers** = LLM only answers from retrieved context, prevents hallucination
