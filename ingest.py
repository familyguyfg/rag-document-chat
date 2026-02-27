"""
INGEST - Load PDFs into the vector database.
Usage:
    python3 ingest.py paper1.pdf
    python3 ingest.py paper1.pdf paper2.pdf paper3.pdf
    python3 ingest.py --clear paper1.pdf   (clears database first)
"""

import sys
import os
import chromadb
from loader import load_and_chunk_pdf
from embeddings import add_documents, get_vector_store
from config import PERSIST_DIRECTORY, COLLECTION_NAME


def clear_database():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Database cleared.")
    except Exception:
        print("Database was already empty.")


def ingest(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nLoading: {file_path}")
    chunks = load_and_chunk_pdf(file_path)
    add_documents(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest.py <pdf1> <pdf2> ...")
        print("       python3 ingest.py --clear <pdf1> <pdf2> ...")
        sys.exit(1)

    args = sys.argv[1:]

    if "--clear" in args:
        clear_database()
        args.remove("--clear")

    for file_path in args:
        ingest(file_path)

    collection = get_vector_store()
    print(f"\nDone! Database has {collection.count()} total chunks.")
    print("Run: python3 agent.py to start chatting.")
    