"""
EMBEDDINGS - Convert text to vectors and store in ChromaDB.
"""

import chromadb
from config import PERSIST_DIRECTORY, COLLECTION_NAME, TOP_K


def get_vector_store():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

def get_vector_store():
    """Connect to (or create) the ChromaDB database."""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="documents")
    return collection


def add_documents(chunks):
    """Embed and store chunks in the vector database."""
    
    collection = get_vector_store()
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{chunk['metadata']['chunk_index']}" for chunk in chunks]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Stored {len(documents)} chunks in vector database")

def search(query, top_k=TOP_K):
    """Find the most relevant chunks for a question."""
    
    collection = get_vector_store()
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results