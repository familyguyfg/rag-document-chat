"""
RETRIEVER - The tool layer for our RAG agent.
"""

from embeddings import search
from config import TOP_K


def retrieve_context(query, top_k=TOP_K):
    """
    Search documents and return formatted context.
    This is what the agent will call.
    """
    
    results = search(query, top_k)
    
    if not results['documents'][0]:
        return "No relevant documents found."
    
    context_parts = []
    for i in range(len(results['documents'][0])):
        source = results['metadatas'][0][i].get('source', 'unknown')
        page = results['metadatas'][0][i].get('page', '?')
        text = results['documents'][0][i]
        
        context_parts.append(
            f"[Source: {source}, Page {page}]\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)
