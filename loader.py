"""
LOADER - Load PDFs and split into chunks.
"""

import os
import fitz
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path):
    """Open a PDF and extract text from each page."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    doc = fitz.open(file_path)
    filename = os.path.basename(file_path)
    
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": filename
            })
    
    doc.close()
    print(f"Loaded {len(pages)} pages from {filename}")
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]    # already small enough
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap    # move forward 450 words
    
    return chunks


def load_and_chunk_pdf(file_path):
    """
    Full pipeline: PDF -> Pages -> Chunks with metadata.
    This is what the rest of the system will use.
    """
    
    pages = load_pdf(file_path)
    
    all_chunks = []
    chunk_index = 0
    
    for page in pages:
        chunks = chunk_text(page["text"])
        
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index
                }
            })
            chunk_index += 1
    
    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
