"""
AGENT - The brain. Same ReACT loop, new tools.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import glob
import anthropic
from retriever import retrieve_context
from embeddings import get_vector_store, add_documents
from loader import load_and_chunk_pdf
from config import MODEL, MAX_TOKENS, MAX_STEPS, ANTHROPIC_API_KEY

# --- Tool Definitions (the menu for the LLM) ---
tools = [
    {
        "name": "retrieve_context",
        "description": "Search the loaded documents for information relevant to a query. Returns the most relevant text chunks with source and page number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific."
                }
            },
            "required": ["query"]
        }
    }
]

# --- System Prompt ---
SYSTEM_PROMPT = """You are a document Q&A assistant. You answer questions based on the documents loaded in your database.

RULES:
1. ALWAYS use the retrieve_context tool before answering.
2. Base your answers ONLY on the retrieved context.
3. If the context doesn't contain the answer, say "I couldn't find this in the loaded documents."
4. Always mention which page the information came from.
"""

# --- Tool Execution ---
def execute_tool(tool_name, tool_input):
    """Run the tool and return result."""
    if tool_name == "retrieve_context":
        return retrieve_context(tool_input["query"])
    else:
        return f"Unknown tool: {tool_name}"


# --- The ReACT Loop ---
def chat(user_message):
    """Process a question through the ReACT loop."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": user_message}]

    max_steps = MAX_STEPS
    step = 0

    while step < max_steps:
        step += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Searching: {block.input.get('query', '')}")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

    return "Reached maximum steps."


# --- Interactive Chat ---
if __name__ == "__main__":
    DOCS_FOLDER = "./documents"

    print("=" * 50)
    print("  RAG Document Chat Agent")
    print("=" * 50)
    print("\nThis agent answers questions from your PDF")
    print("documents using Retrieval Augmented Generation.")

    # Create documents folder if it doesn't exist
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"\nCreated '{DOCS_FOLDER}' folder.")
        print("Drop your PDF files there and run again.")
        sys.exit(0)

    # Find all PDFs in the folder
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))

    if not pdf_files:
        print(f"\nNo PDFs found in '{DOCS_FOLDER}' folder.")
        print("Drop your PDF files there and run again.")
        sys.exit(0)

    # Show what we found
    print(f"\nFound {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  - {os.path.basename(f)}")

    # Check if already loaded
    collection = get_vector_store()
    if collection.count() > 0:
        print(f"\nDatabase already has {collection.count()} chunks.")
        reload = input("Reload documents? (y/n): ").strip().lower()
        if reload == "y":
            import chromadb
            from config import PERSIST_DIRECTORY, COLLECTION_NAME
            client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            client.delete_collection(name=COLLECTION_NAME)
            print("Database cleared.")
            print("\nUpdate your PDFs in the 'documents' folder if needed.")
            input("Press Enter when ready to continue...")
        else:
            print()

    # Load PDFs if database is empty
    collection = get_vector_store()
    if collection.count() == 0:
        print("\nLoading documents...")
        for pdf_file in pdf_files:
            print(f"\n  Loading {os.path.basename(pdf_file)}...")
            chunks = load_and_chunk_pdf(pdf_file)
            add_documents(chunks)

        collection = get_vector_store()
        print(f"\nTotal chunks loaded: {collection.count()}")

    print("\nReady! Ask questions about your documents.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nThinking...")
        answer = chat(user_input)
        print(f"\nAssistant: {answer}\n")