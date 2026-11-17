# LangChain Vector Store with ChromaDB

A hands-on Python example of storing documents as vector embeddings and retrieving semantically similar information using **LangChain**, **ChromaDB**, and **OpenAI Embeddings**.

## Project Overview

The `use_chromadb.py` script creates three LangChain `Document` objects describing **Texas, California, and New York**, each with state metadata. It uses OpenAI embeddings and a persistent ChromaDB collection named `usa_state_info` to store and search these documents.

## What This Example Covers

| Concept | How it is used |
| --- | --- |
| `Document` | Stores text (`page_content`) and state metadata |
| `OpenAIEmbeddings` | Converts document text and queries into numeric vectors |
| `Chroma` | Stores vectors in a collection named `usa_state_info` |
| `persist_directory` | Saves the database locally under `chroma_db/` |
| `add_documents()` | Inserts the three state documents |
| `get()` | Retrieves stored documents, metadata, and embeddings |
| `similarity_search()` | Finds the closest matching documents for a query |
| `similarity_search_with_score()` | Returns matches together with distance scores |

## How It Works

1. Create `Document` objects with content and metadata.
2. Generate embeddings with OpenAI's embedding model.
3. Store the vectors and metadata in a persistent ChromaDB collection.
4. Search using a natural-language query such as **"which one is located in south?"** with `k=1`.
5. Inspect the matching document and its score.

**Note:** Chroma's returned score is a *distance* (lower generally means closer), not a percentage confidence value.

## Getting Started

```bash
git clone https://github.com/priyankvora8274/langchain-vectorstore.git
cd langchain-vectorstore
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux or `.venv\\Scripts\\activate` on Windows), then install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

Run the example:

```bash
python use_chromadb.py
```

An OpenAI API key with access to embeddings is required and API usage may incur charges. Keep `.env` out of version control.

## Advantages and Limitations

**Advantages**
- Semantic search can match related meaning, not just exact keywords.
- Metadata can help organize and filter stored documents.
- Local persistence lets the vector collection survive across runs.
- Useful building block for document search and Retrieval-Augmented Generation (RAG).

**Limitations**
- Embedding generation requires an external API and may have a cost.
- Search relevance depends on the embedding model and document quality.
- Vector similarity alone does not verify factual accuracy.
- Re-running inserts may add duplicate records unless document IDs and ingestion are managed.

## Important Code Note

The current `use_chromadb.py` ends with an incomplete call:

```python
chromadb.update_document(document_id="")
```

`update_document()` requires a valid document ID **and** a replacement `Document`. Remove this line to run the search demonstration as-is, or implement an update with both arguments. The script also creates the `chroma_db/` directory locally; avoid committing generated database files unless intentional.

## Why Vector Stores Matter for RAG

In a RAG pipeline, relevant document chunks are embedded and stored in a vector database. When a user asks a question, the application retrieves related chunks and provides them as context to an LLM. **This repository demonstrates the storage and retrieval foundation**, not a complete RAG question-answering pipeline.

## Tech Stack

Python · LangChain · ChromaDB · OpenAI Embeddings · python-dotenv
