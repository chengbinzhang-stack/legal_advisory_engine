# Legal Data Protection Engine and Advisory Chatbot

A comprehensive legal analysis tool that processes website terms of use, privacy policies, and robots.txt files to provide actionable insights on data scraping, storage, display, and redistribution permissions.

## Features

- **Legal Data Protection Engine**: Scrapes and analyzes legal documents from websites
- **RAG Pipeline**: Stores legal documents in a vector database for semantic search
- **7-Parameter Analysis**: Evaluates permissions for scraping, collection, storage, display, and redistribution
- **4 Category Buckets**: Classifies websites into permission categories
- **Legal Advisory Chatbot**: Interactive Q&A about website legal terms

## Architecture

### Data Flow

```
URL Input
    ↓
LegalDataEngine.process_website()
    ├── Scrapers: TermsScraper + PrivacyScraper + RobotsScraper
    ├── LegalClassifier.classify_permissions() → 7-parameter analysis → bucket category
    ├── DocumentStore.store_website_data() → chunk → embed → ChromaDB
    └── _save_analysis() → JSON to ./data/summaries/
```

```
User Question (Chatbot)
    ↓
ResponseGenerator.generate_response()
    ├── QueryEngine.query() → ChromaDB similarity search
    ├── PromptBuilder.build_query_prompt() → build prompt with context
    └── MiniMaxClient.chat() → LLM response
```

### RAG Pipeline (ChromaDB)

ChromaDB is the core vector database for the RAG (Retrieval-Augmented Generation) pipeline:

| Module | Role |
|--------|------|
| `DocumentStore` | Chunks legal documents, generates embeddings via scikit-learn TF-IDF, stores in ChromaDB |
| `QueryEngine` | Retrieves top-k similar document chunks from ChromaDB based on query embedding |

### Permission Categories

| Bucket | Scraping | Storing | Display | Redistribute |
|--------|----------|---------|---------|--------------|
| 1 | Yes      | Yes     | Yes     | Yes          |
| 2 | Yes      | Yes     | Yes     | No           |
| 3 | Yes      | Yes     | No      | No           |
| 4 | No       | No      | No      | No           |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

```bash
export MINIMAX_API_KEY=your_api_key_here
streamlit run app.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MINIMAX_API_KEY` | MiniMax API key for LLM integration |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `EMBEDDING_DEVICE` | Device for embedding generation (`cuda`/`cpu`) |
| `STREAMLIT_CLOUD` | Set to `1` to use in-memory ChromaDB (avoids inotify limits on Streamlit Cloud) |

## Project Structure

```
.
├── app.py                        # Streamlit main application
├── config.py                     # Configuration settings
├── src/
│   ├── legal_data_engine.py      # Main orchestration class
│   ├── scraper/                  # Document scrapers (terms, privacy, robots.txt)
│   ├── parser/                   # Text chunking and parsing
│   ├── embeddings/
│   │   ├── chroma_client.py      # ChromaDB wrapper
│   │   └── embedding_generator.py # TF-IDF embedding generation
│   ├── classifier/               # 7-parameter permission classification
│   ├── rag/
│   │   ├── document_store.py    # RAG document storage
│   │   └── query_engine.py      # RAG similarity search
│   └── chatbot/
│       ├── response_generator.py # LLM response generation
│       └── prompt_builder.py     # Prompt construction
├── data/summaries/               # Analysis results (JSON)
└── chroma_db/                    # ChromaDB persistent storage (local only)
```

## License

MIT