# Legal Data Protection Engine and Advisory Chatbot

A comprehensive legal analysis tool that processes website terms of use, privacy policies, and robots.txt files to provide actionable insights on data scraping, storage, display, and redistribution permissions.

## Features

- **Legal Data Protection Engine**: Scrapes and analyzes legal documents from websites
- **RAG Pipeline**: Stores legal documents in a vector database for semantic search
- **7-Parameter Analysis**: Evaluates permissions for scraping, collection, storage, display, and redistribution
- **4 Category Buckets**: Classifies websites into permission categories
- **Legal Advisory Chatbot**: Interactive Q&A about website legal terms

## Installation



## Usage

### Local Development



### Docker Deployment



## Project Structure



## Permission Categories

| Bucket | Scraping | Storing | Display | Redistribute |
|--------|----------|---------|---------|--------------|
| 1 | Yes      | Yes     | Yes     | Yes          |
| 2      | Yes      | Yes     | Yes     | No           |
| 3      | Yes      | Yes     | No      | No           |
| 4      | No       | No      | No      | No           |

## Environment Variables

- : Anthropic API key for LLM integration
- : Device for embedding generation (cuda/cpu)

## License

MIT