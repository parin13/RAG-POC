# PDF RAG (Retrieval Augmented Generation) System

This project implements a RAG system that processes PDF documents and enables semantic search through vector embeddings. The system offers two storage options:

1. Local Storage (`ragExample1.py`)
2. Qdrant Vector Database (`quadrantDb.py`)

## Project Goal

The main objective of this project is to:
- Process and store PDF documents as vector embeddings
- Enable semantic search and question-answering capabilities
- Provide flexible storage options (local or vector database)
- Demonstrate RAG (Retrieval Augmented Generation) implementation using LlamaIndex

## Setup and Requirements

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # For Unix/Mac
# or
.venv\Scripts\activate  # For Windows

# Install dependencies
pip install llama-index python-dotenv qdrant-client
```

### 2. Environment Variables
Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=localhost:6333  # Required only for Qdrant DB implementation
```

### 3. Data Directory
Place your PDF files in the `Rag/pdfData` directory.

## Storage Options

### Option 1: Local Storage
Uses `ragExample1.py` which stores vector embeddings locally in the `storage` directory.

```bash
python Rag/ragExample1.py
```

### Option 2: Qdrant Vector Database (Recommended for Production)
Uses `quadrantDb.py` which requires Qdrant setup:

1. Install Qdrant using Docker:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

2. Run the Qdrant implementation:
```bash
python Rag/quadramtDb.py
```

## Project Structure
```
ragLLm/
├── Rag/
│   ├── pdfData/          # Directory for PDF files
│   ├── storage/          # Local storage for vector embeddings (gitignored)
│   ├── ragExample1.py    # Local storage implementation
│   └── quadramtDb.py     # Qdrant DB implementation
├── .env                  # Environment variables
└── README.md
```

## Important Notes
- The `storage` directory is git-ignored as it contains generated vector embeddings
- For production use, it's recommended to use Qdrant or another vector database instead of local storage
- Make sure to have sufficient OpenAI API credits as the system uses it for embeddings and queries
- The Qdrant implementation provides better scalability and persistence compared to local storage
