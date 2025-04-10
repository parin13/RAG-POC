# RAG LLM Project

This project implements a Retrieval-Augmented Generation (RAG) system using Large Language Models (LLMs) to process and analyze PDF documents.

## Setup Instructions



2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.sample` to `.env`
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     HUGGING_FACE_ACCESS_TOKEN=your_huggingface_token
     ```

4. **Add PDF Documents**
   - Place your PDF files in the `Rag/pdfData/` directory
   - Supported formats: PDF files
   - Note: The `pdfData` directory is gitignored for privacy

## Project Structure

```
ragLLm/
├── Rag/
│   ├── pdfData/          # Place your PDF files here
│   ├── .env.sample       # Environment variables template
│   └── ...              # Other project files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

[Add usage instructions once the main functionality is implemented]

## Dependencies

- Python 3.10.14
- OpenAI API
- Hugging Face Transformers
- [Add other major dependencies]
