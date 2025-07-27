# RAG PDF Chatbot with Flask and Groq

A Flask-based RAG (Retrieval-Augmented Generation) chatbot that answers questions about PDF documents using Groq API.

## Features
- PDF document processing and embedding
- ChromaDB vector storage
- Groq API for fast inference
- Web interface with Flask

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your Groq API key to `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python app.py
```

## Usage
- Open http://localhost:5000
- Upload a PDF document using the file upload form
- Ask questions about the uploaded document
- Get AI-powered responses based on document content