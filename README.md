# Document Synthesis RAG System

Welcome to the **Document Synthesis RAG (Retrieval-Augmented Generation) System**!  
This repository contains a full-stack, production-ready platform for uploading, indexing, querying, and synthesizing answers from large collections of documents (PDFs, images, and text files). The backend leverages FastAPI, MongoDB, FAISS, and the Groq Llama 3 LLM for efficient and explainable retrieval and synthesis. The frontend is a modern JavaScript web application for seamless user interaction.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Pipeline Diagram](#pipeline-diagram)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Frontend Usage](#frontend-usage)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Multi-format document ingestion**: Supports PDFs, image files (with OCR), and plain text files.
- **Automated text extraction**: Uses OCR for scanned/image-based documents and smart loaders for digital files.
- **Granular chunking and embedding**: Splits documents into manageable chunks and generates vector embeddings for each.
- **Semantic search**: FAISS-powered vectorstore enables lightning-fast, semantically rich retrieval.
- **Reranking**: CrossEncoder reranker improves retrieval quality for more relevant context.
- **LLM-powered synthesis**: Integrates Groq Llama 3 for high-quality, context-aware answer generation with explicit citations.
- **Theme extraction**: Summarizes main themes across all uploaded documents.
- **Per-document and multi-document querying**: Answer questions across all or selected documents.
- **Full CRUD file management**: Upload, list, and delete documents from the web interface.
- **Modern, user-friendly frontend**: Drag-and-drop uploads, real-time file listing, and interactive querying.
- **Transparent and explainable**: All answers are grounded in the source with page and paragraph citations.
- **Modular and scalable**: Easily extend or swap components (e.g., embedding models, LLMs, storage).

---

## Architecture Overview

This system is composed of two main components:

- **Backend (FastAPI)**
  - Handles file uploads, document processing, embedding, vector search, reranking, and LLM-based synthesis.
  - Stores document metadata in MongoDB and vector embeddings in FAISS.
  - Exposes RESTful API endpoints for all frontend operations.

- **Frontend (JavaScript/HTML/CSS)**
  - Provides a modern web interface for document management and querying.
  - Communicates with the backend via REST APIs.

---

## Pipeline Diagram

Below is a conceptual diagram of the document synthesis RAG pipeline:

```
User Uploads (Web Frontend)
        |
        v
FastAPI Backend (File Save & Metadata)
        |
        +---------------------> MongoDB (Document Metadata)
        |
        v
Document Loader & OCR
        |
        v
Text Splitter
        |
        v
Embedding Model
        |
        v
FAISS Vectorstore
        |
        v
Retriever + Reranker
        |
        v
Groq Llama 3 (LLM)
        |
        v
Synthesized Answer (with Citations)
        |
        v
Web Frontend (Display Answer)
```

---

## How It Works

1. **Document Upload & Storage**
   - Users upload files via the web interface.
   - Files are saved to disk and metadata (filename, type, upload time, etc.) is stored in MongoDB.

2. **Text Extraction**
   - The backend detects the file type.
   - For images and scanned PDFs, OCR (pytesseract) extracts text.
   - For digital PDFs and text files, direct loaders extract text.

3. **Chunking & Embedding**
   - Extracted text is split into smaller chunks for efficient retrieval.
   - Each chunk is embedded using a configurable embedding model.
   - Embeddings are stored in a FAISS vectorstore.

4. **Semantic Retrieval & Reranking**
   - User queries are matched to relevant chunks using FAISS.
   - A CrossEncoder reranker refines the retrieved results for contextual relevance.

5. **LLM Synthesis**
   - The most relevant chunks are sent, along with the user query, to Groq Llama 3.
   - The LLM generates a concise answer, always citing page and paragraph numbers.

6. **Result Display**
   - The synthesized answer and citations are returned to the frontend and displayed to the user.

7. **Theme Extraction**
   - Users can request a summary of main themes across all documents, synthesized by the LLM.

8. **File Management**
   - Users can list, select, and delete documents from the web interface.
   - Deletions update both the file store and MongoDB.

---

## Installation

### Prerequisites

- Python 3.9+
- Node.js (for frontend, optional if using static files)
- MongoDB (local or cloud)
- [Groq API access](https://groq.com/) (or compatible LLM endpoint)
- (Recommended) Virtual environment for Python

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/document-synthesis-rag.git
   cd document-synthesis-rag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Create a `.env` file in the root directory.
   - Set the following variables:
     ```
     MONGODB_URI=mongodb://localhost:27017/
     GROQ_API_KEY=your_groq_api_key
     EMBEDDING_MODEL=your_embedding_model_name
     ```

5. **Start the backend server:**
   ```bash
   uvicorn app:app --reload
   ```

### Frontend Setup

1. **Install dependencies (if using a build tool):**
   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend (if applicable):**
   ```bash
   npm start
   ```
   Or, simply open `index.html` in your browser if using static files.

---

## Configuration

- **MongoDB**: Ensure your MongoDB instance is running and accessible.
- **FAISS Vectorstore**: The vector index is stored locally; ensure the backend has write permissions.
- **OCR**: Tesseract must be installed and accessible in your system path for OCR to function.
- **Groq API**: Obtain an API key from [Groq](https://groq.com/) and configure it in your `.env`.
- **Embedding Model**: Set the embedding model name (e.g., OpenAI, HuggingFace, etc.) in your `.env`.

---

## Usage

### Uploading Documents

- Drag and drop files into the upload area or use the file picker.
- Uploaded files will appear in the file list with metadata.

### Querying

- Type your question in the query box and submit.
- Select specific documents for per-document queries, or query across all uploaded documents.
- Answers will be displayed with citations (page and paragraph).

### Theme Extraction

- Click the "Extract Themes" button to receive a bullet-point summary of main topics across all documents.

### Managing Files

- Use the file list to select or delete documents.
- Deletions remove both the file and its metadata from the system.

---

## API Endpoints

| Endpoint            | Method | Description                                      |
|---------------------|--------|--------------------------------------------------|
| `/upload`           | POST   | Upload one or more documents                     |
| `/list_files`       | GET    | List all uploaded documents                      |
| `/delete_file`      | POST   | Delete a specific document                       |
| `/query`            | POST   | Submit a query for synthesis                     |
| `/query_per_doc`    | POST   | Query selected documents individually            |
| `/extract_themes`   | POST   | Extract main themes from all documents           |

**Request and response formats are documented in the `docs/` folder and via FastAPI's `/docs` endpoint.**

---

## Frontend Usage

- **Uploading**: Drag and drop or use the upload button.
- **Selecting**: Click checkboxes to select documents for per-document queries or deletion.
- **Querying**: Enter your question and submit; answers appear below with citations.
- **Theme Extraction**: Click to receive a summary of main themes.
- **Feedback**: Errors and status messages are shown via modals and alerts.

---

## Customization

- **Embedding Model**: Swap out for any model supported by your embedding provider.
- **LLM**: Replace Groq Llama 3 with any OpenAI-compatible endpoint.
- **Storage**: Switch MongoDB for another database by updating the backend.
- **Chunking Strategy**: Adjust chunk size and overlap in the backend configuration.
- **Reranker**: Replace or tune the CrossEncoder reranker as needed.
- **Frontend**: Modify styles, add features, or integrate with other platforms as desired.

---

## Troubleshooting

- **OCR not working**: Ensure Tesseract is installed and in your system path.
- **Vectorstore errors**: Check file permissions and available disk space.
- **MongoDB connection issues**: Verify URI, credentials, and network access.
- **LLM errors**: Confirm your API key and endpoint; check rate limits.
- **Frontend not connecting**: Ensure CORS is enabled and frontend points to the correct backend URL.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or documentation improvements.

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear messages.
4. Push to your fork and submit a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [MongoDB](https://www.mongodb.com/)
- [FAISS](https://faiss.ai/)
- [Groq](https://groq.com/)
- [LangChain](https://www.langchain.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## Contact

For questions, issues, or feature requests, please open an issue or contact the maintainer at [your-email@example.com].

---

**Enjoy powerful, explainable, and scalable document synthesis with this RAG system!**

---
Answer from Perplexity: pplx.ai/share
