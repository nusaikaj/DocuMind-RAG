# 📚 DocuMind RAG — Chat With Your Documents

A beginner-friendly **Retrieval-Augmented Generation (RAG)** app built with Python and Streamlit. Upload PDF, TXT, or Markdown documents, ask questions, and get Gemini-generated answers based on passages retrieved from your files, with source/page references.

> Made as a Data Science / Machine Learning portfolio project. This is a working starter, not a production-grade enterprise document search system.

## Features

- Multi-file upload: PDF, `.txt`, `.md`
- Text extraction with PyMuPDF (preserves PDF page numbers)
- Overlapping text chunks
- Local semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity retrieval with NumPy (no vector database required for small datasets)
- Answers via Gemini API (`google-genai` SDK)
- Source citations and expandable supporting passages
- Retrieval-only preview **without an API key**

## How RAG works

```text
Upload document(s)
    ↓
Extract text + page numbers
    ↓
Chunk into overlapping passages
    ↓
Local sentence-transformer embeddings
    ↓
Compare question embedding with document embeddings
    ↓
Retrieve top-k relevant passages
    ↓
Gemini creates answer using only those passages
    ↓
Display response and source references
```

## Requirements

- Python 3.10+ recommended (a supported Python version for PyTorch, Streamlit and Google GenAI)
- Internet connection on first run to download the embedding model (~90 MB) and later for Gemini requests
- Gemini API key, only if you want generative answers. **Retrieval preview works without a key.**

## Windows setup (PowerShell)

From the extracted `documind-rag` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The terminal opens a local site (typically <http://localhost:8501>).

## Gemini setup

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and create an API key.
2. Paste the key into the **Gemini API key** password input in the sidebar, **or** copy `.env.example` to `.env` and replace the placeholder value.
3. Confirm your model is available in your project/region. The default is `gemini-flash-latest`, but available models and API pricing can change.

**Do not** post your API key publicly or commit `.env` to GitHub.

## Usage

1. Upload one or more PDFs or text files.
2. Wait until the passages are indexed.
3. Ask a question such as “What are the main findings in chapter 3?”
4. Read the answer and inspect **Sources retrieved**.

## Test

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## Limitations & next steps

- **No OCR:** image-only/scanned PDFs need OCR to extract text.
- **English-oriented embeddings:** multilingual retrieval requires a multilingual embedding model.
- **Small/medium collections:** uses in-memory NumPy similarity; no persistent vector database.
- **Citations are model-generated:** always inspect the retrieved passages to verify claims.
- **One-question retrieval:** chat interface saves messages but follow-up questions are not automatically rewritten using conversation history.
- **Data privacy:** extraction/embeddings happen locally; when an API key is used, selected snippets are sent to the Gemini API.
- **No persistent account storage:** indexing and chat are stored only during your Streamlit session.
- Strong future additions: ChromaDB, authentication, OCR, multilingual search, query rewriting, evaluation (precision@k), and deployment.

## Tech stack

Python · Streamlit · PyMuPDF · Sentence Transformers · NumPy · Gemini API

## License

MIT — see `LICENSE`.

## Try a demo question

Upload `sample_docs/rag_basics.txt` and ask **"How does RAG retrieve information?"** The app will show the most relevant chunks. Add your Gemini API key to generate a full grounded answer.
