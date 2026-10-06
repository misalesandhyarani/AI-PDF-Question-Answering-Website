# AI PDF Question Answering Website

A Streamlit-based RAG-style PDF question answering application.

## Tech Stack
- Python 3.11.9
- Streamlit
- Pandas / NumPy
- Scikit-learn TF-IDF retrieval
- Google Gemini or Groq API
- Ollama for local LLM
- python-dotenv
- pypdf for PDF text extraction

## Project Flow
PDF Upload -> Text Extraction -> Chunking -> TF-IDF Retrieval -> Relevant Context -> LLM -> Answer + Sources

## Run on Windows

1. Install Python 3.11.9.
2. Open Command Prompt in this folder.
3. Create environment:
   `python -m venv venv`
4. Activate:
   `venv\Scripts\activate`
5. Install:
   `pip install -r requirements.txt`
6. Copy `.env.example` to `.env`.
7. Add a Gemini or Groq API key, OR install Ollama and pull a model.
8. Run:
   `streamlit run app.py`

## Ollama option
Install Ollama, then:
`ollama pull llama3.2`

The app uses:
`http://localhost:11434`

## Important
Do not upload `.env` or expose your API key publicly.
Scanned/image-only PDFs need OCR before text can be retrieved.
