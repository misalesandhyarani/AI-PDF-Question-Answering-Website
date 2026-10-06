
import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(
    page_title="AI PDF Question Answering",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 18px; color: #666; margin-bottom: 25px;}
.answer-box {padding: 18px; border-radius: 12px; border: 1px solid #ddd; background: #fafafa;}
.source-box {padding: 10px; border-left: 4px solid #4F46E5; background: #f5f5ff; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📚 AI PDF Question Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions. The system retrieves relevant content and uses an LLM to generate an answer.</div>',
    unsafe_allow_html=True
)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text or '')
    return text.strip()

def extract_pdf_text(uploaded_file):
    try:
        from pypdf import PdfReader
    except ImportError:
        st.error("pypdf is missing. Run: pip install pypdf")
        return "", []
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append(clean_text(page.extract_text() or ""))
    return "\n".join(pages), pages

def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

def retrieve(question, chunks, top_k=5):
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    q_vector = vectorizer.transform([question])
    scores = cosine_similarity(q_vector, matrix).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i]), i + 1) for i in indices if scores[i] > 0]

def build_prompt(question, retrieved):
    context = "\n\n".join(
        f"[Source {i} | relevance {score:.3f}]\n{chunk}"
        for chunk, score, i in retrieved
    )
    return f"""You are an AI PDF assistant.
Answer the user's question using ONLY the provided PDF context.
If the answer is not present in the context, say:
"I couldn't find the answer in the uploaded PDF."
Do not invent facts. Give a clear, concise answer.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def call_gemini(prompt, api_key, model):
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def call_groq(prompt, api_key, model):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer questions from supplied PDF context only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def call_ollama(prompt, model, base_url):
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]

with st.sidebar:
    st.header("⚙️ AI Settings")
    provider = st.selectbox("LLM Provider", ["Gemini", "Groq", "Ollama"])
    if provider == "Gemini":
        api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        model = st.text_input("Gemini Model", value="gemini-2.0-flash")
    elif provider == "Groq":
        api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        model = st.text_input("Groq Model", value="llama-3.1-8b-instant")
    else:
        api_key = ""
        model = st.text_input("Ollama Model", value="llama3.2")
        base_url = st.text_input("Ollama URL", value="http://localhost:11434")

    top_k = st.slider("Retrieved chunks", 2, 8, 5)
    st.caption("Tip: Ollama runs locally and does not require an API key.")

uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    text, pages = extract_pdf_text(uploaded_file)
    chunks = chunk_text(text)

    if not chunks:
        st.error("No readable text was found. If this is a scanned PDF, OCR is required.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", len(pages))
    col2.metric("Text Characters", len(text))
    col3.metric("Chunks", len(chunks))

    st.success(f"PDF ready: {uploaded_file.name}")

    st.subheader("💬 Ask a question")
    question = st.chat_input("Example: What are the main objectives discussed in this PDF?")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retrieved = retrieve(question, chunks, top_k)

        with st.chat_message("assistant"):
            if not retrieved:
                answer = "I couldn't find relevant information in the uploaded PDF."
                st.warning(answer)
            else:
                prompt = build_prompt(question, retrieved)
                try:
                    with st.spinner("Thinking..."):
                        if provider == "Gemini":
                            if not api_key:
                                raise ValueError("Enter your Gemini API key in the sidebar.")
                            answer = call_gemini(prompt, api_key, model)
                        elif provider == "Groq":
                            if not api_key:
                                raise ValueError("Enter your Groq API key in the sidebar.")
                            answer = call_groq(prompt, api_key, model)
                        else:
                            answer = call_ollama(prompt, model, base_url)

                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    with st.expander("🔎 Retrieved PDF context"):
                        for chunk, score, index in retrieved:
                            st.markdown(
                                f'<div class="source-box"><b>Chunk {index}</b> — relevance {score:.3f}<br>{chunk}</div>',
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    answer = f"Error: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF to start.")
    st.markdown("""
### ✨ Features
- 📄 Upload PDF documents
- 🔍 TF-IDF based relevant-text retrieval
- 🤖 Gemini, Groq, or local Ollama LLM support
- 💬 Chat-style question answering
- 🔎 Shows retrieved PDF context
- 🧩 Simple Streamlit interface
- 🔐 API keys can be stored in `.env`
""")
