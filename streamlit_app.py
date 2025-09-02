# main.py
# Vidur Bot - Indian Legal AI Assistant
# Fully fixed for Streamlit Cloud deployment

# MUST BE TOP: Fix Chroma sqlite3 issue
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass

import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import requests
import urllib.parse
from gtts import gTTS
import feedparser
from datetime import datetime

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config("Vidur Bot - Legal Advisor", layout="centered")
st.title("Vidur Bot")
st.markdown("An AI-powered legal assistant for Indian laws.")

# -------------------------------
# Secrets & Keys
# -------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except:
    st.error("Set `GROQ_API_KEY` and `YOUTUBE_API_KEY` in secrets.")
    st.stop()

# -------------------------------
# Constants
# -------------------------------
PDF_FOLDER = "pdfs"
TEMP_DIR = "/tmp" if os.name != "nt" else "."
os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml"
}

LEGAL_NEWS_FEEDS = [
    "https://www.livelaw.in/rss",
    "https://indiankanoon.org/docsource/?docsource=Supreme%20Court",
    "https://www.lawcrossing.com/rss/india/",
    "https://www.barandbench.com/feed"
]

# -------------------------------
# LLM Setup
# -------------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

# -------------------------------
# Vector DB with RAG
# -------------------------------
@st.cache_resource
def load_vector_db():
    if not os.path.exists(PDF_FOLDER):
        st.warning("PDF folder not found. RAG disabled.")
        return None

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdfs:
        st.warning("No PDFs in 'pdfs/' folder. RAG disabled.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = []
        for file in pdfs:
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            documents.extend(loader.load())

        if not documents:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        db = Chroma.from_documents(texts, embedding=embeddings)
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.warning(f"Vector DB failed: {e}")
        return None

# -------------------------------
# YouTube Videos
# -------------------------------
def fetch_yt_videos(query):
    q = f"{query} Indian law site:youtube.com"
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"part=snippet&maxResults=2&q={urllib.parse.quote(q)}"
        f"&key={YOUTUBE_API_KEY}&type=video&regionCode=IN"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        return [
            f"[{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})"
            for item in resp.get("items", [])
        ]
    except:
        return []

# -------------------------------
# PDF Text Extractor
# -------------------------------
def extract_pdf_text(uploaded_file):
    try:
        path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        from pypdf import PdfReader
        reader = PdfReader(path)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])[:10000]
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""

# -------------------------------
# TTS
# -------------------------------
def speak(text, lang="en", file="resp.mp3"):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        path = os.path.join(TEMP_DIR, file)
        tts.save(path)
        return path
    except Exception as e:
        st.warning(f"Voice error: {e}")
        return None

# -------------------------------
# News Fetcher (Cached)
# -------------------------------
@st.cache_data(ttl=600)
def fetch_news(max_items=5):
    items = []
    seen = set()
    for feed_url in LEGAL_NEWS_FEEDS:
        if len(items) >= max_items:
            break
        try:
            feed = feedparser.parse(feed_url.strip())
            for entry in feed.entries:
                if len(items) >= max_items or not entry.title:
                    break
                title = entry.title.strip()
                if title in seen:
                    continue
                seen.add(title)
                date = datetime(*entry.published_parsed[:6]).strftime("%b %d") if hasattr(entry, "published_parsed") else "Recent"
                items.append({
                    "title": title,
                    "summary": entry.summary[:150] + "..." if len(entry.summary) > 150 else entry.summary,
                    "link": entry.link,
                    "date": date
                })
        except:
            continue
    return items[:max_items]

# -------------------------------
# QA Chain with Fallback
# -------------------------------
def get_qa_chain(lang="English"):
    llm = get_llm()
    retriever = load_vector_db()

    prompt_template = f"""You are Vidur Bot, a legal advisor in Indian law.
Answer accurately using the context.
Mention laws: IPC, CrPC, Constitution, etc.
Respond in {lang}.

Context:
{{context}}

Question:
{{question}}

Vidur Bot:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    if retriever is None:
        # Fallback without RAG
        prompt = PromptTemplate(
            template=f"Respond in {lang}.\nQuestion: {{question}}\nVidur Bot:",
            input_variables=["question"]
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.dummy_retriever,
            chain_type_kwargs={"prompt": prompt},
            input_key="question"
        )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Dummy retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, run_manager):
        return [Document(page_content="")]
if "dummy_retriever" not in st.session_state:
    st.session_state.dummy_retriever = DummyRetriever()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Upload PDF")
uploaded = st.sidebar.file_uploader("Upload legal document", type="pdf")
pdf_text = extract_pdf_text(uploaded) if uploaded else ""

st.sidebar.header("Language")
lang = st.sidebar.selectbox("Response language", options=SUPPORTED_LANGUAGES.keys(), index=0)

st.sidebar.header("Voice")
voice = st.sidebar.checkbox("Enable voice", True)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["💬 Chat", "📰 News"])

# -------------------------------
# Chat Tab
# -------------------------------
with tab1:
    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("audio") and voice:
                st.audio(msg["audio"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Searching laws..."):
            try:
                chain = get_qa_chain(lang)
                inp = f"Document: {pdf_text}\n\nQuestion: {prompt}" if pdf_text else prompt
                response = chain.run(inp)
            except Exception as e:
                response = "I'm unable to process this right now."

            videos = fetch_yt_videos(prompt)
            if videos:
                response += "\n\n### Learn More:\n" + "\n".join(videos)

            audio = speak(response, SUPPORTED_LANGUAGES[lang], f"resp_{len(st.session_state.history)}.mp3") if voice else None

            st.session_state.history.append({
                "role": "assistant",
                "content": response,
                "audio": audio
            })
            with st.chat_message("assistant"):
                st.write(response)
                if voice and audio:
                    st.audio(audio)

# -------------------------------
# News Tab
# -------------------------------
with tab2:
    st.subheader("Latest Legal News in India")
    with st.spinner("Fetching..."):
        news = fetch_news()
    if not news:
        st.info("Could not fetch news.")
    else:
        for item in news:
            with st.expander(f"📘 {item['title']} ({item['date']})"):
                st.write(item["summary"])
                st.markdown(f"[Read more]({item['link']})")
