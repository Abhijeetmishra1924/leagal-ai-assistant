# main.py
# Vidur Bot - Indian Legal AI Assistant (with Voice & News)
# Fully fixed for Streamlit Cloud

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
st.set_page_config(page_title="Vidur Bot - Legal Advisor", layout="centered")
st.title("Vidur Bot")
st.markdown("An AI-powered legal assistant for Indian laws.")

# -------------------------------
# Secrets & Keys
# -------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except Exception as e:
    st.error("Set `GROQ_API_KEY` and `YOUTUBE_API_KEY` in Streamlit secrets.")
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
        st.warning("PDF folder 'pdfs/' not found. RAG disabled.")
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
            docs = loader.load()
            documents.extend(docs)

        if not documents:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        db = Chroma.from_documents(texts, embedding=embeddings)
        return db.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.warning(f"Failed to load vector DB: {str(e)}")
        return None

# -------------------------------
# YouTube Videos
# -------------------------------
def fetch_youtube_videos(query):
    q = f"{query} Indian law explained site:youtube.com"
    encoded_q = urllib.parse.quote(q)
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"part=snippet&maxResults=2&q={encoded_q}"
        f"&key={YOUTUBE_API_KEY}&type=video&regionCode=IN&relevanceLanguage=en"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        return [
            f"[{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})"
            for item in resp.get("items", [])
        ]
    except Exception as e:
        st.warning(f"YouTube fetch failed: {e}")
        return []

# -------------------------------
# Extract Text from Uploaded PDF
# -------------------------------
def extract_pdf_text(uploaded_file):
    try:
        path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text[:10000]
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# -------------------------------
# Text-to-Speech
# -------------------------------
def text_to_speech(text, lang="en", filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filepath = os.path.join(TEMP_DIR, filename)
        tts.save(filepath)
        return filepath
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")
        return None

# -------------------------------
# Fetch Legal News (Cached)
# -------------------------------
@st.cache_data(ttl=600)  # 10 minutes
def fetch_legal_news(max_items=5):
    news_items = []
    seen_titles = set()

    for feed_url in LEGAL_NEWS_FEEDS:
        if len(news_items) >= max_items:
            break
        try:
            feed = feedparser.parse(feed_url.strip())
            for entry in feed.entries:
                if len(news_items) >= max_items:
                    break
                title = entry.get("title", "No title").strip()
                link = entry.get("link", "#")
                summary = entry.get("summary", "").replace("\n", " ")[:200] + "..."

                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                try:
                    pub_date = datetime(*entry.published_parsed[:6]).strftime("%b %d, %Y")
                except:
                    pub_date = "Recent"

                source = urllib.parse.urlparse(link).netloc

                news_items.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "date": pub_date,
                    "source": source
                })
        except Exception as e:
            st.warning(f"Feed failed: {feed_url} → {str(e)}")
            continue

    return news_items[:max_items]

# -------------------------------
# QA Chain with Fallback
# -------------------------------
def get_qa_chain(language="English"):
    llm = get_llm()
    retriever = load_vector_db()

    instruction = f"Respond in {language}."
    if language != "English":
        instruction += " Use local context if needed."

    template = f"""You are Vidur Bot, a knowledgeable legal advisor in Indian law.
Answer accurately using the provided context.
Mention relevant laws: IPC, BNS, CrPC, Constitution, etc.
Suggest actionable steps and rights.
Keep responses clear and concise.

{instruction}

Context:
{{context}}

Question:
{{question}}

Vidur Bot:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Fallback if no retriever
    if retriever is None:
        fallback_prompt = PromptTemplate(
            template=f"Respond in {language}.\n\nQuestion: {{question}}\n\nVidur Bot:",
            input_variables=["question"]
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.dummy_retriever,
            chain_type_kwargs={"prompt": fallback_prompt},
            input_key="question"
        )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )


# Dummy retriever for fallback
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="")]

if "dummy_retriever" not in st.session_state:
    st.session_state.dummy_retriever = DummyRetriever()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("📄 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a legal document", type="pdf", label_visibility="collapsed")
pdf_text = extract_pdf_text(uploaded_pdf) if uploaded_pdf else ""

st.sidebar.header("🗣️ Language")
selected_lang = st.sidebar.selectbox(
    "Response language",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0
)

st.sidebar.header("🔊 Voice Output")
enable_voice = st.sidebar.checkbox("Enable voice", value=True)

# -------------------------------
# Tabs: Chat & News
# -------------------------------
tab1, tab2 = st.tabs(["💬 Chat", "📰 Latest Legal News"])

# -------------------------------
# Tab 1: Chat
# -------------------------------
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("audio") and enable_voice:
                st.audio(msg["audio"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Searching laws and preparing response..."):
            try:
                qa_chain = get_qa_chain(selected_lang)
                input_query = f"Document Summary:\n{pdf_text}\n\nQuestion: {prompt}" if pdf_text else prompt

                # ✅ Use invoke() instead of run()
                result = qa_chain.invoke({"query": input_query})
                response = result["result"]

                if not response.strip():
                    response = "I could not find a relevant answer."

            except Exception as e:
                st.error(f"Error: {str(e)}")  # Optional: remove in prod
                response = "I'm currently unable to process your request."

            # Add YouTube links
            youtube_links = fetch_youtube_videos(prompt)
            if youtube_links:
                response += "\n\n### Learn More:\n" + "\n".join(youtube_links)

            # Generate voice
            audio_file = None
            if enable_voice:
                lang_code = SUPPORTED_LANGUAGES[selected_lang]
                audio_file = text_to_speech(response, language=lang_code, filename=f"resp_{len(st.session_state.chat_history)}.mp3")

            # Save and display
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "audio": audio_file
            })
            with st.chat_message("assistant"):
                st.write(response)
                if enable_voice and audio_file:
                    st.audio(audio_file)

# -------------------------------
# Tab 2: Legal News
# -------------------------------
with tab2:
    st.subheader("Latest Legal News in India")
    with st.spinner("Fetching updates..."):
        news_items = fetch_legal_news()

    if not news_items:
        st.info("Could not fetch news at this time. Check back later.")
    else:
        for item in news_items:
            with st.expander(f"📘 {item['title']} ({item['date']}) · {item['source']}"):
                st.write(item["summary"])
                st.markdown(f"[Read more]({item['link']})")
