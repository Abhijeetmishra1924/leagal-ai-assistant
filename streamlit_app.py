# main.py
# Vidur Bot - Indian Legal AI Assistant (with Voice & News)
# Deployment-ready for Streamlit Cloud
# #lovableinit

# MUST BE TOP OF FILE: Fix sqlite3 for Chroma
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass  # for local dev

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
import PyPDF2
from gtts import gTTS
import base64
from datetime import datetime
import feedparser


# -------------------------------
# Configuration
# -------------------------------

st.set_page_config(page_title="Vidur Bot - Legal Advisor", layout="centered")
st.title("Vidur Bot")
st.markdown("An AI-powered legal assistant for Indian laws.")


# Load API keys
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except Exception:
    st.error("API keys not found. Please set GROQ_API_KEY and YOUTUBE_API_KEY in secrets.")
    st.stop()


# Paths
PDF_FOLDER = "pdfs"
TEMP_DIR = "/tmp" if os.name != "nt" else "."
os.makedirs(TEMP_DIR, exist_ok=True)


# Supported languages for TTS and response
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


# Legal news RSS feeds (cleaned URLs)
LEGAL_NEWS_FEEDS = [
    "https://www.livelaw.in/rss",
    "https://indiankanoon.org/docsource/?docsource=Supreme%20Court",
    "https://www.lawcrossing.com/rss/india/",
    "https://www.barandbench.com/feed"
]


# -------------------------------
# Initialize LLM
# -------------------------------
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        st.stop()


# -------------------------------
# Load PDFs and Build Vector DB
# -------------------------------
@st.cache_resource
def load_vector_database():
    try:
        if not os.path.exists(PDF_FOLDER):
            st.warning(f"PDF folder '{PDF_FOLDER}' not found. RAG disabled.")
            return None

        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
        if not pdf_files:
            st.warning("No PDFs found in 'pdfs/' folder. RAG disabled.")
            return None

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = []

        for file in pdf_files:
            file_path = os.path.join(PDF_FOLDER, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.warning(f"Could not load {file}: {str(e)}")

        if not documents:
            st.warning("No documents loaded. RAG disabled.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(texts, embedding=embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.warning(f"Could not load vector database: {str(e)}")
        return None


# -------------------------------
# Fetch YouTube Videos
# -------------------------------
def fetch_youtube_videos(query):
    query = f"{query} Indian law explained site:youtube.com"
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"part=snippet&maxResults=2&q={encoded_query}"
        f"&key={YOUTUBE_API_KEY}&type=video&regionCode=IN&relevanceLanguage=en&safeSearch=strict"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        items = response.json().get("items", [])
        return [
            f"[{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})"
            for item in items
        ]
    except Exception as e:
        st.warning(f"YouTube fetch failed: {str(e)}")
        return []


# -------------------------------
# Extract Text from Uploaded PDF
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text[:10000]  # Limit context
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


# -------------------------------
# Text-to-Speech Function
# -------------------------------
def text_to_speech(text, language="en", filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        filepath = os.path.join(TEMP_DIR, filename)
        tts.save(filepath)
        return filepath
    except Exception as e:
        st.warning(f"Voice generation failed: {str(e)}")
        return None


# -------------------------------
# Fetch Latest Legal News (Cached)
# -------------------------------
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_legal_news(max_items=5):
    news_items = []
    seen_titles = set()

    for feed_url in LEGAL_NEWS_FEEDS:
        if len(news_items) >= max_items:
            break
        try:
            feed = feedparser.parse(feed_url.strip(), timeout=10)
            for entry in feed.entries:
                if len(news_items) >= max_items:
                    break
                title = entry.get("title", "No title").strip()
                link = entry.get("link", "#")
                summary = entry.get("summary", "").replace("\n", " ")[:200] + "..."

                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                # Parse date
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
            st.warning(f"Failed to fetch from {feed_url}: {str(e)}")
            continue

    return news_items[:max_items]


# -------------------------------
# Build QA Chain with Fallback
# -------------------------------
def get_qa_chain(language="English"):
    llm = get_llm()
    retriever = load_vector_database()

    instruction = f"Respond in {language}."
    if language != "English":
        instruction += f" Use local language context if needed."

    template = f"""You are Vidur Bot, a knowledgeable legal advisor in Indian law.
Answer accurately based on the provided context.
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

    if retriever is None:
        # Fallback: Simple LLM without RAG
        simple_template = f"""You are Vidur Bot, a legal advisor in Indian law.
Respond in {language}.
Answer to the best of your knowledge.

Question: {{question}}

Vidur Bot:"""
        simple_prompt = PromptTemplate(template=simple_template, input_variables=["question"])
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.get("dummy_retriever"),
            chain_type_kwargs={"prompt": simple_prompt},
            input_key="question"
        )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )


# Dummy retriever to avoid None
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
st.sidebar.header("Document Upload")
uploaded_pdf = st.sidebar.file_uploader("Upload a legal document (PDF)", type="pdf", label_visibility="collapsed")

pdf_text = ""
if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
    if pdf_text:
        st.sidebar.success("PDF text extracted and ready.")

st.sidebar.header("Language")
selected_language = st.sidebar.selectbox(
    "Choose response language",
    options=list(SUPPORTED_LANGUAGES.keys()),
    index=0
)

st.sidebar.header("Voice Output")
enable_voice = st.sidebar.checkbox("Enable voice response", value=True)


# -------------------------------
# Tabs: Chat | News
# -------------------------------
tab1, tab2 = st.tabs(["💬 Chat", "📰 Latest Legal News"])

# -------------------------------
# Tab 1: Chat
# -------------------------------
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("audio") and enable_voice:
                st.audio(message["audio"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Searching laws and preparing response..."):
            try:
                qa_chain = get_qa_chain(selected_language)
                input_query = f"Document Summary:\n{pdf_text}\n\nQuestion: {prompt}" if pdf_text else prompt

                response = qa_chain.run(input_query)
                if not response or response.strip() == "":
                    response = "I could not find a relevant answer based on the document or my knowledge."

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                response = "I'm currently unable to process your request. Please try again later."

            # Add YouTube links
            youtube_links = fetch_youtube_videos(prompt)
            if youtube_links:
                response += "\n\n### Learn More:\n" + "\n".join(youtube_links)

            # Generate voice if enabled
            audio_file = None
            if enable_voice:
                lang_code = SUPPORTED_LANGUAGES[selected_language]
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
