import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from src.etl.cleaner import DocumentCleaner
from src.database.chunker import SemanticChunker
from src.database.vector_db import VectorDatabaseManager
from src.rag.retriever import RAGRetriever
from src.rag.llm_twin_generator import LLMTwinGenerator

st.set_page_config(
    page_title="LLM Twin - AI Personal Character",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    .hero-banner {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin-right: 8px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline State
@st.cache_resource
def init_pipeline():
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    vector_db = VectorDatabaseManager()
    vector_db.index_chunks(chunks)
    retriever = RAGRetriever(vector_db)
    generator = LLMTwinGenerator(retriever)
    return generator

generator = init_pipeline()

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🤖 LLM Twin — AI Personal Character Assistant</div>
    <div style="color: #94a3b8; font-size: 15px;">
        An intelligent AI twin trained on personal technical writing, grounded by <b>Qdrant Vector RAG</b>, and served via <b>Groq Llama 3</b>.
    </div>
    <div>
        <span class="badge">⚡ Groq Llama 3.3 70B</span>
        <span class="badge">🔍 Qdrant Vector Cloud</span>
        <span class="badge">🧠 QLoRA Fine-Tuned Adapter</span>
        <span class="badge">✅ LLM-as-a-Judge Gate Passed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
with st.sidebar:
    st.header("💡 Prompt Suggestions")
    if st.button("⚡ Production RAG with Qdrant"):
        st.session_state.prompt_input = "How do you build a production RAG pipeline with Qdrant?"
    if st.button("🧠 QLoRA Fine-Tuning Workflow"):
        st.session_state.prompt_input = "Explain your QLoRA Fine-Tuning workflow"
    if st.button("📊 LLM-as-a-judge Gate"):
        st.session_state.prompt_input = "How do you set up an LLM-as-a-judge gate?"
        
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
prompt = st.chat_input("💬 Ask LLM Twin anything...")
if "prompt_input" in st.session_state and st.session_state.prompt_input:
    prompt = st.session_state.prompt_input
    del st.session_state.prompt_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vector memory & generating response..."):
            res = generator.generate_response(prompt)
            answer = res["answer"]
            if res["sources"]:
                answer += "\n\n---\n**📚 Context Retrieved from Past Writing:**\n"
                for s in res["sources"]:
                    answer += f"- [{s['title']}]({s['source_url'] or '#'}) (Relevance: `{s['score']}`)\n"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
