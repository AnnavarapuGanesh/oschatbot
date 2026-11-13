import os
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
# Load environment variables
load_dotenv()

def get_config(key, default=None):
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

GROQ_API_KEY = get_config("GROQ_API_KEY")
PINECONE_API_KEY = get_config("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_config(
    "PINECONE_INDEX_NAME", "os-rag-chatbot"
)

# Initialize Groq
LLM_MODEL = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# UI Setup
st.set_page_config(page_title="OS RAG Chatbot", page_icon="💻")
st.title("💻 Operating Systems Expert Chatbot")
st.markdown("Ask anything about Operating Systems.")

# -----------------------------
# LOAD RESOURCES (CACHED)
# -----------------------------
@st.cache_resource
def get_pinecone_index():
    if not PINECONE_API_KEY:
        return None
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def get_bm25():
    bm25 = BM25Encoder()
    try:
        bm25.load("bm25_params.json")
    except:
        st.warning("Run ingest.py first (bm25 missing)")
    return bm25

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

index = get_pinecone_index()
bm25 = get_bm25()
embedding_model = get_embedding_model()

# -----------------------------
# FUNCTIONS
# -----------------------------
def hybrid_score(dense, sparse, alpha=0.75):
    return (
        [v * alpha for v in dense],
        {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        },
    )

def web_search(query):
    try:
        results = DDGS().text(query, max_results=3)
        return "\n\n".join(
            [f"{r['body']} (Source: {r['href']})" for r in results]
        )
    except Exception as e:
        return f"Web search error: {e}"

def generate_answer(context, query):
    if not client:
        return "❌ GROQ_API_KEY not set"

    prompt = f"""
You are an Operating Systems expert.

Answer ONLY from context.
If not found, say: Not found in knowledge base.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fallback" not in st.session_state:
    st.session_state.fallback = None

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(s)

# -----------------------------
# FALLBACK SEARCH
# -----------------------------
if st.session_state.fallback:
    st.warning("No answer found. Search web?")
    if st.button("Yes"):
        web = web_search(st.session_state.fallback)
        answer = generate_answer(web, st.session_state.fallback)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": ["Web"]
        })
        st.session_state.fallback = None
        st.rerun()

# -----------------------------
# USER INPUT
# -----------------------------
if query := st.chat_input("Ask something..."):
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        st.error("Set API keys in .env")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        start = time.time()

        try:
            dense = embedding_model.encode([query])[0].tolist()
            sparse = bm25.encode_queries(query)

            hd, hs = hybrid_score(dense, sparse)

            res = index.query(
                vector=hd,
                sparse_vector=hs,
                top_k=5,
                include_metadata=True
            )

            contexts, sources = [], []

            for m in res["matches"]:
                contexts.append(m["metadata"]["text"])
                sources.append(f"{m['metadata']['source']} (Score: {m['score']:.2f})")

            if not contexts:
                st.session_state.fallback = query
                st.rerun()

            context_text = "\n\n".join(contexts)

            answer = generate_answer(context_text, query)

            if "Not found" in answer:
                st.session_state.fallback = query
                st.rerun()

            st.markdown(answer)

            with st.expander(f"Sources (⏱ {time.time()-start:.2f}s)"):
                for s in sources:
                    st.write(s)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error: {e}")
