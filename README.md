# Operating Systems Expert Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to answer complex questions about Operating Systems. Built with Streamlit, Groq (Llama 3), and Pinecone for advanced hybrid search capabilities.

## 🌟 Features

- **Hybrid Search Retrieval**: Combines dense embeddings (Sentence Transformers) and sparse vectors (BM25) for highly accurate context retrieval.
- **Lightning Fast LLM**: Utilizes the Groq API (Llama-3.3-70b-versatile) for rapid, high-quality response generation.
- **Web Search Fallback**: Automatically falls back to DuckDuckGo web search if the local knowledge base lacks the required information.
- **Source Citations**: Transparently displays the sources and relevance scores for each generated answer.
- **Clean UI**: Interactive and responsive chat interface built with Streamlit.

## 🛠️ Tech Stack

- **Frontend/App Framework**: Python, Streamlit
- **LLM Inference**: Groq API
- **Vector Database**: Pinecone
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers), BM25
- **Data Processing**: PyPDF, NLTK

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/AnnavarapuGanesh/oschatbot.git
cd oschatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your API keys:
```ini
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=os-rag-chatbot
# Note: Ensure you have your Pinecone index setup with dimension 384 and metric dotproduct.
```

### 5. Ingest Knowledge Base (Optional if already ingested)
If you need to process your text and PDF data into the vector database:
```bash
python ingest.py
```

### 6. Run the Application
```bash
streamlit run app.py
```

## 🔒 Security Best Practices
- Never commit your `.env` file to version control.
- API keys are handled securely via environment variables.

## 🚀 Deployment (Upcoming)
This project is structured and ready to be deployed on platforms like **Render** or **Streamlit Community Cloud**. Deployment instructions will be added in future stages.

---
**Author**: [Annavarapu Ganesh](https://github.com/AnnavarapuGanesh)
