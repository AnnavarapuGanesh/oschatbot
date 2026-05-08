import os
import re
import time
import uuid
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "os-rag-chatbot")

# --- Configuration ---
CHUNK_SIZE_CHARS = 1600  # approx 400 tokens
OVERLAP_CHARS = 600      # approx 150 tokens
EMBEDDING_DIMENSION = 384
TARGET_TOPICS = [
    "process management", "memory management", 
    "deadlock", "scheduling", "file system"
]

def clean_text(text):
    # Remove HTML, repeated nav, and unwanted formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove markdown links
    text = re.sub(r'Recaptcha requires verification.', '', text)
    text = re.sub(r'protected by \*\*reCAPTCHA\*\*', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # Remove short useless lines
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20 or line.strip().endswith(':')]
    return '\n'.join(lines)

def get_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_pdf_chapters(pdf_path):
    print("Extracting relevant chapters from PDF...")
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        # Process only pages that contain target topics to avoid ingesting everything
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_lower = text.lower()
                if any(topic in text_lower for topic in TARGET_TOPICS):
                    extracted_text += text + "\n"
        print(f"Extracted {len(extracted_text)} characters from relevant PDF pages.")
        return extracted_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_txt(txt_path):
    print("Extracting text from local file...")
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Extracted {len(text)} characters from text file.")
        return text
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def main():
    if not GEMINI_API_KEY or not PINECONE_API_KEY:
        print("Error: Missing API keys in .env")
        return
        
    print("Step 1: Data Preprocessing...")
    txt_content = extract_txt("os_clean_data.txt")
    pdf_content = extract_pdf_chapters("Abraham-Silberschatz-Operating-System-Concepts-10th-2018.pdf")
    
    txt_clean = clean_text(txt_content)
    pdf_clean = clean_text(pdf_content)
    
    txt_chunks = get_chunks(txt_clean, CHUNK_SIZE_CHARS, OVERLAP_CHARS)
    pdf_chunks = get_chunks(pdf_clean, CHUNK_SIZE_CHARS, OVERLAP_CHARS)
    
    all_chunks = []
    for chunk in txt_chunks:
        all_chunks.append({"text": chunk, "source": "scraped"})
    for chunk in pdf_chunks:
        all_chunks.append({"text": chunk, "source": "book"})
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Step 2: Prepare Vector DB
    print("Step 2: Initialize Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists and delete it (dimension mismatch)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Deleting existing Pinecone index '{PINECONE_INDEX_NAME}' due to dimension change...")
        pc.delete_index(PINECONE_INDEX_NAME)
        time.sleep(5)
        
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION, 
        metric="dotproduct", # Recommended for hybrid search sparse/dense scaling
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to be ready...")
    time.sleep(10) # wait for index to be ready
        
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Fit BM25 Encoder
    print("Fitting BM25 Encoder...")
    bm25 = BM25Encoder()
    corpus = [chunk["text"] for chunk in all_chunks]
    
    # Avoid fitting error if corpus is empty
    if not corpus:
        print("No valid text found to ingest. Exiting.")
        return
        
    bm25.fit(corpus)
    # Save bm25 values if needed for app.py
    bm25.dump('bm25_params.json')
    
    print("Step 3 & 4: Embeddings and Upsert...")
    print("Loading Hugging Face model: all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]
        
        try:
            # Dense embeddings locally
            dense_embeds = embedding_model.encode(texts).tolist()
            
            # Sparse embeddings
            sparse_embeds = bm25.encode_documents(texts)
            
            # Prepare vectors
            vectors = []
            for j, (dense, sparse, item) in enumerate(zip(dense_embeds, sparse_embeds, batch)):
                # Normalize dense vector for hybrid search
                # Dotproduct is used, meaning scores are a combination of dense and sparse weights
                vec = {
                    "id": str(uuid.uuid4()),
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": {"text": item["text"], "source": item["source"]}
                }
                vectors.append(vec)
                
            index.upsert(vectors=vectors)
            print(f"Upserted batch {i//batch_size + 1}/{len(all_chunks)//batch_size + 1}")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
