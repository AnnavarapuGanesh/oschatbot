
import os
import re
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "os-rag-chatbot"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

CHUNK_SIZE_CHARS = 1600
OVERLAP_CHARS = 600
BATCH_SIZE = 32

PDF_FILE = BASE_DIR / "Abraham-Silberschatz-Operating-System-Concepts-10th-2018.pdf"
TXT_FILE = BASE_DIR / "os_clean_data.txt"
BM25_FILE = BASE_DIR / "bm25_params.json"

TARGET_TOPICS = [
    "process management",
    "memory management",
    "deadlock",
    "scheduling",
    "file system"
]


# ==========================================
# 2. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'Recaptcha requires verification.', '', text)
    text = re.sub(r'protected by \*\*reCAPTCHA\*\*', '', text)

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) > 20
        or line.strip().endswith(':')
    ]

    return '\n'.join(lines)


# ==========================================
# 3. CHUNKING
# ==========================================

def get_chunks(text):

    chunks = []

    step = CHUNK_SIZE_CHARS - OVERLAP_CHARS

    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    for start in range(0, len(text), step):

        chunk = text[start:start + CHUNK_SIZE_CHARS].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# ==========================================
# 4. EXTRACT TXT
# ==========================================

def extract_txt():

    print("Extracting TXT data...")

    if not TXT_FILE.exists():
        print(f"Missing file: {TXT_FILE}")
        return ""

    try:
        with open(TXT_FILE, "r", encoding="utf-8") as file:
            text = file.read()

        print(f"TXT characters: {len(text)}")
        return text

    except Exception as e:
        print(f"TXT extraction error: {e}")
        return ""


# ==========================================
# 5. EXTRACT PDF
# ==========================================

def extract_pdf():

    print("Extracting PDF data...")

    if not PDF_FILE.exists():
        print(f"Missing file: {PDF_FILE}")
        return ""

    try:

        reader = PdfReader(str(PDF_FILE))

        extracted_text = ""

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if not text:
                continue

            text_lower = text.lower()

            if any(topic in text_lower for topic in TARGET_TOPICS):

                extracted_text += (
                    f"\n[PDF Page {page_number + 1}]\n{text}\n"
                )

        print(f"PDF characters: {len(extracted_text)}")

        return extracted_text

    except Exception as e:

        print(f"PDF extraction error: {e}")
        return ""


# ==========================================
# 6. CREATE CHUNKS
# ==========================================

def prepare_documents():

    print("\nStep 1: Data preprocessing")

    txt_content = clean_text(extract_txt())
    pdf_content = clean_text(extract_pdf())

    txt_chunks = get_chunks(txt_content)
    pdf_chunks = get_chunks(pdf_content)

    documents = []

    for chunk in txt_chunks:
        documents.append({
            "text": chunk,
            "source": "scraped"
        })

    for chunk in pdf_chunks:
        documents.append({
            "text": chunk,
            "source": "book"
        })

    print(f"TXT chunks: {len(txt_chunks)}")
    print(f"PDF chunks: {len(pdf_chunks)}")
    print(f"Total chunks: {len(documents)}")

    return documents


# ==========================================
# 7. INITIALIZE PINECONE
# ==========================================

def initialize_pinecone():

    print("\nStep 2: Initializing Pinecone")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [
        item.name for item in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print("Creating new Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    else:

        print("Existing Pinecone index found.")

        description = pc.describe_index(PINECONE_INDEX_NAME)

        if description.dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                "Pinecone dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"found {description.dimension}. "
                "Create a separate correctly configured index."
            )

    # Wait until index is ready
    for attempt in range(60):

        description = pc.describe_index(PINECONE_INDEX_NAME)

        if description.status["ready"]:
            break

        time.sleep(2)

    else:
        raise TimeoutError("Pinecone index did not become ready.")

    index = pc.Index(PINECONE_INDEX_NAME)

    print("Pinecone connected successfully.")

    return index


# ==========================================
# 8. BM25 ENCODER
# ==========================================

def initialize_bm25(documents):

    print("\nStep 3: Fitting BM25")

    corpus = [
        item["text"] for item in documents
    ]

    bm25 = BM25Encoder()
    bm25.fit(corpus)

    bm25.dump(str(BM25_FILE))

    print(f"BM25 saved: {BM25_FILE}")

    return bm25


# ==========================================
# 9. LOAD EMBEDDING MODEL
# ==========================================

def initialize_embedding_model():

    print("\nStep 4: Loading embedding model")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ==========================================
# 10. GENERATE STABLE VECTOR ID
# ==========================================

def generate_vector_id(source, text, chunk_number):

    raw = f"{source}:{chunk_number}:{text}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ==========================================
# 11. UPSERT VECTORS
# ==========================================

def upsert_documents(index, bm25, model, documents):

    print("\nStep 5: Generating embeddings and uploading")

    total = len(documents)

    for start in range(0, total, BATCH_SIZE):

        batch = documents[start:start + BATCH_SIZE]

        texts = [
            item["text"] for item in batch
        ]

        try:

            dense_embeddings = model.encode(
                texts,
                show_progress_bar=False
            ).tolist()

            sparse_embeddings = bm25.encode_documents(texts)

            vectors = []

            for j, (dense, sparse, item) in enumerate(
                zip(dense_embeddings, sparse_embeddings, batch)
            ):

                chunk_number = start + j

                vector_id = generate_vector_id(
                    item["source"],
                    item["text"],
                    chunk_number
                )

                vectors.append({
                    "id": vector_id,
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": {
                        "text": item["text"],
                        "source": item["source"]
                    }
                })

            index.upsert(vectors=vectors)

            print(
                f"Uploaded {min(start + BATCH_SIZE, total)}/{total}"
            )

        except Exception as e:

            print(
                f"Batch {start // BATCH_SIZE + 1} failed: {e}"
            )

            raise

    print("\nAll vectors uploaded successfully.")


# ==========================================
# 12. MAIN
# ==========================================

def main():

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing.")

    documents = prepare_documents()

    if not documents:
        raise ValueError("No documents available for ingestion.")

    index = initialize_pinecone()

    bm25 = initialize_bm25(documents)

    model = initialize_embedding_model()

    upsert_documents(
        index,
        bm25,
        model,
        documents
    )

    print("\n================================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("================================")

    print(f"Index: {PINECONE_INDEX_NAME}")
    print(f"Documents: {len(documents)}")
    print(f"BM25 file: {BM25_FILE}")


if __name__ == "__main__":
    main()
