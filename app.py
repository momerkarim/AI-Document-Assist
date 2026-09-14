import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Document Assistant")
st.caption("Upload documents or load supported files from a Google Drive link, then ask questions about them.")


# -----------------------------
# Model and API client
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Small, fast model that works well for a simple RAG demo.
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    # Never hard-code the API key. Add GROQ_API_KEY to .streamlit/secrets.toml.
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                    "source": "Local Upload",
                }
            )

    return pages


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
            "source": "Local Upload",
        }
    ]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
            "source": "Local Upload",
        }
    ]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
            "source": "Local Upload",
        }
    ]


def extract_document(file_bytes, filename, source="Local Upload"):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        pages = extract_pdf(file_bytes, filename)
    elif extension == ".docx":
        pages = extract_docx(file_bytes, filename)
    elif extension == ".txt":
        pages = extract_txt(file_bytes, filename)
    elif extension == ".md":
        pages = extract_md(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    # Make the source correct for Google Drive files too.
    for item in pages:
        item["source"] = source

    return pages


# -----------------------------
# Text chunking
# -----------------------------
def chunk_text(pages, chunk_size=800, overlap=150):
    chunks = []

    for page_data in pages:
        text = re.sub(r"\s+", " ", page_data["text"]).strip()

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "filename": page_data["filename"],
                        "page": page_data["page"],
                        "source": page_data["source"],
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# -----------------------------
# Embeddings and FAISS
# -----------------------------
def build_vector_store(chunks):
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "what", "who", "when",
    "where", "why", "how", "does", "do", "did", "can", "could",
    "this", "that", "these", "those", "about", "from", "it", "its",
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_score(question, text):
    question_words = important_words(question)
    if not question_words:
        return 0.0

    text_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
    matches = sum(1 for word in question_words if word in text_words)

    return matches / len(question_words)


# -----------------------------
# Hybrid semantic + keyword search
# -----------------------------
def hybrid_search(question, chunks, index, top_k=5):
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Get a wider semantic candidate set, then combine scores.
    candidate_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, semantic_ids = index.search(question_embedding, candidate_k)

    candidates = []

    for score, chunk_id in zip(semantic_scores[0], semantic_ids[0]):
        if chunk_id < 0:
            continue

        chunk = chunks[int(chunk_id)]
        kw_score = keyword_score(question, chunk["text"])

        # Both scores are normalized to roughly 0-1.
        combined_score = (0.75 * float(score)) + (0.25 * kw_score)

        candidates.append(
            {
                "chunk": chunk,
                "semantic_score": float(score),
                "keyword_score": float(kw_score),
                "score": combined_score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Google Drive loading
# -----------------------------
def download_drive_files(link):
    temp_dir = tempfile.mkdtemp(prefix="document_assistant_")

    # gdown supports public/shared Google Drive file and folder URLs.
    downloaded = gdown.download_folder(
        url=link,
        output=temp_dir,
        quiet=True,
        use_cookies=False,
    )

    if not downloaded:
        # It may be a single file rather than a folder.
        output_file = os.path.join(temp_dir, "drive_file")
        downloaded_file = gdown.download(
            url=link,
            output=output_file,
            quiet=True,
            fuzzy=True,
        )

        downloaded = [downloaded_file] if downloaded_file else []

    supported = {".pdf", ".docx", ".txt", ".md"}
    files = []

    for path in downloaded or []:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in supported:
            files.append(path)

    return files


# -----------------------------
# Session-state document pipeline
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


def add_documents(file_items):
    """Process only new files. Existing file hashes are skipped."""
    new_pages = []
    added_names = []

    for filename, file_bytes, source in file_items:
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in st.session_state.documents:
            continue

        pages = extract_document(file_bytes, filename, source)

        st.session_state.documents[file_hash] = {
            "filename": filename,
            "source": source,
            "bytes": len(file_bytes),
            "pages": len(pages),
        }

        new_pages.extend(pages)
        added_names.append(filename)

    if not new_pages:
        return added_names, 0

    new_chunks = chunk_text(new_pages)

    # For a simple MVP, rebuild the FAISS index only when new documents arrive.
    # Questions reuse this existing index and embeddings.
    st.session_state.chunks.extend(new_chunks)
    st.session_state.index, st.session_state.embeddings = build_vector_store(
        st.session_state.chunks
    )

    return added_names, len(new_chunks)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Uploads", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one document.")
        else:
            file_items = [
                (file.name, file.getvalue(), "Local Upload")
                for file in uploaded_files
            ]

            try:
                names, count = add_documents(file_items)
                if names:
                    st.success(f"Added {len(names)} document(s) and {count} new chunks.")
                else:
                    st.info("These documents were already processed.")
            except Exception as e:
                st.error(f"Document processing failed: {e}")

    st.divider()

    st.subheader("Google Drive")
    drive_link = st.text_input(
        "Paste a public/shared Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )

    if st.button("Load from Google Drive", use_container_width=True):
        if not drive_link.strip():
            st.warning("Please paste a Google Drive link.")
        else:
            with st.spinner("Loading Google Drive files..."):
                try:
                    drive_paths = download_drive_files(drive_link)

                    if not drive_paths:
                        st.warning(
                            "No supported files were found. The Drive link must be "
                            "accessible and contain PDF, DOCX, TXT or MD files."
                        )
                    else:
                        file_items = [
                            (
                                path.name,
                                path.read_bytes(),
                                "Google Drive",
                            )
                            for path in drive_paths
                        ]

                        names, count = add_documents(file_items)
                        if names:
                            st.success(
                                f"Loaded {len(names)} Drive file(s) and {count} new chunks."
                            )
                        else:
                            st.info("The Drive files were already processed.")
                except Exception as e:
                    st.error(f"Google Drive loading failed: {e}")

    st.divider()

    st.metric("Documents", len(st.session_state.documents))
    st.metric("Created Chunks", len(st.session_state.chunks))

    if st.button("Clear All Documents", use_container_width=True):
        st.session_state.documents = {}
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.embeddings = None
        st.rerun()


# -----------------------------
# Document information
# -----------------------------
st.subheader("Document Information")

if st.session_state.documents:
    rows = []
    for item in st.session_state.documents.values():
        rows.append(
            {
                "Filename": item["filename"],
                "Source": item["source"],
                "Pages / Sections": item["pages"],
                "Size (bytes)": item["bytes"],
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info(
        f"Total chunks created: **{len(st.session_state.chunks)}**. "
        "Embeddings are created when documents are processed, not for every question."
    )
else:
    st.info("No documents loaded yet.")


# -----------------------------
# Question answering
# -----------------------------
st.subheader("Ask Your Documents")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the main objectives of this document?",
)

top_k = st.slider("Number of source chunks", 1, 8, 5)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.chunks:
        st.warning("Please upload or load at least one document first.")
    else:
        results = hybrid_search(
            question,
            st.session_state.chunks,
            st.session_state.index,
            top_k=top_k,
        )

        context_parts = []

        for i, result in enumerate(results, start=1):
            chunk = result["chunk"]
            page_text = (
                f", page {chunk['page']}"
                if chunk["page"] is not None
                else ""
            )
            context_parts.append(
                f"[Source {i}: {chunk['filename']}{page_text}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        client = get_groq_client()

        if not client:
            st.error(
                "GROQ_API_KEY is missing. Add it to "
                ".streamlit/secrets.toml."
            )
        else:
            system_prompt = """You are a document question-answering assistant.

Answer ONLY from the provided CONTEXT.
Do not use outside knowledge.
If the answer is not available in the context, say:
"I could not find that information in the provided documents."

Keep the answer clear and concise.
"""

            user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}
"""

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    max_tokens=800,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                answer = response.choices[0].message.content

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved Sources")

                for i, result in enumerate(results, start=1):
                    chunk = result["chunk"]
                    page = (
                        str(chunk["page"])
                        if chunk["page"] is not None
                        else "N/A"
                    )

                    with st.expander(
                        f"{i}. {chunk['filename']} | Page: {page} | "
                        f"Hybrid score: {result['score']:.3f}"
                    ):
                        st.caption(
                            f"Source: {chunk['source']} | "
                            f"Semantic: {result['semantic_score']:.3f} | "
                            f"Keyword: {result['keyword_score']:.3f}"
                        )
                        st.write(chunk["text"])

            except Exception as e:
                st.error(f"Groq request failed: {e}")
