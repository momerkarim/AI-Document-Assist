# 📚 AI Document Assistant

A simple Streamlit-based **AI Document Assistant / RAG application** for a Generative AI course or hackathon.

It supports:

- PDF upload
- DOCX upload
- TXT upload
- Markdown (MD) upload
- Google Drive file/folder links
- Text extraction
- Text chunking with overlap
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid semantic + keyword search
- Groq-powered question answering
- Retrieved source display
- Streamlit session-state caching so embeddings are not recreated for every question

## 1. Project files

The project intentionally uses only three application files:

```text
document-assistant/
├── app.py
├── requirements.txt
└── readme.md
```

For local development, you will also create this secret file:

```text
.streamlit/
└── secrets.toml
```

Do **not** commit `secrets.toml` to GitHub.

---

## 2. How the application works

The application follows a simple RAG pipeline:

```text
PDF / DOCX / TXT / MD
          │
          ▼
   Text Extraction
          │
          ▼
   Text Chunking
   (800 chars / 150 overlap)
          │
          ▼
 Sentence Transformer
      Embeddings
          │
          ▼
      FAISS Index
          │
          │
User Question ───────► Question Embedding
          │
          ▼
   Semantic Search
          │
          ├────────► Keyword Search
          │
          ▼
     Hybrid Ranking
          │
          ▼
 Relevant Document Chunks
          │
          ▼
        Groq LLM
          │
          ▼
       Answer + Sources
```

The same pipeline is used for both local uploads and Google Drive files.

---

## 3. Install the application

Make sure Python 3.10+ is installed.

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Configure the Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

The key is deliberately **not** stored in `app.py`.

You can also set `GROQ_API_KEY` as an environment variable.

For Streamlit Community Cloud, add `GROQ_API_KEY` under the application's **Secrets** settings.

---

## 5. Run the application

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

---

## 6. Supported documents

### PDF

PDF text is extracted page by page.

The page number is preserved in the metadata, so retrieved sources can show:

```text
Filename: report.pdf
Page: 4
```

### DOCX

Text is extracted from Word paragraphs.

DOCX files do not have reliable PDF-style page boundaries using the simple `python-docx` approach, so the page value is shown as `N/A`.

### TXT

The complete text file is extracted.

### MD

Markdown is read as text. Markdown syntax is preserved so the original document structure remains available to the LLM.

### Google Drive

Paste a public/shared Google Drive file or folder link.

Supported Drive files:

- PDF
- DOCX
- TXT
- MD

The application downloads the supported files and sends them through the **same extraction → chunking → embedding → FAISS → hybrid search** pipeline.

For a Drive folder, the files need to be accessible to the link/download mechanism. Private files requiring an authenticated Google account may need a more advanced Google Drive API implementation.

---

## 7. How chunking works

The MVP uses:

```text
Chunk size: 800 characters
Overlap:    150 characters
```

For example:

```text
Chunk 1: characters 0–800
Chunk 2: characters 650–1450
Chunk 3: characters 1300–2100
```

The overlap helps preserve context when an important sentence crosses a chunk boundary.

Every chunk keeps:

```python
{
    "text": "...",
    "filename": "...",
    "page": 4,
    "source": "Local Upload"
}
```

This allows the application to show the source after answering a question.

---

## 8. Embeddings

The application uses:

```text
sentence-transformers
all-MiniLM-L6-v2
```

The model converts every document chunk into a numerical vector.

The embeddings are stored in:

```python
st.session_state.embeddings
```

The FAISS index is stored in:

```python
st.session_state.index
```

Therefore, asking another question does **not** recreate document embeddings.

Only the user's new question is embedded during a search.

The embedding model itself is loaded with:

```python
@st.cache_resource
```

so Streamlit can reuse the model during the application session.

---

## 9. FAISS semantic search

FAISS stores the document vectors and performs similarity search.

The application uses:

```python
faiss.IndexFlatIP
```

with normalized embeddings.

This makes inner-product similarity equivalent to cosine similarity for the normalized vectors.

---

## 10. Keyword search

Semantic search is useful, but keyword matching can help when a question contains an exact name, number, product, acronym, or technical term.

The application:

1. Removes common stop words.
2. Extracts important words from the question.
3. Counts how many appear in each candidate chunk.
4. Produces a simple keyword score.

Example:

```text
Question:
What is the project budget for Phase 2?

Important words:
project
budget
phase
2
```

---

## 11. Hybrid search

The application combines both scores:

```text
Hybrid Score =
    75% Semantic Score
  + 25% Keyword Score
```

The top hybrid results are sent to Groq as context.

This is intentionally simple and easy to explain for a course/hackathon MVP.

---

## 12. Groq question answering

The retrieved chunks are included in the prompt sent to Groq.

The system instruction tells the model:

```text
Answer ONLY from the provided CONTEXT.
Do not use outside knowledge.
If the answer is not available in the context, say:
"I could not find that information in the provided documents."
```

This helps reduce hallucination and keeps the application focused on document-based answers.

---

## 13. Retrieved sources

After every successful answer, the application displays the chunks used as context.

Each source shows:

```text
Filename
Page number when available
Source type
Semantic score
Keyword score
Hybrid score
Retrieved text chunk
```

This provides basic RAG transparency and makes it easier to explain why the answer was generated.

---

## 14. Important optimization

Documents are processed only when they are new.

The application calculates a SHA-256 hash for every file.

If the same file is uploaded again during the session:

```text
File hash already exists
        ↓
Skip extraction
        ↓
Skip chunking
        ↓
Skip embedding
```

Questions reuse the existing:

```text
chunks
embeddings
FAISS index
```

This is much more efficient than embedding all documents every time the user asks a question.

---

## 15. Simple architecture

The code is intentionally kept in one file:

```text
app.py
```

The major functions are:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()

extract_document()

chunk_text()

build_vector_store()

keyword_score()

hybrid_search()

download_drive_files()

add_documents()
```

This makes the application easy to explain in a 6-week Generative AI course or hackathon.

---

## 16. Suggested hackathon explanation

You can describe the project in five simple layers:

### Layer 1 — Ingestion

"Users can upload documents or provide a Google Drive link."

### Layer 2 — Processing

"We extract text and split it into overlapping chunks while preserving source metadata."

### Layer 3 — Retrieval

"We convert chunks into embeddings and store them in FAISS. We combine semantic similarity with keyword matching."

### Layer 4 — Generation

"We send the user's question plus the most relevant chunks to Groq."

### Layer 5 — Grounding

"The LLM is instructed to answer only from retrieved context, and the application displays the source chunks after every answer."

---

## 17. Future improvements

This MVP can later be extended with:

- Persistent FAISS index on disk
- SQLite/PostgreSQL metadata store
- Better document fingerprinting
- PDF table extraction
- OCR for scanned PDFs
- Better Markdown cleaning
- DOCX page-number extraction
- Reranking models
- BM25 keyword search
- Multi-query retrieval
- Conversation memory
- Streaming Groq responses
- Google Drive OAuth
- User authentication
- Multiple vector databases
- Evaluation metrics such as Recall@K and MRR
- Document deletion/update management
- Cloud deployment

The current version deliberately avoids these complexities so the core RAG architecture remains easy to understand.
