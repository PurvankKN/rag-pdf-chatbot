# RAG Document Q&A ChatBot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask natural-language questions about its contents. Built with **LangChain**, **Google Gemini**, **ChromaDB**, and **Streamlit**.

---
## Interface
<img width="1902" height="822" alt="Image" src="https://github.com/user-attachments/assets/22e2c32c-5138-45bb-bbee-cf322f5124b7" />

---
## Features

- **PDF Upload & Parsing** — Upload any PDF directly through the browser; text is extracted using `PyPDFLoader`.
- **Automatic Chunking** — Documents are split into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter` for better retrieval accuracy.
- **MMR Retrieval with ChromaDB** — Chunks are embedded and stored in a session-only Chroma vector store. Maximum Marginal Relevance (MMR) selects four relevant, diverse chunks per query to improve context coverage and reduce redundancy.
- **Gemini-Powered Answers** — Uses Google's `gemini-3.1-flash-lite` model to generate grounded answers strictly from the retrieved context.
- **Context-Only Responses** — The system prompt explicitly instructs the model to avoid hallucination: if the answer isn't in the document, it says so instead of guessing.
- **Conversational UI** — Full chat interface built with Streamlit's `st.chat_message` and `st.chat_input`, with persistent message history for the session.
- **Small-Talk Handling** — Greetings, thanks, and goodbyes are detected and answered directly, without triggering unnecessary retrieval calls.
- **Transparency Controls** — A sidebar toggle lets users inspect the exact chunks retrieved for each answer (with page numbers), useful for debugging or trust-building.
- **Document Insights Sidebar** — Displays page count, chunk count, embedding model, and vector store info at a glance.
- **Session Management** — Users can clear the chat or upload a new document at any time, resetting state cleanly.
- **Input Validation** — Detects and rejects PDFs with no extractable text (e.g., scanned/image-only documents) with a clear error message.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI / App Framework | Streamlit |
| LLM | Google Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) |
| Embeddings | Google Generative AI Embeddings (`gemini-embedding-2-preview`) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| PDF Parsing | PyPDFLoader (langchain-community) |

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/urshashi09/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Copy `.env.example` to `.env`, then add your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## How It Works

1. **Upload** — User uploads a PDF; it's temporarily saved to disk for processing.
2. **Extract & Validate** — Text is extracted page-by-page; if no readable text is found, the app raises a clear error (helps catch scanned PDFs).
3. **Chunk** — The document is split into overlapping 1000-character chunks to preserve context across boundaries.
4. **Embed & Store** — Each chunk is embedded via Google's embedding model and stored in Chroma.
5. **Query** — On each user question, MMR considers 10 candidate chunks and retrieves 4 relevant, diverse chunks.
6. **Generate** — Retrieved chunks are injected into a structured prompt that instructs Gemini to answer only from context, preserve exact figures/dates/names, and flag missing information.
7. **Respond** — The answer is displayed in the chat, with an optional view of the underlying retrieved chunks for transparency.

---

## Production Considerations

This project is designed as a lightweight, single-user RAG application. The following enhancements would be considered for a production deployment:

### Conversational Retrieval
The chat history is stored for the UI, but retrieval uses only the latest user query. A production system could rewrite follow-up questions into standalone queries before retrieval to improve multi-turn conversations.

### Managed Vector Database
The application uses a session-only Chroma vector store, which is suitable for local development. In production, a managed vector database such as Pinecone, PgVector, or a hosted Chroma instance would provide better scalability and persistence.

---

