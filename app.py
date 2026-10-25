import tempfile

from dotenv import load_dotenv
load_dotenv()  

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import streamlit as st
from time import sleep
import os


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):
    loader= PyPDFLoader(path)
    docs= loader.load()
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError(
            "The PDF contains no readable text. It may be a scanned document."
        )
    page_count = len(docs)



    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
        )
    

    docs= text_splitter.split_documents(docs)
    chunk_count = len(docs)

    embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True
    st.session_state.page_count = page_count
    st.session_state.chunk_count = chunk_count

st.subheader("📃 RAG Document Q&A ChatBot 🤖")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


#upload pdf file
if not st.session_state.document_uploaded:
    file= st.file_uploader(label="Upload your PDF file", type="pdf")
    if file:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        with st.spinner("Processing the document..."):

            try:

                document_process(temp_path)

            except Exception as e:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                st.error(f"Unable to process PDF.\n{e}")
                st.stop()

        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.markdown ("Document processed successfully!")

        sleep(2)
        st.rerun()


if st.session_state.document_uploaded:

    st.sidebar.subheader("📄 Document Info")

    st.sidebar.write(
        f"Pages: {st.session_state.page_count}"
    )

    st.sidebar.write(
        f"Chunks: {st.session_state.chunk_count}"
    )

    st.sidebar.write(
        "Embedding: Gemini Embedding"
    )

    st.sidebar.write(
        "Vector Store: Chroma"
    )

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("📄 Upload New Document"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.document_uploaded = False
        st.rerun()
    show_chunks = st.sidebar.checkbox(
    "Show Retrieved Chunks"
)


if st.session_state.document_uploaded and st.session_state.vector_db:


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    query = st.chat_input("Ask a question about the document")

    if query:

        
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        greetings = {
            "hi", "hello", "hey", "hii", "heyy",
            "good morning", "good afternoon", "good evening"
        }

        thanks = {
            "thanks", "thank you", "thankyou"
        }

        bye = {
            "bye", "goodbye", "see you"
        }

        user_query = query.lower().strip()

        if user_query in greetings:
            answer = "Hello! 👋 I'm your PDF RAG chatbot. Ask me anything about the uploaded document."

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

        elif user_query in thanks:
            answer = "You're welcome! 😊"
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)

        elif user_query in bye:
            answer = "Goodbye! 👋 Have a great day!"
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)

        else:

            retriever = st.session_state.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10
                }
            )

            documents = retriever.invoke(query)
            if show_chunks:

                with st.sidebar.expander("Retrieved Chunks"):

                    for i, doc in enumerate(documents, start=1):
                        page = doc.metadata.get("page", 0) + 1
                        st.write(f"### Chunk {i}")
                        st.write(f"📄 Page: {page}")
                        st.code(doc.page_content)

                        st.divider()

            context = ""

            for doc in documents:
                context += doc.page_content + "\n\n"

            prompt = f"""
You are a helpful AI assistant that answers questions about an uploaded PDF.

Instructions:
- Answer using only the provided context.
- If the context does not contain the answer, reply:
  "I couldn't find that information in the uploaded document."
- Do not make up or assume information.
- If the context partially answers the question, provide the available information and clearly mention what is missing.
- Keep answers clear, concise, and well-structured.
- If the question requires calculations, use values from the context and show the calculation before the final answer.
- If the user asks for a list, summary, or comparison, format the response with Markdown bullets or tables when appropriate.
- Preserve names, dates, and numbers exactly as they appear in the document.
- If the question is unrelated to the uploaded document, politely explain that you can only answer questions based on the uploaded document.

Context:
{context}

Question:
{query}

Answer:
"""

            with st.spinner("Thinking..."):
                try:

                    response = llm.invoke(prompt)

                except Exception as e:

                    st.error(f"Error: {e}")
                    st.stop()

            if isinstance(response.content, str):
                answer = response.content
            else:
                answer = response.content[0]["text"]

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

