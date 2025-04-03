import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import shutil

# ✅ Load API Key from Streamlit Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.title("RAG-Based Q&A")
st.header("RAG Powered Q&A")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a file (PDF recommended)", type=["pdf"])

if uploaded_pdf is not None:
    st.write("PDF uploaded successfully!")

    # Use temporary file instead of fixed path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_pdf.read())
        temp_path = temp_file.name

    # Load PDF and split into chunks
    loader = PyPDFLoader(temp_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # ✅ Ensure a clean ChromaDB storage for new documents
    chroma_dir = "./chromadb"
    shutil.rmtree(chroma_dir, ignore_errors=True)  # Clear existing database

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embedding,
        
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    # User Query Input
    query = st.chat_input("Ask Something: ")

    if query:
        system_prompt = (
            """You are an expert tutor and orator, responsible for answering questions accurately. 
            Use the retrieved context to answer the question. If you cannot find the answer from the context, 
            say that you don't know. Keep responses concise unless asked otherwise.
            \n\n{context}"""
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Create Retrieval-Augmented Generation (RAG) Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])

    # Clean up temporary file
    os.remove(temp_path)
