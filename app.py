
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Please check your .env file.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
st.title("RAG based Q&A")
st.header("RAG Powered Q&A")
uploaded_pdf = st.file_uploader("Upload a file(PDF recommended)",type=["pdf"])
if uploaded_pdf is not None:
    st.write("PDF upload successful")
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.read())
    loader = PyPDFLoader(temp_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding,
    persist_directory="./chromadb"  # Ensures new storage path
    )
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={"k":10}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0,
        max_tokens=None,
        timeout=None,
        google_api_key = GOOGLE_API_KEY
    )
    query = st.chat_input("Ask Something: ")
    prompt = query
    system_prompt = (
        """ You are an expert tutor and orator, with an assistant job for question-answering task.
        Use the following retrieved context to answer the question correctly. If you don't know find the answer form the context
        say that you don't know the answer. Do not make the answer comprehensive keep it precise, if asked then only make it comprehensive
        \n\n
        {context}"""
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    if query:
        question_answer_chain = create_stuff_documents_chain(llm,prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])
    
    os.remove(temp_path)
