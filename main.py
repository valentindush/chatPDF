import os

import streamlit as st
from dotenv import  load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq.chat_models import ChatGroq

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vector_store(chunks):
    inference_api_key = os.environ["HF_API_KEY"]
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatGroq()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF Chat",
        page_icon=":books:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("PDF Chat")
    st.subheader("Ask questions about PDF documents")
    st.chat_input("Ask questions")

    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get the chunks
                text_chunks = get_text_chunks(raw_text)

                # create a vector store
                vector_store = get_vector_store(text_chunks)

                #conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()