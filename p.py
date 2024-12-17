import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (including Gemini API key)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBEDDING_URL = "https://api.gemini.com/v2/embeddings"  # Update this with the correct endpoint

# Custom class for embedding with Gemini API
class GeminiEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed(self, texts):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"texts": texts}
        response = requests.post(GEMINI_EMBEDDING_URL, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json().get("embeddings")
        else:
            raise Exception(f"Failed to fetch embeddings: {response.text}")

# Streamlit app title
st.title("RAG Application with Gemini API ⭐")

# Step 1: Define URLs to scrape
urls = ['https://www.uchicago.edu/', 'https://www.washington.edu/']

# Step 2: Load content from URLs
try:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    st.write("Data loaded successfully from the provided URLs.")
except Exception as e:
    st.error(f"Error loading data from URLs: {e}")
    st.stop()

# Step 3: Split data into manageable chunks
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    st.write(f"Data split into {len(docs)} chunks.")
except Exception as e:
    st.error(f"Error splitting data: {e}")
    st.stop()

# Step 4: Generate embeddings using Gemini API and create a vector store
try:
    gemini_embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)
    embeddings = gemini_embeddings.embed([doc.page_content for doc in docs])  # Embed all documents at once

    # Create vector store using Chroma
    vectorstore = Chroma.from_documents(documents=docs, embeddings=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    st.write("Vector store created successfully with Gemini API.")
except Exception as e:
    st.error(f"Error creating vector store: {e}")
    st.stop()

# Step 5: Define system prompt and template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Step 6: Query handling and response generation
query = st.chat_input("Ask me anything about the provided URLs:")  # User input
if query:
    try:
        # Create retrieval-augmented generation (RAG) chain
        question_answer_chain = create_stuff_documents_chain(
            retriever=retriever, combine_document_chain=prompt_template
        )
        rag_chain = create_retrieval_chain(
            retriever=retriever, combine_document_chain=question_answer_chain
        )

        # Get the response
        response = rag_chain.invoke({"input": query})
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"Error processing query: {e}")
