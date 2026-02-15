import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import json

# 🔑 Set your OpenAI API key
from dotenv import load_dotenv
load_dotenv()

# 1️⃣ Load PDF
loader = PyPDFLoader("cv.pdf")
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
embeddings = OpenAIEmbeddings()

# 4️⃣ Store in vector DB (FAISS)
vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

# 5️⃣ Create retriever
retriever = vectorstore.as_retriever()

# 6️⃣ LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 7️⃣ RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 8️⃣ Ask Question
query = "What is this document about?"
result = qa({"query": query})

print("Answer:\n", result["result"])
