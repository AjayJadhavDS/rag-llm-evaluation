import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# ==========================
# 1️⃣ LOAD + INDEX PDF
# ==========================

loader = PyPDFLoader("cv.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==========================
# 2️⃣ RAG CHAIN (LCEL STYLE)
# ==========================

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "Not mentioned in document."

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | rag_prompt
    | llm
)

# ==========================
# 3️⃣ EVALUATION DATASET
# ==========================

evaluation_dataset = [
    {
        "question": "What is Ajay Jadhav’s primary area of expertise?",
        "ideal_answer": "Ajay specializes in Generative AI, LLM engineering, RAG systems, and scalable AI architecture."
    },
    {
        "question": "What kind of architecture does Ajay prefer?",
        "ideal_answer": "He prefers structured microservice architecture with event-driven systems, observability, and scalability."
    },
    {
        "question": "What companies has Ajay worked with?",
        "ideal_answer": "Ajay has worked with organizations such as Infosys and other enterprise clients."
    },
    {
        "question": "What is Ajay’s experience in blockchain development?",
        "ideal_answer": "Not mentioned in document."
    }
]

# ==========================
# 4️⃣ LLM AS JUDGE
# ==========================

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of RAG systems.

Question:
{question}

Ideal Answer:
{ideal_answer}

RAG Answer:
{rag_answer}

Evaluate:
1. Relevance score (1-10)
2. Correctness score (1-10)
3. Is answer grounded in document? (Yes/No)
4. Short reasoning

Respond strictly in JSON:
{{
    "relevance_score": int,
    "correctness_score": int,
    "grounded": "Yes or No",
    "reason": "text"
}}
""")

judge_chain = judge_prompt | judge_llm | JsonOutputParser()

# ==========================
# 5️⃣ RUN EVALUATION
# ==========================

for item in evaluation_dataset:

    print("\n======================================")
    print("Question:", item["question"])

    # Get RAG answer
    rag_response = rag_chain.invoke(item["question"])
    rag_answer = rag_response.content

    print("RAG Answer:", rag_answer)

    # Judge evaluation
    evaluation = judge_chain.invoke({
        "question": item["question"],
        "ideal_answer": item["ideal_answer"],
        "rag_answer": rag_answer
    })

    print("Evaluation:", evaluation)