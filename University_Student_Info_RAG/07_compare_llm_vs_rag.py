import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

question = input("Ask a university-specific question: ")

print("\n=== 1. LLM WITHOUT RAG ===")
plain = llm.invoke(
    f"""Answer this university question. If you do not actually know the
specific university information, say so rather than inventing it.

Question: {question}"""
)
print(plain.content)

print("\n=== 2. RETRIEVED PRIVATE KNOWLEDGE ===")
docs = store.similarity_search(question, k=4)
context = "\n\n".join(d.page_content for d in docs)
print(context)

print("\n=== 3. LLM WITH RAG ===")
rag = llm.invoke(
    f"""Use ONLY the context below to answer the question.
If the answer is absent, say it is not available in the approved knowledge base.

CONTEXT:
{context}

QUESTION:
{question}"""
)
print(rag.content)
