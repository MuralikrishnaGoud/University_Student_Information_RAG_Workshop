import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY missing. Create .env from .env.example.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a University Student Information Assistant.

Use ONLY the retrieved university context below.

Rules:
1. Do not invent university facts.
2. If information is unavailable, clearly say it is not available in the approved knowledge base.
3. Do not invent a student's attendance, marks, fee-payment status, scholarship approval, or examination result.
4. If the question asks for personal student information, explain that authenticated access to the authorized student system is required.
5. When possible, mention the source file names supplied with the context.
6. Keep answers student-friendly and concise.

RETRIEVED CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
""")

chain = prompt | llm

print("=" * 72)
print("UNIVERSITY STUDENT INFORMATION RAG ASSISTANT")
print("Ask about classes, timetable, fees, events, branches or university policy.")
print("Type 'exit' to stop.")
print("=" * 72)

while True:
    question = input("\nStudent: ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue

    docs = store.similarity_search(question, k=5)

    context_parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        context_parts.append(f"SOURCE: {source}\n{d.page_content}")

    context = "\n\n".join(context_parts)

    print("\n--- What RAG retrieved ---")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source')}")

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAssistant:")
    print(response.content)
