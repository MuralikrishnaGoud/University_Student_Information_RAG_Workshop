from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = input("Student question: ")
docs = store.similarity_search(question, k=4)

print("\n=== RETRIEVED UNIVERSITY INFORMATION ===")
for i, doc in enumerate(docs, 1):
    print(f"\nResult {i}")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))
