from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []
documents.extend(TextLoader("data/university_information.txt", encoding="utf-8").load())
documents.extend(TextLoader("data/specializations.txt", encoding="utf-8").load())
documents.extend(CSVLoader("data/course_fees.csv").load())
documents.extend(CSVLoader("data/class_timetable.csv").load())

chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
).split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = FAISS.from_documents(chunks, embeddings)
store.save_local("faiss_index")

print("University vector store created.")
print("Indexed chunks:", len(chunks))
