from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = []
documents.extend(TextLoader("data/university_information.txt", encoding="utf-8").load())
documents.extend(TextLoader("data/specializations.txt", encoding="utf-8").load())
documents.extend(CSVLoader("data/course_fees.csv").load())
documents.extend(CSVLoader("data/class_timetable.csv").load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = splitter.split_documents(documents)

print("Source documents:", len(documents))
print("Chunks after splitting:", len(chunks))

for i, chunk in enumerate(chunks[:10], 1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk.page_content)
    print("Source:", chunk.metadata.get("source"))
