from langchain_community.document_loaders import TextLoader, CSVLoader

documents = []

for path in [
    "data/university_information.txt",
    "data/specializations.txt"
]:
    documents.extend(TextLoader(path, encoding="utf-8").load())

documents.extend(CSVLoader("data/course_fees.csv").load())
documents.extend(CSVLoader("data/class_timetable.csv").load())

print("Total LangChain documents:", len(documents))

for i, doc in enumerate(documents[:8], 1):
    print(f"\n--- DOCUMENT {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
