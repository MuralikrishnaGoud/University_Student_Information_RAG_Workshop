import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: LOAD GEMINI API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add it to your .env file."
    )

print("Gemini API Key loaded successfully.")


# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================
# We continue using the SAME HuggingFace embedding model.
#
# IMPORTANT:
# The embedding model used for retrieval should match the
# embedding model used when the FAISS index was created.
#
# We are changing only the LLM provider:
# OpenAI/Groq -> Gemini
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# STEP 3: LOAD EXISTING FAISS VECTOR STORE
# ============================================================
# The FAISS index should already have been created using:
#
# python 04_build_vector_store.py
#
# We do NOT need to rebuild FAISS simply because the
# generation model changes from OpenAI/Groq to Gemini.
# ============================================================

print("\nLoading University FAISS Vector Store...")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("University Vector Store loaded successfully.")


# ============================================================
# STEP 4: CONNECT TO GEMINI
# ============================================================

print("\nConnecting to Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

print("Gemini LLM connected successfully.")


# ============================================================
# STEP 5: CREATE RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a University Student Information Assistant.

Answer the student's question using ONLY the university
information provided in the CONTEXT below.

If the answer is not available in the context, say:

"I could not find this information in the available
university data."

Do NOT invent:
- university fees
- class timings
- timetable information
- university events
- university policies
- attendance
- marks
- examination results
- payment information
- scholarship approval
- personal student information

Personal student information requires an authenticated
and authorized university system.

----------------------------------------
CONTEXT
----------------------------------------

{context}

----------------------------------------
STUDENT QUESTION
----------------------------------------

{question}

----------------------------------------
ANSWER
----------------------------------------
"""
)


# ============================================================
# STEP 6: RAG FUNCTION
# ============================================================

def ask_university(question):

    print("\nSearching University Knowledge Base...")


    # --------------------------------------------------------
    # STEP 6A: RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    retrieved_documents = vector_store.similarity_search(
        question,
        k=5
    )

    print(
        f"Retrieved {len(retrieved_documents)} "
        f"relevant university chunks."
    )


    # --------------------------------------------------------
    # STEP 6B: DISPLAY RETRIEVED SOURCES
    # --------------------------------------------------------

    print("\n========================================")
    print("RAG RETRIEVED SOURCES")
    print("========================================")

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown Source"
        )

        print(f"{i}. {source}")


    # --------------------------------------------------------
    # STEP 6C: COMBINE RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(document.page_content)

    context = "\n\n".join(context_parts)


    # --------------------------------------------------------
    # STEP 6D: AUGMENT THE PROMPT
    # --------------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # --------------------------------------------------------
    # STEP 6E: SEND QUESTION + CONTEXT TO GEMINI
    # --------------------------------------------------------

    print("\nSending retrieved context to Gemini...")

    response = llm.invoke(formatted_prompt)


    # --------------------------------------------------------
    # STEP 6F: RETURN GENERATED ANSWER
    # --------------------------------------------------------

    return response.content


# ============================================================
# STEP 7: START UNIVERSITY RAG ASSISTANT
# ============================================================

print("\n========================================")
print(" UNIVERSITY STUDENT RAG ASSISTANT")
print("          GEMINI VERSION")
print("========================================")

print(
    "\nAsk questions about university fees, "
    "timetables, events, policies, services "
    "and CSE specializations."
)

print("\nType 'exit' to stop the program.")


# ============================================================
# STEP 8: INTERACTIVE QUESTION LOOP
# ============================================================

while True:

    question = input(
        "\nStudent Question: "
    ).strip()

    if question.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "\nUniversity RAG Assistant stopped."
        )

        break


    if not question:

        print(
            "Please enter a question."
        )

        continue


    try:

        answer = ask_university(question)

        print("\n========================================")
        print("UNIVERSITY RAG ANSWER")
        print("========================================")

        print(answer)


    except Exception as e:

        print("\nError:")
        print(e)