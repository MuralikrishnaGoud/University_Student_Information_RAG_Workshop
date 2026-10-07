import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: LOAD OPENAI API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Please add it to your .env file."
    )

print("OpenAI API Key loaded successfully.")


# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================
# This is the SAME embedding model used while building
# the FAISS vector database.
#
# Important:
# The embedding model used while searching should be the
# same embedding model used while creating the vector store.
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# STEP 3: LOAD EXISTING FAISS VECTOR STORE
# ============================================================
# The FAISS index should already have been created by:
#
# python 04_build_vector_store.py
#
# FAISS contains embeddings of our university information.
# ============================================================

print("\nLoading University FAISS Vector Store...")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("University Vector Store loaded successfully.")


# ============================================================
# STEP 4: CONNECT TO OPENAI LLM
# ============================================================
# OpenAI replaces ChatGroq here.
#
# If this model is unavailable in your API account,
# replace it with a model available to your account.
# ============================================================

print("\nConnecting to OpenAI...")

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    api_key=api_key,
    temperature=0
)

print("OpenAI LLM connected successfully.")


# ============================================================
# STEP 5: CREATE RAG PROMPT
# ============================================================
# The LLM must answer using ONLY the retrieved university
# information.
#
# This helps prevent the model from inventing university
# fees, timetable details, events or policies.
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
# STEP 6: CREATE QUESTION → RAG FUNCTION
# ============================================================

def ask_university(question):

    print("\nSearching University Knowledge Base...")

    # --------------------------------------------------------
    # STEP 6A: RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------
    # Student question is converted into an embedding.
    #
    # FAISS compares the question embedding with stored
    # university document embeddings.
    #
    # k=5 means retrieve the top 5 relevant chunks.
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
    # STEP 6C: COMBINE RETRIEVED INFORMATION
    # --------------------------------------------------------

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(document.page_content)

    context = "\n\n".join(context_parts)


    # --------------------------------------------------------
    # STEP 6D: ADD CONTEXT TO PROMPT
    # --------------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # --------------------------------------------------------
    # STEP 6E: SEND AUGMENTED PROMPT TO OPENAI
    # --------------------------------------------------------

    print("\nSending retrieved context to OpenAI...")

    response = llm.invoke(formatted_prompt)


    # --------------------------------------------------------
    # STEP 6F: RETURN FINAL ANSWER
    # --------------------------------------------------------

    return response.content


# ============================================================
# STEP 7: START UNIVERSITY RAG ASSISTANT
# ============================================================

print("\n========================================")
print("   UNIVERSITY STUDENT RAG ASSISTANT")
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