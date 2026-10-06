# University Student Information RAG Assistant

## Purpose

A classroom RAG project for B.Tech CSE and specialized branches.

IMPORTANT: All university names, fees, timetables, policies and events in this
project are FICTIONAL DEMO DATA. Replace them with approved university data for
a real deployment.

## Why this problem gets student attention

Students already ask questions such as:
- What class do I have at 10 AM?
- Where is my lab?
- What is the fee for CSE-AI?
- When is the hackathon?
- How many WFH days? (wrong domain question to test retrieval)
- What is the attendance requirement?
- Has my semester fee been paid?
- What are my marks?

The last two demonstrate an important enterprise distinction:
static knowledge can be handled by RAG, but personal/current student records
require authenticated tools/APIs to the Student Information System.

## Architecture

Student Question
    -> Embedding
    -> FAISS Vector Search
    -> University documents / CSV data
    -> Relevant Context
    -> LangChain Prompt
    -> Groq-hosted LLM
    -> Grounded Student Answer

## Data sources in this demo

- university_information.txt: timings, attendance, exams, events, services
- specializations.txt: CSE and specialized branch descriptions
- course_fees.csv: demo course fees
- class_timetable.csv: demo Semester 7 Monday timetable

This lets you teach structured + unstructured data in the same RAG project.

## Setup

    python -m venv venv

Command Prompt:

    venv\Scripts\activate

PowerShell:

    .\venv\Scripts\Activate.ps1

Install:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Copy `.env.example` to `.env` and add the student's own Groq key:

    GROQ_API_KEY=your_actual_key

Never commit `.env`.

## Execute

    python 01_explore_university_data.py
    python 02_load_documents.py
    python 03_chunk_documents.py
    python 04_build_vector_store.py
    python 05_test_retrieval.py
    python 06_university_rag_assistant.py
    python 07_compare_llm_vs_rag.py

Run 04 before 05-07.

## Live classroom questions

### Easy / direct
- What are the regular class timings?
- What is the lunch break?
- Where is the Placement Cell?
- When is AI Innovation Week?

### Branch / timetable
- I am a Semester 7 CSE-AI student. What classes do I have on Monday?
- Where is my AI Project Lab?
- What subjects are shown for CSE Data Science on Monday?
- What is the focus of Cyber Security specialization?

### Fee
- What is the annual tuition fee for CSE-AIML?
- Compare the demo fees for regular CSE and CSE-AI.
- Where should semester fees be paid?

### RAG grounding
- What is the university's hostel fee?
- What is the Diwali vacation schedule?

These are absent. The assistant should not invent answers.

### Real-time / personal questions
- What is my attendance today?
- Did I pay my semester fee?
- What marks did I get?
- Is my scholarship approved?

Explain: RAG alone is not enough. These need authenticated access to live
student systems through APIs/tools.

## Best transition to Agentic AI

Student asks:

"Check whether I have paid my fee. If not, tell me the due amount and send me
a reminder."

RAG can explain the fee policy, but it cannot know that student's current
payment status or send a reminder.

That creates the next architecture:

RAG Knowledge
    +
Student Information System Tool
    +
Fee API
    +
Notification Tool
    +
Agent
    +
Authentication / Guardrails

This is the natural bridge from RAG to Tools and Agents.

## Security discussion

A public RAG assistant can answer approved general university information.
It should NOT expose:
- another student's marks
- phone numbers
- payment records
- personal attendance
- scholarship decisions
- credentials

For personal data:
Authentication -> Authorization -> API/Tool -> Minimum required data ->
Audit logging.

## Model note

The sample uses `openai/gpt-oss-20b` through Groq because it matches the
workshop flow. Model availability can change. If unavailable in the student's
Groq account, replace the model name with a currently available compatible
Groq chat model.
