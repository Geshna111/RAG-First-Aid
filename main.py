import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# SETUP
# ---------------------------
MAX_QUESTIONS = 5

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ AI First Aid Assistant Ready\n")

# ---------------------------
# DYNAMIC QUESTION LOGIC
# ---------------------------
asked_questions = set()

def get_dynamic_question(context):
    context = context.lower()

    if "collapse" in context or "unconscious" in context:
        questions = [
            "Is the person breathing properly?",
            "Is the person conscious?"
        ]

    elif "poison" in context or "swallowed" in context:
        questions = [
            "Is the person vomiting?",
            "Is the person conscious?"
        ]

    elif "burn" in context:
        questions = [
            "Is the burn large or deep?",
            "Is there blistering?"
        ]

    elif "bleeding" in context or "cut" in context:
        questions = [
            "Is the bleeding heavy?",
            "Has it stopped?"
        ]

    elif "breathing" in context or "asthma" in context:
        questions = [
            "Is breathing getting worse?",
            "Is there chest tightness?"
        ]

    elif "dizzy" in context or "sweating" in context:
        questions = [
            "Is the person feeling weak or about to faint?",
            "When did this start?"
        ]

    else:
        questions = [
            "Is the person conscious?",
            "Is the person breathing properly?"
        ]

    # Avoid repetition
    for q in questions:
        if q not in asked_questions:
            asked_questions.add(q)
            return q

    return "When did this start?"


# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    print("\n--- New Case ---\n")

    question_count = 0
    user_context = ""
    asked_questions.clear()

    # ---------------------------
    # FIXED QUESTIONS
    # ---------------------------
    q1 = input("AI: What happened?\nYou: ")
    user_context += " " + q1
    question_count += 1

    q2 = input("AI: What is the age and gender of the person?\nYou: ")
    user_context += " " + q2
    question_count += 1

    q3 = input("AI: Do they have any medical conditions or take regular medication?\nYou: ")
    user_context += " " + q3
    question_count += 1

    # ---------------------------
    # DYNAMIC QUESTIONS
    # ---------------------------
    while question_count < MAX_QUESTIONS:
        question = get_dynamic_question(user_context)
        answer = input(f"AI: {question}\nYou: ")

        user_context += " " + answer
        question_count += 1

    # ---------------------------
    # FINAL ANSWER (RAG OUTPUT)
    # ---------------------------
    docs = retriever.invoke(user_context)

    print("\n💡 Final Advice:\n")

    for doc in docs:
        print("🔴", doc.page_content.strip())
        print("\n-----------------\n")

    print("\n=========================\n")