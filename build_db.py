import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load JSON data
with open("first_aid_data.json", "r") as file:
    data = json.load(file)

documents = []

for item in data:
    content = f"""
    Title: {item['title']}
    Identify: {item['identify']}
    Steps: {" ".join(item['immediate_steps'])}
    Avoid: {" ".join(item['avoid'])}
    Emergency: {item['emergency']}
    Source: {item['source']}
    """
    
    documents.append(Document(page_content=content))

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS DB
vectorstore = FAISS.from_documents(documents, embeddings)

# Save
vectorstore.save_local("faiss_index")

print("✅ FAISS database rebuilt successfully!")