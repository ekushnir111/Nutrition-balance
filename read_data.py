from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
# Load environment
env_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=env_path, override=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Define the path to your ChromaDB
chroma_db_path = "/Users/ekushnir/Documents/Food/Eugene/Database/chroma_db"
# Initialize embeddings and connect to persistent ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
vectorstore = Chroma(
    collection_name="food_summaries",
    embedding_function=embeddings,
    persist_directory=chroma_db_path  # <-- This is the path
)

results = vectorstore.get()
# Now you can use vectorstore.get() to read all data
if not results["ids"]:
    print("No entries found in ChromaDB")


print(f"\n{'='*60}")
print(f"Total entries in ChromaDB: {len(results['ids'])}")
print(f"{'='*60}\n")

for i, (doc_id, document, metadata) in enumerate(zip(
    results["ids"], 
    results["documents"], 
    results["metadatas"]
)):
    print(f"--- Entry {i+1}: {doc_id} ---")
    print(f"Date: {metadata.get('date', 'N/A')}")
    print(f"Meal Type: {metadata.get('meal_type', 'N/A')}")
    print(f"Person: {metadata.get('person', 'N/A')}")
    print(f"Image: {metadata.get('image_path', 'N/A')}")
    print(f"Summary: {document}")
    print()