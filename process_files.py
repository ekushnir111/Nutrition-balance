from IPython.display import Image, display
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
import os 
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage, AnyMessage, SystemMessage
from pprint import pprint
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import PIL.Image
import base64
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


env_path=find_dotenv(usecwd=True)
loaded=load_dotenv(dotenv_path=env_path, override=True)


os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Nutrition balance"
gemini_api_key=os.getenv("GEMINI_API_KEY")

llm_img = ChatGoogleGenerativeAI( model="gemini-2.0-flash", google_api_key=gemini_api_key)
llm_doc = ChatGoogleGenerativeAI( model="gemini-2.5-pro", google_api_key=gemini_api_key)
folder_path = "/Users/ekushnir/Documents/Food/Eugene"
chroma_db_path = "/Users/ekushnir/Documents/Food/Eugene/Database/chroma_db"

# Initialize embeddings and persistent ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
vectorstore = Chroma(
    collection_name="food_summaries",
    embedding_function=embeddings,
    persist_directory=chroma_db_path
)

def get_food_files(folder_path: str) -> dict:
    """
    Reads files from a folder and organizes them by prefix (img/tr).
    
    Returns a dictionary where:
    - Key: timestamp identifier (e.g., "20260201174235")
    - Value: dict with "img" and "tr" keys containing file paths (or None if not present)
    
    Example return:
    {
        "20260201174235": {"img": "/path/to/img_20260201174235.jpg", "tr": "/path/to/tr_20260201174235.txt"},
        "20260130123120": {"img": "/path/to/img_20260130123120.jpg", "tr": None}
    }
    """
    files_dict = {}
    
    # Get all files in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Extract prefix and identifier
        if filename.startswith("img_"):
            # Extract identifier (everything after "img_" and before extension)
            identifier = filename[4:].rsplit(".", 1)[0]
            if identifier not in files_dict:
                files_dict[identifier] = {"img": None, "tr": None}
            files_dict[identifier]["img"] = filepath
            
        elif filename.startswith("tr_"):
            # Extract identifier (everything after "tr_" and before extension)
            identifier = filename[3:].rsplit(".", 1)[0]
            if identifier not in files_dict:
                files_dict[identifier] = {"img": None, "tr": None}
            files_dict[identifier]["tr"] = filepath
    
    return files_dict


def get_images_without_transcript(files_dict: dict) -> list:
    """Returns list of identifiers that have img but no associated tr file."""
    return [id for id, files in files_dict.items() if files["img"] and not files["tr"]]


def get_images_with_transcript(files_dict: dict) -> list:
    """Returns list of identifiers that have both img and tr files."""
    return [id for id, files in files_dict.items() if files["img"] and files["tr"]]

def get_nutrition_summary(image_path, llm):

    # 2. Encode image to Base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # 3. Create the multimodal message
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this food image. Create a detailed description of the food. Concetrate on whole portion size and ratio of ingredients. "},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )

    # 4. Invoke the model
    response = llm.invoke([message])
    
    return response.content



def process_food_entries(files_dict: dict, llm) -> dict:
    """
    Processes each entry in the food files dictionary.
    
    For each entry:
    - Extracts the image path and runs it through get_nutrition_summary
    - Reads the transcript file (if exists) into notes variable
    - Stores the food summary in ChromaDB with metadata
    
    Args:
        files_dict: Dictionary with timestamp keys and img/tr file paths as values
        llm: The LLM model to use for nutrition summary
    """
    for identifier, files in files_dict.items():
        image_path = files["img"]
        tr_path = files["tr"]
        
        # Skip if no image exists
        if not image_path:
            continue
        
        # Get nutrition summary from image
        nutrition_summary = get_nutrition_summary(image_path, llm)
        
        # Read transcript/notes if file exists
        notes = None
        if tr_path:
            with open(tr_path, "r", encoding="utf-8") as f:
                notes = f.read()
        

        food_summary = llm_img.invoke([HumanMessage(
            content=[
                {"type": "text", "text": f"Give me summary of that text: {nutrition_summary}, also combine them with these notes: {notes}. "
                 f"Highlight in the beginning of the document depending on that timestamp {identifier} if it's breakfast(05:00-11:00), lunch(11:00-17:00) or dinner(17:00-05:00)."}])])

        # Determine meal type from timestamp (HHMMSS portion)
        time_part = identifier[8:12]  # Extract HHMM
        hour = int(time_part[:2])
        if 5 <= hour < 11:
            meal_type = "breakfast"
        elif 11 <= hour < 17:
            meal_type = "lunch"
        else:
            meal_type = "dinner"
        
        # Extract date from timestamp (YYYYMMDD)
        date_str = f"{identifier[:4]}-{identifier[4:6]}-{identifier[6:8]}"
        
        # Store in ChromaDB with metadata
        vectorstore.add_texts(
            texts=[food_summary.content],
            metadatas=[{
                "timestamp": identifier,
                "date": date_str,
                "meal_type": meal_type,
                "person": "Eugene",
                "image_path": image_path
            }],
            ids=[identifier]  # Use timestamp as unique ID
        )
        
        print(f"Stored: {identifier} ({meal_type})")


def get_all_food_entries():
    """
    Retrieves and prints all food entries stored in ChromaDB.
    """
    # Get all documents from the vectorstore
    results = vectorstore.get()
    
    if not results["ids"]:
        print("No entries found in ChromaDB")
        return
    
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


files_dict = get_food_files(folder_path)
process_food_entries(files_dict, llm_img)
get_all_food_entries()





