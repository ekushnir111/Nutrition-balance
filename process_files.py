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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


env_path=find_dotenv(usecwd=True)
loaded=load_dotenv(dotenv_path=env_path, override=True)


os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Nutrition balance"
gemini_api_key=os.getenv("GEMINI_API_KEY")

llm_img = ChatGoogleGenerativeAI( model="gemini-2.0-flash", google_api_key=gemini_api_key)
llm_doc = ChatGoogleGenerativeAI( model="gemini-2.5-pro", google_api_key=gemini_api_key)
folder_path = "/Users/ekushnir/Documents/Food/Eugene"

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
            {"type": "text", "text": "Analyze this food image. Create a detailed description of the food. Concetrate on portion whole portion size and ratio of ingredients. Also add how much input and output tokens have been used for that request"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )

    # 4. Invoke the model
    response = llm.invoke([message])
    
    return response.content


summary = get_nutrition_summary("/Users/ekushnir/Documents/Food/Eugene/img_20260201174535.jpg", llm_img)
print(summary)
result = llm_doc.invoke([HumanMessage(
        content=[
            {"type": "text", "text": f"Give me summary of that text: {summary}"}])])
print(result.content)



dict=get_food_files(folder_path)
print(dict)



