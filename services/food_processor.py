# Food processing service

"""
Service for food image analysis and meal report generation.
"""

import base64
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from exceptions import ImageProcessingError
from models.food_entry import FoodEntry, MealType
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FoodProcessor:
    """Service for processing food images and generating reports."""
    
    # Prompt templates
    IMAGE_ANALYSIS_PROMPT = (
        "Analyze this food image. Create a detailed description of the food. "
        "Concentrate on whole portion size and ratio of ingredients."
    )
    
    MEAL_REPORT_PROMPT = (
        "Using this food description extracted from image: {nutrition_summary}, "
        "combine with these notes: {notes}, which are detailed and specific human generated notes "
        "about that food, generate comprehensive and detailed meal report, which will be used "
        "in future analytics for how balanced and healthy was the food overall. "
        "Highlight in the beginning of the document if it's {meal_type}."
    )
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize the food processor.
        
        Args:
            llm: LangChain LLM for image analysis and text generation
        """
        self.llm = llm
    
    def analyze_image(self, image_path: Path) -> str:
        """
        Analyze a food image and generate a description.
        
        Args:
            image_path: Path to the food image
        
        Returns:
            Text description of the food
        
        Raises:
            ImageProcessingError: If analysis fails
        """
        try:
            # Encode image to Base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Create multimodal message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.IMAGE_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            )
            
            # Invoke the model
            response = self.llm.invoke([message])
            logger.debug(f"Analyzed image: {image_path.name}")
            return response.content
            
        except FileNotFoundError as e:
            raise ImageProcessingError(f"Image not found: {image_path}") from e
        except Exception as e:
            raise ImageProcessingError(f"Failed to analyze image {image_path}: {e}") from e
    
    def generate_meal_report(
        self,
        nutrition_summary: str,
        notes: Optional[str],
        meal_type: MealType
    ) -> str:
        """
        Generate a comprehensive meal report.
        
        Args:
            nutrition_summary: AI-generated food description
            notes: Optional human notes/transcript
            meal_type: Type of meal (breakfast, lunch, dinner)
        
        Returns:
            Comprehensive meal report
        """
        try:
            prompt = self.MEAL_REPORT_PROMPT.format(
                nutrition_summary=nutrition_summary,
                notes=notes or "No additional notes provided",
                meal_type=meal_type.value
            )
            
            message = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([message])
            
            logger.debug(f"Generated meal report for {meal_type.value}")
            return response.content
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to generate meal report: {e}") from e
    
    def process_food_entry(
        self,
        identifier: str,
        image_path: Path,
        notes: Optional[str],
        person: str
    ) -> FoodEntry:
        """
        Process a complete food entry from image to report.
        
        Args:
            identifier: Timestamp identifier
            image_path: Path to food image
            notes: Optional transcript notes
            person: Name of the person
        
        Returns:
            Complete FoodEntry with generated summary
        """
        # Analyze the image
        nutrition_summary = self.analyze_image(image_path)
        
        # Determine meal type from timestamp
        hour = int(identifier[8:10])
        meal_type = MealType.from_hour(hour)
        
        # Generate comprehensive report
        summary = self.generate_meal_report(nutrition_summary, notes, meal_type)
        
        # Create and return FoodEntry
        entry = FoodEntry.from_timestamp(
            timestamp=identifier,
            person=person,
            image_path=image_path,
            summary=summary,
            notes=notes
        )
        
        logger.info(f"Processed food entry: {identifier} ({meal_type.value})")
        return entry
