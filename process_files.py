#!/usr/bin/env python3
# Food Processing Script - Enterprise Grade

"""
Main script for processing food images and storing summaries in ChromaDB.

Usage:
    python process_files.py [--person NAME]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings, configure_langchain
from exceptions import NutritionBalanceError
from services.file_service import FileService
from services.food_processor import FoodProcessor
from services.vectorstore_service import VectorStoreService
from utils.logging_config import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process food images and store summaries in ChromaDB"
    )
    parser.add_argument(
        "--person",
        type=str,
        default=None,
        help="Name of the person (overrides config default)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for food processing.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        settings = get_settings()
        configure_langchain(settings)
        
        # Override person if provided via CLI
        person = args.person or settings.default_person
        
        # Initialize services
        logger.info("Initializing services...")
        
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_image_model,
            google_api_key=settings.gemini_api_key
        )
        
        file_service = FileService(settings.food_folder_path)
        food_processor = FoodProcessor(llm)
        vectorstore = VectorStoreService(
            persist_directory=settings.chroma_db_path,
            collection_name=settings.collection_name,
            api_key=settings.gemini_api_key,
            embedding_model=settings.embedding_model
        )
        
        # Scan for food files
        logger.info(f"Scanning folder: {settings.food_folder_path}")
        entries = file_service.scan_food_files()
        
        if not entries:
            logger.warning("No food files found")
            return 0
        
        # Process each entry
        processed_count = 0
        for identifier, file_entry in entries.items():
            if not file_entry.has_image:
                logger.debug(f"Skipping {identifier}: no image")
                continue
            
            # Read transcript if available
            notes = None
            if file_entry.has_transcript:
                notes = file_service.read_transcript(file_entry.transcript_path)
            
            # Process the food entry
            logger.info(f"Processing: {identifier}")
            food_entry = food_processor.process_food_entry(
                identifier=identifier,
                image_path=file_entry.image_path,
                notes=notes,
                person=person
            )
            
            # Store in vector database
            vectorstore.add_entry(food_entry)
            processed_count += 1
        
        # Summary
        logger.info(f"Processing complete. Stored {processed_count} entries.")
        logger.info(f"Total entries in database: {vectorstore.count}")
        
        return 0
        
    except NutritionBalanceError as e:
        logger.error(f"Processing error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
