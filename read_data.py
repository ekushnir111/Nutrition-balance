#!/usr/bin/env python3
# Data Reading Script - Enterprise Grade

"""
Script for reading and querying food entries from ChromaDB.

Usage:
    python read_data.py [--meal-type TYPE] [--date DATE] [--person NAME] [--search QUERY]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from exceptions import NutritionBalanceError
from services.vectorstore_service import VectorStoreService
from utils.logging_config import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Read and query food entries from ChromaDB"
    )
    parser.add_argument(
        "--meal-type", "-m",
        type=str,
        choices=["breakfast", "lunch", "dinner"],
        help="Filter by meal type"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Filter by date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--person", "-p",
        type=str,
        help="Filter by person name"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Semantic search query"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum results for search (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def print_entry(entry: dict, index: int) -> None:
    """Pretty print a food entry."""
    metadata = entry.get("metadata", {})
    document = entry.get("document", "")
    entry_id = entry.get("id", metadata.get("timestamp", "Unknown"))
    
    print(f"\n{'='*60}")
    print(f"Entry {index}: {entry_id}")
    print(f"{'='*60}")
    print(f"Date:      {metadata.get('date', 'N/A')}")
    print(f"Meal Type: {metadata.get('meal_type', 'N/A')}")
    print(f"Person:    {metadata.get('person', 'N/A')}")
    print(f"Image:     {metadata.get('image_path', 'N/A')}")
    print(f"\nSummary:\n{document}")


def main() -> int:
    """
    Main entry point for data reading.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    setup_logging(level=log_level)
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        settings = get_settings()
        
        # Initialize vector store
        vectorstore = VectorStoreService(
            persist_directory=settings.chroma_db_path,
            collection_name=settings.collection_name,
            api_key=settings.gemini_api_key,
            embedding_model=settings.embedding_model
        )
        
        # Semantic search or filter-based query
        if args.search:
            # Build filters for search
            filters = {}
            if args.meal_type:
                filters["meal_type"] = args.meal_type
            if args.person:
                filters["person"] = args.person
            
            entries = vectorstore.similarity_search(
                query=args.search,
                k=args.limit,
                **filters
            )
            print(f"\n🔍 Semantic search results for: '{args.search}'")
        else:
            # Filter-based query
            entries = vectorstore.get_by_filter(
                meal_type=args.meal_type,
                date=args.date,
                person=args.person
            )
            
            filter_desc = []
            if args.meal_type:
                filter_desc.append(f"meal_type={args.meal_type}")
            if args.date:
                filter_desc.append(f"date={args.date}")
            if args.person:
                filter_desc.append(f"person={args.person}")
            
            if filter_desc:
                print(f"\n📋 Filtered results: {', '.join(filter_desc)}")
            else:
                print(f"\n📋 All entries")
        
        # Print results
        if not entries:
            print("\nNo entries found.")
            return 0
        
        print(f"\nTotal entries: {len(entries)}")
        
        for i, entry in enumerate(entries, 1):
            print_entry(entry, i)
        
        return 0
        
    except NutritionBalanceError as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1


# Utility functions for interactive use
def update_meal_type(entry_id: str, new_meal_type: str) -> None:
    """Update the meal type for a specific entry."""
    settings = get_settings()
    vectorstore = VectorStoreService(
        persist_directory=settings.chroma_db_path,
        collection_name=settings.collection_name,
        api_key=settings.gemini_api_key
    )
    
    # Get current entry
    entries = vectorstore.get_by_filter()
    for entry in entries:
        if entry["id"] == entry_id:
            metadata = entry["metadata"]
            metadata["meal_type"] = new_meal_type
            vectorstore.update_metadata(entry_id, metadata)
            print(f"✅ Updated {entry_id} to {new_meal_type}")
            return
    
    print(f"❌ Entry {entry_id} not found")


def delete_entry(entry_id: str) -> None:
    """Delete an entry by ID."""
    settings = get_settings()
    vectorstore = VectorStoreService(
        persist_directory=settings.chroma_db_path,
        collection_name=settings.collection_name,
        api_key=settings.gemini_api_key
    )
    vectorstore.delete_entry(entry_id)
    print(f"✅ Deleted {entry_id}")


def add_tag(entry_id: str, tag_name: str, tag_value: str) -> None:
    """Add a new metadata tag to an entry."""
    settings = get_settings()
    vectorstore = VectorStoreService(
        persist_directory=settings.chroma_db_path,
        collection_name=settings.collection_name,
        api_key=settings.gemini_api_key
    )
    
    entries = vectorstore.get_by_filter()
    for entry in entries:
        if entry["id"] == entry_id:
            metadata = entry["metadata"]
            metadata[tag_name] = tag_value
            vectorstore.update_metadata(entry_id, metadata)
            print(f"✅ Added {tag_name}={tag_value} to {entry_id}")
            return
    
    print(f"❌ Entry {entry_id} not found")


if __name__ == "__main__":
    sys.exit(main())