# File operations service

"""
Service for file discovery and reading operations.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from exceptions import FileOperationError
from models.food_entry import FoodFileEntry
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FileService:
    """Service for file operations related to food tracking."""
    
    def __init__(self, base_folder: Path):
        """
        Initialize the file service.
        
        Args:
            base_folder: Base folder containing food images and transcripts
        """
        self.base_folder = Path(base_folder)
        if not self.base_folder.exists():
            raise FileOperationError(f"Base folder does not exist: {base_folder}")
    
    def scan_food_files(self) -> Dict[str, FoodFileEntry]:
        """
        Scan the folder for food image and transcript files.
        
        Returns:
            Dictionary mapping timestamp identifiers to FoodFileEntry objects
        """
        entries: Dict[str, FoodFileEntry] = {}
        
        try:
            for filename in os.listdir(self.base_folder):
                filepath = self.base_folder / filename
                
                # Skip directories
                if filepath.is_dir():
                    continue
                
                # Process image files
                if filename.startswith("img_"):
                    identifier = filename[4:].rsplit(".", 1)[0]
                    if identifier not in entries:
                        entries[identifier] = FoodFileEntry(identifier=identifier)
                    entries[identifier].image_path = filepath
                    logger.debug(f"Found image: {filename}")
                
                # Process transcript files
                elif filename.startswith("tr_"):
                    identifier = filename[3:].rsplit(".", 1)[0]
                    if identifier not in entries:
                        entries[identifier] = FoodFileEntry(identifier=identifier)
                    entries[identifier].transcript_path = filepath
                    logger.debug(f"Found transcript: {filename}")
            
            logger.info(f"Scanned {len(entries)} food entries in {self.base_folder}")
            return entries
            
        except OSError as e:
            raise FileOperationError(f"Error scanning folder: {e}") from e
    
    def read_transcript(self, transcript_path: Path) -> Optional[str]:
        """
        Read transcript file contents.
        
        Args:
            transcript_path: Path to the transcript file
        
        Returns:
            Transcript contents or None if file doesn't exist
        """
        if not transcript_path or not transcript_path.exists():
            return None
        
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Read transcript: {transcript_path.name}")
            return content
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Error reading transcript {transcript_path}: {e}")
            return None
    
    def get_entries_with_images(self) -> List[FoodFileEntry]:
        """Get all entries that have images."""
        entries = self.scan_food_files()
        return [e for e in entries.values() if e.has_image]
    
    def get_entries_without_transcripts(self) -> List[FoodFileEntry]:
        """Get entries that have images but no transcripts."""
        entries = self.scan_food_files()
        return [e for e in entries.values() if e.has_image and not e.has_transcript]
