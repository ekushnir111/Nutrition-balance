# Data models for food entries

"""
Type-safe data models for food entries and related types.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime


class MealType(Enum):
    """Enumeration of meal types."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    
    @classmethod
    def from_hour(cls, hour: int) -> "MealType":
        """Determine meal type from hour of day."""
        if 5 <= hour < 11:
            return cls.BREAKFAST
        elif 11 <= hour < 17:
            return cls.LUNCH
        else:
            return cls.DINNER


@dataclass
class FoodFileEntry:
    """Represents a pair of image and transcript files."""
    identifier: str
    image_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    
    @property
    def has_image(self) -> bool:
        return self.image_path is not None
    
    @property
    def has_transcript(self) -> bool:
        return self.transcript_path is not None


@dataclass
class FoodEntry:
    """Represents a processed food entry with all metadata."""
    timestamp: str
    date: str
    meal_type: MealType
    person: str
    image_path: Path
    summary: str
    notes: Optional[str] = None
    
    @classmethod
    def from_timestamp(
        cls,
        timestamp: str,
        person: str,
        image_path: Path,
        summary: str,
        notes: Optional[str] = None
    ) -> "FoodEntry":
        """
        Create a FoodEntry from a timestamp identifier.
        
        Args:
            timestamp: Format YYYYMMDDHHMMSS (e.g., "20260201174535")
            person: Name of the person
            image_path: Path to the food image
            summary: AI-generated food summary
            notes: Optional transcript notes
        
        Returns:
            FoodEntry with parsed date and meal type
        """
        # Parse timestamp
        date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        hour = int(timestamp[8:10])
        meal_type = MealType.from_hour(hour)
        
        return cls(
            timestamp=timestamp,
            date=date_str,
            meal_type=meal_type,
            person=person,
            image_path=image_path,
            summary=summary,
            notes=notes
        )
    
    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata format."""
        return {
            "timestamp": self.timestamp,
            "date": self.date,
            "meal_type": self.meal_type.value,
            "person": self.person,
            "image_path": str(self.image_path)
        }
