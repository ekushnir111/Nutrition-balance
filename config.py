# Configuration management for Nutrition Balance application

"""
Centralized configuration with type-safe settings using dataclasses.
Loads values from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from exceptions import ConfigurationError


@dataclass
class Settings:
    """Application settings with validation."""
    
    # API Keys
    gemini_api_key: str
    langchain_api_key: Optional[str] = None
    
    # Paths
    food_folder_path: Path = field(default_factory=lambda: Path("/Users/ekushnir/Documents/Food/Eugene"))
    chroma_db_path: Path = field(default_factory=lambda: Path("/Users/ekushnir/Documents/Food/Eugene/Database/chroma_db"))
    
    # Model settings
    llm_image_model: str = "gemini-2.0-flash"
    llm_doc_model: str = "gemini-2.5-pro"
    embedding_model: str = "models/embedding-001"
    
    # ChromaDB settings
    collection_name: str = "food_summaries"
    
    # LangChain settings
    langchain_tracing: bool = True
    langchain_project: str = "Nutrition balance"
    
    # Default person (can be overridden per-run)
    default_person: str = "Eugene"
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is required")
        
        # Ensure paths are Path objects
        if isinstance(self.food_folder_path, str):
            self.food_folder_path = Path(self.food_folder_path)
        if isinstance(self.chroma_db_path, str):
            self.chroma_db_path = Path(self.chroma_db_path)
    
    def validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.food_folder_path.exists():
            raise ConfigurationError(f"Food folder does not exist: {self.food_folder_path}")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """
    Load settings from environment variables.
    
    Returns:
        Settings object with all configuration values.
    
    Raises:
        ConfigurationError: If required settings are missing.
    """
    # Load .env file
    env_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Get API keys
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ConfigurationError("GEMINI_API_KEY environment variable is not set")
    
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    
    # Get paths from env or use defaults
    food_folder = os.getenv("FOOD_FOLDER_PATH", "/Users/ekushnir/Documents/Food/Eugene")
    chroma_db = os.getenv("CHROMA_DB_PATH", "/Users/ekushnir/Documents/Food/Eugene/Database/chroma_db")
    
    # Get optional settings
    default_person = os.getenv("DEFAULT_PERSON", "Eugene")
    
    settings = Settings(
        gemini_api_key=gemini_api_key,
        langchain_api_key=langchain_api_key,
        food_folder_path=Path(food_folder),
        chroma_db_path=Path(chroma_db),
        default_person=default_person,
    )
    
    return settings


def configure_langchain(settings: Settings) -> None:
    """Configure LangChain environment variables."""
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
