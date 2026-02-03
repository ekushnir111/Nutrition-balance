# Vector store service for ChromaDB operations

"""
Service for ChromaDB vector store operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from exceptions import VectorStoreError
from models.food_entry import FoodEntry, MealType
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """Service for ChromaDB vector store operations."""
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        api_key: str,
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize the vector store service.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            api_key: Google API key for embeddings
            embedding_model: Model to use for embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"Initialized VectorStoreService: {collection_name} at {persist_directory}")
    
    def add_entry(self, entry: FoodEntry) -> None:
        """
        Add a food entry to the vector store.
        
        Args:
            entry: FoodEntry to store
        """
        try:
            self.vectorstore.add_texts(
                texts=[entry.summary],
                metadatas=[entry.to_metadata()],
                ids=[entry.timestamp]
            )
            logger.info(f"Stored entry: {entry.timestamp} ({entry.meal_type.value})")
        except Exception as e:
            raise VectorStoreError(f"Failed to add entry {entry.timestamp}: {e}") from e
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all entries from the vector store.
        
        Returns:
            List of dictionaries with id, document, and metadata
        """
        try:
            results = self.vectorstore.get()
            
            if not results["ids"]:
                logger.info("No entries found in vector store")
                return []
            
            entries = []
            for doc_id, document, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"]
            ):
                entries.append({
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata
                })
            
            logger.info(f"Retrieved {len(entries)} entries")
            return entries
            
        except Exception as e:
            raise VectorStoreError(f"Failed to retrieve entries: {e}") from e
    
    def get_by_filter(
        self,
        meal_type: Optional[str] = None,
        date: Optional[str] = None,
        person: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entries filtered by metadata.
        
        Args:
            meal_type: Filter by meal type (breakfast, lunch, dinner)
            date: Filter by date (YYYY-MM-DD)
            person: Filter by person name
        
        Returns:
            List of matching entries
        """
        try:
            # Build filter
            filters = []
            if meal_type:
                filters.append({"meal_type": meal_type})
            if date:
                filters.append({"date": date})
            if person:
                filters.append({"person": person})
            
            # Combine filters
            where = None
            if len(filters) == 1:
                where = filters[0]
            elif len(filters) > 1:
                where = {"$and": filters}
            
            results = self.vectorstore.get(where=where)
            
            entries = []
            for doc_id, document, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"]
            ):
                entries.append({
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata
                })
            
            logger.info(f"Retrieved {len(entries)} entries with filter: {where}")
            return entries
            
        except Exception as e:
            raise VectorStoreError(f"Failed to filter entries: {e}") from e
    
    def update_metadata(self, entry_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update metadata for an existing entry.
        
        Args:
            entry_id: ID of the entry to update
            metadata: New metadata dictionary
        """
        try:
            self.vectorstore._collection.update(
                ids=[entry_id],
                metadatas=[metadata]
            )
            logger.info(f"Updated metadata for entry: {entry_id}")
        except Exception as e:
            raise VectorStoreError(f"Failed to update entry {entry_id}: {e}") from e
    
    def delete_entry(self, entry_id: str) -> None:
        """
        Delete an entry from the vector store.
        
        Args:
            entry_id: ID of the entry to delete
        """
        try:
            self.vectorstore._collection.delete(ids=[entry_id])
            logger.info(f"Deleted entry: {entry_id}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete entry {entry_id}: {e}") from e
    
    def delete_by_filter(self, **kwargs) -> None:
        """
        Delete entries matching filter criteria.
        
        Args:
            **kwargs: Filter criteria (meal_type, date, person)
        """
        try:
            where = kwargs
            self.vectorstore._collection.delete(where=where)
            logger.info(f"Deleted entries matching: {where}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete entries: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5, **filters) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            **filters: Metadata filters
        
        Returns:
            List of similar entries
        """
        try:
            filter_dict = filters if filters else None
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            entries = []
            for doc in results:
                entries.append({
                    "document": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Found {len(entries)} similar entries for query")
            return entries
            
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    @property
    def count(self) -> int:
        """Get total number of entries in the store."""
        return len(self.vectorstore.get()["ids"])
