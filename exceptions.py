# Custom exceptions for the Nutrition Balance application

"""
Custom exception classes for enterprise-grade error handling.
All exceptions inherit from NutritionBalanceError for easy catching.
"""


class NutritionBalanceError(Exception):
    """Base exception for all Nutrition Balance errors."""
    pass


class ConfigurationError(NutritionBalanceError):
    """Raised when configuration is missing or invalid."""
    pass


class ImageProcessingError(NutritionBalanceError):
    """Raised when image analysis fails."""
    pass


class VectorStoreError(NutritionBalanceError):
    """Raised when ChromaDB operations fail."""
    pass


class FileOperationError(NutritionBalanceError):
    """Raised when file operations fail."""
    pass
