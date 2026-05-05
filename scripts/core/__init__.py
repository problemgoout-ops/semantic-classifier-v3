"""
Semantic Classifier v3 - Core Components

Compatibility layer for v2:
- ClassificationResult matches mdm-classifier v2 format
- classify() method signature matches v2
"""

from .semantic_router_v3 import (
    SemanticClassifierV3,
    ClassificationRequest,
    ClassificationResult
)
from .vector_store import VectorStore
from .attribute_extractor_v3 import AttributeExtractorV3

__version__ = "3.0.0"
__all__ = [
    'SemanticClassifierV3',
    'ClassificationRequest',
    'ClassificationResult',
    'VectorStore',
    'AttributeExtractorV3'
]