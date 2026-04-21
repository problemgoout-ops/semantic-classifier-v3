"""
Semantic Classifier v3 - Core classification engine.
Учитывает опыт mdm-classifier v2:
- N=N validation (принцип: число сущностей = числу атрибутов)
- Fallback на rules при низком confidence
- Self-learning через feedback
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from .vector_store import VectorStore
from .attribute_extractor_v3 import AttributeExtractorV3


@dataclass
class ClassificationRequest:
    """Запрос на классификацию."""
    name: str
    code: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_embedding_text(self) -> str:
        """Текст для embedding (только наименование)."""
        return f"ETALON:{self.name}"


@dataclass
class ClassificationResult:
    """Результат классификации (как в v2 для совместимости)."""
    code: str
    name: str
    cls: str = ""  # class_name для совместимости с v2
    class_name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    matched_etalon: str = ""
    similarity: float = 0.0
    source: str = "semantic_v3"
    errors: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Для совместимости с v2
        if self.cls and not self.class_name:
            self.class_name = self.cls
        if self.class_name and not self.cls:
            self.cls = self.class_name


class SemanticClassifierV3:
    """
    Semantic Classifier v3.
    
    Улучшения по сравнению с v2:
    1. Semantic search вместо keyword matching
    2. Confidence score для каждого результата
    3. Pattern extraction из эталонов
    4. Self-learning через feedback
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        fallback_threshold: float = 0.6,
        top_k: int = 5
    ):
        self.vector_store = vector_store or VectorStore()
        self.fallback_threshold = fallback_threshold
        self.top_k = top_k
        self.attribute_extractor = AttributeExtractorV3()
        
        # Кэш для embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_size = 1000
    
    def classify(self, code: str, name: str) -> ClassificationResult:
        """
        Классифицировать наименование.
        
        Returns:
            ClassificationResult совместимый с v2
        """
        request = ClassificationRequest(name=name, code=code)
        
        # Шаг 1: Semantic search (как в v3)
        embedding = self._get_embedding(request)
        neighbors = self.vector_store.search(embedding, k=self.top_k)
        
        if not neighbors:
            return self._fallback_empty(code, name)
        
        # Шаг 2: Class aggregation
        class_name, confidence = self._aggregate_class(neighbors)
        
        if confidence < self.fallback_threshold:
            return self._fallback_low_confidence(code, name, confidence)
        
        # Шаг 3: Attribute extraction (как в v2, но через соседей)
        attributes = self.attribute_extractor.extract(name, neighbors, class_name)
        
        # Шаг 4: N=N validation (из v2)
        validation_notes = self._validate_nn(name, attributes)
        
        # Шаг 5: Формирование результата (совместимый с v2)
        best_neighbor = neighbors[0]
        
        return ClassificationResult(
            code=code,
            name=name,
            cls=class_name,
            class_name=class_name,
            attributes=attributes,
            confidence=confidence,
            matched_etalon=best_neighbor.name,
            similarity=best_neighbor.similarity,
            source="semantic_v3",
            validation_notes=validation_notes
        )
    
    def _get_embedding(self, request: ClassificationRequest) -> List[float]:
        """Получить или сгенерировать embedding."""
        text = request.to_embedding_text()
        
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Fallback: простой hashing для демонстрации
        import random
        random.seed(text)
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        
        # Сохранить в кэш
        self._embedding_cache[text] = embedding
        if len(self._embedding_cache) > self._cache_size:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        
        return embedding
    
    def _aggregate_class(self, neighbors: List) -> Tuple[str, float]:
        """Агрегация классов через majority voting (как в v3)."""
        if not neighbors:
            return "", 0.0
        
        # Подсчитать голоса
        class_votes: Dict[str, float] = {}
        for n in neighbors:
            cls = n.class_name
            weight = n.similarity  # Вес = similarity
            class_votes[cls] = class_votes.get(cls, 0) + weight
        
        if not class_votes:
            return "", 0.0
        
        # Найти победителя
        best_class = max(class_votes, key=class_votes.get)
        total_weight = sum(class_votes.values())
        best_weight = class_votes[best_class]
        
        # Confidence = доля голосов победителя
        confidence = best_weight / total_weight if total_weight > 0 else 0
        
        return best_class, confidence
    
    def _validate_nn(self, name: str, attributes: Dict) -> List[str]:
        """
        Валидация N=N (из v2).
        Принцип: число сущностей в названии = числу атрибутов.
        """
        notes = []
        
        # Подсчитать сущности в названии
        entities_in_name = len(re.findall(r'\b\d+\b', name)) + \
                          len(re.findall(r'[A-ZА-Я]{2,}\d+', name))
        
        # Подсчитать атрибуты
        attr_count = len(attributes)
        
        if entities_in_name != attr_count:
            notes.append(
                f"N=N: {entities_in_name} сущностей → {attr_count} атрибутов. "
                f"Возможно: {abs(entities_in_name - attr_count)} {'лишних' if attr_count > entities_in_name else 'пропущенных'}"
            )
        
        return notes
    
    def _fallback_empty(self, code: str, name: str) -> ClassificationResult:
        """Fallback: нет соседей в базе."""
        return ClassificationResult(
            code=code,
            name=name,
            errors=["Нет похожих эталонов в базе"],
            source="semantic_v3_fallback"
        )
    
    def _fallback_low_confidence(self, code: str, name: str, confidence: float) -> ClassificationResult:
        """Fallback: низкий confidence."""
        return ClassificationResult(
            code=code,
            name=name,
            errors=[f"Низкий confidence: {confidence:.2%} (порог: {self.fallback_threshold:.0%})"],
            source="semantic_v3_fallback"
        )
    
    def add_feedback(
        self,
        name: str,
        predicted_class: str,
        user_class: str,
        user_attributes: Dict,
        confidence: float
    ):
        """Добавить feedback для дообучения (новое в v3)."""
        self.vector_store.add(
            code=f"feedback_{name[:20]}",
            name=name,
            class_name=user_class,
            attributes=user_attributes,
            embedding=self._get_embedding(ClassificationRequest(name=name))
        )