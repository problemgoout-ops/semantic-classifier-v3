"""
Semantic Router - core classification engine.
Vector-first approach: find similar embeddings → infer class and attributes.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import re

from .vector_store import VectorStore
from .attribute_extractor import AttributeExtractor


@dataclass
class ClassificationRequest:
    """Запрос на классификацию."""
    name: str
    code: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_embedding_text(self) -> str:
        """Преобразовать в текст для эмбеддинга."""
        # Добавляем контекст если есть
        ctx_parts = []
        if self.context:
            if 'prev_class' in self.context:
                ctx_parts.append(f"prev:{self.context['prev_class']}")
        
        ctx = f" [{' '.join(ctx_parts)}]" if ctx_parts else ""
        return f"CLS:{self.name}{ctx}"
    
    def get_cache_key(self) -> str:
        """Ключ для LRU кэша."""
        text = self.to_embedding_text()
        return hashlib.md5(text.encode()).hexdigest()


@dataclass
class ClassificationResult:
    """Результат классификации."""
    code: str
    name: str
    class_name: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    matched_neighbors: List[Dict] = field(default_factory=list)
    source: str = "vector_match"  # vector_match, pattern_fallback, unknown
    validation_notes: List[str] = field(default_factory=list)


class SemanticRouter:
    """
    Semantic Router v3 - vector-first classification.
    
    Алгоритм:
    1. Преобразовать запрос в embedding
    2. Найти top-K соседей в VectorStore
    3. Агрегировать: majority voting для класса
    4. Извлечь атрибуты из анализа соседей
    5. Вернуть result с confidence
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        fallback_threshold: float = 0.6,
        top_k: int = 5
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.fallback_threshold = fallback_threshold
        self.top_k = top_k
        self.attribute_extractor = AttributeExtractor()
        
        # LRU кэш для embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_size = 1000
    
    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """
        Классифицировать наименование.
        
        Args:
            request: ClassificationRequest с name и опциональным context
            
        Returns:
            ClassificationResult с class, confidence, attributes
        """
        # Шаг 1: Получить embedding
        embedding = self._get_embedding(request)
        
        # Шаг 2: Поиск соседей
        neighbors = self.vector_store.search(embedding, k=self.top_k)
        
        # Шаг 3: Агрегация результатов
        class_name, confidence = self._aggregate_class(neighbors)
        
        # Если confidence низкий → fallback
        if confidence < self.fallback_threshold:
            return self._fallback_classification(request)
        
        # Шаг 4: Извлечь атрибуты
        attributes = self.attribute_extractor.extract(
            request.name, 
            neighbors,
            class_name
        )
        
        # Шаг 5: Валидация N=N
        validation_notes = self._validate_nn(request.name, attributes)
        
        return ClassificationResult(
            code=request.code or "",
            name=request.name,
            class_name=class_name,
            confidence=confidence,
            attributes=attributes,
            matched_neighbors=[n.to_dict() for n in neighbors[:3]],
            source="vector_match",
            validation_notes=validation_notes
        )
    
    def _get_embedding(self, request: ClassificationRequest) -> List[float]:
        """Получить или сгенерировать embedding."""
        cache_key = request.get_cache_key()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        text = request.to_embedding_text()
        
        if self.embedding_provider == "ollama":
            embedding = self._get_ollama_embedding(text)
        else:
            # Fallback: simple hashing (not real embedding)
            import random
            random.seed(text)
            embedding = [random.uniform(-1, 1) for _ in range(768)]
        
        # Сохранить в кэш
        self._embedding_cache[cache_key] = embedding
        if len(self._embedding_cache) > self._cache_size:
            # LRU: удалить случайный ключ
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        
        return embedding
    
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Получить embedding через Ollama API."""
        import requests
        import os
        
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/embeddings')
        
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get('embedding', [])
        except Exception as e:
            # Fallback: случайный вектор (для тестирования)
            import random
            random.seed(text)
            return [random.uniform(-1, 1) for _ in range(768)]
    
    def _aggregate_class(self, neighbors: List) -> Tuple[str, float]:
        """
        Агрегировать классы соседей через majority voting.
        
        Returns:
            (class_name, confidence)
        """
        if not neighbors:
            return "", 0.0
        
        # Подсчитать голоса
        class_votes: Dict[str, float] = {}
        for n in neighbors:
            cls = n.class_name
            # Вес = similarity
            weight = n.similarity
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
    
    def _fallback_classification(self, request: ClassificationRequest) -> ClassificationResult:
        """Fallback при низком confidence."""
        return ClassificationResult(
            code=request.code or "",
            name=request.name,
            class_name="",
            confidence=0.0,
            attributes={},
            source="unknown",
            validation_notes=["Low confidence, no pattern matched"]
        )
    
    def _validate_nn(self, name: str, attributes: Dict) -> List[str]:
        """Валидация N=N: число сущностей = числу атрибутов."""
        notes = []
        
        # Подсчитать сущности в названии
        entities_in_name = len(re.findall(r'\b\d+\b', name)) + \
                          len(re.findall(r'[A-ZА-Я]{2,}\d+', name))
        
        # Подсчитать атрибуты
        attr_count = len(attributes)
        
        if entities_in_name != attr_count:
            notes.append(f"N=N warning: {entities_in_name} entities vs {attr_count} attributes")
        
        return notes
    
    def add_training_example(
        self,
        name: str,
        class_name: str,
        attributes: Dict,
        code: str = ""
    ):
        """Добавить новый пример для обучения."""
        request = ClassificationRequest(name=name, code=code)
        embedding = self._get_embedding(request)
        
        self.vector_store.add(
            code=code,
            name=name,
            class_name=class_name,
            attributes=attributes,
            embedding=embedding
        )