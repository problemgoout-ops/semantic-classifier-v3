"""
Attribute Extractor - извлечение атрибутов через анализ соседей.
Не regex, а inference из похожих записей.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import json


class AttributeExtractor:
    """
    Извлечение атрибутов через pattern inference из соседей.
    
    Алгоритм:
    1. Анализ соседних записей: что общего?
    2. Infer паттерны атрибутов из соседей
    3. Применить паттерны к входному тексту
    4. Вернуть атрибуты с confidence
    """
    
    def __init__(self):
        # Кэш инференсов
        self._pattern_cache: Dict[str, Dict] = {}
        
        # Универсальные паттерны (fallback)
        self.universal_patterns = {
            'dimensions': {
                'regex': r'(\d+[\.,]?\d*)\s*[xхX\*]\s*(\d+[\.,]?\d*)(?:\s*[xхX\*]\s*(\d+[\.,]?\d*))?',
                'extractor': self._extract_dimensions
            },
            'diameter': {
                'regex': r'[dDдД][\s\.]*(\d+[\.,]?\d*)',
                'extractor': self._extract_diameter
            },
            'marka': {
                'regex': r'\b([A-ZА-Я]\d{3,4}[A-ZА-Я]?)\b',
                'extractor': self._extract_marka
            },
            'gost': {
                'regex': r'[Гг][ОоOo][СсCc][Тт]\s*[RР]?\s*(\d+(?:[-–]\d+)?)',
                'extractor': self._extract_gost
            },
            'color': {
                'regex': r'(серый|белый|черный|красный|синий|зеленый|желтый|коричневый)',
                'extractor': self._extract_color,
                'normalize': True
            }
        }
    
    def extract(
        self,
        name: str,
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Any]:
        """
        Извлечь атрибуты из названия через анализ соседей.
        
        Args:
            name: Наименование для извлечения
            neighbors: Список похожих записей (VectorRecord)
            detected_class: Определённый класс
            
        Returns:
            Словарь атрибутов
        """
        attributes = {}
        
        # Шаг 1: Infer паттерны из соседей
        class_patterns = self._infer_patterns_from_neighbors(neighbors, detected_class)
        
        # Шаг 2: Применить выведенные паттерны
        for attr_name, pattern_info in class_patterns.items():
            value = pattern_info['extractor'](name, pattern_info.get('regex'))
            if value:
                attributes[attr_name] = value
        
        # Шаг 3: Универсальные паттерны (fallback)
        for pattern_name, pattern_def in self.universal_patterns.items():
            if pattern_name not in attributes:
                value = pattern_def['extractor'](name, pattern_def.get('regex'))
                if value:
                    # Normalize если нужно
                    if pattern_def.get('normalize') and isinstance(value, str):
                        value = value.lower()
                    attributes[pattern_name] = value
        
        # Шаг 4: Нормализация ключей
        normalized = {}
        for key, value in attributes.items():
            norm_key = self._normalize_key(key)
            normalized[norm_key] = value
        
        return normalized
    
    def _infer_patterns_from_neighbors(
        self,
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Dict]:
        """
        Вывести паттерны атрибутов из анализа соседей.
        
        Смотрим: какие атрибуты есть у соседей того же класса?
        Какие паттерны встречаются часто?
        """
        # Ключ кэша
        cache_key = f"{detected_class}_{len(neighbors)}"
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        
        # Анализ атрибутов соседей
        attribute_frequency: Dict[str, Dict] = defaultdict(lambda: {'count': 0, 'examples': []})
        
        for neighbor in neighbors:
            if neighbor.class_name != detected_class:
                continue
                
            for attr_name, attr_value in neighbor.attributes.items():
                attribute_frequency[attr_name]['count'] += 1
                if len(attribute_frequency[attr_name]['examples']) < 3:
                    attribute_frequency[attr_name]['examples'].append(attr_value)
        
        # Выбрать атрибуты которые встречаются чаще всего (>50%)
        total_neighbors = len([n for n in neighbors if n.class_name == detected_class])
        if total_neighbors == 0:
            total_neighbors = 1
        
        inferred = {}
        for attr_name, freq_info in attribute_frequency.items():
            frequency = freq_info['count'] / total_neighbors
            if frequency >= 0.5:
                # Infer паттерн из примеров
                examples = freq_info['examples']
                pattern = self._infer_pattern_from_examples(examples)
                
                inferred[attr_name] = {
                    'frequency': frequency,
                    'regex': pattern,
                    'extractor': self._create_extractor(attr_name, pattern),
                    'examples': examples
                }
        
        self._pattern_cache[cache_key] = inferred
        return inferred
    
    def _infer_pattern_from_examples(self, examples: List[str]) -> Optional[str]:
        """Infer regex pattern из примеров."""
        if not examples:
            return None
        
        # Анализ common patterns
        # Например: "200", "250", "300" → pattern "\d+"
        # "А500С", "А400С" → pattern "[A-Z]\d{3}[A-Z]"
        
        # Проверка: все числа?
        if all(re.match(r'^\d+[\.,]?\d*$', str(e)) for e in examples):
            return r'(\d+[\.,]?\d*)'
        
        # Проверка: марка стали (А500С)?
        if all(re.match(r'^[A-ZА-Я]\d{3,4}[A-ZА-Я]?$', str(e)) for e in examples):
            return r'\b([A-ZА-Я]\d{3,4}[A-ZА-Я]?)\b'
        
        # Проверка: размеры (200х200)?
        if all(re.match(r'^\d+[xхX]\d+$', str(e)) for e in examples):
            return r'(\d+[xхX]\d+)'
        
        return None
    
    def _create_extractor(self, attr_name: str, pattern: Optional[str]) -> callable:
        """Создать extractor function для атрибута."""
        
        def extractor(text: str, regex: Optional[str]) -> Optional[str]:
            if not regex:
                return None
            
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                return match.group(1)
            return None
        
        return extractor
    
    def _normalize_key(self, key: str) -> str:
        """Нормализовать ключ атрибута."""
        # lowercase, replace spaces with underscore
        normalized = key.lower().strip().replace(' ', '_')
        
        # Common aliases
        aliases = {
            'diameter': 'd',
            'диаметр': 'd',
            'length': 'l',
            'длина': 'l',
            'width': 'w',
            'ширина': 'w',
            'height': 'h',
            'высота': 'h',
            'thickness': 't',
            'толщина': 't',
        }
        
        return aliases.get(normalized, normalized)
    
    # Extractor functions
    
    def _extract_dimensions(self, text: str, regex: str) -> Optional[str]:
        """Извлечь размеры ДхШхВ."""
        match = re.search(regex, text)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return 'x'.join(filter(None, groups))
        return None
    
    def _extract_diameter(self, text: str, regex: str) -> Optional[str]:
        """Извлечь диаметр."""
        match = re.search(regex, text)
        if match:
            return match.group(1)
        return None
    
    def _extract_marka(self, text: str, regex: str) -> Optional[str]:
        """Извлечь марку стали."""
        match = re.search(regex, text)
        if match:
            return match.group(1)
        return None
    
    def _extract_gost(self, text: str, regex: str) -> Optional[str]:
        """Извлечь ГОСТ."""
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_color(self, text: str, regex: str) -> Optional[str]:
        """Извлечь цвет."""
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None