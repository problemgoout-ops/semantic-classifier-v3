"""
Attribute Extractor v3 - improved pattern extraction from neighbors.

Improvements over v2:
- Pattern inference from neighbor examples instead of regex
- Better N=N compliance
- Attribute normalization
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json


class AttributeExtractorV3:
    """
    Extract attributes from name using patterns learned from neighbors.
    """
    
    def __init__(self):
        self._pattern_cache: Dict[str, Dict] = {}
        
        # Universal patterns (fallback when no neighbors match)
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
                'extractor': self._extract_color
            }
        }
    
    def extract(
        self,
        name: str,
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Any]:
        """
        Extract attributes from name using neighbor examples.
        
        Args:
            name: Item name to process
            neighbors: List of similar VectorRecords
            detected_class: Predicted class name
            
        Returns:
            Dictionary of extracted attributes
        """
        attributes = {}
        
        # Step 1: Try to infer patterns from same-class neighbors
        class_neighbors = [n for n in neighbors if n.class_name == detected_class]
        
        if class_neighbors:
            # Infer attribute patterns from neighbors
            class_patterns = self._infer_patterns_from_neighbors(class_neighbors)
            
            # Apply inferred patterns
            for attr_name, pattern_info in class_patterns.items():
                value = self._apply_pattern(name, attr_name, pattern_info)
                if value:
                    attributes[self._normalize_key(attr_name)] = value
        
        # Step 2: Universal patterns for common attributes
        for pattern_name, pattern_def in self.universal_patterns.items():
            if pattern_name not in attributes:
                value = pattern_def['extractor'](name, pattern_def.get('regex'))
                if value:
                    attributes[self._normalize_key(pattern_name)] = value
        
        # Step 3: Extract from name structure
        # e.g., "Адаптер 2А 220В/24В YDS48"
        # → 2А, 220В/24В, YDS48 might be attributes
        if not attributes:
            attributes = self._extract_structural(name, detected_class)
        
        return attributes
    
    def _infer_patterns_from_neighbors(self, neighbors: List) -> Dict[str, Dict]:
        """Infer attribute patterns from neighbor examples."""
        # Group attributes by name
        attr_examples: Dict[str, List] = defaultdict(list)
        
        for neighbor in neighbors:
            if hasattr(neighbor, 'attributes') and neighbor.attributes:
                for attr_name, attr_value in neighbor.attributes.items():
                    attr_examples[attr_name].append(str(attr_value))
        
        # Infer patterns for each attribute
        patterns = {}
        for attr_name, examples in attr_examples.items():
            if len(examples) >= 2:  # Need at least 2 examples
                pattern = self._infer_pattern(examples)
                if pattern:
                    patterns[attr_name] = {
                        'pattern': pattern,
                        'examples': examples[:3]  # Keep first 3 examples
                    }
        
        return patterns
    
    def _infer_pattern(self, examples: List[str]) -> Optional[str]:
        """Infer regex pattern from examples."""
        if not examples:
            return None
        
        # Check if all are numbers
        if all(re.match(r'^\d+[\.,]?\d*$', str(e)) for e in examples):
            return r'(\d+[\.,]?\d*)'
        
        # Check if all are steel grades (А500С pattern)
        if all(re.match(r'^[A-ZА-Я]\d{3,4}[A-ZА-Я]?$', str(e)) for e in examples):
            return r'\b([A-ZА-Я]\d{3,4}[A-ZА-Я]?)\b'
        
        # Check if dimensions (200х200 pattern)
        if all(re.match(r'^\d+[xхX]\d+$', str(e)) for e in examples):
            return r'(\d+[xхX]\d+)'
        
        # Check if voltage/current (220В/24В pattern)
        if all(re.search(r'\d+[ВV]', str(e)) for e in examples):
            return r'(\d+[ВV](?:/\d+[ВV])?)'
        
        return None
    
    def _apply_pattern(self, name: str, attr_name: str, pattern_info: Dict) -> Optional[str]:
        """Apply inferred pattern to extract attribute."""
        pattern = pattern_info.get('pattern')
        if not pattern:
            return None
        
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_structural(self, name: str, detected_class: str) -> Dict[str, str]:
        """Extract attributes based on name structure."""
        attrs = {}
        
        # Extract numbers that could be dimensions, sizes, etc.
        numbers = re.findall(r'\b(\d+[\.,]?\d*)\b', name)
        if numbers:
            # First number often represents diameter/size
            attrs['размер'] = numbers[0]
        
        # Extract model numbers (alphanumeric patterns)
        models = re.findall(r'\b([A-ZА-Я]+[\d-]+[A-ZА-Я\d-]*)\b', name)
        if models:
            attrs['модель'] = models[0]
        
        # Extract GOST references
        gost_match = re.search(r'[Гг][ОоCc][СсCc][Тт]\s*(\d+(?:[-–]\d+)?)', name)
        if gost_match:
            attrs['гост'] = gost_match.group(1)
        
        return attrs
    
    def _normalize_key(self, key: str) -> str:
        """Normalize attribute key."""
        normalized = key.lower().strip().replace(' ', '_')
        
        aliases = {
            'diameter': 'd',
            'диаметр': 'd',
            'length': 'l',
            'длина': 'l',
            'width': 'w',
            'ширина': 'w',
        }
        
        return aliases.get(normalized, normalized)
    
    # Universal extractors
    def _extract_dimensions(self, text: str, regex: str) -> Optional[str]:
        match = re.search(regex, text)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return 'x'.join(filter(None, groups))
        return None
    
    def _extract_diameter(self, text: str, regex: str) -> Optional[str]:
        match = re.search(regex, text)
        return match.group(1) if match else None
    
    def _extract_marka(self, text: str, regex: str) -> Optional[str]:
        match = re.search(regex, text)
        return match.group(1) if match else None
    
    def _extract_gost(self, text: str, regex: str) -> Optional[str]:
        match = re.search(regex, text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_color(self, text: str, regex: str) -> Optional[str]:
        match = re.search(regex, text, re.IGNORECASE)
        return match.group(1).lower() if match else None