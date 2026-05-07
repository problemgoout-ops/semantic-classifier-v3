"""
Wrapper для интеграции semantic-classifier-v3.
Упрощённый интерфейс для вызова из других скиллов и API.
"""

import sys
import json
import os
from pathlib import Path

# Добавить путь к core модулям
SKILL_DIR = Path(__file__).parent.parent.parent  # skills/semantic-classifier-v3
sys.path.insert(0, str(SKILL_DIR / 'scripts'))

from core.semantic_router_v3 import SemanticClassifierV3
from core.vector_store import VectorStore


def classify_item(name: str, code: str = "") -> dict:
    """
    Классифицировать наименование номенклатуры.
    
    Args:
        name: Наименование материала
        code: Код позиции (опционально)
        
    Returns:
        dict с полями:
            - class_name: название класса
            - attributes: извлечённые атрибуты
            - confidence: уверенность (0-1)
            - matched_etalon: лучший похожий эталон
            - similarity: косинусное сходство
            - source: источник (semantic_v3 или fallback)
            - errors: список ошибок (если есть)
            - validation_notes: замечания валидации
    """
    classifier = SemanticClassifierV3()
    result = classifier.classify(code=code, name=name)
    
    return {
        'class_name': result.class_name,
        'attributes': result.attributes,
        'confidence': result.confidence,
        'matched_etalon': result.matched_etalon,
        'similarity': result.similarity,
        'source': result.source,
        'errors': result.errors,
        'validation_notes': result.validation_notes
    }


def correct_class(name: str, class_name: str) -> dict:
    """
    Исправить класс по указанию пользователя.
    НЕ записывает в БД. Использует эталоны правильного класса для извлечения атрибутов.
    """
    classifier = SemanticClassifierV3()
    return classifier.feedback_correct_class(name, class_name)


def correct_attributes(name: str, class_name: str, attributes: dict) -> dict:
    """
    Исправить атрибуты по указанию пользователя.
    НЕ записывает в БД. Проверяет что атрибуты соответствуют спецификации класса.
    """
    classifier = SemanticClassifierV3()
    return classifier.feedback_correct_attributes(name, class_name, attributes)


def confirm_and_learn(name: str, class_name: str, attributes: dict = None, source: str = None) -> dict:
    """
    Подтвердить результат и дообучить классификатор.
    Записывает пример в базу эталонов с атрибутами.
    """
    if attributes is None:
        attributes = {}
    
    # Нормализация: title case для класса
    normalized_class = class_name.strip()
    if normalized_class and normalized_class[0].islower():
        normalized_class = normalized_class[0].upper() + normalized_class[1:]
    
    classifier = SemanticClassifierV3()
    classifier.confirm_and_learn(name=name, class_name=normalized_class, attributes=attributes, source=source)
    return {'status': 'ok', 'class_name': normalized_class}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Semantic Classifier v3 Wrapper')
    parser.add_argument('--name', required=True, help='Наименование для классификации')
    parser.add_argument('--code', default='', help='Код позиции')
    parser.add_argument('--learn', action='store_true', help='Режим дообучения (требует --class)')
    parser.add_argument('--class', dest='class_name', help='Имя класса для дообучения')
    parser.add_argument('--attributes', default='{}', help='Атрибуты в формате JSON')
    parser.add_argument('--source', default=None, help='Источник подтверждения (например, user_confirmed)')
    args = parser.parse_args()
    
    if args.learn:
        if not args.class_name:
            print(json.dumps({'error': 'Для дообучения укажите --class'}, ensure_ascii=False))
            sys.exit(1)
        try:
            attrs = json.loads(args.attributes)
        except json.JSONDecodeError:
            attrs = {}
        result = confirm_and_learn(args.name, args.class_name, attrs)
        print(json.dumps(result, ensure_ascii=False))
    else:
        result = classify_item(args.name, args.code)
        print(json.dumps(result, ensure_ascii=False))
