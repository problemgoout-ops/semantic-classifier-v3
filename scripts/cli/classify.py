#!/usr/bin/env python3
"""
CLI для Semantic Classifier v3.

Использование:
    python3 classify.py --name "Муфта аксиальная Ду32"
    python3 classify.py --input items.json --output results.json
    python3 classify.py --name "..." --verbose
"""

import sys
import json
import argparse
from pathlib import Path

# Добавляем путь к core
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from core import SemanticRouter, ClassificationRequest, VectorStore


def format_result(result, verbose=False) -> str:
    """Форматировать результат для вывода."""
    lines = []
    lines.append("=" * 70)
    lines.append("SEMANTIC CLASSIFIER v3 RESULT")
    lines.append("=" * 70)
    lines.append(f"Наименование: {result.name}")
    lines.append(f"Код: {result.code}")
    lines.append(f"Класс: {result.class_name or '❌ Не определен'}")
    lines.append(f"Confidence: {result.confidence:.2%}")
    lines.append(f"Source: {result.source}")
    
    lines.append(f"\n📋 Атрибуты ({len(result.attributes)}):")
    if result.attributes:
        for k, v in sorted(result.attributes.items()):
            lines.append(f"  • {k}: {v}")
    else:
        lines.append("  (атрибуты не извлечены)")
    
    if verbose and result.matched_neighbors:
        lines.append(f"\n🔍 Соседи (Top-3):")
        for i, neighbor in enumerate(result.matched_neighbors[:3], 1):
            lines.append(f"  {i}. {neighbor.get('name', 'N/A')[:50]}...")
            lines.append(f"     Класс: {neighbor.get('class_name', 'N/A')}, "
                        f"Similarity: {neighbor.get('similarity', 0):.2%}")
    
    if result.validation_notes:
        lines.append(f"\n⚠️ Валидация:")
        for note in result.validation_notes:
            lines.append(f"  • {note}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Semantic Classifier v3 - vector-based MDM classification'
    )
    parser.add_argument('--name', type=str, help='Наименование для классификации')
    parser.add_argument('--code', type=str, default='', help='Код МДМ (опционально)')
    parser.add_argument('--input', type=Path, help='JSON файл с позициями')
    parser.add_argument('--output', type=Path, help='JSON файл для результатов')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--init-db', action='store_true', help='Инициализировать схему БД')
    
    args = parser.parse_args()
    
    # Инициализация
    vector_store = VectorStore()
    router = SemanticRouter(vector_store=vector_store)
    
    # Инициализация БД
    if args.init_db:
        print("Инициализация схемы PostgreSQL...")
        vector_store.init_schema()
        print("✅ Готово")
        return 0
    
    # Проверка аргументов
    if not args.name and not args.input:
        print("Ошибка: укажите --name или --input")
        parser.print_help()
        return 1
    
    # Обработка одного наименования
    if args.name:
        request = ClassificationRequest(name=args.name, code=args.code)
        result = router.classify(request)
        print(format_result(result, verbose=args.verbose))
        return 0
    
    # Обработка файла
    if args.input:
        print(f"Обработка файла: {args.input}")
        
        with open(args.input, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        results = []
        for i, item in enumerate(items, 1):
            print(f"  {i}/{len(items)}: {item.get('name', 'N/A')[:40]}...", end=' ')
            
            request = ClassificationRequest(
                name=item.get('name', ''),
                code=item.get('code', '')
            )
            result = router.classify(request)
            
            results.append({
                'code': result.code,
                'name': result.name,
                'class': result.class_name,
                'confidence': result.confidence,
                'attributes': result.attributes,
                'source': result.source
            })
            
            print(f"✓ {result.class_name or 'FAIL'}")
        
        # Сохранить результаты
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Сохранено: {args.output}")
        else:
            print(f"\n{'='*70}")
            print(json.dumps(results, ensure_ascii=False, indent=2))
        
        return 0
    
    return 0


if __name__ == '__main__':
    sys.exit(main())