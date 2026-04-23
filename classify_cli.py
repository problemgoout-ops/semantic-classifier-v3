#!/usr/bin/env python3
"""CLI для semantic-classifier-v3 с форматированным выводом для контроля"""
import sys
import json
sys.path.insert(0, '/home/clawd/.openclaw/skills/semantic-classifier-v3/scripts')

from core.semantic_router_v3 import SemanticClassifierV3

def format_result(name, result):
    """Форматирует результат в требуемом формате"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Наименование: {name}")
    lines.append("")
    lines.append(f"Класс: {result.class_name or '❌ Не определен'}")
    lines.append(f"Confidence: {result.confidence:.2%}")
    lines.append("")
    lines.append("Характеристики:")
    
    if result.attributes:
        for attr, value in sorted(result.attributes.items()):
            lines.append(f"  • {attr}: {value}")
    else:
        lines.append("  • Характеристики не извлечены")
    
    if result.validation_notes:
        lines.append("")
        lines.append(f"⚠️ Примечание: {result.validation_notes[0]}")
    
    lines.append("")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print(f"  python3 {sys.argv[0]} 'Наименование для классификации'")
        print()
        print("Примеры:")
        print('  python3 classify_cli.py "Муфта аксиальная Ду32"')
        print('  python3 classify_cli.py "Тройник стальной Ду50"')
        sys.exit(1)
    
    name = sys.argv[1]
    
    print("⏳ Инициализация классификатора...")
    classifier = SemanticClassifierV3()
    
    print("⏳ Классификация...")
    result = classifier.classify(code="TEST", name=name)
    
    print(format_result(name, result))

if __name__ == "__main__":
    main()
