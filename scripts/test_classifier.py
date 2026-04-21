#!/usr/bin/env python3
"""
Тестовый скрипт для semantic-classifier-v3.
Проверяет работу классификатора после миграции.
"""

import sys
sys.path.insert(0, '/home/clawd/.openclaw/skills/semantic-classifier-v3/scripts')

from core.semantic_router_v3 import SemanticClassifierV3

print("🧪 Тестирование semantic-classifier-v3")
print("=" * 50)

# Инициализация
classifier = SemanticClassifierV3()

# Тестовые примеры
test_items = [
    ("000001", "Муфта аксиальная Ду32"),
    ("000002", "Адаптер грувлочный Динарм AFG060 DN50"),
    ("000003", "Арматура А500С D12"),
]

for code, name in test_items:
    print(f"\n📋 Тест: {name}")
    print("-" * 40)
    
    result = classifier.classify(code=code, name=name)
    
    print(f"Класс: {result.cls or '❌ Не определен'}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Source: {result.source}")
    
    if result.attributes:
        print(f"Атрибуты ({len(result.attributes)}):")
        for k, v in list(result.attributes.items())[:5]:
            print(f"  • {k}: {v}")
    else:
        print("Атрибуты: не извлечены")
    
    if result.validation_notes:
        print(f"⚠️ {result.validation_notes[0]}")

print("\n" + "=" * 50)
print("✅ Тест завершен")

# Статистика БД
from core.vector_store import VectorStore
store = VectorStore()
stats = store.get_stats()
print(f"\n📊 Статистика БД:")
for k, v in stats.items():
    print(f"  {k}: {v}")