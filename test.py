#!/usr/bin/env python3
"""CLI для тестирования semantic-classifier-v3"""
import sys
sys.path.insert(0, '/home/clawd/.openclaw/skills/semantic-classifier-v3/scripts')

from core.semantic_router_v3 import SemanticClassifierV3

def main():
    classifier = SemanticClassifierV3()
    
    print("=" * 60)
    print("Semantic Classifier v3 - Тестирование")
    print("=" * 60)
    print()
    
    # Примеры по умолчанию
    defaults = [
        "Муфта аксиальная Ду32",
        "Адаптер грувлочный Динарм AFG060 DN50",
        "Арматура А500С D12",
        "Решетка вентиляционная фасадная оцинкованная сталь s0,55мм 2400х750мм",
    ]
    
    print("Примеры (введи цифру):")
    for i, name in enumerate(defaults, 1):
        print(f"  {i}. {name}")
    print()
    
    while True:
        user_input = input("Введи номер или свое наименование (или 'exit' для выхода): ").strip()
        
        if user_input.lower() in ('exit', 'quit', 'q'):
            print("\nДо свидания!")
            break
        
        # Выбор примера по номеру
        if user_input.isdigit() and 1 <= int(user_input) <= len(defaults):
            name = defaults[int(user_input) - 1]
        else:
            name = user_input
        
        print(f"\n📋 Тестируем: {name}")
        print("-" * 60)
        
        try:
            result = classifier.classify(code="TEST", name=name)
            
            print(f"✅ Класс: {result.class_name or 'Не определен'}")
            print(f"📊 Confidence: {result.confidence:.2%}")
            
            if result.attributes:
                print(f"🔧 Атрибуты ({len(result.attributes)}):")
                for k, v in list(result.attributes.items())[:10]:
                    print(f"   • {k}: {v}")
            else:
                print("🔧 Атрибуты: не извлечены")
            
            if result.validation_notes:
                print(f"⚠️ Примечания: {result.validation_notes[0]}")
            
            print(f"📚 Источник: {result.source}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print()

if __name__ == "__main__":
    main()
