#!/usr/bin/env python3
"""Миграция эталонов с реальными Ollama embeddings."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from core.vector_store import VectorStore
from migrate_excel import EtalonParser

def get_ollama_embedding(text: str) -> list:
    """Получить embedding через Ollama."""
    import ollama
    try:
        response = ollama.embed(model='mxbai-embed-large', input=text)
        return response['embeddings'][0]
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return None

def migrate_with_ollama():
    """Основная функция миграции."""
    excel_path = Path('/home/clawd/.openclaw/skills/semantic-classifier-v3/data/etalons.xlsx')
    
    print(f"🚀 Начинаю миграцию из {excel_path}")
    
    # Парсинг
    parser = EtalonParser(excel_path)
    etalons = parser.parse_all()
    print(f"📊 Найдено {len(etalons)} эталонов")
    
    # Подключение к БД
    vector_store = VectorStore(password="postgres")
    
    # Инициализация схемы
    print("🗄️ Проверка схемы...")
    vector_store.init_schema()
    
    # Очистка существующих данных для пересоздания с реальными embeddings
    print("🧹 Очистка старых данных...")
    conn = vector_store._get_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE classifications;")
    conn.commit()
    cur.close()
    
    # Миграция с Ollama
    print("💾 Миграция в PostgreSQL с Ollama embeddings...")
    for i, etalon in enumerate(etalons, 1):
        if i % 100 == 0:
            print(f"  Прогресс: {i}/{len(etalons)} ({i*100//len(etalons)}%)")
        
        # Получить embedding через Ollama
        embedding = get_ollama_embedding(etalon['example_name'])
        if embedding is None:
            print(f"  ⚠️ Пропуск {i}: не удалось получить embedding")
            continue
        
        # Сохранение
        try:
            vector_store.add(
                code=f"etalon_{i:04d}",
                name=etalon['example_name'],
                class_name=etalon['class_name'],
                attributes=etalon['attributes'],
                embedding=embedding
            )
        except Exception as e:
            print(f"  ⚠️ Ошибка {i}: {e}")
    
    # Итоги
    print("✅ Миграция завершена!")
    stats = vector_store.get_stats()
    print(f"\nСтатистика:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

if __name__ == '__main__':
    migrate_with_ollama()
