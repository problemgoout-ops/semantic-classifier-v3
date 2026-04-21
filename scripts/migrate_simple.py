#!/usr/bin/env python3
"""
Миграция эталонов из Excel в PostgreSQL.
Упрощённая версия: случайные векторы вместо реальных embeddings.
"""

import sys
from pathlib import Path
from typing import List, Dict
import random

sys.path.insert(0, str(Path(__file__).parent))
from core.vector_store import VectorStore
from migrate_excel import EtalonParser


def generate_simple_embedding(text: str) -> List[float]:
    """Генерация простого embedding для демонстрации."""
    random.seed(text)
    return [random.uniform(-1, 1) for _ in range(768)]


def migrate_etalons(excel_path: Path, db_config: Dict):
    """Основная функция миграции."""
    print(f"🚀 Начинаю миграцию из {excel_path}")
    
    # Парсинг
    parser = EtalonParser(excel_path)
    etalons = parser.parse_all()
    print(f"📊 Найдено {len(etalons)} эталонов")
    
    # Подключение к БД
    vector_store = VectorStore(
        host=db_config.get('host', 'localhost'),
        database=db_config.get('database', 'nomenclature_v3'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', '')
    )
    
    # Инициализация схемы
    print("🗄️ Проверка схемы...")
    vector_store.init_schema()
    
    # Миграция
    print("💾 Миграция в PostgreSQL...")
    for i, etalon in enumerate(etalons, 1):
        if i % 100 == 0:
            print(f"  Прогресс: {i}/{len(etalons)} ({i*100//len(etalons)}%)")
        
        # Генерация embedding
        embedding = generate_simple_embedding(etalon['example_name'])
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Миграция эталонов в PostgreSQL')
    parser.add_argument('--excel', type=Path, default=Path('data/etalons.xlsx'),
                       help='Путь к Excel файлу')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-name', default='nomenclature_v3')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-password', default='')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    migrate_etalons(args.excel, db_config)