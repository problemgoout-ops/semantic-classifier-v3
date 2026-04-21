---
name: semantic-classifier-v3
description: "Векторный классификатор МДМ v3. Обучается на 956 эталонных примерах из Excel. Semantic search + pattern extraction."
triggers:
  - "semantic-classify"
  - "классифицируй v3"
  - "semantic classifier"
version: "3.0.0"
---

# Semantic Classifier v3

## Особенности

- **Semantic search** по 956 эталонам (embeddings)
- **Majority voting** для определения класса
- **Pattern extraction** из соседних примеров
- **N=N validation** с возможностью дообучения

## Источник данных

Excel-файл `etalons.xlsx` с разметкой 1057 классов.

## Архитектура

```
Наименование → Embeddings → Top-K эталонов → Агрегация класса → Извлечение атрибутов → Валидация
```

## API

```python
from semantic_classifier_v3 import SemanticClassifierV3

classifier = SemanticClassifierV3()
result = classifier.classify("Муфта компрессионная PN16 Ду50")

print(result.class_name)   # "Муфта"
print(result.confidence)   # 0.94
print(result.attributes)   # {"тип": "компрессионная", "давление": "PN16", "диаметр": "Ду50"}
```

## Установка

1. Убедиться что PostgreSQL + pgvector установлены
2. Запустить `scripts/setup_database.py`
3. Запустить `scripts/migrate_excel.py`

## Требования

- PostgreSQL 14+
- pgvector
- Python 3.10+
- sentence-transformers или Ollama