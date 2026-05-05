---
name: semantic-classifier-v3
description: "Векторный классификатор МДМ v3. Только классификация и извлечение атрибутов — без поиска в МДМ."
triggers:
  - "semantic-classify"
  - "классифицируй v3"
  - "semantic classifier"
version: "3.2.0"
---

# Semantic Classifier v3

## Назначение

Скилл выполняет ТОЛЬКО классификацию номенклатуры:
- Определяет класс материала по наименованию (semantic search + majority voting)
- Извлекает атрибуты (только из входного текста, НЕ добавляет из эталонов)
- Работает на основе 6800+ эталонных примеров в PostgreSQL+pgvector

## Архитектура

```
Входное наименование
       ↓
OpenAI text-embedding-3-small (1536d)
       ↓
PostgreSQL + pgvector HNSW (Top-20 ближайших)
       ↓
Majority voting по корню класса (группировка)
       ↓
Класс + Confidence + Атрибуты
```

## Дообучение

### Правила обратной связи:
- ✅ Пользователь подтвердил → `confirm_and_learn(name, class_name)` — записать пример в базу
- ✏️ Пользователь поправил → `confirm_and_learn(name, corrected_class)` — записать поправленный вариант
- ❌ Пользователь ничего не сказал → НЕ записывать
- ⚠️ Атрибуты = только то что на входе. Не добавлять из эталонов!

### API дообучения:
```python
classifier.confirm_and_learn(name="Муфта аксиальная Ду32", class_name="Муфта аксиальная")
```

## Что НЕ делает

- НЕ ищет в справочнике МДМ (это mdm-nomenclature)
- НЕ возвращает коды МДМ
- НЕ добавляет атрибуты из эталонов

## Конфигурация

| Параметр | Значение |
|----------|----------|
| Embedding модель | OpenAI text-embedding-3-small |
| Размерность | 1536d |
| База данных | nomenclature_v3 |
| Таблица | etalons |
| Индекс | HNSW (vector_cosine_ops) |
| Top-K | 20 соседей |
| Fallback threshold | 0.3 |

## Файлы

- `scripts/core/semantic_router_v3.py` — основной классификатор + дообучение
- `scripts/core/attribute_extractor_v3.py` — извлечение атрибутов
- `scripts/core/vector_store.py` — PostgreSQL+pgvector хранилище (1536d)
- `scripts/integration/wrapper.py` — обёртка для вызова извне
- `scripts/migrate_excel.py` — миграция из Excel (OpenAI embeddings)
- `data/etalons.xlsx` — эталонные примеры
