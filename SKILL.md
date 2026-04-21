---
name: semantic-classifier-v3
description: "Векторный классификатор МДМ v3. Только классификация и извлечение атрибутов — без поиска в МДМ."
triggers:
  - "semantic-classify"
  - "классифицируй v3"
  - "semantic classifier"
version: "3.0.0"
---

# Semantic Classifier v3

## Назначение

Скилл выполняет ТОЛЬКО классификацию номенклатуры:
- Определяет класс материала по наименованию
- Извлекает атрибуты (диаметр, давление, тип и т.д.)
- Работает на основе 956 эталонных примеров

## Что НЕ делает

- НЕ ищет в справочнике МДМ (это отдельный скилл mdm-nomenclature)
- НЕ возвращает коды МДМ
- НЕ проверяет наличие в базе

## API

```python
from scripts.core.semantic_router_v3 import SemanticClassifierV3

classifier = SemanticClassifierV3()
result = classifier.classify(name="Муфта компрессионная PN16 Ду50")

print(result.cls)           # "Муфта"
print(result.confidence)    # 0.94
print(result.attributes)    # {"тип": "компрессионная", "давление": "PN16", "диаметр": "Ду50"}
```

## Файлы

- `scripts/core/semantic_router_v3.py` — основной классификатор
- `scripts/core/attribute_extractor_v3.py` — извлечение атрибутов
- `scripts/core/vector_store.py` — векторное хранилище
- `data/etalons.xlsx` — эталонные примеры (956 шт.)
- `scripts/cli/classify.py` — CLI для тестирования

## Использование с другими скиллами

```
semantic-classifier-v3.classify(name) → {class, attributes, confidence}
mdm-nomenclature.search(name)         → [коды МДМ, статусы]
```
