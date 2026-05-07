"""
Attribute Extractor v3 - извлечение атрибутов из наименования.

Принцип: АТРИБУТЫ = ТОЛЬКО ТО ЧТО НА ВХОДЕ. Не добавлять из эталонов.

Три источника для маппинга:
1. Спецификация класса (class_specs) — какие атрибуты возможны
2. Соседние эталоны с атрибутами — какие значения уже стоят в атрибутах
3. Regex-паттерны — извлечение значений из текста наименования

Алгоритм:
1. Найти спецификацию класса → список возможных атрибутов
2. Посмотреть атрибуты у соседей → понять маппинг слов→атрибуты
3. Извлечь значения regex-паттернами
4. Сопоставить извлечённые + оставшиеся слова с атрибутами спецификации
   через маппинг, выученный из соседей
"""

import re
import json
import os
import psycopg2
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class AttributeExtractorV3:
    """
    Извлечение атрибутов из наименования номенклатуры.
    
    Использует:
    - Спецификацию класса (class_specs) для определения возможных атрибутов
    - Атрибуты соседних эталонов для обучения маппинга слов→атрибуты
    - Regex-паттерны для извлечения значений из текста
    """
    
    def __init__(self):
        self._specs_cache: Dict[str, List[str]] = {}
        self._neighbor_map_cache: Dict[str, Dict[str, str]] = {}
        self._load_specs()
        
        # Универсальные паттерны для извлечения
        self.extractors = [
            # ГОСТ/ТУ
            ('гост', r'ГОСТ\s*[RР]?\s*(\d+(?:[-–]\d+)?)', 'ГОСТ {0}'),
            ('ту', r'ТУ\s*(\d+(?:[.\-/]\d+)*)', 'ТУ {0}'),
            
            # Давление
            ('давление_мпа', r'[PР]\s*=\s*(\d+[,\.]?\d*)\s*МПа', '{0} МПа'),
            ('pn', r'PN\s*(\d+)', 'PN{0}'),
            
            # Диаметры
            ('dn', r'DN\s*(\d+)', 'DN{0}'),
            ('du', r'Ду\s*(\d+)', 'Ду{0}'),
            ('d_mm', r'[DdД]\s*(\d+)\s*мм', '{0}'),
            ('d_num', r'[DdД]\s*(\d+)(?!\s*[xх*\d])', '{0}'),
            
            # Длина
            ('l_mm', r'[Ll]\s*(\d+)\s*мм', '{0}'),
            ('l_num', r'[Ll]\s*=\s*(\d+)', '{0}'),  # только L=число, не L5015000 (артикул)
            
            # Толщина стенки
            ('s_mm', r'[SsТт][олщ\.]?\s*(\d+[,\.]?\d*)\s*мм', '{0}'),
            ('s_num', r'[Ss]\s*(\d+[,\.]?\d*)(?!\s*МПа)', '{0}'),
            ('толщина', r'толщ\.?\s*(\d+[,\.]?\d*)\s*мм', '{0}мм'),
            
            # Размеры (с поддержкой десятичных: 32х1,2)
            # Спецформат для труб: DхS (диаметр × толщина стенки) — ловим раньше
            ('размеры_tube_ds', r'(\d+)\s*[xхX*]\s*(\d+(?:[,\.]\d+)?)\s*мм', '{0}x{1}'),
            ('размеры_3d', r'(\d+)\s*[xхX*]\s*(\d+(?:[,\.]\d+)?)\s*[xхX*]\s*(\d+(?:[,\.]\d+)?)', '{0}x{1}x{2}'),
            ('размеры_2d', r'(\d+)\s*[xхX*]\s*(\d+(?:[,\.]\d+)?)', '{0}x{1}'),
            
            # Марки
            ('марка_стали', r'\b(А\d{3,4}[A-ZА-Я]?)\b', '{0}'),
            ('pe', r'\b(ПЭ\d+)\b', '{0}'),
            ('sdr', r'\bSDR\s*(\d+)\b', 'SDR{0}'),
            
            # Покрытие
            ('покрытие_вус', r'\b(ВУС(?:-?Т)?)\b', '{0}'),
            ('покрытие_полимер', r'(с полимерным покрытием)', '{0}'),
            ('покрытие_цинк', r'\b(оцинкованн[а-яё]+)\b', '{0}'),
            ('покрытие_битум', r'\b(битумно-мастичн[а-яё]+)\b', '{0}'),
            ('покрытие_эпоксид', r'\b(с эпоксидным покрытием)\b', '{0}'),
            
            # Материал
            ('материал', r'\b(латун[а-яё]+|чугун[а-яё]+|стальн[а-яё]+|полимерн[а-яё]+|асбестоцементн[а-яё]+|железобетонн[а-яё]+|нержавеющ[а-яё]+|медн[а-яё]+|пластиков[а-яё]+|полиэтиленов[а-яё]+|полипропиленов[а-яё]+)\b', '{0}'),
            
            # Тип соединения
            ('соединение', r'\b(фланцев[а-яё]+|муфтов[а-яё]+|сварн[а-яё]+|резьбов[а-яё]+|компрессионн[а-яё]+|грувлочн[а-яё]+)\b', '{0}'),
            
            # Тип резьбы
            ('резьба_наружная', r'(наружная резьба|с наружной резьбой)', '{0}'),
            ('резьба_внутренняя', r'(внутренняя резьба|с внутренней резьбой)', '{0}'),
            
            # Тип конструкции трубы
            ('конструкция', r'\b(электросварн[а-яё]+|прямошовн[а-яё]+|бесшовн[а-яё]+|спиралешовн[а-яё]+)\b', '{0}'),
            
            # Тип люка
            ('тип_люка', r'\bтип\s*[«""]?\s*([ТЛСМСАМ]+)\s*[»""]?', '{0}'),
            
            # Тип: многословные характеристики (визуально проницаемые, сварной гладкий, и т.д.)
            ('тип_многословный', r'\b(визуально\s+проницаем(?:ые|ая|ый|ое)|сварн(?:ой|ая|ый)\s+гладк(?:ий|ая|ое)|нержавеющ(?:ий|ая|ое)\s+сталь\w*)\b', '{0}'),
            
            # Номер знака / номер ГОСТ (формат X.X.X, X.X и т.д.)
            ('номер_знака', r'\b(\d+(?:\.\d+)+)\b', '{0}'),
            
            # Текст в кавычках/ёлочках — название, назначение, наименование
            ('название_в_кавычках', r'[«""\']([^«""\'»]+)[»""\']', '{0}'),
            
            # Высота (для лестниц и т.п.)
            ('высота_м', r'(\d+[,\.]?\d*)\s*м(?![аПМп])', '{0}'),
            
            # Мощность/напряжение
            ('мощность', r'(\d+[,\.]?\d*)\s*кВт', '{0} кВт'),
            ('напряжение', r'(\d+)\s*[ВV]', '{0}В'),
            
            # Бренд
            ('бренд', r'\b(GROSS|Gross|Barus|One Plus|Warmin|Динарм|Огнеборец|Danfoss|Vitra|EKF|IEK|DKC|KEAZ|Rexant|Fortisflex|Proxima|Cabeus|E-Line|Beward|Eltis|LTV|Hikvision|Varton|Kinco|Ridan|Татполимер|Промрукав|АЗСМ)\b', '{0}'),
            
            # Модель/артикул
            ('модель', r'\b([A-ZА-Я]+[\d-]+[A-ZА-Я\d-]*)\b', '{0}'),
        ]
        
        # Базовый маппинг: извлечённый тип → атрибут спецификации
        self.spec_mapping = {
            'гост': 'Нормативно-технический документ',
            'ту': 'Нормативно-технический документ',
            'давление_мпа': 'Давление номинальное',
            'pn': 'Давление номинальное',
            'dn': 'Диаметр номинальный',
            'du': 'Диаметр номинальный',
            'd_mm': 'Диаметр наружный',
            'd_num': 'Диаметр наружный',
            'l_mm': 'Длина',
            'l_num': 'Длина',
            's_mm': 'Толщина стенки',
            's_num': 'Толщина стенки',
            'толщина': 'Толщина стенки',
            'размеры_tube_ds': 'Диаметр наружный',  # будет разбит на Диаметр + Толщина в _map_to_spec
            'размеры_2d': 'Размер',
            'размеры_3d': 'Размер',
            'марка_стали': 'Марка стали',
            'pe': 'Материал',
            'sdr': 'Дополнительные характеристики',
            'покрытие_вус': 'Покрытие',
            'покрытие_полимер': 'Покрытие',
            'покрытие_цинк': 'Покрытие',
            'покрытие_битум': 'Покрытие',
            'покрытие_эпоксид': 'Покрытие',
            'материал': 'Материал',
            'соединение': 'Тип присоединения',
            'резьба_наружная': 'Тип присоединения',
            'резьба_внутренняя': 'Тип присоединения',
            'конструкция': 'Способ изготовления',
            'тип_люка': 'Тип',
            'тип_многословный': 'Тип',
            'номер_знака': 'Номер',
            'название_в_кавычках': 'Назначение',
            'высота_м': 'Высота',
            'мощность': 'Дополнительные характеристики',
            'напряжение': 'Дополнительные характеристики',
            'бренд': 'Бренд/Производитель',
            'модель': 'Марка/Модель',
        }
        
        # Межклассовый маппинг: слово/фраза → атрибут спецификации
        # Выучен из эталонов: какие атрибуты заполняются для этих значений
        self.global_word_map = {
            # Тип присоединения
            'фланцевый': 'Тип присоединения',
            'фланцевая': 'Тип присоединения',
            'фланцевое': 'Тип присоединения',
            'муфтовый': 'Тип присоединения',
            'муфтовая': 'Тип присоединения',
            'сварной': 'Тип присоединения',
            'резьбовой': 'Тип присоединения',
            'межфланцевый': 'Тип присоединения',
            'компрессионный': 'Тип присоединения',
            'грувлочный': 'Тип присоединения',
            'под муфту': 'Тип присоединения',
            
            # Материал
            'стальной': 'Материал',
            'стальная': 'Материал',
            'стальное': 'Материал',
            'стальные': 'Материал',
            'латунный': 'Материал',
            'латунная': 'Материал',
            'чугунный': 'Вид чугуна',
            'чугунная': 'Вид чугуна',
            'чугунное': 'Вид чугуна',
            
            # Исполнение
            'со второй крышкой': 'Исполнение',
            'с второй крышкой': 'Исполнение',
            'двойная крышка': 'Исполнение',
            'удлиненная': 'Исполнение конструктивное',
            'литая': 'Исполнение конструктивное',
            'магнитный': 'Исполнение конструктивное',
            'магнитная': 'Исполнение конструктивное',
            'безнапорная': 'Исполнение конструктивное',
            'напорная': 'Исполнение конструктивное',
            'под фланец': 'Исполнение',
            
            # Покрытие
            'с полимерным покрытием': 'Покрытие',
            
            # Обозначение
            'БНТ': 'Обозначение',
            
            # Класс метрологический
            'СВХ': 'Класс метрологический',
            'СВГ': 'Класс метрологический',
            'СВМ': 'Класс метрологический',
            
            # Назначение
            'визуально проницаемые': 'Тип',
            'железобетонная': 'Назначение',
            'железобетонный': 'Назначение',
            'железобетонное': 'Назначение',
            'перекрытия': 'Назначение',
            'стеновое': 'Назначение',
            
            # Высота
            # (число + м уже обрабатывается через regex высота_м)
        }
    
    def _load_specs(self):
        """Загрузить спецификации классов из БД."""
        try:
            conn = psycopg2.connect(host='localhost', database='nomenclature_v3', user='postgres', password='')
            cur = conn.cursor()
            cur.execute("SELECT class_name, attributes FROM class_specs")
            for cls, attrs in cur.fetchall():
                self._specs_cache[cls] = attrs if isinstance(attrs, list) else []
            cur.close()
            conn.close()
        except:
            pass
    
    def get_spec(self, class_name: str) -> List[str]:
        """Получить спецификацию атрибутов для класса."""
        if class_name in self._specs_cache:
            return self._specs_cache[class_name]
        
        # Попробовать найти по корню класса
        root = class_name.split()[0] if class_name else ''
        for cls, attrs in self._specs_cache.items():
            if cls.startswith(root):
                return attrs
        
        return []
    
    def _learn_from_neighbors(self, neighbors: List, class_name: str) -> Dict[str, str]:
        """
        Выучить маппинг слов→атрибуты из ВСЕХ эталонов данного класса.
        
        Использует и переданных соседей, и все эталоны из БД.
        Возвращает: {слово_или_фраза: атрибут_спецификации}
        """
        if class_name in self._neighbor_map_cache:
            return self._neighbor_map_cache[class_name]
        
        word_to_attr: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Собираем все примеры: сначала соседи, потом из БД
        all_examples = []
        
        for neighbor in neighbors:
            if hasattr(neighbor, 'attributes') and neighbor.attributes and hasattr(neighbor, 'example_name') and neighbor.example_name:
                all_examples.append((neighbor.example_name, neighbor.attributes))
        
        # Дозагрузить из БД ВСЕ эталоны этого класса с атрибутами
        try:
            conn = psycopg2.connect(host='localhost', database='nomenclature_v3', user='postgres', password='')
            cur = conn.cursor()
            cur.execute("""
                SELECT example_name, attributes FROM etalons 
                WHERE class_name = %s AND attributes != '{}'::jsonb
            """, (class_name,))
            for name, attrs in cur.fetchall():
                if attrs:
                    all_examples.append((name, attrs))
            cur.close()
            conn.close()
        except:
            pass
        
        # Строим маппинг: значение атрибута → имя атрибута
        for name, attrs in all_examples:
            for attr_name, attr_value in attrs.items():
                if not attr_value or str(attr_value) == 'Нет':
                    continue
                
                val_str = str(attr_value).strip()
                
                # Полное значение как фраза
                if val_str and val_str in name:
                    word_to_attr[val_str][attr_name] += 1
                
                # Отдельные слова значения
                for word in val_str.split():
                    word = word.strip('.,;:')
                    if len(word) >= 2 and word in name:
                        word_to_attr[word][attr_name] += 1
        
        # Выбрать наиболее частый атрибут для каждого слова
        mapping = {}
        for word, attr_counts in word_to_attr.items():
            if attr_counts:
                best_attr = max(attr_counts, key=attr_counts.get)
                if attr_counts[best_attr] >= 1:
                    mapping[word] = best_attr
        
        self._neighbor_map_cache[class_name] = mapping
        return mapping

    def extract(
        self,
        name: str,
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Any]:
        """
        Извлечь атрибуты из наименования через LLM.
        
        Алгоритм:
        1. Загрузить эталоны данного класса из БД (с заполненными атрибутами)
        2. Отдать LLM эталоны как примеры + новое название
        3. LLM возвращает атрибуты в том же формате
        
        Fallback: если OpenAI недоступен — использовать старый regex-подход.
        """
        if _OPENAI_AVAILABLE:
            llm_attrs = self._extract_with_llm(name, detected_class)
            if llm_attrs:
                return llm_attrs
        
        # Fallback: старый regex-подход
        return self._extract_fallback(name, neighbors, detected_class)
    
    def _extract_all(self, name: str) -> Dict[str, Tuple[str, Tuple[int,int]]]:
        """Извлечь все значения из текста с их позициями."""
        extracted = {}
        used_spans = []
        
        for ext_name, pattern, fmt in self.extractors:
            for match in re.finditer(pattern, name, re.IGNORECASE):
                span = match.span()
                
                # Проверка на пересечение
                if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in used_spans):
                    continue
                
                groups = match.groups()
                try:
                    value = fmt.format(*groups)
                except (IndexError, ValueError):
                    value = match.group(0)
                
                if not value:
                    continue
                
                base_name = ext_name
                key = ext_name
                if ext_name in extracted:
                    i = 2
                    while f"{ext_name}_{i}" in extracted:
                        i += 1
                    key = f"{ext_name}_{i}"
                
                extracted[key] = (value, span)
                used_spans.append(span)
        
        return extracted
    
    def _map_to_spec(
        self,
        extracted: Dict[str, Tuple[str, Tuple[int,int]]],
        spec: List[str],
        name: str,
        neighbor_map: Dict[str, str],
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Any]:
        """Распределить извлечённые значения по атрибутам спецификации."""
        attributes = {}
        used_spec_attrs = set()
        used_extracted = set()
        
        # === Фаза 1: Точное сопоставление через spec_mapping (с приоритетом global_word_map) ===
        for ext_name, (value, span) in extracted.items():
            base_name = ext_name.rsplit('_', 1)[0] if '_' in ext_name and ext_name[-1].isdigit() else ext_name
            
            # Проверяем global_word_map
            value_lower = value.lower()
            if value_lower in self.global_word_map:
                gwm_attr = self.global_word_map[value_lower]
                if gwm_attr in spec and gwm_attr not in used_spec_attrs:
                    attributes[gwm_attr] = value
                    used_spec_attrs.add(gwm_attr)
                    used_extracted.add(ext_name)
                    continue
                # Если gwm_attr нет в spec — fallback через _find_spec_attr
                elif gwm_attr not in spec:
                    target_attr = self._find_spec_attr(gwm_attr, spec, used_spec_attrs)
                    if target_attr:
                        attributes[target_attr] = value
                        used_spec_attrs.add(target_attr)
                        used_extracted.add(ext_name)
                        continue
            
            spec_attr = self.spec_mapping.get(base_name)
            
            # Спецобработка: DхS для труб — разбить на Диаметр + Толщина
            if base_name == 'размеры_tube_ds' and value and 'x' in value:
                parts = value.split('x')
                if len(parts) == 2:
                    # Первое число = Диаметр, второе = Толщина стенки
                    diam_attr = self._find_spec_attr('Диаметр наружный', spec, used_spec_attrs)
                    if diam_attr:
                        attributes[diam_attr] = parts[0]
                        used_spec_attrs.add(diam_attr)
                    thick_attr = self._find_spec_attr('Толщина стенки', spec, used_spec_attrs)
                    if thick_attr:
                        attributes[thick_attr] = parts[1]
                        used_spec_attrs.add(thick_attr)
                    used_extracted.add(ext_name)
                    continue
            
            if spec_attr:
                # Найти подходящий атрибут в спецификации (с fallback)
                target_attr = self._find_spec_attr(spec_attr, spec, used_spec_attrs)
                if target_attr:
                    attributes[target_attr] = value
                    used_spec_attrs.add(target_attr)
                    used_extracted.add(ext_name)
        
        # === Фаза 2: Маппинг из соседей для оставшихся извлечённых ===
        for ext_name, (value, span) in extracted.items():
            if ext_name in used_extracted:
                continue
            
            base_name = ext_name.rsplit('_', 1)[0] if '_' in ext_name and ext_name[-1].isdigit() else ext_name
            spec_attr = self.spec_mapping.get(base_name)
            
            if spec_attr and spec_attr in spec and spec_attr not in used_spec_attrs:
                attributes[spec_attr] = value
                used_spec_attrs.add(spec_attr)
                used_extracted.add(ext_name)
            elif value in neighbor_map:
                mapped_attr = neighbor_map[value]
                if mapped_attr in spec and mapped_attr not in used_spec_attrs:
                    attributes[mapped_attr] = value
                    used_spec_attrs.add(mapped_attr)
                    used_extracted.add(ext_name)
        
        # === Фаза 3: Распределить оставшиеся слова/фразы из названия ===
        remaining = self._get_remaining_words(name, extracted)
        
        for word, span in remaining:
            matched = False
            
            # 3a: Точное совпадение в global_word_map (самый приоритет)
            word_lower = word.lower()
            if word_lower in self.global_word_map:
                mapped_attr = self.global_word_map[word_lower]
                # Ищем подходящий атрибут в спецификации
                target_attr = self._find_spec_attr(mapped_attr, spec, used_spec_attrs)
                if target_attr:
                    attributes[target_attr] = word
                    used_spec_attrs.add(target_attr)
                    continue
            
            # 3b: Точное совпадение в neighbor_map
            if word in neighbor_map:
                mapped_attr = neighbor_map[word]
                if mapped_attr in spec and mapped_attr not in used_spec_attrs:
                    attributes[mapped_attr] = word
                    used_spec_attrs.add(mapped_attr)
                    continue
            
            # 3c: Частичное совпадение в global_word_map
            for map_word, mapped_attr in self.global_word_map.items():
                if map_word in word_lower or word_lower in map_word:
                    target_attr = self._find_spec_attr(mapped_attr, spec, used_spec_attrs)
                    if target_attr:
                        attributes[target_attr] = word
                        used_spec_attrs.add(target_attr)
                        matched = True
                        break
            if matched:
                continue
            
            # 3d: Частичное совпадение в neighbor_map
            for map_word, mapped_attr in neighbor_map.items():
                if map_word in word or word in map_word:
                    if mapped_attr in spec and mapped_attr not in used_spec_attrs:
                        attributes[mapped_attr] = word
                        used_spec_attrs.add(mapped_attr)
                        matched = True
                        break
            if matched:
                continue
            
            # 3e: Fallback: keyword эвристика
            matched_attr = self._match_word_to_spec(word, spec, used_spec_attrs)
            if matched_attr:
                attributes[matched_attr] = word
                used_spec_attrs.add(matched_attr)
        
        # === Фаза 4: Заполнить из соседей для пустых атрибутов спецификации ===
        # НЕ добавляем — принцип "только из входного текста"
        
        return attributes
    
    def _extract_with_llm(self, name: str, class_name: str) -> Optional[Dict[str, Any]]:
        """Извлечь атрибуты через LLM, используя эталоны класса как примеры."""
        etalons = self._get_etalons_with_attrs(class_name, limit=10)
        if not etalons:
            return None
        
        examples = []
        for ename, eattrs in etalons:
            attrs_obj = eattrs if isinstance(eattrs, dict) else (json.loads(str(eattrs)) if eattrs else {})
            examples.append({
                "название": ename,
                "характеристики": attrs_obj
            })
        
        # Спецификация класса
        spec = self.get_spec(class_name)
        spec_note = ""
        if spec:
            spec_note = f"""
Доступные ключи атрибутов для класса "{class_name}":
{json.dumps(spec, ensure_ascii=False)}

СМОТРИ НА ПРИМЕРЫ НИЖЕ чтобы понять какое слово из названия в какой ключ попадает.
Например: «двустворчатые» → Исполнение конструктивное, «визуально проницаемые» → Тип.

"""

        prompt = f"""Класс материала: "{class_name}"
{spec_note}Вот примеры эталонов этого класса (формат для справки):

{json.dumps(examples, ensure_ascii=False, indent=2)}

---
Новое название: "{name}"

Задача: извлеки характеристики из нового названия. ВНИМАТЕЛЬНО прочитай название и распредели КАЖДОЕ значимое слово в подходящий атрибут из списка выше, ориентируясь на примеры ниже.

ЖЁСТКОЕ ПРАВИЛО — N=N:
- КАЖДОЕ слово-характеристика из названия должно попасть в какой-то атрибут.
- Если слово описывает КОНСТРУКТИВНУЮ ОСОБЕННОСТЬ (одностворчатые, двустворчатые, распашные, въездные и т.д.) → это Исполнение конструктивное.
- Смотри на ПРИМЕРЫ чтобы понять какое слово куда идёт.
- НЕ пропускай слова. Если в названии есть «распашные» — оно ДОЛЖНО быть в ответе.

ВАЖНО:
- Используй ТОЛЬКО те значения, которые ЯВНО есть в названии.
- Не додумывай и не предполагай значения которых нет в тексте.
- Ключи бери из эталонов (не выдумывай новые).
- Если характеристика не указана в названии — не включай её.

ФОРМАТ РАЗМЕРОВ:
- ДВА числа через х (например 302x302) → одна характеристика "Размеры": "302x302"
- ТРИ числа через х (например 400х1200х10) → три характеристики: "Длина", "Ширина", "Толщина" по отдельности

Формат ответа: ТОЛЬКО JSON словарь с характеристиками, без markdown, без вложенных структур.
Пример ответа: {{"Толщина": "80", "Цвет": "серый"}}
Ответ:"""

        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            raw = resp.choices[0].message.content.strip()
            if '```' in raw:
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            # Unwrap if LLM nested everything under 'характеристики' or similar wrapper key
            if isinstance(result, dict):
                for key in list(result.keys()):
                    if key.lower() in ('характеристики', 'attributes', 'атрибуты', 'characteristics'):
                        inner = result[key]
                        if isinstance(inner, dict) and len(inner) > 0:
                            return inner
                # Also unwrap single-key dict
                if len(result) == 1:
                    only_key = list(result.keys())[0]
                    inner = result[only_key]
                    if isinstance(inner, dict) and len(inner) > 1:
                        return inner
            return result
        except Exception:
            return None
    
    def _get_etalons_with_attrs(self, class_name: str, limit: int = 10) -> List:
        """Достать эталоны класса с заполненными атрибутами."""
        try:
            conn = psycopg2.connect(host='localhost', database='nomenclature_v3', user='postgres', password='')
            cur = conn.cursor()
            cur.execute("""
                SELECT example_name, attributes FROM etalons 
                WHERE class_name = %s AND attributes IS NOT NULL AND attributes::text <> '{}'::text
                ORDER BY RANDOM()
                LIMIT %s
            """, (class_name, limit))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows
        except Exception:
            return []
    
    def _extract_fallback(
        self,
        name: str,
        neighbors: List,
        detected_class: str
    ) -> Dict[str, Any]:
        """Fallback: старый regex-подход если LLM недоступен."""
        spec = self.get_spec(detected_class)
        neighbor_map = self._learn_from_neighbors(neighbors, detected_class)
        extracted = self._extract_all(name)
        
        if spec:
            return self._map_to_spec(extracted, spec, name, neighbor_map, neighbors, detected_class)
        else:
            attributes = {}
            for ext_type, (value, span) in extracted.items():
                key = self.spec_mapping.get(ext_type, ext_type)
                attributes[key] = value
            return attributes
    
    def _get_remaining_words(
        self,
        name: str,
        extracted: Dict[str, Tuple[str, Tuple[int,int]]]
    ) -> List[Tuple[str, Tuple[int,int]]]:
        """Получить слова из названия, не вошедшие в извлечённые атрибуты."""
        used_spans = [span for value, span in extracted.values()]
        
        # НЕ расширять used_spans — точное совпадение спанов
        extended_spans = list(used_spans)
        
        remaining = []
        
        # === СНАЧАЛА фразы (приоритетнее отдельных слов) ===
        phrase_patterns = [
            r'со второй крышкой',
            r'с второй крышкой',
            r'с резиновым уплотнением',
            r'резиновым уплотнительным кольцом',
            r'с полимерным покрытием',
            r'с эпоксидным покрытием',
            r'наружная резьба',
            r'внутренняя резьба',
            r'с наружной резьбой',
            r'с внутренней резьбой',
            r'под фланец',
            r'под муфту',
            r'безнапорная',
            r'напорная',
        ]
        
        for phrase_pat in phrase_patterns:
            for match in re.finditer(phrase_pat, name, re.IGNORECASE):
                phrase = match.group(0)
                span = match.span()
                if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in extended_spans):
                    continue
                remaining.append((phrase, span))
                extended_spans.append(span)
        
        # === ПОТОМ отдельные слова (не входящие в фразы) ===
        for match in re.finditer(r'[\w\-/]+(?:\.[\w\-/]+)*|[«"][^»"]+[»"]', name):
            word = match.group(0).strip('«»""')
            span = match.span()
            
            # Пропустить если пересекается с извлечёнными ИЛИ фразами
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in extended_spans):
                continue
            
            # Пропустить служебные
            if word.lower() in {'и', 'с', 'по', 'для', 'из', 'на', 'в', 'под', 'исп', 'х', 'со', 'без'}:
                continue
            
            # Пропустить односимвольные цифры (мусор от размеров типа "Ду32 х 1")
            if re.match(r'^\d$', word):
                continue
            
            # Пропустить первые 1-2 слова (класс)
            if span[0] < 3:
                continue
            
            remaining.append((word, span))
        
        return remaining
    
    def _match_word_to_spec(
        self,
        word: str,
        spec: List[str],
        used: set
    ) -> Optional[str]:
        """
        Сопоставить слово с атрибутом спецификации по семантике.
        """
        word_lower = word.lower()
        
        # Расширенный маппинг ключевых слов → атрибуты спецификации
        keyword_map = {
            'аксиальн': 'Тип',
            'прямая': 'Тип',
            'угловая': 'Тип',
            'переходн': 'Тип',
            'комбинированн': 'Тип',
            'равнопроходн': 'Тип',
            'крыльчаты': 'Тип',
            'стеновое': 'Тип',
            'перекрытия': 'Тип',
            'удлиненн': 'Исполнение конструктивное',
            'магнитн': 'Исполнение конструктивное',
            'бнт': 'Тип',
            'сдвоенн': 'Исполнение конструктивное',
            'грязевик': 'Дополнительные характеристики',
            'абонентск': 'Дополнительные характеристики',
            'прямошовн': 'Способ изготовления',
            'электросварн': 'Способ изготовления',
            'бесшовн': 'Способ изготовления',
            'спиралешовн': 'Способ изготовления',
        }
        
        for keyword, spec_attr in keyword_map.items():
            if keyword in word_lower and spec_attr in spec and spec_attr not in used:
                return spec_attr
        
        return None
    
    def _find_spec_attr(self, target: str, spec: List[str], used: set) -> Optional[str]:
        """
        Найти атрибут в спецификации, соответствующий целевому.
        
        Если точного совпадения нет — найти ближайший по семантике.
        Например: target='Вид чугуна' → найдёт 'Вид чугуна' если есть, иначе 'Материал'.
        """
        # Точное совпадение
        if target in spec and target not in used:
            return target
        
        # Синонимы и фоллбэки
        fallbacks = {
            'Вид чугуна': ['Материал', 'Марка материала', 'Исполнение конструктивное'],
            'Тип присоединения': ['Исполнение конструктивное', 'Тип', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Материал': ['Марка материала', 'Марка стали', 'Исполнение конструктивное', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Исполнение': ['Исполнение конструктивное', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Обозначение': ['Обозначение/Артикул', 'Марка/Модель', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Номер': ['Тип', 'Типоразмер', 'Обозначение', 'Обозначение/Артикул', 'Марка/Модель'],
            'Покрытие': ['Назначение', 'Исполнение конструктивное', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Способ изготовления': ['Исполнение конструктивное', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Диаметр наружный': ['Диаметр', 'Диаметр номинальный', 'Диаметр условного прохода', 'Размер присоединения', 'Дополнительные характеристики'],
            'Толщина стенки': ['Толщина', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Размер': ['Ширина', 'Длина', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Диаметр номинальный': ['Диаметр условного прохода', 'Диаметр', 'Размер присоединения', 'Дополнительные характеристики', 'Дополнительные технические характеристики'],
            'Диаметр': ['Диаметр номинальный', 'Диаметр условного прохода', 'Диаметр наружный', 'Размер присоединения', 'Дополнительные характеристики'],
        }
        
        # Partial match: если target содержит ключевые слова из spec
        # НО: исключаем "Тип присоединения" из partial match на "Тип" — слишком общее
        if target != 'Тип присоединения':
            target_lower = target.lower()
            for attr in spec:
                if attr not in used:
                    # «Диаметр» в target + «Диаметр» в attr
                    key_words = ['диаметр', 'материал', 'покрытие', 'исполнение', 'тип']
                    for kw in key_words:
                        if kw in target_lower and kw in attr.lower() and attr not in used:
                            return attr
        
        for fb in fallbacks.get(target, []):
            if fb in spec and fb not in used:
                return fb
        
        # Последний fallback — Дополнительные характеристики
        for attr in spec:
            if 'Дополнительн' in attr and attr not in used:
                return attr
        
        return None