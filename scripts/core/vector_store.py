"""
Vector Store - PostgreSQL + pgvector адаптер.
Хранит векторные представления классов с метаданными.

Схема БД (nomenclature_v3.etalons):
    - id SERIAL PRIMARY KEY
    - class_name TEXT
    - example_name TEXT (наименование)
    - unit TEXT
    - status TEXT
    - attributes JSONB
    - attribute_specs JSONB
    - embedding VECTOR(1536)
    - excel_row INTEGER
    - created_at TIMESTAMP
    - code TEXT
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import psycopg2
from psycopg2.extras import Json


@dataclass
class VectorRecord:
    """Запись в векторном хранилище."""
    code: str
    name: str  # соответствует example_name в БД
    class_name: str
    attributes: Dict[str, Any]
    embedding: List[float]
    similarity: float = 0.0  # Заполняется при поиске
    
    def to_dict(self) -> Dict:
        """Преобразовать в словарь для сериализации."""
        return {
            'code': self.code,
            'name': self.name,
            'class_name': self.class_name,
            'attributes': self.attributes,
            'similarity': self.similarity
        }


class VectorStore:
    """
    PostgreSQL + pgvector store для эмбеддингов номенклатуры.
    Использует таблицу etalons (1017 классов, 2560d vectors для qwen3).
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "nomenclature_v3",
        user: str = "postgres",
        password: str = ""
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None
        
    def _get_connection(self):
        """Получить соединение с БД."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        return self._conn
    
    def init_schema(self):
        """Инициализировать схему БД (существующая таблица etalons)."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Создать расширение pgvector если нужно
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Создать индексы если их нет
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_etalons_embedding 
            ON etalons USING hnsw (embedding vector_cosine_ops);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_etalons_class 
            ON etalons (class_name);
        """)
        
        conn.commit()
        cur.close()
    
    def add(
        self,
        code: str,
        name: str,
        class_name: str,
        attributes: Dict,
        embedding: List[float]
    ):
        """Добавить новую запись (для обратной совместимости с migrate_excel.py)."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Преобразовать embedding в строку для PostgreSQL
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        cur.execute("""
            INSERT INTO etalons (code, example_name, class_name, attributes, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (code) DO UPDATE SET
                example_name = EXCLUDED.example_name,
                class_name = EXCLUDED.class_name,
                attributes = EXCLUDED.attributes,
                embedding = EXCLUDED.embedding;
        """, (code, name, class_name, Json(attributes), embedding_str))
        
        conn.commit()
        cur.close()
    
    def search(
        self,
        embedding: List[float],
        k: int = 5,
        class_filter: Optional[str] = None
    ) -> List[VectorRecord]:
        """
        Поиск k ближайших соседей.
        
        Args:
            embedding: Вектор запроса (2000d для qwen3)
            k: Количество результатов
            class_filter: Опциональная фильтрация по классу
            
        Returns:
            Список VectorRecord с заполненным similarity
        """
        conn = self._get_connection()
        cur = conn.cursor()
        
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        if class_filter:
            # Поиск с фильтром по классу
            cur.execute("""
                SELECT code, example_name, class_name, attributes, embedding,
                       1 - (embedding <=> %s::vector) as similarity
                FROM etalons
                WHERE class_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, class_filter, embedding_str, k))
        else:
            # Поиск без фильтра
            cur.execute("""
                SELECT code, example_name, class_name, attributes, embedding,
                       1 - (embedding <=> %s::vector) as similarity
                FROM etalons
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, k))
        
        results = []
        for row in cur.fetchall():
            code, example_name, class_name, attrs, emb, similarity = row
            results.append(VectorRecord(
                code=code or '',
                name=example_name or '',
                class_name=class_name or '',
                attributes=attrs if isinstance(attrs, dict) else (json.loads(attrs) if attrs else {}),
                embedding=emb,
                similarity=float(similarity or 0)
            ))
        
        cur.close()
        return results
    
    def get_by_code(self, code: str) -> Optional[VectorRecord]:
        """Получить запись по коду."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT code, example_name, class_name, attributes, embedding
            FROM etalons
            WHERE code = %s;
        """, (code,))
        
        row = cur.fetchone()
        cur.close()
        
        if row:
            code, example_name, class_name, attrs, emb = row
            return VectorRecord(
                code=code or '',
                name=example_name or '',
                class_name=class_name or '',
                attributes=attrs if isinstance(attrs, dict) else (json.loads(attrs) if attrs else {}),
                embedding=emb
            )
        return None
    
    def get_neighbors(
        self,
        code: str,
        k: int = 5
    ) -> List[VectorRecord]:
        """Найти соседей для существующей записи."""
        record = self.get_by_code(code)
        if record and record.embedding:
            return self.search(record.embedding, k=k)
        return []
    
    def count(self) -> int:
        """Общее количество записей."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM etalons;")
        count = cur.fetchone()[0]
        cur.close()
        return count
    
    def get_stats(self) -> Dict:
        """Статистика по базе."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(DISTINCT class_name) FROM etalons;")
        class_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM etalons WHERE embedding IS NOT NULL;")
        with_embedding = cur.fetchone()[0]
        
        cur.close()
        
        return {
            'total_records': self.count(),
            'unique_classes': class_count,
            'with_embeddings': with_embedding
        }
