"""
Vector Store - PostgreSQL + pgvector адаптер.
Хранит векторные представления классов с метаданными.
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
    name: str
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
    PostgreSQL + pgvector store for classification embeddings.
    
    Schema:
        CREATE TABLE classifications (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            attributes JSONB,
            embedding VECTOR(768)
        );
        
        CREATE INDEX idx_classifications_embedding ON classifications 
        USING hnsw (embedding vector_cosine_ops);
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "nomenclature_kb",
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
        """Инициализировать схему БД."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Создать расширение pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Создать таблицу
        cur.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                attributes JSONB DEFAULT '{}',
                embedding VECTOR(768)
            );
        """)
        
        # Создать HNSW индекс
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_classifications_embedding 
            ON classifications USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Индексы для фильтрации
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_classifications_class 
            ON classifications (class_name);
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
        """Добавить новую запись."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Преобразовать embedding в строку для PostgreSQL
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        cur.execute("""
            INSERT INTO classifications (code, name, class_name, attributes, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (code) DO UPDATE SET
                name = EXCLUDED.name,
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
            embedding: Вектор запроса
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
                SELECT code, name, class_name, attributes, embedding,
                       1 - (embedding <=> %s::vector) as similarity
                FROM etalons
                WHERE class_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, class_filter, embedding_str, k))
        else:
            # Поиск без фильтра
            cur.execute("""
                SELECT code, name, class_name, attributes, embedding,
                       1 - (embedding <=> %s::vector) as similarity
                FROM etalons
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, k))
        
        results = []
        for row in cur.fetchall():
            code, name, class_name, attrs, emb, similarity = row
            results.append(VectorRecord(
                code=code,
                name=name,
                class_name=class_name,
                attributes=attrs if isinstance(attrs, dict) else json.loads(attrs),
                embedding=emb,
                similarity=float(similarity)
            ))
        
        cur.close()
        return results
    
    def get_by_code(self, code: str) -> Optional[VectorRecord]:
        """Получить запись по коду."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT code, name, class_name, attributes, embedding
            FROM classifications
            WHERE code = %s;
        """, (code,))
        
        row = cur.fetchone()
        cur.close()
        
        if row:
            code, name, class_name, attrs, emb = row
            return VectorRecord(
                code=code,
                name=name,
                class_name=class_name,
                attributes=attrs if isinstance(attrs, dict) else json.loads(attrs),
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
        cur.execute("SELECT COUNT(*) FROM classifications;")
        count = cur.fetchone()[0]
        cur.close()
        return count
    
    def get_stats(self) -> Dict:
        """Статистика по базе."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(DISTINCT class_name) FROM classifications;")
        class_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM classifications WHERE embedding IS NOT NULL;")
        with_embedding = cur.fetchone()[0]
        
        cur.close()
        
        return {
            'total_records': self.count(),
            'unique_classes': class_count,
            'with_embeddings': with_embedding
        }
    
    def bulk_import_from_jsonl(
        self,
        jsonl_path: Path,
        embedding_fn = None
    ):
        """
        Импорт из JSONL файла.
        
        Args:
            jsonl_path: Путь к файлу с записями
            embedding_fn: Функция для генерации embeddings (name -> vector)
        """
        import ollama
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        batch = []
        batch_size = 100
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                code = record.get('code', '')
                name = record.get('name', '')
                class_name = record.get('class', '')
                attributes = {k: v for k, v in record.items() 
                             if k not in ['code', 'name', 'class']}
                
                # Генерировать embedding
                if embedding_fn:
                    embedding = embedding_fn(name)
                else:
                    # Fallback: случайный вектор
                    import random
                    random.seed(name)
                    embedding = [random.uniform(-1, 1) for _ in range(768)]
                
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                batch.append((code, name, class_name, Json(attributes), embedding_str))
                
                if len(batch) >= batch_size:
                    cur.executemany("""
                        INSERT INTO classifications (code, name, class_name, attributes, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (code) DO UPDATE SET
                            name = EXCLUDED.name,
                            class_name = EXCLUDED.class_name,
                            attributes = EXCLUDED.attributes,
                            embedding = EXCLUDED.embedding;
                    """, batch)
                    conn.commit()
                    batch = []
        
        # Остаток
        if batch:
            cur.executemany("""
                INSERT INTO classifications (code, name, class_name, attributes, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (code) DO UPDATE SET
                    name = EXCLUDED.name,
                    class_name = EXCLUDED.class_name,
                    attributes = EXCLUDED.attributes,
                    embedding = EXCLUDED.embedding;
            """, batch)
            conn.commit()
        
        cur.close()
        print(f"Imported {self.count()} records")