#!/usr/bin/env python3
"""Initialize database schema for semantic-classifier-v3"""
import sys
sys.path.insert(0, '/home/clawd/.openclaw/skills/semantic-classifier-v3/scripts')

from core.vector_store import VectorStore

store = VectorStore(password="postgres")
store.init_schema()
print("✅ Database schema initialized successfully")

# Show stats
stats = store.get_stats()
print(f"Total records: {stats['total_records']}")
print(f"Unique classes: {stats['unique_classes']}")
print(f"With embeddings: {stats['with_embeddings']}")
