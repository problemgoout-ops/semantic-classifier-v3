-- Semantic Classifier v3 Database Schema
-- PostgreSQL + pgvector

-- Включить расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица эталонов (956 записей из Excel)
CREATE TABLE IF NOT EXISTS etalons (
    id SERIAL PRIMARY KEY,
    class_name TEXT NOT NULL,
    example_name TEXT NOT NULL,
    unit TEXT,
    status TEXT DEFAULT 'Эталон',
    attributes JSONB DEFAULT '{}',
    attribute_specs JSONB DEFAULT '{}',
    embedding VECTOR(768),
    excel_row INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW индекс для semantic search
CREATE INDEX IF NOT EXISTS idx_etalons_embedding 
ON etalons USING hnsw (embedding vector_cosine_ops);

-- Индекс для поиска по классу
CREATE INDEX IF NOT EXISTS idx_etalons_class 
ON etalons (class_name);

-- Статистика
CREATE TABLE IF NOT EXISTS etalon_stats (
    total_count INTEGER DEFAULT 0,
    unique_classes INTEGER DEFAULT 0,
    last_migration TIMESTAMP
);

INSERT INTO etalon_stats (total_count, unique_classes, last_migration)
VALUES (0, 0, NOW())
ON CONFLICT DO NOTHING;

-- Функция для поиска похожих
CREATE OR REPLACE FUNCTION find_similar_etalons(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.6,
    max_results INT DEFAULT 5
)
RETURNS TABLE (
    id INT,
    class_name TEXT,
    example_name TEXT,
    attributes JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.class_name,
        e.example_name,
        e.attributes,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM etalons e
    WHERE e.embedding IS NOT NULL
      AND 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Представление для статистики
CREATE OR REPLACE VIEW v_etalon_stats AS
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT class_name) as unique_classes,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
    MAX(created_at) as last_update
FROM etalons;