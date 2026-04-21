#!/bin/bash
echo "🚀 Setting up semantic-classifier-v3..."

# Check PostgreSQL
echo "🔍 Checking PostgreSQL..."
if ! command -v psql &\u003e /dev/null; then
    echo "⚠️ Installing PostgreSQL..."
    apt-get update && apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
fi

# Check pgvector
echo "📦 Checking pgvector..."
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;" 2\u003e/dev/null || {
    echo "⚠️ Installing pgvector..."
    apt-get install -y postgresql-14-pgvector || apt-get install -y postgresql-16-pgvector
}

# Create database
echo "🗄️ Creating database..."
sudo -u postgres psql -c "CREATE DATABASE nomenclature_v3;" 2\u003e/dev/null || echo "Database exists"
sudo -u postgres psql -d nomenclature_v3 -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run schema
SCHEMA="/home/clawd/.openclaw/skills/semantic-classifier-v3/data/schema.sql"
if [ -f "$SCHEMA" ]; then
    echo "📋 Running schema..."
    sudo -u postgres psql -d nomenclature_v3 -f "$SCHEMA"
else
    echo "⚠️ Schema not found"
fi

echo "✅ Setup complete!"
echo "Next: Copy Excel to data/etalons.xlsx and run migrate_excel.py"
