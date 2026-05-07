#!/usr/bin/env python3
"""Переиндексация всех эталонов через OpenAI text-embedding-3-small."""
import sys, os, time
from pathlib import Path
import psycopg2
from openai import OpenAI

API_KEY = os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    print('❌ OPENAI_API_KEY не установлен', file=sys.stderr)
    sys.exit(1)
client = OpenAI(api_key=API_KEY)

conn = psycopg2.connect(host='localhost', database='nomenclature_v3', user='postgres', password='')
cur = conn.cursor()
cur.execute('SELECT id, example_name FROM etalons ORDER BY id')
rows = cur.fetchall()
total = len(rows)
print(f'Переиндексация {total} записей через OpenAI text-embedding-3-small...', flush=True)

errors = 0
for idx, (row_id, name) in enumerate(rows, 1):
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[name])
        emb = resp.data[0].embedding
        emb_str = '[' + ','.join(map(str, emb)) + ']'
        cur.execute('UPDATE etalons SET embedding = %s::vector WHERE id = %s', (emb_str, row_id))
        conn.commit()
    except Exception as e:
        errors += 1
        conn.rollback()
        print(f'  ❌ #{idx} id={row_id}: {e}')
        if errors > 10:
            print('Слишком много ошибок, останавливаюсь')
            break
    
    if idx % 100 == 0:
        print(f'  Прогресс: {idx}/{total} ({idx*100//total}%)', flush=True)
        sys.stdout.flush()
conn.commit()
cur.close()
conn.close()
print(f'\n✅ Готово: {total} записей, ошибок: {errors}')
