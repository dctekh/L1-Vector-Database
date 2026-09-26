import time
import json
import psycopg2
import numpy as np
from config import load_config

CHUNK_FILE = "bookcorpus_chunk.jsonl"


def load_chunk_from_file(path=CHUNK_FILE):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sentences.append(row["sentence"])
    return sentences


def execute_g0():
    db_config = load_config(section='postgresql')
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cursor.execute("DROP TABLE IF EXISTS sentence_embeddings_pgvector;")
    cursor.execute("""
        CREATE TABLE sentence_embeddings_pgvector (
            id SERIAL PRIMARY KEY,
            sentence TEXT NOT NULL,
            embedding vector(384)
        );
    """)
    conn.commit()

    sentences = load_chunk_from_file()
    print(f"Frases cargadas desde {CHUNK_FILE}: {len(sentences)}")

    text_times = []
    print("Ejecutando [G0]: Guardando texto en PostgreSQL (pgvector)...")
    for s in sentences:
        t0 = time.perf_counter()
        cursor.execute("INSERT INTO sentence_embeddings_pgvector (sentence) VALUES (%s);", (s,))
        conn.commit()
        text_times.append(time.perf_counter() - t0)

    print("\n--- Estadísticas de tiempo [G0]: Texto ---")
    print(f"Mínimo: {np.min(text_times):.6f} s")
    print(f"Máximo: {np.max(text_times):.6f} s")
    print(f"Promedio: {np.mean(text_times):.6f} s")
    print(f"Desviación Estándar: {np.std(text_times):.6f} s")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_g0()
