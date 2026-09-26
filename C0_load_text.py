import time
import numpy as np
import psycopg2
import chromadb
from chromadb.config import Settings
from config import load_config

EMBEDDING_DIM = 384 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sentence_embeddings_chroma"
BATCH_SIZE = 500


def fetch_sentences_from_postgres():
    db_config = load_config(section='postgresql')
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT id, sentence FROM sentence_embeddings ORDER BY id;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def execute_c0():
    print("Leyendo el mismo chunk de datos desde PostgreSQL...")
    rows = fetch_sentences_from_postgres()
    total = len(rows)
    print(f"Frases a cargar en Chroma: {total}")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    placeholder = [0.0] * EMBEDDING_DIM

    batch_times = []
    print(f"Ejecutando [C0]: Guardando texto en Chroma por lotes de {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        ids = [str(row_id) for row_id, _ in chunk]
        documents = [sentence for _, sentence in chunk]
        embeddings = [placeholder] * len(chunk)

        t0 = time.perf_counter()
        collection.add(ids=ids, documents=documents, embeddings=embeddings)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

        print(f"  Lote {i // BATCH_SIZE + 1}: {len(chunk)} docs en {elapsed:.4f} s")

    total_time = sum(batch_times)
    print("\n--- Estadisticas de tiempo [C0]: Texto (por lote) ---")
    print(f"Minimo: {np.min(batch_times):.6f} s")
    print(f"Maximo: {np.max(batch_times):.6f} s")
    print(f"Promedio: {np.mean(batch_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(batch_times):.6f} s")
    print(f"Tiempo total: {total_time:.4f} s | Tiempo medio por documento: {total_time / total:.6f} s")

if __name__ == '__main__':
    execute_c0()