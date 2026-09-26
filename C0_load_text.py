import time
import numpy as np
import psycopg2
import chromadb
from config import load_config

EMBEDDING_DIM = 384 
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sentence_embeddings_chroma"
PROGRESS_EVERY = 500


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

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    placeholder = [0.0] * EMBEDDING_DIM

    text_times = []
    print("Ejecutando [C0]: Guardando texto (con embedding placeholder) en Chroma...")
    for i, (row_id, sentence) in enumerate(rows, start=1):
        t0 = time.perf_counter()
        collection.add(
            ids=[str(row_id)],
            documents=[sentence],
            embeddings=[placeholder],
        )
        text_times.append(time.perf_counter() - t0)

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  ... {i}/{total} procesadas")

    print("\n--- Estadisticas de tiempo [C0]: Texto ---")
    print(f"Minimo: {np.min(text_times):.6f} s")
    print(f"Maximo: {np.max(text_times):.6f} s")
    print(f"Promedio: {np.mean(text_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(text_times):.6f} s")


if __name__ == '__main__':
    execute_c0()