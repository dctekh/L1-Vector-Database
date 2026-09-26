import time
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from config import load_config, hf_login


def execute_g1():
    db_config = load_config(section='postgresql')
    hf_login()
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    register_vector(conn)  # activa la conversión automática numpy <-> vector
    cursor = conn.cursor()

    print("Cargando el modelo (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    cursor.execute("SELECT id, sentence FROM sentence_embeddings_pgvector ORDER BY id;")
    rows = cursor.fetchall()

    generation_times = []
    embedding_times = []
    print("Ejecutando [G1]: Generando e insertando embeddings...")
    for row_id, sentence_text in rows:
        tg0 = time.perf_counter()
        vector = model.encode(sentence_text).astype(np.float32)  
        generation_times.append(time.perf_counter() - tg0)

        t0 = time.perf_counter()

        cursor.execute(
            "UPDATE sentence_embeddings_pgvector SET embedding = %s WHERE id = %s;",
            (vector, row_id)
        )
        conn.commit()
        embedding_times.append(time.perf_counter() - t0)

    print("\n--- Estadísticas de tiempo [G1]: Generación del embedding ---")
    print(f"Mínimo: {np.min(generation_times):.6f} s")
    print(f"Máximo: {np.max(generation_times):.6f} s")
    print(f"Promedio: {np.mean(generation_times):.6f} s")
    print(f"Desviación Estándar: {np.std(generation_times):.6f} s")

    print("\n--- Estadísticas de tiempo [G1]: Almacenamiento del embedding ---")
    print(f"Mínimo: {np.min(embedding_times):.6f} s")
    print(f"Máximo: {np.max(embedding_times):.6f} s")
    print(f"Promedio: {np.mean(embedding_times):.6f} s")
    print(f"Desviación Estándar: {np.std(embedding_times):.6f} s")

    print("\nEjecutando VACUUM ANALYZE sobre sentence_embeddings_pgvector...")
    old_autocommit = conn.autocommit
    conn.autocommit = True
    cursor.execute("VACUUM ANALYZE sentence_embeddings_pgvector;")
    conn.autocommit = old_autocommit

    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_g1()
