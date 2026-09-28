import time
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from config import load_config, hf_login

def execute_p1():
    db_config = load_config(section='postgresql')
    hf_login()  
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    cursor = conn.cursor()

    print("Cargando el modelo (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    cursor.execute("SELECT id, sentence FROM sentence_embeddings ORDER BY id;")
    rows = cursor.fetchall()

    generation_times = []  
    embedding_times = []   
    print("Ejecutando [P1]: Generando e insertando embeddings...")
    for row_id, sentence_text in rows:
        # (a) coste de generar el embedding (modelo de ML, en Python)
        tg0 = time.time()
        vector = model.encode(sentence_text).tolist()
        generation_times.append(time.time() - tg0)

        # (b) coste de almacenarlo en Postgres (lo que pide el enunciado)
        t0 = time.time()
        cursor.execute(
            "UPDATE sentence_embeddings SET embedding = %s WHERE id = %s;",
            (vector, row_id)
        )
        conn.commit()
        embedding_times.append(time.time() - t0)

    print("\n--- Estadísticas de tiempo [P1]: Generación del embedding (extra) ---")
    print(f"Mínimo: {np.min(generation_times):.6f} s")
    print(f"Máximo: {np.max(generation_times):.6f} s")
    print(f"Promedio: {np.mean(generation_times):.6f} s")
    print(f"Desviación Estándar: {np.std(generation_times):.6f} s")

    print("\n--- Estadísticas de tiempo [P1]: Almacenamiento del embedding ---")
    print(f"Mínimo: {np.min(embedding_times):.6f} s")
    print(f"Máximo: {np.max(embedding_times):.6f} s")
    print(f"Promedio: {np.mean(embedding_times):.6f} s")
    print(f"Desviación Estándar: {np.std(embedding_times):.6f} s")

    print("\nEjecutando VACUUM ANALYZE sobre sentence_embeddings...")
    old_autocommit = conn.autocommit
    conn.autocommit = True
    cursor.execute("VACUUM ANALYZE sentence_embeddings;")
    conn.autocommit = old_autocommit

    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_p1()