import time
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from config import load_config


def execute_g2():
    db_config = load_config(section='postgresql')
    conn = psycopg2.connect(**db_config)
    register_vector(conn)
    cursor = conn.cursor()

    print("Creando índices HNSW (coseno y euclídea)...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_cosine
        ON sentence_embeddings_pgvector USING hnsw (embedding vector_cosine_ops);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_l2
        ON sentence_embeddings_pgvector USING hnsw (embedding vector_l2_ops);
    """)
    conn.commit()

    times_cos = []
    times_euc = []

    print("\n=== EJECUCIÓN [G2]: TOP-2 CON PGVECTOR (índice HNSW, sin traer a memoria) ===")
    for t_id in range(1, 11):  
        cursor.execute("SELECT sentence, embedding FROM sentence_embeddings_pgvector WHERE id = %s;", (t_id,))
        t_sentence, query_vec = cursor.fetchone()

        # <=> : distancia coseno nativa (1 - similitud coseno; menor = más parecido)
        t0 = time.perf_counter()
        cursor.execute("""
            SELECT id, sentence, embedding <=> %s AS distance
            FROM sentence_embeddings_pgvector
            WHERE id != %s
            ORDER BY embedding <=> %s
            LIMIT 2;
        """, (query_vec, t_id, query_vec))
        top2_cos = cursor.fetchall()
        times_cos.append(time.perf_counter() - t0)

        # <-> : distancia euclídea (L2) nativa
        t1 = time.perf_counter()
        cursor.execute("""
            SELECT id, sentence, embedding <-> %s AS distance
            FROM sentence_embeddings_pgvector
            WHERE id != %s
            ORDER BY embedding <-> %s
            LIMIT 2;
        """, (query_vec, t_id, query_vec))
        top2_euc = cursor.fetchall()
        times_euc.append(time.perf_counter() - t1)

        print(f"\nFrase Objetivo (id={t_id}): '{t_sentence}'")
        print(f"  -> Top-2 Coseno: {top2_cos}")
        print(f"  -> Top-2 Euclídea: {top2_euc}")

    print("\n--- Estadísticas de tiempo [G2]: Consulta Coseno ---")
    print(f"Mínimo: {np.min(times_cos):.6f} s | Máximo: {np.max(times_cos):.6f} s | "
          f"Promedio: {np.mean(times_cos):.6f} s | Desv. Est: {np.std(times_cos):.6f} s")

    print("\n--- Estadísticas de tiempo [G2]: Consulta Euclídea ---")
    print(f"Mínimo: {np.min(times_euc):.6f} s | Máximo: {np.max(times_euc):.6f} s | "
          f"Promedio: {np.mean(times_euc):.6f} s | Desv. Est: {np.std(times_euc):.6f} s")


    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_g2()
