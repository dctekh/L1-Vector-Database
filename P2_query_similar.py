import time
import psycopg2
import numpy as np
from config import load_config

def execute_p2():
    db_config = load_config(section='postgresql')
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, sentence, embedding
        FROM sentence_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY id;
    """)
    all_data = cursor.fetchall()

    # --- FIX respecto a la versión original ---
    # Antes: por cada una de las 10 consultas, y por cada una de las N
    # frases, y por cada una de las 2 métricas, se hacía np.array(r_emb).
    # Eso son ~ 10 * N * 2 conversiones redundantes de lista -> array,
    # y contaminaba la medición (se medía sobre todo overhead de Python,
    # no el coste real de comparar vectores).
    # Ahora: construimos la matriz de embeddings UNA sola vez, fuera del
    # bucle cronometrado, y calculamos todas las distancias de una consulta
    # contra el resto con una operación numpy vectorizada (esto es,
    # de hecho, una aproximación a lo que hacen internamente las bases de
    # datos vectoriales: operar sobre el array completo, no fila a fila).
    ids = np.array([row[0] for row in all_data])
    sentences = [row[1] for row in all_data]
    matrix = np.array([row[2] for row in all_data], dtype=np.float64)  # (N, d)
    norms = np.linalg.norm(matrix, axis=1)  # precalculado, se reutiliza en coseno

    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    sample_queries = list(zip(ids[:10], [sentences[i] for i in range(10)]))

    times_cos = []
    times_euc = []

    print("\n=== EJECUCIÓN [P2]: BÚSQUEDA TOP-2 SIMILITUD ===")
    for t_id, t_sentence in sample_queries:
        q_idx = id_to_idx[t_id]
        query_vec = matrix[q_idx]

        # 1. Similitud coseno (vectorizado)
        t0 = time.time()
        query_norm = norms[q_idx]
        sims = (matrix @ query_vec) / (norms * query_norm + 1e-10)
        order_cos = np.argsort(-sims)  # descendente: mayor similitud primero
        top2_cos_idx = [i for i in order_cos if ids[i] != t_id][:2]
        top2_cos = [(sims[i], sentences[i]) for i in top2_cos_idx]
        times_cos.append(time.time() - t0)

        # 2. Distancia euclídea (vectorizado)
        t1 = time.time()
        dists = np.linalg.norm(matrix - query_vec, axis=1)
        order_euc = np.argsort(dists)  # ascendente: menor distancia primero
        top2_euc_idx = [i for i in order_euc if ids[i] != t_id][:2]
        top2_euc = [(dists[i], sentences[i]) for i in top2_euc_idx]
        times_euc.append(time.time() - t1)

        print(f"\nFrase Objetivo (id={t_id}): '{t_sentence}'")
        print(f"  -> Top-2 Coseno: {top2_cos}")
        print(f"  -> Top-2 Euclídea: {top2_euc}")

    print("\n--- Estadísticas de tiempo [P2]: Consulta Coseno ---")
    print(f"Mínimo: {np.min(times_cos):.6f} s | Máximo: {np.max(times_cos):.6f} s | "
          f"Promedio: {np.mean(times_cos):.6f} s | Desv. Est: {np.std(times_cos):.6f} s")

    print("\n--- Estadísticas de tiempo [P2]: Consulta Euclídea ---")
    print(f"Mínimo: {np.min(times_euc):.6f} s | Máximo: {np.max(times_euc):.6f} s | "
          f"Promedio: {np.mean(times_euc):.6f} s | Desv. Est: {np.std(times_euc):.6f} s")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_p2()