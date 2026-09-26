import time
import psycopg2
import numpy as np
from config import load_config

# Cálculo DENTRO de PostgreSQL
# Requiere P0/P1 con la columna embedding de tipo REAL[].

DDL = """
CREATE OR REPLACE FUNCTION cosine_sim(a REAL[], b REAL[]) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT sum(x::float8 * y::float8)
           / NULLIF(sqrt(sum(x::float8 * x::float8)) * sqrt(sum(y::float8 * y::float8)), 0)
    FROM unnest(a, b) AS t(x, y)
$$;

CREATE OR REPLACE FUNCTION euclid_dist(a REAL[], b REAL[]) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT sqrt(sum((x::float8 - y::float8) ^ 2)) FROM unnest(a, b) AS t(x, y)
$$;

CREATE OR REPLACE FUNCTION top2_cosine(q_id INT)
RETURNS TABLE (r_id INT, r_sentence TEXT, r_score DOUBLE PRECISION)
LANGUAGE sql STABLE AS $$
    SELECT s.id, s.sentence, cosine_sim(s.embedding, q.embedding) AS sc
    FROM sentence_embeddings s
    CROSS JOIN (SELECT embedding FROM sentence_embeddings WHERE id = q_id) q
    WHERE s.id <> q_id AND s.embedding IS NOT NULL
    ORDER BY sc DESC
    LIMIT 2
$$;

CREATE OR REPLACE FUNCTION top2_euclid(q_id INT)
RETURNS TABLE (r_id INT, r_sentence TEXT, r_score DOUBLE PRECISION)
LANGUAGE sql STABLE AS $$
    SELECT s.id, s.sentence, euclid_dist(s.embedding, q.embedding) AS d
    FROM sentence_embeddings s
    CROSS JOIN (SELECT embedding FROM sentence_embeddings WHERE id = q_id) q
    WHERE s.id <> q_id AND s.embedding IS NOT NULL
    ORDER BY d ASC
    LIMIT 2
$$;
"""


def execute_p2_sql():
    db_config = load_config(section='postgresql')
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(DDL)
    conn.commit()

    times_cos = []
    times_euc = []

    print("\n=== EJECUCIÓN [P2-SQL]: TOP-2 DENTRO DE POSTGRESQL ===")
    for t_id in range(1, 11):   
        cursor.execute("SELECT sentence FROM sentence_embeddings WHERE id = %s;", (t_id,))
        t_sentence = cursor.fetchone()[0]

        t0 = time.perf_counter()
        cursor.execute("SELECT * FROM top2_cosine(%s);", (t_id,))
        top2_cos = cursor.fetchall()
        times_cos.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        cursor.execute("SELECT * FROM top2_euclid(%s);", (t_id,))
        top2_euc = cursor.fetchall()
        times_euc.append(time.perf_counter() - t1)

        print(f"\nFrase Objetivo (id={t_id}): '{t_sentence}'")
        print(f"  -> Top-2 Coseno: {top2_cos}")
        print(f"  -> Top-2 Euclídea: {top2_euc}")

    print("\n--- Estadísticas de tiempo [P2-SQL]: Consulta Coseno ---")
    print(f"Mínimo: {np.min(times_cos):.6f} s | Máximo: {np.max(times_cos):.6f} s | "
          f"Promedio: {np.mean(times_cos):.6f} s | Desv. Est: {np.std(times_cos):.6f} s")

    print("\n--- Estadísticas de tiempo [P2-SQL]: Consulta Euclídea ---")
    print(f"Mínimo: {np.min(times_euc):.6f} s | Máximo: {np.max(times_euc):.6f} s | "
          f"Promedio: {np.mean(times_euc):.6f} s | Desv. Est: {np.std(times_euc):.6f} s")


    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_p2_sql()
