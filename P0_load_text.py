import time
import json
import psycopg2
import numpy as np
import nltk
from datasets import load_dataset
from config import load_config, hf_login

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

TARGET_N_SENTENCES = 10000
CHUNK_EXPORT_FILE = "bookcorpus_chunk.jsonl"

def build_sentence_chunk(target_n=TARGET_N_SENTENCES):
    print(f"Descargando bookCorpus (hasta {target_n} frases)...")
    ds = load_dataset(
        "bookcorpus/bookcorpus",
        split=f"train[:{target_n}]",
        revision="refs/convert/parquet",
    )

    sentences = []
    for row in ds:
        raw_text = row["text"]
        for sent in nltk.sent_tokenize(raw_text):
            sent = sent.strip()
            if len(sent) > 5:
                sentences.append(sent)
        if len(sentences) >= target_n:
            break

    return sentences[:target_n]


def export_chunk(sentences, path=CHUNK_EXPORT_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences, start=1):
            f.write(json.dumps({"id": i, "sentence": sentence}, ensure_ascii=False) + "\n")
    print(f"Chunk exportado a {path} ({len(sentences)} frases)")


def execute_p0():
    db_config = load_config(section='postgresql')
    hf_login()

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentence_embeddings (
            id SERIAL PRIMARY KEY,
            sentence TEXT NOT NULL,
            embedding DOUBLE PRECISION[]
        );
    """)
    cursor.execute("TRUNCATE TABLE sentence_embeddings RESTART IDENTITY;")
    conn.commit()

    sentences = build_sentence_chunk()
    print(f"Frases obtenidas tras sent_tokenize: {len(sentences)}")
    export_chunk(sentences)

    text_times = []
    print("Ejecutando [P0]: Guardando texto en PostgreSQL...")
    for s in sentences:
        t0 = time.perf_counter()
        cursor.execute("INSERT INTO sentence_embeddings (sentence) VALUES (%s);", (s,))
        conn.commit()
        text_times.append(time.perf_counter() - t0)

    print("\n--- Estadisticas de tiempo [P0]: Texto ---")
    print(f"Minimo: {np.min(text_times):.6f} s")
    print(f"Maximo: {np.max(text_times):.6f} s")
    print(f"Promedio: {np.mean(text_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(text_times):.6f} s")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    execute_p0()