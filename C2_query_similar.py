import time
import numpy as np
import chromadb
from chromadb.config import Settings

CHROMA_PATH = "./chroma_db"
SOURCE_COLLECTION = "sentence_embeddings_chroma"
N_QUERIES = 10
BUILD_BATCH_SIZE = 500


def build_metric_collections(client, ids, documents, embeddings):
    for name in ["c2_cosine", "c2_l2"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    coll_cosine = client.create_collection(
        name="c2_cosine",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    coll_l2 = client.create_collection(
        name="c2_l2",
        embedding_function=None,
        metadata={"hnsw:space": "l2"},
    )

    for i in range(0, len(ids), BUILD_BATCH_SIZE):
        batch_ids = ids[i:i + BUILD_BATCH_SIZE]
        batch_docs = documents[i:i + BUILD_BATCH_SIZE]
        batch_embs = embeddings[i:i + BUILD_BATCH_SIZE]
        coll_cosine.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs)
        coll_l2.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs)

    return coll_cosine, coll_l2


def run_queries(collection, query_ids, query_embeddings, query_docs, label):
    times = []
    print(f"\n=== Consultas top-2 en coleccion '{label}' ===")
    for qid, qvec, qdoc in zip(query_ids, query_embeddings, query_docs):
        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["documents", "distances"],
        )
        times.append(time.perf_counter() - t0)

        ids_res = results["ids"][0]
        docs_res = results["documents"][0]
        dist_res = results["distances"][0]

        top2 = [(d, doc) for rid, d, doc in zip(ids_res, dist_res, docs_res) if rid != qid][:2]

        print(f"\nFrase Objetivo (id={qid}): '{qdoc}'")
        print(f"  -> Top-2 ({label}): {top2}")

    print(f"\n--- Estadisticas de tiempo [C2]: Consulta '{label}' ---")
    print(f"Minimo: {np.min(times):.6f} s | Maximo: {np.max(times):.6f} s | "
          f"Promedio: {np.mean(times):.6f} s | Desv. Est: {np.std(times):.6f} s")


def execute_c2():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    source = client.get_collection(SOURCE_COLLECTION)

    data = source.get(include=["documents", "embeddings"])
    ids = data["ids"]
    documents = data["documents"]
    embeddings = data["embeddings"]

    print("Construyendo colecciones auxiliares (cosine / l2) por lotes...")
    coll_cosine, coll_l2 = build_metric_collections(client, ids, documents, embeddings)

    query_ids = ids[:N_QUERIES]
    query_docs = documents[:N_QUERIES]
    query_embeddings = embeddings[:N_QUERIES]

    run_queries(coll_cosine, query_ids, query_embeddings, query_docs, "cosine")
    run_queries(coll_l2, query_ids, query_embeddings, query_docs, "l2 (euclidea al cuadrado)")


if __name__ == '__main__':
    execute_c2()