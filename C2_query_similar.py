import time
import numpy as np
import chromadb

CHROMA_PATH = "./chroma_db"
SOURCE_COLLECTION = "sentence_embeddings_chroma"
N_QUERIES = 10


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

    batch = 500
    for i in range(0, len(ids), batch):
        coll_cosine.add(
            ids=ids[i:i + batch],
            documents=documents[i:i + batch],
            embeddings=embeddings[i:i + batch],
        )
        coll_l2.add(
            ids=ids[i:i + batch],
            documents=documents[i:i + batch],
            embeddings=embeddings[i:i + batch],
        )

    return coll_cosine, coll_l2


def run_queries(collection, query_ids, query_embeddings, query_docs, label):
    times = []
    print(f"\n=== Consultas top-2 en colección '{label}' ===")
    for qid, qvec, qdoc in zip(query_ids, query_embeddings, query_docs):
        t0 = time.time()
        results = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["documents", "distances"],
        )
        times.append(time.time() - t0)

        ids_res = results["ids"][0]
        docs_res = results["documents"][0]
        dist_res = results["distances"][0]

        top2 = [(d, doc) for rid, d, doc in zip(ids_res, dist_res, docs_res) if rid != qid][:2]

        print(f"\nFrase Objetivo (id={qid}): '{qdoc}'")
        print(f"  -> Top-2 ({label}): {top2}")

    print(f"\n--- Estadísticas de tiempo [C2]: Consulta '{label}' ---")
    print(f"Mínimo: {np.min(times):.6f} s | Máximo: {np.max(times):.6f} s | "
          f"Promedio: {np.mean(times):.6f} s | Desv. Est: {np.std(times):.6f} s")


def execute_c2():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    source = client.get_collection(SOURCE_COLLECTION)

    data = source.get(include=["documents", "embeddings"])
    ids = data["ids"]
    documents = data["documents"]
    embeddings = data["embeddings"]

    print("Construyendo colecciones auxiliares (cosine / l2)...")
    coll_cosine, coll_l2 = build_metric_collections(client, ids, documents, embeddings)

    query_ids = ids[:N_QUERIES]
    query_docs = documents[:N_QUERIES]
    query_embeddings = embeddings[:N_QUERIES]

    run_queries(coll_cosine, query_ids, query_embeddings, query_docs, "cosine")
    run_queries(coll_l2, query_ids, query_embeddings, query_docs, "l2 (euclídea al cuadrado)")


if __name__ == '__main__':
    execute_c2()