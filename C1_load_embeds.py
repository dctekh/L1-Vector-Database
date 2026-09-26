import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config import hf_login

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sentence_embeddings_chroma"
PROGRESS_EVERY = 500


def execute_c1():
    hf_login()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print("Cargando el modelo (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    data = collection.get(include=["documents"])
    ids = data["ids"]
    documents = data["documents"]

    generation_times = []
    storage_times = []

    total = len(ids)
    print(f"Ejecutando [C1]: Generando e insertando embeddings para {total} frases...")
    for i, (doc_id, sentence) in enumerate(zip(ids, documents), start=1):
        t0 = time.perf_counter()
        vector = model.encode(sentence).tolist()
        generation_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        collection.update(ids=[doc_id], embeddings=[vector])
        storage_times.append(time.perf_counter() - t1)

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  ... {i}/{total} procesadas")

    print("\n--- Estadisticas de tiempo [C1]: Generacion del embedding ---")
    print(f"Minimo: {np.min(generation_times):.6f} s")
    print(f"Maximo: {np.max(generation_times):.6f} s")
    print(f"Promedio: {np.mean(generation_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(generation_times):.6f} s")

    print("\n--- Estadisticas de tiempo [C1]: Almacenamiento del embedding ---")
    print(f"Minimo: {np.min(storage_times):.6f} s")
    print(f"Maximo: {np.max(storage_times):.6f} s")
    print(f"Promedio: {np.mean(storage_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(storage_times):.6f} s")


if __name__ == '__main__':
    execute_c1()