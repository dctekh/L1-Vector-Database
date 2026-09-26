import time
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import hf_login

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sentence_embeddings_chroma"
BATCH_SIZE = 500


def execute_c1():
    hf_login()
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    print("Cargando el modelo (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    data = collection.get(include=["documents"])
    ids = data["ids"]
    documents = data["documents"]
    total = len(ids)

    generation_times = [] 
    storage_times = []     

    print(f"Ejecutando [C1]: Generando y actualizando embeddings por lotes de {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]

        t0 = time.perf_counter()
        vectors = model.encode(batch_docs).tolist()
        generation_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        collection.update(ids=batch_ids, embeddings=vectors)
        storage_times.append(time.perf_counter() - t1)

        print(f"  Lote {i // BATCH_SIZE + 1}: {len(batch_ids)} docs procesados")

    print("\n--- Estadisticas de tiempo [C1]: Generacion del embedding (por lote) ---")
    print(f"Minimo: {np.min(generation_times):.6f} s")
    print(f"Maximo: {np.max(generation_times):.6f} s")
    print(f"Promedio: {np.mean(generation_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(generation_times):.6f} s")

    print("\n--- Estadisticas de tiempo [C1]: Almacenamiento del embedding (por lote) ---")
    print(f"Minimo: {np.min(storage_times):.6f} s")
    print(f"Maximo: {np.max(storage_times):.6f} s")
    print(f"Promedio: {np.mean(storage_times):.6f} s")
    print(f"Desviacion Estandar: {np.std(storage_times):.6f} s")

    total_gen = sum(generation_times)
    total_store = sum(storage_times)
    print(f"\nTiempo medio de generacion por documento: {total_gen / total:.6f} s")
    print(f"Tiempo medio de almacenamiento por documento: {total_store / total:.6f} s")


if __name__ == '__main__':
    execute_c1()