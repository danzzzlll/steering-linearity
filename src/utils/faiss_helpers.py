import faiss
import numpy as np

def knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    index = faiss.index_factory(X.shape[1], "HNSW32")
    # index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    _, knn = index.search(X, k + 1)
    return knn[:, 1:]
