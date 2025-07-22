from sklearn.decomposition import PCA

def pca_fit(X, svd_solver: str, n=None):
    pca = PCA(n_components=n, random_state=42, svd_solver=svd_solver)
    return pca.fit(X)
