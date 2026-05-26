import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class GeometryAnalyser:
    def analyse(self, embeddings: np.ndarray) -> dict:
        if embeddings.ndim != 2 or embeddings.shape[1] != 4:
            raise ValueError(f'Expected (N, 4) array, got {embeddings.shape}')
        n = embeddings.shape[0]
        pca = PCA().fit(embeddings)
        pc1_var = float(pca.explained_variance_ratio_[0])
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim95 = int(np.searchsorted(cumvar, 0.95) + 1)
        curvature = self._estimate_curvature(embeddings, pca)
        neighbor_purity = self._neighbor_purity(embeddings, pca)
        geo_euclidean_corr = self._geo_euclidean_corr(embeddings)
        return {
            'pc1': round(pc1_var, 4),
            'dim95': dim95,
            'curvature': round(curvature, 4),
            'neighbor_purity': round(neighbor_purity, 4),
            'geo_euclidean_corr': round(geo_euclidean_corr, 4),
        }

    def _estimate_curvature(self, embeddings: np.ndarray, pca: PCA) -> float:
        proj = pca.transform(embeddings)
        recon_1d = proj[:, :1] @ pca.components_[:1] + pca.mean_
        residual = embeddings - recon_1d
        return float(np.mean(np.linalg.norm(residual, axis=1)))

    def _neighbor_purity(self, embeddings: np.ndarray, pca: PCA, k: int = 5) -> float:
        if embeddings.shape[0] < k + 1:
            return 1.0
        proj_1d = pca.transform(embeddings)[:, :1]
        nn_full = NearestNeighbors(n_neighbors=k).fit(embeddings)
        nn_1d = NearestNeighbors(n_neighbors=k).fit(proj_1d)
        idx_full = nn_full.kneighbors(embeddings, return_distance=False)
        idx_1d = nn_1d.kneighbors(proj_1d, return_distance=False)
        matches = sum(len(set(idx_full[i]) & set(idx_1d[i])) for i in range(len(embeddings)))
        return matches / (len(embeddings) * k)

    def _geo_euclidean_corr(self, embeddings: np.ndarray) -> float:
        from scipy.spatial.distance import pdist
        from scipy.stats import pearsonr
        geo = pdist(embeddings, metric='euclidean')
        proj_1d = PCA(n_components=1).fit_transform(embeddings)
        euclidean = pdist(proj_1d, metric='euclidean')
        corr, _ = pearsonr(geo, euclidean)
        return float(corr)
