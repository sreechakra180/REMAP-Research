from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import warnings

def run_kmeans(features, n_clusters):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_

def run_spectral(features, n_clusters):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(features)
    return labels

def compute_silhouette(features, labels):
    if len(set(labels)) > 1:
        return silhouette_score(features, labels)
    return -1.0
