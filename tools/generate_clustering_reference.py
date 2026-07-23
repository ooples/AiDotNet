#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def create_two_cluster_blobs(points_per_cluster=12, spacing=12.0):
    data = []
    for i in range(points_per_cluster):
        offset = i * 0.2
        data.append([offset, offset])
    for i in range(points_per_cluster):
        offset = i * 0.2
        data.append([spacing + offset, spacing + offset])
    return np.array(data, dtype=float)


def create_with_outlier():
    base = create_two_cluster_blobs(points_per_cluster=4, spacing=10.0)
    outlier = np.array([[30.0, 30.0]], dtype=float)
    return np.vstack([base, outlier])


def main(output_path):
    two_cluster = create_two_cluster_blobs(points_per_cluster=12, spacing=12.0)
    with_outlier = create_with_outlier()

    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        random_state=42,
        max_iter=300,
    )
    kmeans_labels = kmeans.fit_predict(two_cluster)

    dbscan = DBSCAN(eps=1.6, min_samples=3)
    dbscan_labels = dbscan.fit_predict(with_outlier)

    metrics = {
        "silhouette": float(silhouette_score(two_cluster, kmeans_labels)),
        "davies_bouldin": float(davies_bouldin_score(two_cluster, kmeans_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(two_cluster, kmeans_labels)),
    }

    payload = {
        "version": {
            "sklearn": "1.8.0",
            "numpy": "2.4.0",
        },
        "datasets": {
            "two_cluster_blobs_12_12": {
                "data": two_cluster.tolist(),
            },
            "with_outlier": {
                "data": with_outlier.tolist(),
            },
        },
        "kmeans": {
            "two_cluster_blobs_12_12": {
                "labels": kmeans_labels.tolist(),
                "centers": kmeans.cluster_centers_.tolist(),
                "inertia": float(kmeans.inertia_),
            },
        },
        "dbscan": {
            "with_outlier_eps_1_6_min_3": {
                "labels": dbscan_labels.tolist(),
                "num_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
                "num_noise": int(list(dbscan_labels).count(-1)),
            },
        },
        "metrics": {
            "two_cluster_blobs_12_12": metrics,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    default_output = Path(__file__).resolve().parents[1] / "tests" / "AiDotNet.Tests" / "IntegrationTests" / "Clustering" / "ReferenceData" / "sklearn_clustering_reference.json"
    main(default_output)
