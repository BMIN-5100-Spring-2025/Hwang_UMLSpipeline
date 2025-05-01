"""Command‑line utility to visualise saved document vectors with UMAP + HDBSCAN."""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from hdbscan import HDBSCAN


def parse_args():
    p = argparse.ArgumentParser(description="Document vector visualiser")
    p.add_argument('--vectors', required=True, help='.npy file with doc vectors')
    p.add_argument('--meta', required=True, help='CSV with metadata (row_id column)')
    p.add_argument('--out-html', required=True, help='Output HTML file')
    p.add_argument('--n-neighbors', type=int, default=15)
    p.add_argument('--min-dist', type=float, default=0.1)
    p.add_argument('--hdb-min-cluster-size', type=int, help='HDBSCAN min_cluster_size (default: 5)')
    p.add_argument('--hdb-min-samples', type=int, help='HDBSCAN min_samples (optional)')
    p.add_argument('--cluster', choices=['hdbscan', 'gmm', 'spectral'], default='hdbscan')
    return p.parse_args()


def main():
    args = parse_args()
    vecs = np.load(args.vectors)
    meta = pd.read_csv(args.meta)

    #  Calculate Isotropy 
    try:
        u, s, vh = np.linalg.svd(vecs, full_matrices=False)
        total_variance = np.sum(s**2)
        if total_variance > 1e-9:
            isotropy_score = 1.0 - (s[0]**2 / total_variance)
            print(f"Isotropy score (1 - λ1/Σλ): {isotropy_score:.4f}")
        else:
            print("Warning: Total variance of singular values is near zero, cannot calculate isotropy.")
    except Exception as e:
        print(f"Warning: Isotropy calculation failed: {e}")

    reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    emb3 = reducer.fit_transform(vecs)

    if args.cluster == 'hdbscan':
        hdb_kwargs = {
            'metric': 'euclidean',
            'allow_single_cluster': True,
            'min_cluster_size': args.hdb_min_cluster_size if args.hdb_min_cluster_size is not None else 5
        }
        if args.hdb_min_samples is not None:
            hdb_kwargs['min_samples'] = args.hdb_min_samples
        print(f"Running HDBSCAN with params: {hdb_kwargs}")
        clusterer = HDBSCAN(**hdb_kwargs)
        labels = clusterer.fit_predict(emb3)
    elif args.cluster == 'gmm':
        from sklearn.mixture import GaussianMixture
        n_components = 10
        labels = GaussianMixture(n_components=n_components, covariance_type='full').fit_predict(emb3)
    else:
        from sklearn.cluster import SpectralClustering
        labels = SpectralClustering(n_clusters=10, affinity='nearest_neighbors').fit_predict(emb3)

    df = pd.DataFrame({
        'row_id': meta['row_id'],
        'x': emb3[:, 0], 'y': emb3[:, 1], 'z': emb3[:, 2],
        'cluster': labels
    })

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', hover_data=['row_id'])
    fig.write_html(args.out_html)
    print(f"Saved visualisation to {args.out_html}")


if __name__ == '__main__':
    main() 