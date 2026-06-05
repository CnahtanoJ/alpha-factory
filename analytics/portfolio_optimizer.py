import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def get_inverse_variance_weights(cov_matrix):
    """
    Computes inverse-variance weights for a given covariance matrix.
    """
    ivp = 1.0 / np.diag(cov_matrix)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov_matrix, cluster_items):
    """
    Computes cluster variance given its items.
    """
    cov_slice = cov_matrix.iloc[cluster_items, cluster_items]
    weights = get_inverse_variance_weights(cov_slice)
    c_var = np.dot(np.dot(weights, cov_slice), weights)
    return c_var


def get_quasi_diag(link):
    """
    Sort clustered items by distance.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def get_rec_bipart(cov_matrix, sort_ix):
    """
    Computes HRP allocation from a sorted index.
    """
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while len(clusters) > 0:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1]

            c0_var = get_cluster_var(cov_matrix, c0)
            c1_var = get_cluster_var(cov_matrix, c1)

            alpha = 1 - c0_var / (c0_var + c1_var)

            weights[c0] *= alpha
            weights[c1] *= 1 - alpha

    return weights


def compute_hrp_weights(returns_df, top_k_symbols, alphas=None):
    """
    Computes the Hierarchical Risk Parity weights for the given returns.

    Args:
        returns_df (pd.DataFrame): DataFrame where index is time and columns are asset symbols, values are returns.
        top_k_symbols (list): The list of symbols that the alpha model has pre-selected.
        alphas (pd.Series, optional): Alpha scores to blend with risk weights. Defaults to None.

    Returns:
        dict: A mapping of symbol to allocated weight.
    """
    # 1. Ensure we only operate on the selected universe
    active_returns = returns_df[top_k_symbols].copy()

    # 2. Drop any columns that are entirely NaN or have 0 variance, and fill remaining NaNs
    active_returns = active_returns.fillna(0.0)
    for col in active_returns.columns:
        if active_returns[col].var() == 0:
            # Inject tiny noise to prevent singular matrix errors during clustering
            active_returns[col] += np.random.normal(0, 1e-8, size=len(active_returns))

    # 3. Compute Covariance and Correlation Matrices
    cov_matrix = active_returns.cov()
    corr_matrix = active_returns.corr().fillna(0)

    # 4. Compute Distance Matrix
    # Dist = sqrt(0.5 * (1 - corr))
    dist_matrix = np.sqrt(np.clip((1 - corr_matrix) / 2, a_min=0.0, a_max=1.0))

    # 5. Hierarchical Clustering (Ward linkage)
    # squareform converts a symmetric square matrix into a condensed distance vector
    dist_condensed = squareform(dist_matrix, checks=False)
    cluster_links = linkage(dist_condensed, method="ward")

    # 6. Quasi-Diagonalization
    sort_ix = get_quasi_diag(cluster_links)

    # 7. Recursive Bisection
    hrp_weights = get_rec_bipart(cov_matrix, sort_ix)
    hrp_weights.index = cov_matrix.columns[
        hrp_weights.index
    ]  # Map positional index back to symbol names

    # 8. (Optional) Alpha Overlay
    if alphas is not None:
        # Scale HRP weights by alpha strength (e.g., higher alpha -> shift weight upwards)
        # We ensure weights remain positive and sum to 1.
        combined = hrp_weights * np.exp(alphas[top_k_symbols])
        combined = combined / combined.sum()
        final_weights = combined.to_dict()
    else:
        final_weights = hrp_weights.to_dict()

    return final_weights
