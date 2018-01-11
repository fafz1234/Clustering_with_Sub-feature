import numpy as np
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.cluster import _k_means


def Subspace_iter(X, n_clusters, init='k-means++', max_iter=300, tol=1e-4, tol_eig=-1e-10, x_squared_norms=None, random_state=None):
    random_state = check_random_state(random_state)
    centers = _init_centroids(X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms)

    new_labels, new_inertia, new_centers = None, None, None

    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
    d_shape = X.shape[1]
    randomval = random_state.random_sample(d_shape ** 2).reshape(d_shape, d_shape)
    V_val, _ = np.linalg.qr(randomval, mode='complete')
    m_val = d_shape // 2
    S_D = np.dot(X.T, X)
    P_Cluster = np.eye(m_val, M=d_shape).T
    for i in range(max_iter):
        centers_old = centers.copy()
        X_values = np.dot(np.dot(X, V_val), P_Cluster)
        centers_c = np.dot(np.dot(centers, V_val), P_Cluster)
        labels, _ = pairwise_distances_argmin_min(X = X_values, Y = centers_c,  metric='euclidean',metric_kwargs={'squared': True})
        labels = labels.astype(np.int32)
        centers = _k_means._centers_dense(X, labels, n_clusters, distances)
        S = np.zeros((d_shape, d_shape))
        for it in range(n_clusters):
            X_it = X[:][labels == it] - centers[:][it]
            S += np.dot(X_it.T, X_it)
        Sigma = S - S_D
        EV, _ = np.linalg.eigh(Sigma)
        m = len(np.where(EV < tol_eig)[0])
        P_Cluster = np.eye(m, M=d_shape).T
        inertia = 0.0
        for j in range(n_clusters):
            inertia += row_norms( X[:][labels == j] - centers[:][j],squared=True).sum()

        if new_inertia is None or inertia < new_inertia:
            new_labels = labels.copy()
            new_centers = centers.copy()
            new_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            break

    if center_shift_total > 0:
        new_labels, new_inertia = _labels_inertia(X, x_squared_norms, new_centers,
                            precompute_distances=False,
                            distances=distances)
    return new_labels, new_inertia, new_centers, i + 1
