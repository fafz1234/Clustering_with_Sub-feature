import numpy as np
from sklearn.cluster.k_means_ import _tolerance
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import as_float_array
from sklearn.cluster.k_means_ import _labels_inertia
from feature_sub import Subspace_iter

def SubKmeans_cluster(X_Matrix, n_clusters, init='', n_init=10, max_iter=300, tol=1e-4, tol_eig=-1e-10, random_state=None, copy_x=True,return_n_iter=False):

    X = as_float_array(X_Matrix, copy=copy_x)
    tol = _tolerance(X, tol)
    X_mean = X.mean(axis=0)
    X -= X_mean
    x_squared_norms = row_norms(X, squared=True)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)

    new_labels, new_inertia, new_centers = None, None, None

    for it in range(n_init):
        labels, inertia, centers, n_iter_ = Subspace_iter(X, n_clusters, init= 'k-means++', max_iter=300, tol=1e-4, tol_eig=-1e-10, x_squared_norms=x_squared_norms, random_state=seeds[it])
        if new_inertia is None or inertia < new_inertia:
            new_labels = labels.copy()
            new_centers = centers.copy()
            new_inertia = inertia
            new_n_iter = n_iter_


    if not copy_x:
        X += X_mean
    new_centers += X_mean

    if return_n_iter:
        return new_centers, new_labels, new_inertia, new_n_iter
    else:
        return new_centers, new_labels, new_inertia
