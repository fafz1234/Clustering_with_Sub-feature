import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.cluster.k_means_ import _tolerance
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _validate_center_shape
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.cluster import _k_means
from sklearn.utils import as_float_array
from sklearn.cluster.k_means_ import _labels_inertia
from Subspace import Subspace_iter
from SKcluster import SubKmeans_cluster

class KMeans_Sub(KMeans):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, tol_eig=-1e-10, random_state=None, copy_x=True ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.tol_eig = tol_eig
        self.n_init = n_init
        self.random_state = random_state
        self.copy_x = copy_x

    def _transform(self, X):
        return np.dot(X, self.V_)

    def fit(self, X_Matrix, y=None):
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X_Matrix)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = SubKmeans_cluster(
                X,
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                tol_eig=self.tol_eig,
                random_state=random_state,
                copy_x=self.copy_x,
                return_n_iter=True
            )

        d_shape = X.shape[1]
        S_D = np.dot(X.T, X)
        S = np.zeros((d_shape, d_shape))
        for i in range(self.n_clusters):
            X_i = X[:][self.labels_ == i] - self.cluster_centers_[:][i]
            S += np.dot(X_i.T, X_i)
        Sigma = S - S_D
        self.feature_importances_, self.V_ = np.linalg.eigh(Sigma)
        self.m_ = len(np.where(self.feature_importances_ < self.tol_eig)[0])

        return self


