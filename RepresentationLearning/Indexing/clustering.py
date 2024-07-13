import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Union, Optional, Tuple

class K_Means(object):
    def __init__(self, 
                 vectors: NDArray, 
                 num_clusters: int = -1,
                 max_iters: int = 100,
                 tol: float= 1e-4,
                 verbose:bool=False) -> None:
        self.vectors = vectors # [N,d]
        self.dim = self.vectors.shape[1]
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.clusters = None # [k, d]
        self.verbose = verbose

        if self.num_clusters <= 0:
            TypeError("Num clusters should be positive")
            exit(1)

    def fit(self) -> None:
        "fit and identify the cluster centers"
        # random initialization of cluster centers
        clusters = np.random.rand(self.num_clusters, self.dim)
        mins = np.array([ np.min(self.vectors[:,d]) for d in range(self.dim)]) #[d]
        maxs = np.array([ np.max(self.vectors[:,d]) for d in range(self.dim)]) #[d]
        # rescale random clusters to be between mins and maxs
        self.clusters = clusters * (maxs - mins) + mins
        # loop {
        # (1) compute labels of all data points 
        # (2) compute centroid of each cluster and update cluster center        
        # }
        iter = 0
        err = 1E6
        while iter < self.max_iters:
            # compute labels 
            labels = self.predict(self.vectors) #[N]
            # compute centroids
            centroids = np.zeros_like(clusters)
            for k in range(self.num_clusters):
                ids_for_k = np.where(labels==k)[0]
                centroids[k] = np.mean(self.vectors[ids_for_k,:], axis=0)
            # compute err
            err = np.mean(np.linalg.norm(centroids - self.clusters, axis=1))
            if self.verbose:
                print("%d\t %0.4f"%(iter, err), self.clusters)
            # compute err and check for tol
            if err < self.tol:
                # we are done
                break
            self.clusters = centroids
            iter += 1
        return

    def predict(self, query:NDArray) -> NDArray:
        """
        predict the closest cluster each sample in query belongs to
        Args:
            query (NDArray) [Q, d]
        Return:
            labels: (NDArray[np.int]) [Q] -- predicted index of cluster center
        """
        # find distance of each query point from every cluster
        d = self._distance(query, self.clusters) #[q, k]
        # find argmin 
        labels = np.argmin(d, axis=1)
        return labels

    def _distance(self, vectors:NDArray, clusters:NDArray) -> NDArray:
        """
        Args:
            vectors [N,d]
            clusters [K, d]
        Return:
            distance [N, K]
        """
        N, d = vectors.shape
        K = clusters.shape[0]
        distances = np.zeros((N, K ))
        for k in range(K):
            # compute distance of all vectors for this cluster
            d_k = vectors - clusters[k,:] # [N,d]
            distances[:,k] = np.linalg.norm(d_k, axis=1)
        return distances
    
    def get_cluster_centers(self) -> NDArray:
        return self.clusters #[K, d]