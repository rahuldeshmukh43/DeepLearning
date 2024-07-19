import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Union, Optional, Tuple

from clustering import K_Means


"""
Indexing is a critical part of retrieval and search systems. IVFPQ is one of the most commonly used method because of its parallelization and stable performance. Some common libraries used for index at scale are faiss (facebook) , scann (google) and annoy (spotify). 

Here is a simple (not the most efficient) implementation of IVFPQ 
"""

class IVFPQ_Index(object):
    """
    Inverted File Product Quantization Indexing
    """
    def __init__(self,
                 dim,
                 num_partitions:int,
                 # pq related
                subvector_size: int,
                num_clusters_in_subvector: int, 
                ) -> None:
        self.dim = dim
        self.num_partitions = num_partitions
        self.subvector_size = subvector_size
        self.num_cluster = num_clusters_in_subvector

        self.pq_index = [PQ_Index(dim, subvector_size, num_clusters_in_subvector) for _ in range(num_partitions)]

    def train(self, vectors:NDArray):
        """
        train a kmeans for IVF partitioning of vectors. Then train individual pq_index for each partition
        Args:
            vectors (NDArray) [N, dim]
        """
        # train a kmeans for IVF partitioning of vectors
        self.top_partition = K_Means(vectors,
                                     self.num_partitions)
        self.top_partition.fit()
        self.idx_top_partition = self.top_partition.predict(vectors) #[N]

        # train the kmeans for each partition 
        for i_part in range(self.num_partitions):
            this_pq_index = self.pq_index[i_part]
            # train this pq_index using the vectors corresponding to this partition
            this_partition_idx = np.where(self.idx_top_partition == i_part)
            this_pq_index.train(vectors[this_partition_idx, :])
        return
        
    def add(self, vectors:NDArray):
        """
        add vectors to the index
        """
        for i_part in range(self.num_partitions):
            this_pq_index = self.pq_index[i_part]
            # add vectors to the index
            this_partition_idx = np.where(self.idx_top_partition == i_part)
            this_pq_index.add(vectors[this_partition_idx, :], labels=this_partition_idx)
        return

    def query(self, query_vectors:NDArray, k:int) -> NDArray:
        """
        Find k closest approximate neighbors for each query vector        
        Args:
            query_vectors (NDArray) [Q, d]
            k (int)
        Return:
            out (NDArray) [Q, k] indices 
        """
        Q, dim = query_vectors.shape
        out = np.zeros((Q, k), dtype=np.int32)

        # (1) compute the correct top partition 
        idx_query_top_part = self.top_partition.predict(query_vectors)

        # (2) For each partition, compute the k neighbors
        for i_part in range(self.num_partitions):
            this_part_query_idx = np.where(idx_query_top_part == i_part)
            if len(this_part_query_idx) == 0 :
                continue
            # use this partition's pq to find the closest neighbors
            pq_index = self.pq_index[i_part]
            _out = pq_index.query(query_vectors[this_part_query_idx, :], k)
            out[this_part_query_idx, :] = _out
        return out

class PQ_Index(object):
    """
    Product Quantization Indexing
    """
    def __init__(self,
                dim,
                subvector_size: int,
                num_clusters_in_subvector: int,
                ) -> None:
        """
        Args:
            dim (int) dimension of data embeddings
            vectors (NDArray) [N, d]
            subvector_size (int) dimension of the subvector
            num_clusters_in_subvector (int) 
        """
        self.dim = dim
        self.subvector_size = subvector_size
        self.num_cluster = num_clusters_in_subvector       

        # checks
        assert self.dim % self.subvector_size == 0, "Data dimension (%d) not divisible by subvector size(%d)"%(self.dim, self.subvector_size) 

        # computed attibutes
        self.num_parts = self.dim // self.subvector_size

    def train(self, vectors:NDArray) -> None:
        """
        train the kmeans clustering for each part in PQ index
        Args:
            vectors (NDArray) [N, dim]
        """
        N, dim = vectors.shape
        assert dim == self.dim, "data dimension does not match"

        # partition and create k-means cluster for each partition
        self.clusters = []
        for i_part in range(self.num_parts):
            # instantiate k-means for this part
            kmean = K_Means(vectors[i_part*self.subvector_size:(i_part+1)*self.subvector_size,:],
                            self.num_cluster,
                            verbose=False)
            
            # compute cluster centers
            kmean.fit()
            self.clusters.append(kmean)

        self.cluster_centers = [ c.get_cluster_centers() for c in self.clusters] # List[NDArray[num_cluster, subvector_size]]
        return
    
    def add(self, vectors:NDArray, labels:NDArray =None) -> None:
        """
        add the vectors to index using trained kmeans
        Args:
            vectors (NDArray) [N, dim]
            labels (NDArray) [N]
        """
        N, dim = vectors.shape
        assert dim == self.dim, "data dimension does not match"
        
        if labels == None:
            self.labels = np.arange(N)
        else:
            self.labels = labels

        # build index
        self.index = np.zeros((N, self.num_parts))
        for i_part in range(self.num_parts): #NOTE: this can be parallelized
            # get ids for vectors for this part
            _ids = self.clusters[i_part].predict(vectors[:, i_part*self.subvector_size:(i_part + 1) * self.subvector_size]) #[N]
            self.index[:,i_part] = _ids
        return
    
    def _PQ_distance_matrix(self, query:NDArray) -> NDArray:
        """
        compute the PQ distance matrix of query vector using the cluster centers of partitions
        Args:
            query (NDArray) [d] single query vector
        Return:
            distances (NDArray) [num_clusters, num_parts]
        """
        # compute the distance matrix for query vectors
        distance = np.array((self.num_cluster, self.num_parts))
        # for each part compute distance of query from the cluster centers
        for i_part in range(self.num_parts): #NOTE: this can be parallelized
            q = query[i_part * self.subvector_size : ( i_part + 1 ) * self.subvector_size] #[subvector_size]
            cluster_center = self.cluster_centers[i_part] #[num_clusters, subvector_size]
            d = np.linalg.norm(cluster_center - q, axis=1) #[num_clusters]
            distance[:,i_part] = d        
        return distance

    def _PQ_distance_metric(self, query:NDArray, distance_matrix:NDArray) -> NDArray:
        """
        Compute the pq distance metric using the distance matrix and index for the query vector
        Args:
            query (NDArray) [d] single query vector
            distances (NDArray) [num_clusters, num_parts]
        Return:
            out (NDArray) [N] where N is training dataset size. pq distance of query from each data point
        """
        datasize = self.index.shape[0]
        out = np.zeros(datasize)
        for i in range(datasize):
            #NOTE: this can be easily parallelized
            v = self.index[i,:] # [1, num_parts]
            d = np.sum(distance_matrix[v])
            out[i] = d
        return out

    def query(self, query_vectors:NDArray, k:int) -> NDArray:
        """
        compute k closest approximate pq neighbors for the query vectors
        Args:
            query_vectors (NDArray) [Q, d]
            k (int) number of neighbors to find
        Return:
            out (NDArray) [Q, k] indices of dataset vectors which are close the query vectors
        """
        Q, dim = query_vectors.shape
        assert dim == self.dim, "data dimension does not match"

        out = np.array((Q,k), dtype=np.int32)
        for iq in range(Q):
            #NOTE: This can be parallelized/vectorized
            q = query_vectors[iq, :]
            # compute the PQ distance matrix for this q
            _pq_dist_mat = self._PQ_distance_matrix(q)
            # compute the pq distance for this q
            _pq_dist = self._PQ_distance_metric(q, _pq_dist_mat) #[N]
            # find idx of closest k datapoints
            _idx = np.argsort(_pq_dist)[:k]
            out[iq, :] = self.labels[_idx]
        return out