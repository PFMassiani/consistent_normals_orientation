# Code by Pierre-François Massiani

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from queue import LifoQueue

import time

def local_PCA(points):
    """
        Performs Principal Component Analysis on the points
        Parameters:
            points : np array : Nx3
        Outputs:
            eigenvalues : the eigenvalues of the cloud
            eigenvectors : the eigenvectors of the cloud
        Remark :
            The order of the eigenvalues and eigenvectors is similar to the one
            of the np.linalg.eigh method.
    """

    N,d = points.shape

    p_ = points.mean(axis=0)

    Q = points - p_
    cov = Q.T @ Q / N

    eigenvalues,eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius=None, n_neighbors=None, verbose=False):
    """
        Performs PCA on the neighborhoods of all query_points in cloud_points
        Parameters:
            query_points : np array : Px3 : the points we query
            cloud_points : np array : Nx3 : the points used to compute the neighborhoods
            radius : float : the radius used to compute the neighborhood. Should be specified iff n_neighbors is None.
            n_neighbors : int : the number of neighbors used to compute the neighborhoor. Should be specified iff radius is None.
            verbose : boolean : verbosity
        Outputs:
            all_eigenvalues: np array : Px3 : the eigenvalues of the cloud around the query points, from smallest to greatest
            all_eigen_vectors : np array : Px3x3 : the eigenvectors of the cloud arount the query points, in the same order as the eigenvalues.
    """
    if radius is None and n_neighbors is None:
        raise ValueError('radius and n_neighbors cannot be both None')
    if radius is not None and n_neighbors is not None:
        raise ValueError('radius and n_neighbors cannot be both set')

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    if verbose:
        print('Computing KDTree...')
        t0 = time.time()
    tree = KDTree(cloud_points,leaf_size = 40)
    if verbose:
        t1 = time.time()
        print('Done. ({:.3f} s)'.format(t1-t0))
        print('Querying the tree to compute the neighbors...')
        t0 = time.time()

    if radius is not None:
        neighborhoods = tree.query_radius(query_points,r = radius)
    if n_neighbors is not None:
        neighborhoods = tree.query(query_points,k=n_neighbors,return_distance=False)
    if verbose:
        t1 = time.time()
        print('Done. ({:.3f} s)'.format(t1-t0))
        print('Computing the normals using the neighborhoods...')
        t0 = time.time()
    for k in range(len(neighborhoods)):
    	N_indexes = neighborhoods[k]
    	N = cloud_points[N_indexes,:]
    	lambdas,eigens = local_PCA(N)
    	l3,l2,l1 = lambdas
    	v3,v2,v1 = eigens.T

    	all_eigenvalues[k] = [l1,l2,l3]
    	all_eigenvectors[k] = [v1,v2,v3]

    if verbose:
        t1 = time.time()
        print('Done. ({:.3f} s)'.format(t1-t0))

    return all_eigenvalues, all_eigenvectors


def compute_normals(cloud,radius=None,n_neighbors=None,verbose=False):
    """
        Computes the normals of the points cloud using PCA.
        Parameters:
            cloud : np array : Nx3 : the list of points of which the normals are going to be computed
            radius : float : the radius used to compute the neighborhood. Should be specified iff n_neighbors is None.
            n_neighbors : int : the number of neighbors used to compute the neighborhoor. Should be specified iff radius is None.
            verbose : boolean : verbosity
    """
    if verbose:
        print("Computing normal for the cloud of size",cloud.shape)
        print('----------------')
        t0 = time.time()
    all_eigenvalues,all_eigenvectors = neighborhood_PCA(cloud, cloud, radius,n_neighbors, verbose)
    normals = all_eigenvectors[:,-1,:]

    normalization = np.linalg.norm(normals,axis=1)
    normals = normals / normalization[:,np.newaxis]

    if verbose:
        print('----------------')
        t1 = time.time()
        print('Done. ({:.3f} s)'.format(t1-t0))
    return normals

def slow_emst(cloud):
    """
        Inefficient implementation of a Euclidean MST finding algorithm. Has memory and time cost of O(N**2).
        Parameters:
            cloud : np array : Nx3
        Outputs:
            emst : scipy.sparse matrix : NxN : the EMST
    """
    euclidean_graph = np.linalg.norm(cloud[:,np.newaxis,:] - cloud[np.newaxis,:,:],axis=2)
    emst = minimum_spanning_tree(euclidean_graph,overwrite=True) # overwrite = True for performance
    return emst

def fast_emst(cloud):
    """
        Efficient implementation of a Euclidean MST finding algorithm.
        Parameters:
            cloud : np array : Nx3
        Outputs:
            emst : scipy.sparse matrix : NxN : the EMST
    """
    # The euclidean minimum spanning graph is a subgraph of the Delaunay graph
    # See https://fr.wikipedia.org/wiki/Triangulation_de_Delaunay#Applications
    delaunay_triangulation = Delaunay(cloud)
    extract_edges = ((0,1),(1,2),(0,2))
    delaunay_edges = None
    for edge_extraction in extract_edges:
        if delaunay_edges is None:
            delaunay_edges = delaunay_triangulation.simplices[:,edge_extraction]
        else:
            delaunay_edges = np.vstack((delaunay_edges,delaunay_triangulation.simplices[:,edge_extraction]))
    euclidean_weights = np.linalg.norm(
                            cloud[delaunay_edges[:,0],:] - cloud[delaunay_edges[:,1]],
                            axis = 1
                        )
    delaunay_euclidean_graph = csr_matrix((euclidean_weights,delaunay_edges.T), shape=(len(cloud),len(cloud)))
    emst = minimum_spanning_tree(delaunay_euclidean_graph,overwrite=True)
    return emst

def compute_emst(cloud):
    """
        Computes the Euclidean Minimum Spanning Tree of the points cloud.
        Parameters:
            cloud : np array : Nx3
        Outputs:
            emst : scipy.sparse.matrix : NxN : the EMST
        Remarks :
            1. The EMST that this function outputs is NOT symmetric, but triangular (superior)
    """
    # The Delaunay triangulation method requires at least 5 points in the cloud
    if len(cloud) > 5:
        return fast_emst(cloud)
    else:
        return slow_emst(cloud)


def symmetric_kneighbors_graph(cloud,n_neighbors):
    """
        Computes a graph whose edge (i,j) is nonzero iff j is in the
        k-neighborhood of i OR i is in the k-neighborhood of j.
        Parameters:
            cloud : np array : Nx3
            n_neighbors : int
        Outputs:
            kgraph : scipy.sparse.matrix : NxN
        Remark :
            The values of the edges are not significant.
    """
    kgraph = kneighbors_graph(cloud,n_neighbors, mode='connectivity')
    kgraph = kgraph.tocoo()
    return kgraph + kgraph.transpose()

def acyclic_graph_dfs_iterator(graph,seed):
    """
        Computes an iterator for iterating depth-first in an unoriented acyclic graph
        Parameters:
            graph : scipy.sparse.csr_matrix : NxN
            seed : int : the seed that should be considered as the root of the graph
        Outputs:
            graph_iterator : iterator
        Remark :
            This function uses the fact that the graph is unoriented. Hence, its
            matrix has to be symmetric.
    """
    graph = csr_matrix(graph)

    stack = LifoQueue()
    stack.put((None,seed))

    while not stack.empty():
        parent,child = stack.get()
        connected_to_child = graph[child,:].nonzero()[1]
        for second_order_child in connected_to_child:
            if second_order_child != parent:
                stack.put((child,second_order_child))
        yield parent,child
