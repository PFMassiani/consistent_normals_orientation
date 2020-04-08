# Code by Pierre-François Massiani

import numpy as np
from scipy.sparse import csr_matrix

from normals.common import *

def compute_riemannian_mst(cloud,normals,n_neighbors,eps=1e-4,verbose=False):
    """
        Computes the Riemannian Minimum Spannine Tree for the Hoppe method
        Parameters:
            cloud : numpy array : Nx3 : Should not have duplicated points
            normals : numpy array : Nx3 : Should be normalized
            n_neighbors : int : used to compute the k-neighbors graph
            eps : float : the value added to the weight of every edge of the Riemannian MST. Should be small, and strictly positive.
            verbose : boolean : verbosity
    """
    # See the link below for some explanation on the riemannian graph used in the article
    # https://math.stackexchange.com/questions/2101217/what-is-the-exact-definition-of-a-riemannian-graph

    emst = compute_emst(cloud)
    symmetric_emst = (emst + emst.T)

    symmetric_kgraph = symmetric_kneighbors_graph(cloud,n_neighbors)
    enriched = symmetric_emst + symmetric_kgraph

    enriched = enriched.tocoo()
    connected_l = enriched.row
    connected_r = enriched.col
    riemannian_weights = [
        1 + eps - np.abs(
                np.dot(
                    normals[connected_l[k],:],
                    normals[connected_r[k],:]
                )) for k in range(len(connected_l))] # Can be optimized : we do each operation two times because we do not exploit the symmetry of the riemannian graph

    riemannian_graph = csr_matrix((riemannian_weights,(connected_l,connected_r)),shape = (cloud.shape[0],cloud.shape[0]))
    riemannian_mst = minimum_spanning_tree(riemannian_graph,overwrite = True)
    # The scipy.minimum_spanning_tree function returns a triangular (superior) graph.
    # This is more memory-efficient, but out implementation of the graph search requires
    # a symmetric graph.
    riemannian_mst = riemannian_mst + riemannian_mst.T
    return riemannian_mst


def hoppe_orientation(cloud,normals,n_neighbors,verbose=False):
    """
        Orients the normals using the Hoppe method presented in "Surface Reconstruction from Unorganized Points" (1992).
        Parameters:
            cloud : np array : Nx3 : should not contain duplicates
            normals : np array : Nx3 : should be normalized
            n_neighbors : int : the number of neighbors used in the graph computation
            verbose : boolean : whether the algorithm is verbose. Mainly provides information about computation times.
        Outputs:
            normals_o : np array : Nx3 : the oriented normals
    """
    normals_o = normals.copy() # oriented normals

    # Step 1 : compute the riemannian mst of the cloud
    riemannian_mst = compute_riemannian_mst(cloud,normals,n_neighbors,verbose)

    # Step 2 : select seed and its orientation
    seed_index = np.argmax(cloud[:,2])
    ez = np.array([0,0,1])
    if normals_o[seed_index,:].T @ ez < 0:
        normals_o[seed_index,:] *= -1 # We arbitrarily set the direction of the seed so it points towards increasing z values

    # Step 3 : traverse the MST depth first order by assigning a consistent orientation (scalar product between successors should be positive)
    for parent_index,point_index in acyclic_graph_dfs_iterator(riemannian_mst,seed_index):
        if parent_index is None:
            parent_normal = normals_o[seed_index,:]
        else:
            parent_normal = normals_o[parent_index,:]

        if normals_o[point_index,:] @ parent_normal < 0:
            normals_o[point_index,:] *= -1

    return normals_o
