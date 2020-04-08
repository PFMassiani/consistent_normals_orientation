import numpy as np
from scipy.sparse import csr_matrix

from normals.common import *

def compute_riemannian_mst(cloud,normals,n_neighbors,eps=1e-4,verbose=False):
    # See the link below for some explanation on the riemannian graph used in the article
    # https://math.stackexchange.com/questions/2101217/what-is-the-exact-definition-of-a-riemannian-graph

    # Step 1 : compute the EMST of the cloud
    emst = compute_emst(cloud)
    symmetric_emst = (emst + emst.T)
    # Step 2 : enrich the EMST with the k-neighborhood graph
    symmetric_kgraph = symmetric_kneighbors_graph(cloud,n_neighbors)
    enriched = symmetric_emst + symmetric_kgraph
    # Step 3 : discard the weights, and replace them with 1 - |n_i . n_j|
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
    riemannian_mst = minimum_spanning_tree(riemannian_graph,overwrite = True) # overwrite = True for performance
    riemannian_mst = riemannian_mst + riemannian_mst.T # We symmetrize the graph so it is not oriented
    return riemannian_mst


def hoppe_orientation(cloud,normals,n_neighbors,verbose=False):
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
