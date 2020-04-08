# Code by Pierre-François Massiani

import numpy as np

from normals.common import *
from normals.hoppe import compute_riemannian_mst

from tests.toy_clouds import *

def test_riemannian_mst_computation():
    """
        Tests the function compute_riemannian_mst on a toy example by comparing it with hand-calculated values.
    """
    n_neighbors = 2
    eps = 1e-1
    tolerance = 1e-6

    cloud = np.array([
        [0,0,0],
        [0,1,0],
        [0,1,1],
        [0,1,2]
    ])
    normals = np.array([
        [1,0,0],
        [0,1,0],
        [1,0,0],
        [0,1,0]
    ])

    true_symmetrized_emst = np.array([
        [0,1,0,0],
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0]
    ])
    true_kneighbors_graph = np.array([
        [0,1,1,0],
        [1,0,1,0],
        [0,1,0,1],
        [0,1,1,0]
    ])
    true_symmetrized_kneighbors_graph = np.array([
        [0,1,1,0],
        [1,0,1,1],
        [1,1,0,1],
        [0,1,1,0]
    ])
    true_riemannian_graph = np.array([
        [0,1+eps,eps,0],
        [1+eps,0,1+eps,eps],
        [eps,1+eps,0,1+eps],
        [0,eps,1+eps,0]
    ])
    true_possible_asymetric_rmsts = [
        np.array([
            [0,0,eps,0],
            [0,0,0,eps],
            [0,0,0,1+eps],
            [0,0,0,0],
        ]),
        np.array([
            [0,0,eps,0],
            [0,0,1+eps,eps],
            [0,0,0,0],
            [0,0,0,0],
        ]),
        np.array([
            [0,1+eps,eps,0],
            [0,0,0,eps],
            [0,0,0,0],
            [0,0,0,0],
        ]),
    ]
    true_possible_rmsts = [rmst + rmst.T for rmst in true_possible_asymetric_rmsts]

    actual_rmst = compute_riemannian_mst(cloud=cloud,normals=normals,n_neighbors=n_neighbors,eps=eps)
    is_possible = False
    for possible_rmst in true_possible_rmsts:
        is_possible = is_possible or (np.abs(possible_rmst - actual_rmst) < tolerance).all()
    if is_possible:
        return True
    else:
        print('---- Possible Riemannian MSTs:')
        for rmst in true_possible_rmsts:
            print(rmst)
        print('---- Actual Riemannian MST:')
        print(actual_rmst.toarray())
        return False

def test_iteration_trough_riemannian_mst(eps=1e-4,verbose=False):
    """
        Tests the function common.acyclic_graph_dfs_iterator on the output of compute_riemannian_mst.
        This test emphasizes the role of a non-zero epsilon in the computation of the weights of the riemannian graph.
    """
    n_neighbors = 2
    tolerance = 1e-6

    cloud = np.array([
        [0,0,0],
        [0,1,0],
        [0,1,1],
        [0,1,2]
    ])
    normals = np.array([
        [1,0,0],
        [0,1,0],
        [1,0,0],
        [0,1,0]
    ])
    rmst = compute_riemannian_mst(cloud,normals,n_neighbors,eps)
    # According to the test test_riemannian_mst_computation, the resulting RMST is
    #  0  1+e  e   0
    # 1+e  0   0   e
    #  e   0   0   0
    #  0   e   0   0

    seed = 3 # The highest z coordinate
    actual_traversing_order = [child for parent,child in acyclic_graph_dfs_iterator(graph=rmst,seed=seed)]
    true_traversing_order = [3,1,0,2]
    if actual_traversing_order == true_traversing_order:
        return True
    else:
        if verbose:
            print('---- True traversing order')
            print(true_traversing_order)
            print('---- Actual traversing order')
            print(actual_traversing_order)
        return False
