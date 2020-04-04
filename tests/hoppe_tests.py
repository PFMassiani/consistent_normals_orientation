import numpy as np

from normals.hoppe import compute_emst
from normals.hoppe import acyclic_graph_dfs_iterator
from normals.hoppe import compute_riemannian_mst

from tests.toy_clouds import *

def test_emst_computation():
    cloud = toy_cloud_0()
    N = len(cloud)

    emst = compute_emst(cloud)
    emst = emst.toarray()
    true_emst = np.array([
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ])

    if (emst == true_emst).all():
        return True
    else:
        print('---- Computed EMST\n',emst)
        print('---- True EMST\n',true_emst)
        return False

def test_graph_traversing():
    graph = np.array([
        [0,2,3,1,0],
        [0,0,0,0,0],
        [0,0,0,0,1],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ])
    graph = graph + graph.T

    depth_first_traversing_possible_orders = [
        [0,1,2,4,3],
        [0,1,3,2,4],
        [0,2,4,1,3],
        [0,2,4,3,1],
        [0,3,1,2,4],
        [0,3,2,4,1]
    ]

    actual_traversing_order = [child for parent,child in acyclic_graph_dfs_iterator(graph=graph,seed=0)]
    if actual_traversing_order in depth_first_traversing_possible_orders:
        return True
    else:
        print('---- Possible traversing orders:\n',depth_first_traversing_possible_orders)
        print('---- Actual traversing order\n',actual_traversing_order)
        return False

def test_riemannian_mst_computation():
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


if __name__ == '__main__':
    print('EMST Computation test :',passed(test_emst_computation()))
    print('Graph traversing order test :',passed(test_graph_traversing()))
    print('Riemannian MST computation test :',passed(test_riemannian_mst_computation()))
    eps=1e-4
    print('Riemannian graph traversing (eps={:.4f}):'.format(eps), passed(test_iteration_trough_riemannian_mst(eps)))
    print('Riemannian graph traversing (eps=0):', passed(test_iteration_trough_riemannian_mst(0)))
