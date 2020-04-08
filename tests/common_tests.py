# Code by Pierre-François Massiani

import numpy as np

from normals.common import *
from tests.toy_clouds import *

def test_emst_computation():
    """
        Tests the function compute_emst.
    """
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
    """
        Tests the function acyclic_graph_dfs_iterator.
    """
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
