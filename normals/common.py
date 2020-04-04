import numpy as np
from sklearn.neighbors import KDTree

import time

def local_PCA(points):

    N,d = points.shape

    p_ = points.mean(axis=0)

    Q = points - p_
    cov = Q.T @ Q / N

    eigenvalues,eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius=None, n_neighbors=None, verbose=False):
    if radius is None and n_neighbors is None:
        raise ValueError('radius and n_neighbors cannot be both None')
    if radius is not None and n_neighbors is not None:
        raise ValueError('radius and n_neighbors cannot be both set')
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

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
        Parameters
        ----------
            cloud : numpy array [N x 3] : the list of points of which the normals are going to be computed
            radius : float : the radius of the neighborhood used to compute the normals
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
