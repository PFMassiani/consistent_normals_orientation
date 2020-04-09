# Code by Pierre-François Massiani

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

from normals.common import *

import warnings

def compute_reference_planes(pi_s,pj_s,cloud,normals):
    """
        Computes the features of the reference planes
        Parameters:
            pi_s : numpy array : P : list of the indexes of the cloud whose corresponding points are the parent nodes in the graph
            pj_s : numpy array : P : list of the indexes of the cloud whose corresponding points are the children nodes in the graph
            cloud : numpy array : Nx3 : Should not have duplicated points
            normals : numpy array : Nx3 : Should be normalized
        Outputs:
            reference_planes : numpy array : Px3 : list of the normalized normals for the reference plane_base
            reference_bases : numpy array : Px2x3 : list of the bases for the reference planes such that (reference_bases[k,0,:],reference_bases[k,1,:],reference_planes[k,:]) is orthonormal
    """
    tolerance = 1e-6
    n_couples = len(pi_s)
    norm = np.linalg.norm

    reference_vectors = np.dstack((
                                    cloud[pj_s,:] - cloud[pi_s,:],
                                    normals[pi_s,:],
                                    normals[pj_s,:]
                                ))

    # If the cloud has duplicated points, this loop is going to raise a RuntimeWarning because of a division
    # by 0 for the duplicated couples.
    for d in range(3):
        reference_vectors[:,:,d] = reference_vectors[:,:,d] / np.linalg.norm(reference_vectors[:,:,d],axis=1)[:,np.newaxis]
    reference_vectors = np.swapaxes(reference_vectors,1,2)
    # Now, if n is the index of the couple (i,j), reference_vectors[n,k,:] is (normalized):
    # 1. p_j - p_i if k == 0
    # 2. n_i if k == 1
    # 3. n_j if k == 2

    scalar_e_ni = (reference_vectors[:,0,:] * reference_vectors[:,1,:]).sum(axis=1)
    scalar_e_nj = (reference_vectors[:,0,:] * reference_vectors[:,2,:]).sum(axis=1)
    scalar_ni_nj = (reference_vectors[:,1,:] * reference_vectors[:,2,:]).sum(axis=1)
    determinants = np.linalg.det(reference_vectors)

    are_collinear = np.logical_and(
        np.abs(scalar_e_ni) > 1 - tolerance,
        np.logical_and(
            np.abs(scalar_e_nj) > 1 - tolerance,
            np.abs(scalar_ni_nj) > 1 - tolerance,
        )
    )
    span_plane = np.logical_and(
        np.logical_not(are_collinear),
        np.abs(determinants) < tolerance
    )
    span_volume = np.logical_and(
        np.logical_not(are_collinear),
        np.logical_not(span_plane)
    )

    reference_planes = np.zeros((n_couples,3)) # contains the normals of the reference planes
    reference_bases = np.zeros((n_couples,2,3))

    if are_collinear.any():
        collinears = reference_vectors[are_collinear,:,:].sum(axis=1) / 3
        # We need to find a vector different from collinears to be able to compute
        # a base using cross product. We use a little trick :
        # 1. If collinears is not along ez, we rotate it around this vector
        # 2. Otherwise, we rotate it around ex
        rotation_around_ex = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        rotation_around_ez = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        ez = np.array([0,0,1]).reshape((-1,1))
        equal_to_ez = ((collinears @ ez) > 1 - tolerance).reshape((-1))
        different_from_collinears = collinears.copy()
        different_from_collinears[np.logical_not(equal_to_ez),:] = different_from_collinears[np.logical_not(equal_to_ez),:] @ rotation_around_ez
        different_from_collinears[equal_to_ez,:] = different_from_collinears[equal_to_ez,:] @ rotation_around_ex

        reference_planes[are_collinear,:] = np.cross(collinears,different_from_collinears)
        reference_planes[are_collinear,:] /= np.linalg.norm(reference_planes[are_collinear,:],axis=1)[:,np.newaxis]

        base_y = np.cross(reference_planes[are_collinear,:],collinears)
        base_y /= norm(base_y,axis=1)[:,np.newaxis]
        base_x = collinears / norm(collinears,axis=1)[:,np.newaxis]
        reference_bases[are_collinear,0,:] = base_x
        reference_bases[are_collinear,1,:] = base_y

    if span_plane.any():
        # (eij,ni,nj) span a plane, but there may be collinear vectors among them.
        # Hence, we use two cross products to define the normal, and make sure
        # they don't cancel out when summing by carefully picking the sign.
        eij_v_ni = np.cross(
                            reference_vectors[span_plane,0,:],
                            reference_vectors[span_plane,1,:]
                        )
        ni_v_nj = np.cross(
                            reference_vectors[span_plane,1,:],
                            reference_vectors[span_plane,2,:]
                        )
        sign = (eij_v_ni * ni_v_nj).sum(axis = 1)
        with np.errstate(invalid = 'ignore'):
            sign = (sign / np.abs(sign))[:,np.newaxis]
        sign = np.nan_to_num(sign,nan=1)

        reference_planes[span_plane,:] = eij_v_ni + sign * ni_v_nj
        reference_planes[span_plane,:] /= norm(reference_planes[span_plane,:],axis=1)[:,np.newaxis]

        reference_bases[span_plane,0,:] = reference_vectors[span_plane,1,:]
        reference_bases[span_plane,1,:] = np.cross(reference_planes[span_plane],reference_bases[span_plane,0,:])


    if span_volume.any():
        with np.errstate(invalid='ignore'):
            signs = (scalar_ni_nj / np.abs(scalar_ni_nj))[span_volume].reshape((-1,1))
        signs = np.nan_to_num(signs,nan=1)
        normals_average = (
                            reference_vectors[span_volume,1,:] + signs * reference_vectors[span_volume,2,:]
                        ) / norm(reference_vectors[span_volume,1,:] + signs * reference_vectors[span_volume,2,:],axis=1)[:,np.newaxis]
        reference_planes[span_volume] = np.cross(
                                            reference_vectors[span_volume,1,:],
                                            reference_vectors[span_volume,2,:]
                                        ) + scalar_ni_nj[span_volume,np.newaxis]**2 * np.cross(
                                            normals_average,
                                            reference_vectors[span_volume,0,:]
                                        )
        reference_planes[span_volume,:] /= np.linalg.norm(reference_planes[span_volume,:],axis=1)[:,np.newaxis]

        base_y = np.cross(reference_planes[span_volume,:], normals_average)
        base_y /= norm(base_y,axis=1)[:,np.newaxis]
        base_x = np.cross(base_y,reference_planes[span_volume,:])
        reference_bases[span_volume,0,:] = base_x
        reference_bases[span_volume,1,:] = base_y

    return reference_planes,reference_bases

def compute_angle_differences(vectors):
    """
        Compute the principal unsigned value of the angle differences between the vectors listed column-wise in "vectors".
        Parameters:
            vectors : numpy array : N_pointsxN_curvesxN_pointsx3 : The list of the successive vectors we want to compute the successive angle differences. The vectors are along the fourth coordinate, and the third coordinate is used to list the successive vectors.
        Outputs:
            angular_differences : numpy array : N_pointsxN_curvesx(N_points-1) : The angle differences
    """
    N_points,N_curves,N_critical_points,_ = vectors.shape
    angular_differences = np.zeros((N_points,N_curves,N_critical_points-1))
    normalized_vectors = vectors / np.linalg.norm(vectors,axis=3)[:,:,:,np.newaxis]
    normalized_vectors = np.nan_to_num(normalized_vectors,nan=0)
    cos = np.zeros((N_points,N_curves,N_critical_points-1))
    for k in range(N_critical_points - 1):
        # The clipping in [-1,1] is necessary because rounding errors may introduce nan values and raise RuntimeWarnings
        cos[:,:,k] = np.clip(
                (normalized_vectors[:,:,k,:] * normalized_vectors[:,:,k+1,:]).sum(axis=2),
                -1.,1.
            )
    angular_differences = np.abs(np.arccos(cos))
    return angular_differences

def project_on_plane(vector,plane):
    """
        Projects a vector on a hyperplane
        Parameters:
            vector : np array : Nx3 : the list of the vectors we need to express in the planes
            plane : np array : Nx3 : the list of the normals of the planes
        Outputs :
            projection : np array : Nx3 : the projected vectors
    """
    return vector - (plane * vector).sum(axis=1)[:,np.newaxis] * plane

def express_in_plane_coordinates(vector,plane_base):
    """
        Expresses a N-D vector in a (N-1)-D plane given a base of this plane
        Parameters:
            vector : np array : Nx3 : the list of the vectors we need to express in the planes
            plane : np array : Nx2x3 : the list of the bases of the planes
        Outputs :
            projection : np array : Nx2 : the vectors in the coordinate system of the planes
        Remarks :
            1.The vectors in "vector" are supposed to belong to the plane. Unexpected behaviour may occur if this is not the case.
            2. The plane bases are supposed to be orthonormal
    """
    planar_vectors = (vector[:,np.newaxis,:] * plane_base).sum(axis = 2)
    return planar_vectors

def rotate(vectors,theta):
    """
        Applies the same rotation to a list of vectors
        Parameters :
            vectors : Nx2 : the list of vectors
            theta : float : the angle of the rotation
        Outputs :
            rotated : Nx2 : the rotated vectors
    """
    R = np.array([
                    [np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]
                ])
    return (R @ vectors[:,:,np.newaxis]).squeeze()

def compute_turning_points_coefficients(qijs,ti,tj,tangents_orientations):
    """
        Computes the coefficients a0, a1, and a2 defined in Section 3.3, paragraph "Curve complexity" of the article
        Parameters:
            qijs : np array : Nx2 : the vector difference between vertices i and j (oriented in the sense j - i), projected on the reference plane and expressed in the plane's coordinate system
            ti : np array : Nx2 : the pi/2 rotation of the (projection in the reference plane of the) normal of vertex i expressed in the plane's coordinate system
            tj : np array : Nx2 : same, for vertex j
            tangents_orientations : list(list(int)) : 4x2 : the list of the orientations of ti and tj for each Hermite curve.
        Outputs:
            a2 : np array : Nx4 : the array of the a2 coefficients
            a1 : np array : Nx4 : the array of the a1 coefficients
            a0 : np array : Nx4 : the array of the a0 coefficients
        Remarks:
            1. The flipping order defined in tangents_orientations is conserved in the output coefficients
        Example:
            If we set tangents orientations to ((1,1),(-1,-1),(1,-1),(-1,1)), then :
                1. The first column of each coefficient will correspond to the Hermite curve where the tangents at the origin are exactly ti and tj
                2. The second column is the curve with tangents at the origin -ti,-tj
                3. The third corresponds to ti,-tj
                4. The fourth corresponds to -ti,tj
    """
    # In the article, qij = i - j, whereas our convention is the opposite
    v = lambda t,u: -6 * qijs - 4 * t - 2 * u
    w = lambda t,u: 6 * qijs + 3 * (t + u)

    a0 = np.array([np.linalg.det(
                        np.dstack(
                            (
                                si * ti,
                                v(si*ti,sj*tj)
                            )
                        )
                    ) for si,sj in tangents_orientations]).T
    a1 = np.array([2*np.linalg.det(
                        np.dstack(
                            (
                                si*ti,
                                w(si*ti,sj*tj)
                            )
                        )
                    ) for si,sj in tangents_orientations]).T
    a2 = np.array([np.linalg.det(
                        np.dstack(
                            (
                                v(si*ti,sj*tj),
                                w(si*ti,sj*tj)
                            )
                        )
                    ) for si,sj in tangents_orientations]).T

    return a2,a1,a0

def compute_turning_points(a2,a1,a0):
    """
        Computes the turning points of the Hermite curves (roots of a2*X**2 + a1*X + a0 lying in [0,1]). These roots are the candidates for the turning points of the Hermite curves.
        Special cases :
            1. If the polynomial has no root in [0,1], the values are set to (1/3,2/3)
            2. If the (second order) polynomial has one root (r) in [0,1] AND one less than 0, the values are set to (r/2,r)
            3. If the (second order) polynomial has one root (r) in [0,1] AND one more than 1, the values are set to (r, (r+1)/2)
            4. If the (first order) polynomial has one root (r) in [0,1], the values are set to :
                a. (r/2,r) if r >= 0.5
                b. (r,(r+1)/2) if r < 0.5
        Parameters:
            a2 : Nx4 : the a2 coefficients
            a1 : Nx4 : the a1 coefficients
            a0 : Nx4 : the a0 coefficients
        Outputs:
            turning_points : Nx4x2 : the turning points
        Remark:
            The arbitrary-looking behaviours in the special cases are not problematic because theses values
            are going to be used to compute the points where the curves need to be splitted to compute the curvature. Yet,
            over-splitting the curve leads to a correct answer.
    """
    tolerance = 1e-6
    N,n_curves = a2.shape
    turning_points = np.array([[[1/3,2/3]]*n_curves]*N)

    order_1 = (np.abs(a2) < tolerance)
    order_0 = np.logical_and(order_1,np.abs(a1) < tolerance)
    order_1[order_0] = False
    order_2 = np.logical_not(np.logical_or(order_1,order_0))

    delta = (a1**2 - 4*a2*a0)
    order_2_with_real_roots = np.logical_and(order_2,delta > tolerance)
    sign_a2 = a2[order_2_with_real_roots]/np.abs(a2[order_2_with_real_roots])
    turning_points[order_2_with_real_roots] = np.dstack((
        (-a1[order_2_with_real_roots] - sign_a2 * np.sqrt(delta[order_2_with_real_roots]))/(2*a2[order_2_with_real_roots]),
        (-a1[order_2_with_real_roots] + sign_a2 * np.sqrt(delta[order_2_with_real_roots]))/(2*a2[order_2_with_real_roots])
    ))

    # The following variables are for test purposes
    ## (Recommended) Set to True so the behaviour is the one described in the documentation
    documentation_behaviour = True
    ## Set to True so the behaviour is clipping the values between 0 and 1
    ## (you should set documentation_behaviour to False first)
    clip_01 = False
    ## Set to True so there is no post processing of the roots: the returned values are
    ##  1. The roots of the polynomial, whether or not they are in [0,1], if there are any
    ##  2. [1/3,2/3] otherwise
    ## (you should set documentation_behaviour and clip_01 to False first)
    no_postprocessing = False

    if documentation_behaviour:
        low_zero_high_one = np.logical_and(
            order_2_with_real_roots,
            np.logical_and(
                turning_points[:,:,0] < 0,
                np.logical_and(
                    turning_points[:,:,1] >= 0,
                    turning_points[:,:,1] <= 1
                )
            )
        )
        zero_low_one_high = np.logical_and(
            order_2_with_real_roots,
            np.logical_and(
                turning_points[:,:,1] > 1,
                np.logical_and(
                    turning_points[:,:,0] >= 0,
                    turning_points[:,:,0] <= 1
                )
            )
        )
        both_outside_01 = np.logical_and(
            order_2_with_real_roots,
            np.logical_or(
                (turning_points < 0).all(axis=2),
                np.logical_or(
                    (turning_points > 1).all(axis=2),
                    np.logical_and(
                        turning_points[:,:,0] < 0,
                        turning_points[:,:,1] > 1
                    )
                )
            )
        )

        turning_points[low_zero_high_one,0] = turning_points[low_zero_high_one,1] / 2
        turning_points[zero_low_one_high,1] = (turning_points[zero_low_one_high,0] + 1)/2
        turning_points[both_outside_01,:] = [1/3,2/3]


        turning_points[order_1] = (-a0[order_1] / a1[order_1])[:,np.newaxis]
        less_half = np.logical_and(
            order_1,
            turning_points[:,:,0] < 0.5
        )
        more_half = np.logical_and(
            order_1,
            turning_points[:,:,0] >= 0.5
        )
        outside_01 = np.logical_and(
            order_1,
            np.logical_or(
                turning_points[:,:,0] < 0,
                turning_points[:,:,0] > 1
            )
        )
        if less_half.any():
            turning_points[less_half,:] = np.hstack((turning_points[less_half,0][:,np.newaxis],(turning_points[less_half,0][:,np.newaxis] + 1) / 2))
        if more_half.any():
            turning_points[more_half,:] = np.hstack((turning_points[more_half,1][:,np.newaxis]/2,turning_points[more_half,1][:,np.newaxis]))
        turning_points[outside_01,:] = [1/3,2/3]

    elif clip_01:
        # This option ensures that the turning points are in [0,1], but does not provide a satisfying sampling of the interval.
        # If the curve has a complexity greater than pi, it is going to be invisble in the successive angles difference computation.
        # For this reason, the default behaviour is to provide a sampling of [0,1]
        warnings.warn('You have selected the "clip_01" option for the turning points. This should be only for test purposes.')
        turning_points = np.clip(turning_points,0,1)
    elif no_postprocessing:
        # This option does not ensure that the turning points are in [0,1]. This can lead to errors in evaluating the complexity, because the
        # angular differences may be computed with tangent vectors that actually do not belong to the Hermite curve.
        # I suspect that it is this mode that has been selected in the original authors' article, because one finds the values claimed in the article's
        # Figure 2 when this option is enabled. See the test "tests.konig_tests.test_hermite_curves_complexities" for proof.
        warnings.warn('You have selected the "no_postprocessing" option for the turning points. This should be only for test purposes.')
    else:
        raise ValueError('Please select a postprocessing option for the turning points computation. The default choice should be documentation_behaviour.')


    return turning_points

def compute_tangents_at_critical_points(pis_proj,pjs_proj,ti,tj,turning_points,tangents_orientations):
    """"
        Computes the tangents at the critical points, that is the turning points and 0 and 1
        Parameters:
            pis_proj : np array : Nx2 : the list of the origins of the Hermite curves, expressed in the reference plane's coordinate system
            pjs_proj : np array : Nx2 : the list of the end points of the Hermite curves, expressed in the reference plane's coordinate system
            ti : np array : Nx2 : the pi/2 rotation of the (projection in the reference plane of the) normal of vertex i expressed in the plane's coordinate system
            tj : np array : Nx2 : the pi/2 rotation of the (projection in the reference plane of the) normal of vertex j expressed in the plane's coordinate system
            turning_points : Nx4x2 : the turning points of the Hermite curves
            tangents_orientations :  list(list(int)) : 4x2 : the list of the orientations of ti and tj for each Hermite curve.
        Outputs:
            tangents_at_critical_points : np array : Nx4x4x3 : the tangents at the critical points.
    """
    # There is a typo in the original article in the expression of c_k'(t).
    # One should swap q_2 and m_1 : this is done here.
    hermite_ = lambda t,q1,q2,m1,m2 : (6*t**2 - 6*t) * q1 + (-6*t**2 + 6*t) * q2 + (3*t**2 - 4*t + 1) * m1 + (3*t**2 - 2*t) * m2
    N = pis_proj.shape[0]
    tangents_at_critical_points = np.zeros((N,len(tangents_orientations),4,2))
    for k in range(len(tangents_orientations)):
        si,sj = tangents_orientations[k]
        tangents_at_critical_points[:,k,0,:] = si*ti
        tangents_at_critical_points[:,k,3,:] = sj*tj
        for i in range(1,3):
            tangents_at_critical_points[:,k,i,:] = hermite_(
                    turning_points[:,k,i-1][:,np.newaxis],
                    pis_proj,
                    pjs_proj,
                    si*ti,
                    sj*tj
                )
    return tangents_at_critical_points

def compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_planes,reference_bases):
    """
        Computes the complexities of each Hermite curve for each edge in the graph.
        Parameters:
            pi : np array : Px3 : the list of the origins of the Hermite curves
            pj : np array : Px3 : the list of the end points of the Hermite curves
            cloud : np array : Nx3 : should not have duplicated points
            normals : np array : Nx3 : should be normalized
            reference_planes : np array : Nx3 : the list of the reference planes' normals, normalized.
            reference_bases : np array : Nx2x3 : the list of the reference planes' bases, normalized. The tuple (base,normal) should be orthonormal.
    """

    N = pi_s.shape[0]
    nis = normals[pi_s,:]
    njs = normals[pj_s,:]
    eijs = cloud[pj_s,:] - cloud[pi_s,:]
    tangents_orientations = ((1,1),(-1,-1),(1,-1),(-1,1))

    nis_proj = project_on_plane(nis,reference_planes)
    njs_proj = project_on_plane(njs,reference_planes)
    pis_proj = project_on_plane(cloud[pi_s,:],reference_planes)
    pjs_proj = project_on_plane(cloud[pj_s,:],reference_planes)

    nis_proj = express_in_plane_coordinates(nis_proj,reference_bases)
    njs_proj = express_in_plane_coordinates(njs_proj,reference_bases)
    pis_proj = express_in_plane_coordinates(pis_proj,reference_bases)
    pjs_proj = express_in_plane_coordinates(pjs_proj,reference_bases)
    qijs = pjs_proj - pis_proj

    ti = rotate(nis_proj,np.pi/2)
    tj = rotate(njs_proj,np.pi/2)
    ti = ti * 2*np.linalg.norm(eijs,axis=-1)[:,np.newaxis]
    tj = tj * 2*np.linalg.norm(eijs,axis=-1)[:,np.newaxis]

    a2,a1,a0 = compute_turning_points_coefficients(qijs,ti,tj,tangents_orientations)
    turning_points = compute_turning_points(a2,a1,a0)

    tangents_at_critical_points = compute_tangents_at_critical_points(pis_proj,pjs_proj,ti,tj,turning_points,tangents_orientations)

    angular_differences = compute_angle_differences(tangents_at_critical_points)
    complexities = angular_differences.sum(axis=2)

    return complexities


def compute_riemannian_mst(cloud,normals,n_neighbors,verbose=False):
    """
        Computes the Riemannian Minimum Spanning Tree for the Konig and Gumhold method
        Parameters :
            cloud : numpy array : Nx3 : Should not have duplicated points
            normals : numpy array : Nx3 : Should be normalized
            n_neighbors : int : used to compute the k-neighbors graph
            verbose : boolean : unused
        Outputs :
            riemannian_mst : scipy.sparse.csr_matrix(float) : NxN : Symmetric MST whose weight at edge (i,j) is the unlikelihood to propagate the normal orientation from i to j
            flip_criterion : scipy.sparse.csr_matrix(boolean) : NxN : Boolean array that is True at (i,j) iff the normal of vertex j should be RELATIVELY flipped when visiting it from vertex i. The "relatively" needs to be understood in terms of the orientation of normal i in the parameter "normals"
    """
    epsilon = 1e-4

    emst = compute_emst(cloud)
    symmetric_emst = (emst + emst.T)

    symmetric_kgraph = symmetric_kneighbors_graph(cloud,n_neighbors)
    enriched = symmetric_emst + symmetric_kgraph
    enriched = enriched.tocoo()
    pi_s = enriched.row
    pj_s = enriched.col

    reference_planes,reference_bases = compute_reference_planes(pi_s,pj_s,cloud,normals)
    hermite_curves_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_planes,reference_bases)
    c_keep = np.min(hermite_curves_complexities[:,:2],axis=1).reshape((-1,1))
    c_flip = np.min(hermite_curves_complexities[:,2:],axis=1).reshape((-1,1))
    c_s = np.hstack((c_keep,c_flip))
    # To compute the weights, we add an epsilon compared to the article's proposition
    # because we chose a scipy.sparse matrix for the implementation : if a weight
    # is set to 0, vertices are considered not to be connected (whereas the edge
    # should exist and have a 0 weight)
    riemannian_weights = (np.min(c_s,axis=1) + epsilon)/(np.max(c_s,axis=1)+epsilon)
    flip_criterion_values = (c_flip < c_keep).squeeze()

    riemannian_graph = csr_matrix((riemannian_weights,(pi_s,pj_s)),shape = (cloud.shape[0],cloud.shape[0]))
    riemannian_mst = minimum_spanning_tree(riemannian_graph,overwrite = True) # overwrite = True for performance
    # The scipy.minimum_spanning_tree function returns a triangular (superior) graph.
    # This is more memory-efficient, but our implementation of the graph search requires
    # a symmetric graph.
    riemannian_mst = riemannian_mst + riemannian_mst.T

    flip_criterion = csr_matrix((flip_criterion_values,(pi_s,pj_s)))

    return riemannian_mst,flip_criterion

def konig_orientation(cloud,normals,n_neighbors,verbose=False):
    """
        Orients the normals using the Konig and Gumhold method.
        Parameters:
            cloud : np array : Nx3 : should not contain duplicates
            normals : np array : Nx3 : should be normalized
            n_neighbors : int : the number of neighbors used in the graph computation
            verbose : boolean : unused
        Outputs:
            normals_o : np array : Nx3 : the oriented normals
    """
    normals_o = normals.copy()
    was_flipped = np.zeros(len(cloud),dtype=np.bool)

    riemannian_mst,flip_criterion = compute_riemannian_mst(cloud,normals,n_neighbors,verbose)

    seed_index = np.argmax(cloud[:,2])
    # We arbitrarily set the direction of the seed so it points towards increasing z values
    ez = np.array([0,0,1])
    if normals_o[seed_index,:].T @ ez < 0:
        normals_o[seed_index,:] *= -1
        was_flipped[seed_index] = True

    for parent_index,point_index in acyclic_graph_dfs_iterator(riemannian_mst,seed_index):
        if parent_index is None:
            flip = False
        else:
            # The criterion is True iff we should change the RELATIVE orientation of the normals
            should_flip = flip_criterion[parent_index,point_index]
            flip = (should_flip and not was_flipped[parent_index]) or (was_flipped[parent_index] and not should_flip)
        if flip:
            was_flipped[point_index] = True
            normals_o[point_index,:] *= -1

    return normals_o
