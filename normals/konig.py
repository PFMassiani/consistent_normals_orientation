from scipy.sparse import csr_matrix
import numpy as np

from normals.common import *

def compute_reference_planes(pi_s,pj_s,cloud,normals):
    tolerance = 1e-6
    n_couples = len(pi_s)

    reference_vectors = np.dstack((cloud[pj_s,:] - cloud[pi_s,:],normals[pi_s,:],normals[pj_s,:]))
    for d in range(3):
        reference_vectors[:,:,d] = reference_vectors[:,:,d] / np.linalg.norm(reference_vectors[:,:,d],axis=1)[:,np.newaxis]
    reference_vectors = np.swapaxes(reference_vectors,1,2)
    # Now, reference_vectors[n,k,:] is a vector of the couple (i,j) (in the graph) number k. This vector is :
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

    reference_planes = np.zeros((n_couples,3)) # For each couple, the normal of its reference plane
    reference_bases = np.zeros((n_couples,2,3))
    # For each collinear group of vectors, we find a vector which is different from them and use the cross product to get an orthogonal vector
    if are_collinear.any():
        collinears = reference_vectors[are_collinear,:,:].sum(axis=1) / 3 # we consider the average vector
        rotation_around_ex = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        rotation_around_ez = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        ez = np.array([0,0,1]).reshape((-1,1))
        equal_to_ez = ((collinears @ ez) > 1 - tolerance).reshape((-1)) # these are not going to be affected by the roation around ez, so we need to deal with them differently
        different_from_collinears = collinears.copy()
        # print(equal_to_ez)
        # print(different_from_collinears.shape,equal_to_ez.shape,rotation_around_ez.shape )
        different_from_collinears[np.logical_not(equal_to_ez),:] = different_from_collinears[np.logical_not(equal_to_ez),:] @ rotation_around_ez
        different_from_collinears[equal_to_ez,:] = different_from_collinears[equal_to_ez,:] @ rotation_around_ex
        # Now, different_from_collinears contains vectors that are all significantly different from the ones in collinears
        reference_planes[are_collinear,:] = np.cross(collinears,different_from_collinears)
        reference_planes[are_collinear,:] /= np.linalg.norm(reference_planes[are_collinear,:],axis=1)[:,np.newaxis]
        base_y = np.cross(reference_planes[are_collinear,:],collinears)
        base_y /= np.linalg.norm(base_y,axis=1)[:,np.newaxis]
        base_x = collinears / np.linalg.norm(collinears,axis=1)[:,np.newaxis]
        reference_bases[are_collinear,0,:] = base_x
        reference_bases[are_collinear,1,:] = base_y
        assert (
            np.abs(
                (np.cross(
                    reference_bases[are_collinear,0,:],
                    reference_bases[are_collinear,1,:]
                )*reference_planes[are_collinear]
            ).sum(axis=-1)-1) < tolerance).all()

    if span_plane.any():#reference_planes[span_plane,:]
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
        reference_planes[span_plane,:] /= np.linalg.norm(reference_planes[span_plane,:],axis=1)[:,np.newaxis]
        reference_bases[span_plane,0,:] = reference_vectors[span_plane,1,:]
        reference_bases[span_plane,1,:] = np.cross(reference_planes[span_plane],reference_bases[span_plane,0,:])
        assert (np.abs((np.cross(reference_bases[span_plane,0,:],reference_bases[span_plane,1,:])*reference_planes[span_plane]).sum(axis=-1)-1) < tolerance).all()

    if span_volume.any():
        with np.errstate(invalid='ignore'):
            signs = (scalar_ni_nj / np.abs(scalar_ni_nj))[span_volume].reshape((-1,1))
        signs = np.nan_to_num(signs,nan=1)
        normals_average = (
            reference_vectors[span_volume,1,:] + signs * reference_vectors[span_volume,2,:]
        ) / np.linalg.norm(reference_vectors[span_volume,1,:] + signs * reference_vectors[span_volume,2,:],axis=1)[:,np.newaxis]
        reference_planes[span_volume] = np.cross(
                                            reference_vectors[span_volume,1,:],
                                            reference_vectors[span_volume,2,:]
                                        ) + scalar_ni_nj[span_volume,np.newaxis]**2 * np.cross(
                                            normals_average,
                                            reference_vectors[span_volume,0,:]
                                        )
        reference_planes[span_volume,:] /= np.linalg.norm(reference_planes[span_volume,:],axis=1)[:,np.newaxis]
        base_y = np.cross(reference_planes[span_volume,:], normals_average)
        base_y /= np.linalg.norm(base_y,axis=1)[:,np.newaxis]
        base_x = np.cross(base_y,reference_planes[span_volume,:])
        reference_bases[span_volume,0,:] = base_x
        reference_bases[span_volume,1,:] = base_y
        assert (np.abs((np.cross(reference_bases[span_volume,0,:],reference_bases[span_volume,1,:])*reference_planes[span_volume]).sum(axis=-1)-1) < tolerance).all()

    return reference_planes,reference_bases


def rotation_matrices_around_vectors(u,theta):
    # See https://fr.wikipedia.org/wiki/Matrice_de_rotation#En_dimension_trois for explanations
    # u.shape = (N,3)
    v = u[:,:,np.newaxis] # We have a collection of N column vectors, and we are computing a rotation for each of them
    P = v @ v.swapaxes(1,2) # Equivalent to w @ w.T for each vector w in v. P is a collection of N 3x3 matrices
    I = np.eye(3)
    E01asym = np.array([[0,1,0],[0,-1,0],[0,0,0]]) # For uz
    E02asym = np.array([[0,0,1],[0,0,0],[-1,0,0]]) # For uy
    E12asym = np.array([[0,0,0],[0,0,1],[0,-1,0]]) # For ux
    ex = np.array([1,0,0]).reshape((-1,1))
    ey = np.array([0,1,0]).reshape((-1,1))
    ez = np.array([0,0,1]).reshape((-1,1))
    Q = (ez.T @ v) * E01asym + (ey.T @ v) * E02asym + (ex.T @ v) * E12asym

    return P + np.cos(theta) * (I - P) + np.sin(theta) * Q

def compute_angle_differences(vectors):
    # vectors.shape = (N,4,4,3), and the vectors are along the last coordinate
    # We should compute 3 angular differences
    N_points,N_curves,N_critical_points,_ = vectors.shape
    angular_differences = np.zeros((N_points,N_curves,N_critical_points-1))
    normalized_vectors = vectors / np.linalg.norm(vectors,axis=3)[:,:,:,np.newaxis]
    normalized_vectors = np.nan_to_num(normalized_vectors,nan=0)
    cos = np.zeros((N_points,N_curves,N_critical_points-1))
    for k in range(N_critical_points - 1):
        cos[:,:,k] = np.clip(
                (normalized_vectors[:,:,k,:] * normalized_vectors[:,:,k+1,:]).sum(axis=2),
                -1.,1.
            )
    angular_differences = np.abs(np.arccos(cos))
    return angular_differences

def project_on_plane(vector,plane):
    return vector - (plane * vector).sum(axis=1)[:,np.newaxis] * plane

def express_in_plane_coordinates(vector,plane_base):
    """
    vector.shape = (N,3)
    plane_normal.shape = (N,3)
    plane_base.shape = (N,2,3)
    The plane_base base is supposed to be orthonormal.
    """
    planar_vectors = (vector[:,np.newaxis,:] * plane_base).sum(axis = 2)
    return planar_vectors

def rotate(vectors,theta):
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return (R @ vectors[:,:,np.newaxis]).squeeze()

def compute_turning_points_coefficients(qijs,ti,tj,tangents_orientations):
    # Remark : compared to the article, we have eij = -(q1 - q2), so the signs of eij are changed
    v = lambda t,u: -6 * qijs - 4 * t - 2 * u
    w = lambda t,u: 6 * qijs + 3 * (t + u)
    # Coefficients that define the polynomial defining the turning points
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
    tolerance = 1e-6
    N,n_curves = a2.shape
    turning_points = np.ones((N,n_curves,2)) * 0.5

    order_1 = (np.abs(a2) < tolerance)
    order_0 = np.logical_and(order_1,np.abs(a1) < tolerance)
    order_1[order_0] = False
    order_2 = np.logical_not(np.logical_or(order_1,order_0))

    delta = (a1**2 - 4*a2*a0)
    order_2_with_real_roots = np.logical_and(order_2,delta > tolerance)
    turning_points[order_2_with_real_roots] = np.dstack(((-a1[order_2_with_real_roots] - np.sqrt(delta[order_2_with_real_roots]))/(2*a2[order_2_with_real_roots]),(-a1[order_2_with_real_roots] + np.sqrt(delta[order_2_with_real_roots]))/(2*a2[order_2_with_real_roots])))

    turning_points[order_1] = (-a0[order_1] / a1[order_1])[:,np.newaxis]

    return turning_points

def compute_tangents_at_critical_points(pis_proj,pjs_proj,ti,tj,turning_points,tangents_orientations):
    # Remark : there is a typo in the original article in the expression of c_k'(t). One should swap q_2 and m_1.
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
    N = pi_s.shape[0]
    nis = normals[pi_s,:]
    njs = normals[pj_s,:]
    eijs = cloud[pj_s,:] - cloud[pi_s,:]
    tangents_orientations = ((1,1),(-1,-1),(1,-1),(-1,1))

    # Step 1 : compute projections of normals and points in reference plane
    nis_proj = project_on_plane(nis,reference_planes)
    njs_proj = project_on_plane(njs,reference_planes)
    pis_proj = project_on_plane(cloud[pi_s,:],reference_planes)
    pjs_proj = project_on_plane(cloud[pj_s,:],reference_planes)

    nis_proj = express_in_plane_coordinates(nis_proj,reference_bases)
    njs_proj = express_in_plane_coordinates(njs_proj,reference_bases)
    pis_proj = express_in_plane_coordinates(pis_proj,reference_bases)
    pjs_proj = express_in_plane_coordinates(pjs_proj,reference_bases)
    qijs = pjs_proj - pis_proj
    # Step 2 : compute t_i and t_j
    ti = rotate(nis_proj,np.pi/2)
    tj = rotate(njs_proj,np.pi/2)
    ti = ti * 2*np.linalg.norm(eijs,axis=-1)[:,np.newaxis]
    tj = tj * 2*np.linalg.norm(eijs,axis=-1)[:,np.newaxis]
    # Step 4 : compute the points where there can be turning points
    a2,a1,a0 = compute_turning_points_coefficients(qijs,ti,tj,tangents_orientations)
    turning_points = compute_turning_points(a2,a1,a0)
    # Step 5 : at these points, compute the derivatives and the successive angle differences
    tangents_at_critical_points = compute_tangents_at_critical_points(pis_proj,pjs_proj,ti,tj,turning_points,tangents_orientations)
    # Step 6 : sum the absolute values of these angle differences. The result is the complexity.
    angular_differences = compute_angle_differences(tangents_at_critical_points)
    complexities = angular_differences.sum(axis=2)

    # for n in range(len(turning_points)):
    #     print(ti[n,:])
    #     print(tj[n,:])
    #     for k in range(turning_points.shape[1]):
    #         print('Points\n',turning_points[n,k,:])
    #         print('Tangents\n',tangents_at_critical_points[n,k,:,:])
    #         print('Angles\n',angular_differences[n,k,:])

    return complexities


def compute_riemannian_mst(cloud,normals,n_neighbors,eps=1e-4,verbose=False):
    epsilon = 1e-4
    # Step 1 : compute the EMST of the cloud
    emst = compute_emst(cloud)
    symmetric_emst = (emst + emst.T)
    # Step 2 : enrich the EMST with the k-neighborhood graph
    symmetric_kgraph = symmetric_kneighbors_graph(cloud,n_neighbors)
    enriched = symmetric_emst + symmetric_kgraph
    # Step 3 : discard the weights, and replace them with 1 - |n_i . n_j|
    enriched = enriched.tocoo()
    pi_s = enriched.row
    pj_s = enriched.col

    reference_planes,reference_bases = compute_reference_planes(pi_s,pj_s,cloud,normals) # returns an array of normals of the reference planes of size [len(pi_s), 3]
    # assert (np.abs((reference_planes * np.array([[0,0,1]])).sum(axis=1)) > 1 - 1e-2).all()
    hermite_curves_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_planes,reference_bases) # returns an array of the complexities, in the form [len(pi_s),4]
    c_keep = np.min(hermite_curves_complexities[:,:2],axis=1).reshape((-1,1))
    c_flip = np.min(hermite_curves_complexities[:,2:],axis=1).reshape((-1,1))
    c_s = np.hstack((c_keep,c_flip))
    riemannian_weights = (np.min(c_s,axis=1) + epsilon)/(np.max(c_s,axis=1)+epsilon) # add epsilon here
    flip_criterion_values = (c_flip < c_keep).squeeze()

    riemannian_graph = csr_matrix((riemannian_weights,(pi_s,pj_s)),shape = (cloud.shape[0],cloud.shape[0]))
    riemannian_mst = minimum_spanning_tree(riemannian_graph,overwrite = True) # overwrite = True for performance
    riemannian_mst = riemannian_mst + riemannian_mst.T # We symmetrize the graph so it is not oriented

    flip_criterion = csr_matrix((flip_criterion_values,(pi_s,pj_s)))
    return riemannian_mst,flip_criterion

def konig_orientation(cloud,normals,n_neighbors,verbose=False):
    normals_o = normals.copy() # oriented normals
    was_flipped = np.zeros(len(cloud),dtype=np.bool)

    # Step 1 : compute the riemannian mst of the cloud
    riemannian_mst,flip_criterion = compute_riemannian_mst(cloud,normals,n_neighbors,verbose)

    # Step 2 : select seed and its orientation
    seed_index = np.argmax(cloud[:,2])
    ez = np.array([0,0,1])
    if normals_o[seed_index,:].T @ ez < 0:
        normals_o[seed_index,:] *= -1 # We arbitrarily set the direction of the seed so it points towards increasing z values
        was_flipped[seed_index] = True
    # Step 3 : traverse the MST depth first order by assigning a consistent orientation
    for parent_index,point_index in acyclic_graph_dfs_iterator(riemannian_mst,seed_index):
        if parent_index is None:
            flip = False
        else:
            should_flip = flip_criterion[parent_index,point_index]
            flip = (should_flip and not was_flipped[parent_index]) or (was_flipped[parent_index] and not should_flip)
        if flip:
            was_flipped[point_index] = True
            normals_o[point_index,:] *= -1

    return normals_o
