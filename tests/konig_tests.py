# Code by Pierre-François Massiani

import numpy as np
from normals.konig import *
from normals.common import *
from tests.toy_clouds import *
from utils.ply import write_ply


tolerance = 1e-6

def arediff(v1,v2,tol=1e-6):
    return (np.abs(v1 - v2) > tol).any()

def test_reference_planes_computation(verbose=False):
    """
        Tests the function compute_reference_planes on the examples of Figure 2
    """
    tolerance = 1e-6
    ex = np.array([1,0,0],dtype=np.float)
    ey = np.array([0,1,0],dtype=np.float)
    ez = np.array([0,0,1],dtype=np.float)
    def test_normal(cloud_function,comparison_vector,scalar_with_comparison,test_number):
        cloud,normals = cloud_function()
        pi_s = [0]
        pj_s = [1]

        actual_reference_plane,actual_reference_base = compute_reference_planes(pi_s,pj_s,cloud,normals)

        plane_normal = actual_reference_plane[0,:]
        if np.abs(np.abs(plane_normal @ comparison_vector) - scalar_with_comparison) > tolerance:
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('---- True normal: +/-',comparison_vector)
                print('---- Computed normal:',plane_normal)
                print('---- Difference:',comparison_vector - plane_normal)
            return False

        x = actual_reference_base[0,0,:]
        y = actual_reference_base[0,1,:]
        scalars = np.array([np.abs(x@y),np.abs(x@plane_normal),np.abs(plane_normal@y)])
        if any(scalars > tolerance):
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('Plane base:\n',(x,y))
                print('Plane normal:\n',plane_normal)
                print('Scalar products:\n',scalars)
            return False
        norms = np.linalg.norm(np.vstack((x,y,plane_normal)),axis=1)
        if any(np.abs(norms-1) > tolerance):
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('Plane base:\n',(x,y))
                print('Plane normal:\n',plane_normal)
                print('Norms:\n',norms)
            return False
        return True

    passed = test_normal(toy_cloud_2,ez,1,0)
    passed = test_normal(toy_cloud_3,ez,1,1) and passed
    passed = test_normal(toy_cloud_4,ex,0,2) and passed
    passed = test_normal(toy_cloud_5,ex,1,3) and passed
    return passed

def test_hermite_curves_complexities(verbose=False):
    """
        Tests the function compute_hermite_curves_complexities on the examples of Figure 2
        Remark:
            Considering the difficulty in evaluating a curve's complexity, the true values for
            the complexities in this function have been calibrated using the test "test_u_computation"
    """
    tolerance = 1e-3
    def test_complexity(cloud_function,true_complexities,test_number):
        true_complexities = np.array(true_complexities)
        cloud,normals = cloud_function()
        pi_s = np.array([0])
        pj_s = np.array([1])

        reference_plane,reference_base = compute_reference_planes(pi_s,pj_s,cloud,normals)
        actual_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_plane,reference_base).squeeze()
        if (np.nan_to_num((np.abs(actual_complexities - true_complexities)),nan=np.inf) > tolerance).any():
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('True complexities:',true_complexities)
                print('Actl complexities:',actual_complexities)
                ckeep = np.min(actual_complexities[:2])
                cflip = np.min(actual_complexities[2:])
                u = np.min((ckeep,cflip)) / np.max((ckeep,cflip))
                true_ckeep = np.min(true_complexities[:2])
                true_cflip = np.min(true_complexities[2:])
                true_u = np.min((true_ckeep,true_cflip)) / np.max((true_ckeep,true_cflip))
                print('True u:',true_u)
                print('Actl u:',u)
                print('True keeping:',ckeep<=cflip)
                print('Actl keeping:',true_ckeep<=true_cflip)
            return False
        return True
    pi = np.pi
    passed = test_complexity(toy_cloud_2,(2*pi,0,pi,pi),0)
    passed = test_complexity(toy_cloud_3,(3*pi/2,pi/2,1.1793*pi,1.1793*pi),1) and passed
    passed = test_complexity(toy_cloud_4,(pi,pi,1.3743*pi,1.3742*pi),2) and passed
    passed = test_complexity(toy_cloud_5,(3*pi/2,3*pi/2,3*pi/2,3*pi/2),3) and passed
    return passed

def test_u_computation(verbose = False):
    """
        Tests the computation of the normal propagation criterion on the examples of Figure 2.
    """
    tolerance = 1e-2
    def test_u(cloud_function,true_u,test_number):
        true_u = np.array(true_u)
        cloud,normals = cloud_function()
        pi_s = np.array([0])
        pj_s = np.array([1])

        reference_plane,reference_base = compute_reference_planes(pi_s,pj_s,cloud,normals)
        actual_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_plane,reference_base).squeeze()
        ckeep = np.min(actual_complexities[:2])
        cflip = np.min(actual_complexities[2:])
        u = np.min((ckeep,cflip)) / np.max((ckeep,cflip))
        if np.abs(u-true_u) > tolerance:
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('True u:',true_u)
                print('Actl u:',u)
                print('Complexities:',actual_complexities)
                print('Should keep:',ckeep<=cflip)
            return False
        return True
    pi = np.pi
    passed = test_u(toy_cloud_2,0.,0)
    passed = test_u(toy_cloud_3,0.42,1) and passed
    passed = test_u(toy_cloud_4,0.73,2) and passed
    passed = test_u(toy_cloud_5,1.,3) and passed
    return passed

def figure2_u_discrepancies(verbose = True):
    """
        Shows the discrepancies in the values for u obtained in the cases presented on Figure 2
    """
    def compute_discrepancies(cloud_function,test_number):
        cloud,normals = cloud_function()
        pi_s = np.array([0])
        pj_s = np.array([1])

        reference_plane,reference_base = compute_reference_planes(pi_s,pj_s,cloud,normals)
        nopost_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_plane,reference_base,critical_postprocessing="none").squeeze()
        clip_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_plane,reference_base,critical_postprocessing="clip").squeeze()
        docpost_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_plane,reference_base,critical_postprocessing="documentation").squeeze()
        ckeep_no = np.min(nopost_complexities[:2])
        cflip_no = np.min(nopost_complexities[2:])
        ckeep_cl = np.min(clip_complexities[:2])
        cflip_cl = np.min(clip_complexities[2:])
        ckeep_doc = np.min(docpost_complexities[:2])
        cflip_doc = np.min(docpost_complexities[2:])

        u_no = np.min((ckeep_no,cflip_no)) / np.max((ckeep_no,cflip_no))
        u_clip = np.min((ckeep_cl,cflip_cl)) / np.max((ckeep_cl,cflip_cl))
        u_doc = np.min((ckeep_doc,cflip_doc)) / np.max((ckeep_doc,cflip_doc))
        if verbose:
            print('--- Test {} ---'.format(test_number))

            print('u (nopost):',u_no)
            print('u (clip01):',u_clip)
            print('u (docum.):',u_doc)
            print('Complexities (nopost):',nopost_complexities)
            print('Complexities (clip01):',clip_complexities)
            print('Complexities (docum.):',docpost_complexities)
    if verbose:
        print('------\n\
 Legend:\n* nopost : the critical points are exactly the roots of\
 the polynomial, or [1/3,2/3] if there are none\n* clip01 : The critical points are eh roots\
 of the polynomial, bu clipped between [0,1], with no considerations of sampling\n\* docum. :\
 The critical points are the ones described in the documentation of compute_critical_points.\
 They belong to [0,1]\n------')
    pi = np.pi
    compute_discrepancies(toy_cloud_2,0)
    compute_discrepancies(toy_cloud_3,1)
    compute_discrepancies(toy_cloud_4,2)
    compute_discrepancies(toy_cloud_5,3)


def test_projection(verbose=False):
    """
        Test the function project_on_plane with the projection of the normals and eij on the reference plane on the examples of Figure 2.
    """
    tolerance = 1e-6
    def test_proj(cloud_function,true_nproj,true_piproj,true_pjproj,test_number):
        true_nproj = np.array(true_nproj)
        true_piproj = np.array(true_piproj)
        true_pjproj = np.array(true_pjproj)
        cloud,normals = cloud_function()
        pi = np.array([0])
        pj = np.array([1])
        reference_plane,_ = compute_reference_planes(pi,pj,cloud,normals)
        nproj = project_on_plane(normals,reference_plane)
        piproj = project_on_plane(cloud[pi,:],reference_plane)
        pjproj = project_on_plane(cloud[pj,:],reference_plane)

        norm = np.linalg.norm
        diff = np.array([norm(piproj-true_piproj),norm(pjproj - true_pjproj),norm(nproj[0,:] - true_nproj[0,:]),norm(nproj[1,:] - true_nproj[1,:])])
        if (diff > tolerance).any():
            if verbose:
                print('--- Test {} ---'.format(test_number))
                print('---- True piproj:\n',true_piproj)
                print('---- Actual piproj:\n',piproj)
                print('---- True pjproj:\n',true_pjproj)
                print('---- Actual pjproj:\n',pjproj)
                print('---- True normals:\n',true_nproj)
                print('---- Actual normals:\n',nproj)
            return False
        return True
    passed = test_proj(toy_cloud_2,[
        [0,1,0],
        [0,1,0]
    ],[[0,0,0]],[[1,0,0]],0)
    passed = test_proj(toy_cloud_5,[
        [0,1,0],
        [0,0,1]
    ],[[0,0,0]],[[0,0,0]],1) and passed
    return passed

def test_plane_coordinates(verbose=True):
    """
        Test the function express_in_plane_coordinates
    """
    tol = 1e-6
    u = np.array([[1,0,0]])
    n = np.array([[1,1,0]])/np.sqrt(2)
    base = np.array([[[0,0,1],[1/np.sqrt(2),-1/np.sqrt(2),0]]])
    uproj = project_on_plane(u,n)
    planar_coords = express_in_plane_coordinates(uproj,base)

    true_proj = np.array([[1/2,-1/2,0]])
    true_planar = np.array([[0,1/np.sqrt(2)]])

    if arediff(uproj,true_proj):
        if verbose:
            print('---- True projection:\n',true_proj)
            print('---- Actual projection:\n',uproj)
        return False
    if arediff(planar_coords,true_planar):
        if verbose:
            print('---- True planar:\n',true_planar)
            print('---- Actual planar:\n',planar_coords)
        return False
    return True

def test_rotation(verbose = True):
    """
        Tests the function rotate
    """
    vectors = np.eye(2)
    rotated = rotate(vectors,np.pi/2)
    true_rotated = np.array([[0,1],[-1,0]])
    if arediff(rotated,true_rotated):
        print('---- Rotated:\n',rotated)
        print('---- True rotated:\n',true_rotated)
        return False
    return True

def test_tangents_at_critical_points(verbose=True):
    """
        Tests the function compute_tangents_at_critical_points.
        Since we do not have true values to compare the results with, this test is
        verbose : its output should be carefully inspected to make sure the computed
        values are coherent with the geometry. Hence, the test is not automatic.
    """
    cloud,normals = toy_cloud_2()
    pi_index = np.array([0])
    pj_index = np.array([1])

    n,base = compute_reference_planes(pi_index,pj_index,cloud,normals)
    pi = cloud[pi_index,:]
    pj = cloud[pj_index,:]
    ni = normals[pi_index,:]
    nj = normals[pj_index,:]
    pi_proj = express_in_plane_coordinates(pi,base)
    pj_proj = express_in_plane_coordinates(pj,base)
    ni_proj = express_in_plane_coordinates(ni,base)
    nj_proj = express_in_plane_coordinates(nj,base)
    ti = 2*rotate(ni_proj,np.pi/2)
    tj = 2*rotate(nj_proj,np.pi/2)

    critical_points = np.array([[[0.5,0.5]]*4])
    orientations = ((1,1),(-1,-1),(1,-1),(-1,1))

    tangents = compute_tangents_at_critical_points(pi_proj,pj_proj,ti,tj,critical_points,orientations)
    angular_differences = compute_angle_differences(tangents)
    total_differences = angular_differences.sum(axis=2)

    print('Base:\n',base)
    print('Normals:\n',ni_proj,'\n',nj_proj)
    for n_curve in range(tangents.shape[1]):
        print('-- Curve #{}'.format(n_curve))
        for n_point in range(tangents.shape[2]):
            print(tangents[0,n_curve,n_point,:])
        print('Angles\n',angular_differences[0,n_curve,:],total_differences[0,n_curve])


def test_angle_differences_computations(verbose=False):
    """
        Tests the function compute_angle_differences on toy examples.
    """
    vectors = np.array([
        [
            [
                [1,0],
                [0,1],
                [-1,0],
                [0,-1]
            ],
            [
                [1,0],
                [0,-1],
                [-1,0],
                [0,1]
            ],
            [
                [1,0],
                [np.sqrt(3)/2,1/2],
                [1/2,np.sqrt(3)/2],
                [0,1]
            ],
            [
                [1,0],
                [-1/np.sqrt(2),1/np.sqrt(2)],
                [0,-1],
                [1,0]
            ]
        ]
    ])
    angular_differences = compute_angle_differences(vectors)
    total_differences = angular_differences.sum(axis=2)
    pi = np.pi
    true_diffs = np.array([3*pi/2,3*pi/2,pi/2,2*pi])
    if arediff(total_differences.squeeze(),true_diffs):
        if verbose:
            print('---- True differences:\n',true_diffs)
            print('---- Actual total differences:\n',total_differences)
            print('---- Individual differences:\n',angular_differences)
        return False
    return True

def orientation_of_bases_test(verbose=False):
    """
        Tests the fact that the function compute_reference_planes returns orthonormal bases
    """
    def test_base_orientation(cloud_function):
        cloud,normals = cloud_function()
        pi_index = np.array([0])
        pj_index = np.array([1])
        n,base = compute_reference_planes(pi_index,pj_index,cloud,normals)
        n = n.squeeze()
        bx,by = base[0,:,:]

        iszero = lambda x:np.abs(x) < tolerance
        iseq = lambda x,v: np.abs(x-v) < tolerance
        assert iszero(bx@by)
        assert iszero(bx@n)
        assert iszero(by@n)

        assert iseq(np.cross(bx,by)@n,1)

    test_base_orientation(toy_cloud_2)
    test_base_orientation(toy_cloud_3)
    test_base_orientation(toy_cloud_4)
    test_base_orientation(toy_cloud_5)
    return True

def test_algorithm_on_small_dataset(verbose=False):
    """
        Tests the whole algorithm on a toy example and is extremely verbose about the internal variables.
        This test is useful to do a step-by-step debugging, and is not automatic.
    """
    eps = 1e-4
    tol = 1e-6
    areeq = lambda x,y : (np.abs(x-y)< tol).all()
    sq2 = np.sqrt(2)
    cloud = plane(4)
    normals = np.array([
        [0,0,1],
        [0,0,1],
        [-1,0,0],
        [0,0,1]])
    orientations = np.array([1,-1,1,-1])
    normals = orientations[:,np.newaxis] * normals
    if verbose:
        print('Unoriented normals\n',normals)

    emst = compute_emst(cloud)
    symmetric_emst = (emst + emst.T)
    if verbose:
        print("EMST (sym):\n",symmetric_emst)
    symmetric_kgraph = symmetric_kneighbors_graph(cloud,2)
    enriched = symmetric_emst + symmetric_kgraph
    enriched = enriched.tocoo()
    pi_s = enriched.row
    pj_s = enriched.col
    if verbose:
        print('Enriched EMST with 2-Neighbors:\n',pi_s,'\n',pj_s)

    reference_planes,reference_bases = compute_reference_planes(pi_s,pj_s,cloud,normals) # returns an array of normals of the reference planes of size [len(pi_s), 3]
    if verbose:
        for k in range(len(pi_s)):
            print('---Couple',(pi_s[k],pj_s[k]))
            print('Reference normal',reference_planes[k,:])
            print('Reference base\n',reference_bases[k,:])
    hermite_curves_complexities = compute_hermite_curves_complexities(pi_s,pj_s,cloud,normals,reference_planes,reference_bases) # returns an array of the complexities, in the form [len(pi_s),4]
    if verbose:
        print('Curves complexitites (++,--,+-,-+)')
        for k in range(len(pi_s)):
            print('---Couple',(pi_s[k],pj_s[k]))
            print(hermite_curves_complexities[k,:])
    c_keep = np.min(hermite_curves_complexities[:,:2],axis=1).reshape((-1,1))
    c_flip = np.min(hermite_curves_complexities[:,2:],axis=1).reshape((-1,1))
    c_s = np.hstack((c_keep,c_flip))
    riemannian_weights = (np.min(c_s,axis=1)+eps)/(np.max(c_s,axis=1)+eps)
    if verbose:
        print('Weights in Riemannian graph')
        for k in range(len(pi_s)):
            print((pi_s[k],pj_s[k]),riemannian_weights[k])
    flip_criterion_values = (c_flip < c_keep).squeeze()
    if verbose:
        print('Flip j normal ?')
        for k in range(len(pi_s)):
            print((pi_s[k],pj_s[k]),flip_criterion_values[k])

    riemannian_graph = csr_matrix((riemannian_weights,(pi_s,pj_s)),shape = (cloud.shape[0],cloud.shape[0]))
    if verbose:
        print('Riemannian graph')
        print(riemannian_graph)
    riemannian_mst = minimum_spanning_tree(riemannian_graph,overwrite = True)
    riemannian_mst = riemannian_mst + riemannian_mst.T
    if verbose:
        print('Riemannian MST')
        print(riemannian_mst)
    flip_criterion = csr_matrix((flip_criterion_values,(pi_s,pj_s)))

    normals_o = normals.copy().astype(np.float)
    was_flipped = np.zeros(len(cloud),dtype=np.bool)

    seed_index = np.argmax(cloud[:,2])
    ez = np.array([0,0,1])
    if normals_o[seed_index,:].T @ ez < 0:
        normals_o[seed_index,:] *= -1
        was_flipped[seed_index] = True
    if verbose:
        print('Root of the orientation:',seed_index)
        print('Oriented normal of the root:',normals_o[seed_index,:])
        print('Iteration in graph')
    for parent_index,point_index in acyclic_graph_dfs_iterator(riemannian_mst,seed_index):
        if parent_index is None:
            flip = False
        else:
            should_flip = flip_criterion[parent_index,point_index]
            flip = (should_flip and not was_flipped[parent_index]) or (was_flipped[parent_index] and not should_flip)
        if flip:
            was_flipped[point_index] = True
            normals_o[point_index,:] *= -1
        if verbose:
            print('---- Node',(parent_index,point_index))
            print('Flip:',flip)
            print(normals[point_index,:],'---->',normals_o[point_index,:])

    if verbose:
        print(cloud)
        print(normals_o)
    write_ply('./outputs/test_algorithm_on_small_dataset_output.ply',[cloud,normals_o],['x','y','z','nx','ny','nz'])
