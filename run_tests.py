# Code by Pierre-François Massiani

from tests import common_tests as common
from tests import hoppe_tests as hoppe
from tests import konig_tests as konig

def passed(is_passed):
    if is_passed:
        return 'PASSED'
    else:
        return 'FAILED'

if __name__ == '__main__':
    print('======== Running tests for consistent normal orientation algorithms ========')
    if True:
        print("-------- Common features --------")
        print('EMST Computation test :',passed(common.test_emst_computation()))
        print('Graph traversing order test :',passed(common.test_graph_traversing()))
    if True:
        print("-------- Hoppe method --------")
        print('Riemannian MST computation test :',passed(hoppe.test_riemannian_mst_computation()))
        eps=1e-4
        print('Riemannian graph traversing (eps={:.4f}):'.format(eps), passed(hoppe.test_iteration_trough_riemannian_mst(eps)))
        print('Riemannian graph traversing (eps=0) (should fail):', passed(hoppe.test_iteration_trough_riemannian_mst(0)))
    if True:
        print("-------- Konig method --------")
        print('Reference planes computation test :',passed(konig.test_reference_planes_computation(verbose=True)))
        print('Projection on reference planes test :',passed(konig.test_projection(verbose=True)))
        print('Expression in planar coordinates test :',passed(konig.test_plane_coordinates(verbose=True)))
        print('Rotation test :',passed(konig.test_rotation(verbose=True)))

        print('Angular differences computation test :',passed(konig.test_angle_differences_computations(verbose=True)))
        print('Curves complexities computation test :',passed(konig.test_hermite_curves_complexities(verbose=True)))
        print('Propagation criterion computation test :',passed(konig.test_u_computation(verbose=True)))
        print('Orientation of bases test :',passed(konig.orientation_of_bases_test(verbose=True)))
        print('Proof of the discrepancies between the values of u presented on Figure 2:')
        konig.figure2_u_discrepancies()
        verbose_tests = False
        if verbose_tests:
            print('Tangents at critical points test : please check the values below.')
            konig.test_tangents_at_critical_points(verbose=True)
            print('Small dataset test:')
            konig.test_algorithm_on_small_dataset()
            print('Output written in ../outputs/0_test.ply')
