from tests import hoppe_tests as hoppe

def passed(is_passed):
    if is_passed:
        return 'PASSED'
    else:
        return 'FAILED'

if __name__ == '__main__':
    print('EMST Computation test :',passed(hoppe.test_emst_computation()))
    print('Graph traversing order test :',passed(hoppe.test_graph_traversing()))
    print('Riemannian MST computation test :',passed(hoppe.test_riemannian_mst_computation()))
    eps=1e-4
    print('Riemannian graph traversing (eps={:.4f}):'.format(eps), passed(hoppe.test_iteration_trough_riemannian_mst(eps)))
    print('Riemannian graph traversing (eps=0):', passed(hoppe.test_iteration_trough_riemannian_mst(0)))
