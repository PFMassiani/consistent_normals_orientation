from utils.ply import write_ply
from tests.toy_clouds import *

if __name__ == '__main__':
    # Define the test points clouds
    data_dir = '../data/'
    test_cloud = parabola()
    write_ply(data_dir + 'parabola.ply',[test_cloud],['x','y','z'])
    test_cloud = cube()
    write_ply(data_dir + 'cube.ply',[test_cloud],['x','y','z'])
    test_cloud = close_sheet()
    write_ply(data_dir + 'close_sheet.ply',[test_cloud],['x','y','z'])
    test_cloud = plane()
    write_ply(data_dir + 'plane.ply',[test_cloud],['x','y','z'])
    test_cloud = plane(4)
    write_ply(data_dir + 'small_plane.ply',[test_cloud],['x','y','z'])
