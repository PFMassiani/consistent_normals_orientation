# Code by Pierre-François Massiani

from utils.ply import write_ply
from tests.toy_clouds import *
from os import path,mkdir

if __name__ == '__main__':
    # Define the test points clouds
    data_dir = 'data'
    if not path.exists(data_dir):
        print('Creating directory {:s}...'.format(data_dir))
        mkdir(data_dir)
        print('Done.')
    print('Creating toy clouds...')
    test_cloud = parabola()
    write_ply(path.join(data_dir, 'parabola.ply'),[test_cloud],['x','y','z'])
    test_cloud = cube()
    write_ply(path.join(data_dir, 'cube.ply'),[test_cloud],['x','y','z'])
    test_cloud = close_sheet()
    write_ply(path.join(data_dir, 'close_sheet.ply'),[test_cloud],['x','y','z'])
    test_cloud = plane()
    write_ply(path.join(data_dir, 'plane.ply'),[test_cloud],['x','y','z'])
    test_cloud = plane(4)
    write_ply(path.join(data_dir, 'small_plane.ply'),[test_cloud],['x','y','z'])
    print('Done.')
