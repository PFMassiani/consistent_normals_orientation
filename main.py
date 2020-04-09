# Code by Pierre-François Massiani

import sys
from os import path,mkdir

import numpy as np
from utils.ply import write_ply, read_ply
from normals.hoppe import hoppe_orientation
from normals.common import compute_normals
from normals.konig import konig_orientation


if __name__ == '__main__':
    """
        Main script
        Usage :
            The following command should be executed in the project's root directory
                python3 main.py [CLOUD_NAME [ORIENTATION_METHOD]]

            CLOUD_NAME (optional) : the name of the points cloud, without the ".ply" extension
            ORIENTATION_METHOD : either "hoppe" or "konig"

        Remark :
            1. Point clouds should be in the ./data directory
            2. To use most of the toy datasets available, you should run the following command first :
                python3 create_toy_clouds.py
    """

    print('CONSISTENT PROPAGATION OF NORMAL ORIENTATION IN POINT CLOUDS')
    print('============================================================')

    # Arguments parsing
    if len(sys.argv) > 2:
        if sys.argv[2] == 'konig':
            orientation_method = konig_orientation
            method = 'Hoppe'
        elif sys.argv[2] == 'hoppe':
            orientation_method = hoppe_orientation
            method = 'König and Gumhold'
        else:
            raise ValueError('Invalid orientation method ({:s})'.format(sys.argv[2]))
    else:
        # Change here to select the default orientation method (if not specified in the command line)
        orientation_method = konig_orientation
        method = 'König and Gumhold'

    data_dir = 'data'
    output_dir = 'outputs'
    datasets = {dataset_name:dataset_name + '.ply' for dataset_name in [
        'bunny','sphere','parabola','cube','close_sheet','plane','small_plane'
    ]}
    datasets['lille'] = 'Lille_street_small.ply'

    radii = {}
    radii['sphere'] = 2
    radii['bunny'] = 0.01
    radii['parabola'] = 0.5
    radii['cube'] = 0.3
    radii['close_sheet'] = 0.5
    radii['plane'] = 0.2
    radii['small_plane'] = 1.1
    radii['lille'] = 1

    neighbors = {}
    neighbors['sphere'] = 10
    neighbors['bunny'] = 10
    neighbors['parabola'] = 5
    neighbors['cube'] = 5
    neighbors['close_sheet'] = 3
    neighbors['plane'] = 5
    neighbors['small_plane'] = 2
    neighbors['lille'] = 10

    if len(sys.argv) > 1:
        cloud_name = sys.argv[1]
    else:
        # Change here to select the default cloud (if not specified in command line)
        cloud_name = 'parabola'
    cloud_path = path.join(data_dir, datasets[cloud_name])

    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply['x'],cloud_ply['y'],cloud_ply['z'])).T
    print('Loaded points cloud {:s} successfully.'.format(cloud_path))

    print('Computing normals...')
    normals = compute_normals(cloud,radius=radii[cloud_name],n_neighbors=None,verbose=False)
    # We select a random orientation for the cloud
    np.random.seed(0)
    orientations = np.random.randn(normals.shape[0])
    orientations[orientations<0] = -1
    orientations[orientations >= 0] = 1
    normals = orientations[:,np.newaxis] * normals
    print('Done.')
    print('Orienting normals using the {:s} method'.format(method))
    oriented_normals = orientation_method(cloud,normals,neighbors[cloud_name])
    print('Done.')

    if not path.exists(output_dir):
        print('Creating directory {:s}...'.format(output_dir))
        mkdir(output_dir)
        print('Done.')
    oriented_path = path.join(output_dir, cloud_name+ '_oriented.ply')
    write_ply(oriented_path,[cloud,oriented_normals],['x','y','z','nx','ny','nz'])
    print('Saved the cloud with oriented normals at {:s}'.format(oriented_path))
