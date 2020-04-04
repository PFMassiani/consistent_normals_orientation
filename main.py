import numpy as np
from utils.ply import write_ply, read_ply
from normals.hoppe import hoppe_orientation
from normals.common import compute_normals

if __name__ == '__main__':
    print('CONSISTENT PROPAGATION OF NORMAL ORIENTATION IN POINT CLOUDS')
    print('============================================================')


    data_dir = '../data/'
    output_dir = '../outputs/'
    datasets = {dataset_name:dataset_name + '.ply' for dataset_name in [
        'bunny','bunny_normals','bunny_normals_ascii','dragon','sphere','parabola','cube'
    ]}
    datasets['lille'] = 'Lille_street_small.ply'

    radii = {}
    radii['sphere'] = 2
    radii['lille'] = 0.5
    radii['bunny'] = 0.01
    radii['parabola'] = 0.5
    radii['cube'] = 0.5

    neighbors = {}
    neighbors['sphere'] = 10
    neighbors['lille'] = 10
    neighbors['bunny'] = 10
    neighbors['parabola'] = 5
    neighbors['cube'] = 5

    cloud_name = 'cube'
    cloud_path = data_dir + datasets[cloud_name]

    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply['x'],cloud_ply['y'],cloud_ply['z'])).T
    
    if True:
        normals = compute_normals(cloud,radius=radii[cloud_name],n_neighbors=None,verbose=True)
        oriented_normals = hoppe_orientation(cloud,normals,neighbors[cloud_name],verbose=True)

        unoriented_path = output_dir + cloud_name + '_unoriented_normals.ply'
        oriented_path = output_dir + cloud_name + '_oriented_normals.ply'
        write_ply(unoriented_path,[cloud,normals],['x','y','z','nx','ny','nz'])
        print('Saved the cloud with unoriented normals at {:s}'.format(unoriented_path))
        write_ply(oriented_path,[cloud,oriented_normals],['x','y','z','nx','ny','nz'])
        print('Saved the cloud with oriented normals at {:s}'.format(oriented_path))
        print('Done.')

    if False:
        eigenvalues,eigenvectors = local_PCA(cloud)
        print(eigenvalues,eigenvectors)