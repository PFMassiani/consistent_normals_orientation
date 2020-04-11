import numpy as np
from utils.ply import write_ply, read_ply
from normals.hoppe import hoppe_orientation
from normals.common import compute_normals
from normals.konig import konig_orientation

from tests.toy_clouds import cube,close_sheet

import time
from os import path

if __name__ == '__main__':
    print('CONSISTENT PROPAGATION OF NORMAL ORIENTATION IN POINT CLOUDS')
    print('==================== Benchmark =============================')

    orientation_methods = {'Konig_and_Gumhold':konig_orientation,'Hoppe':hoppe_orientation}
    data_dir = 'data'
    output_dir = 'outputs'
    datasets = {dataset_name:dataset_name + '.ply' for dataset_name in [
        'sphere','cube','close_sheet','bunny','angle'
    ]}
    datasets['lille'] = 'Lille_street_small.ply'

    generated = {'cube':lambda :cube(N_per_dim=10),'close_sheet':lambda :close_sheet(n_points_per_face = 1000),'angle':lambda :close_sheet(n_points_per_face = 1000,angle = np.pi/4)}

    radii = {}
    radii['sphere'] = 2
    radii['cube'] = 0.1
    radii['close_sheet'] = 0.1
    radii['angle'] = 0.5
    radii['bunny'] = 0.01
    radii['lille'] = 1

    neighbors = {}
    neighbors['sphere'] = 10
    neighbors['cube'] = 10
    neighbors['angle'] = 5
    neighbors['close_sheet'] = 5
    neighbors['bunny'] = 10
    neighbors['lille'] = 30

    ground_truths = {'lille':'Lille_street_small_normals.ply','bunny':'bunny_normals.ply'}

    times = {}
    errors = {}
    edges = {}
    vertices = {}

    tolerance = 1e-2
    datasets_to_run = list(datasets.keys())
    print('Orienting datasets...')
    for ds in datasets_to_run:
        generation = generated.get(ds)
        ground = ground_truths.get(ds)
        if (generation is None) and (ground is None):
            cloud_path = path.join(data_dir,datasets[ds])
            cloud_ply = read_ply(cloud_path)
            cloud = np.vstack((cloud_ply['x'],cloud_ply['y'],cloud_ply['z'])).T
        elif ground is not None:
            cloud_path = path.join(data_dir,ground_truths[ds])
            truth_ply = read_ply(cloud_path)
            cloud = np.vstack((truth_ply['x'],truth_ply['y'],truth_ply['z'])).T
            truth_normals = np.vstack((truth_ply['nx'],truth_ply['ny'],truth_ply['nz'])).T
        else:
            cloud = generation()

        if ground is not None:
            normals = truth_normals
        else:
            normals = compute_normals(cloud,radius=radii[ds],n_neighbors=None,verbose=False)
        np.random.seed(0)
        orientations = np.random.randn(normals.shape[0])
        orientations[orientations<0] = -1
        orientations[orientations >= 0] = 1
        normals = orientations[:,np.newaxis] * normals

        for mname in orientation_methods:
            method = orientation_methods[mname]
            t1 = time.time()
            oriented_normals,n_edges = method(cloud,normals,neighbors[ds],return_n_edges = True)
            t2 = time.time()
            if ds in times:
                times[ds][mname] = t2-t1
                edges[ds][mname] = n_edges
                vertices[ds][mname] = len(cloud)
            else:
                times[ds] = {mname:t2-t1}
                edges[ds] = {mname:n_edges}
                vertices[ds] = {mname:len(cloud)}

            if ground is not None:
                number_of_errors = np.min((
                    (np.linalg.norm(oriented_normals - truth_normals,axis=-1) > tolerance).sum(),
                    (np.linalg.norm(-oriented_normals - truth_normals,axis=-1) > tolerance).sum()
                ))

                if ds in errors:
                    errors[ds][mname] = number_of_errors
                else:
                    errors[ds] = {mname:number_of_errors}
            oriented_path = path.join(output_dir, ds+ '_'+mname+'_benchmark.ply')
            write_ply(oriented_path,[cloud,oriented_normals],['x','y','z','nx','ny','nz'])
            del oriented_normals
            print('.')
        del cloud,normals
        if ground is not None:
            del truth_ply,truth_normals
        elif (generation is None) and (ground is None):
            del cloud_ply
    print('RESULTS :')
    for ds in datasets_to_run:
        print("====== Dataset:",ds)
        has_ground = ds in ground_truths
        for mname in orientation_methods:
            print("----- Method:",mname)
            print('Total time: {:.4} s'.format(times[ds][mname]))
            print('Time per edge: {:.4} ms/edge'.format(1000*times[ds][mname]/edges[ds][mname]))
            print('Number of edges: {}'.format(edges[ds][mname]))
            print('Number of vertices: {}'.format(vertices[ds][mname]))
            if has_ground:
                print('Number of misoriented normals: {}'.format(errors[ds][mname]))
                print('Percentage of misoriented normals: {:.4f}%'.format(100*errors[ds][mname]/vertices[ds][mname]))
    print('Done.')
