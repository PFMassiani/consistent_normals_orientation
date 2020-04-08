# Code by Pierre-François Massiani

import numpy as np

def toy_cloud_0():
    return np.array([
        [0,0,0],
        [0,1,0],
        [1,1,0]
    ])
def toy_cloud_1():
    return np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,1]
    ])

def toy_cloud_2():
    cloud = np.array([
        [0,0,0],
        [1,0,0]
    ])
    normals = np.array([
        [0,1,0],
        [0,1,0]
    ])
    return cloud,normals

def toy_cloud_3():
    cloud = np.array([
        [0,0,0],
        [1,0,0]
    ])
    normals = np.array([
        [-1,1,0],
        [1,1,0]
    ]) / np.sqrt(2)
    return cloud,normals
def toy_cloud_4():
    cloud = np.array([
        [0,0,0],
        [1,0,0]
    ])
    normals = np.array([
        [-1,0,0],
        [1,0,0]
    ])
    return cloud,normals

def toy_cloud_5():
    cloud = np.array([
        [0,0,0],
        [1,0,0]
    ])
    normals = np.array([
        [0,1,0],
        [0,0,1]
    ])
    return cloud,normals
def toy_cloud_6():
    cloud = np.array([
        [0,0,0],
        [1,0,0]
    ])
    normals = np.array([
        [0,1,1],
        [1,0,1]
    ])/np.sqrt(2)
    return cloud,normals

def parabola(N_points = 100):
    n = int(np.sqrt(N_points))
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    xy = np.meshgrid(x,y)
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    z = x*y.T
    cloud = np.hstack((
        xy[0].reshape((-1,1)),
        xy[1].reshape((-1,1)),
        z.reshape((-1,1))
    ))
    return cloud

def cube(N_per_dim = 10):
    a = np.linspace(0,1,N_per_dim)
    b = np.linspace(0,1,N_per_dim)
    face_grid = np.meshgrid(a,b)
    a,b = [axis.reshape((-1,1)) for axis in face_grid]
    zero = np.zeros_like(a)
    one = np.ones_like(a)

    cloud = np.hstack((a,b,zero))

    face = np.hstack((a,b,one))
    cloud = np.vstack((cloud,face))

    face = np.hstack((a,zero,b))
    cloud = np.vstack((cloud,face))

    face = np.hstack((a,one,b))
    cloud = np.vstack((cloud,face))

    face = np.hstack((zero,a,b))
    cloud = np.vstack((cloud,face))

    face = np.hstack((one,a,b))
    cloud = np.vstack((cloud,face))

    return np.unique(cloud,axis=0)

def close_sheet(n_points_per_face = 100,angle = np.pi/6):
    n = int(np.sqrt(n_points_per_face))
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    cloud = np.hstack((x,y,np.zeros_like(x)))
    z_max = np.tan(angle)
    z = np.linspace(z_max,0,n)
    z = np.array([z for k in range(n)]).reshape((-1,1))
    upper_face = np.hstack((x,y,z))
    cloud = np.vstack((cloud,upper_face))
    cloud = np.unique(cloud,axis=0)
    return cloud

def plane(n=100):
    n = int(np.sqrt(n))
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    cloud = np.hstack((x,y,np.zeros_like(x)))
    return cloud
