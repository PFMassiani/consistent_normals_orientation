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

    return cloud

if __name__ == '__main__':
    # Define the test points clouds
    data_dir = '../data/'
    test_cloud = parabola()
    write_ply(data_dir + 'parabola.ply',[test_cloud],['x','y','z'])
    test_cloud = cube()
    write_ply(data_dir + 'cube.ply',[test_cloud],['x','y','z'])
