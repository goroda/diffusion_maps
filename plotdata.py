import argparse
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
 
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Generate some data for testing diffusion maps.')
    parser.add_argument('original', type=str, help="Filename for original data")
    parser.add_argument('coordinates', type=str, help="Filename for coordinates")
    parser.add_argument('figname', type=str, help="Filename for figure")
    parser.add_argument('-e', "--epsilon", type=float, default=1.0, help="epsilon for kernel used in spectral embedding")
    
    args = parser.parse_args()    
    file_original = args.original
    file_coord = args.coordinates

    xyz_orig = np.loadtxt(file_original)
    x = xyz_orig[:, 0]
    y = xyz_orig[:, 1]
    z = xyz_orig[:, 2]

    time = np.linspace(0, 1, xyz_orig.shape[0])

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(x, y, z, c=time)
    ax.set_title('Original data')

    ax = fig.add_subplot(1, 3, 2)    
    embedding = SpectralEmbedding(n_components=3, affinity='rbf', gamma=1.0 / args.epsilon)
    xyz_trans = embedding.fit_transform(xyz_orig)
    ax.scatter(xyz_trans[:, 0], xyz_trans[:, 1], c=time)
    ax.set_title('Laplacian via Sklearn SE')

    ax = fig.add_subplot(1, 3, 3)    
    xyz = np.loadtxt(file_coord)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    ax.scatter(y, z, c=time)
    ax.set_title('Diffusion map')

    plt.savefig(args.figname)
    plt.show()
