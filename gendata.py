import argparse
import sys
import numpy as np

def get_helix(uniform=True):
    # plot helix
    npts = 1000
    if uniform is True:
        time = np.linspace(0, 1, npts);
    else:
        np.random.seed(5)
        time = np.sort(np.random.beta(0.5, 0.5, size=(npts)))
        
    amplitude = 1
    freq = 8*2*np.pi
    offset = 2
    radius = offset + amplitude * np.sin(freq * time)
    height = amplitude * np.cos(freq * time)
    theta = 2*np.pi * time;


    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x,y, height


def get_vhelix(uniform=True):
    # plot vertical helix
    # CANNOT REPRODUCE THE COIFMAN PAPER YET
    npts = 1000
    if uniform is True:
        time = np.linspace(0, 1, npts);
    else:
        np.random.seed(5)
        time = np.sort(np.random.beta(0.5, 0.5, size=(npts)))


    height = np.sin(1.0 * np.pi * time)

    radius = np.sin(2.0 * 2.0 * np.pi * time) + 2.0
    x = radius * np.cos(5 * 2 * np.pi * time)
    y = radius * np.sin(5 * 2 * np.pi * time)
    return x,y, height, time
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate some data for testing diffusion maps.')
    parser.add_argument('name', type=str, help="Name of data set choose between: circular-helix, vertical-helix")
    parser.add_argument('-u', "--non_uniform_sampling", action="store_true", help="Generate non-uniform samples")

    args = parser.parse_args()
    uniform = True
    if args.non_uniform_sampling:        
        uniform = False
        
    if args.name == "circular-helix":
        x, y, z = get_helix(uniform=uniform)
        data = np.vstack((x, y, z)).T
        np.savetxt(sys.stdout, data)
    # elif args.name == "vertical-helix":
    #     x, y, z, time = get_vhelix(uniform=uniform)
    #     data = np.vstack((x, y, z)).T
    #     # import matplotlib as mpl
    #     # from mpl_toolkits.mplot3d import Axes3D
    #     # import matplotlib.pyplot as plt
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(projection='3d')
    #     # ax.scatter(x, y, z, c=time)
    #     # plt.show()
    #     np.savetxt(sys.stdout, data)
    else:
        print(f"name {args.name}: is not available")
    
