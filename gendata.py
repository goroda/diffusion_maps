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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate some data for testing diffusion maps.')
    parser.add_argument('name', type=str, help="Name of data set choose between: circular-helix")
    parser.add_argument('-u', "--non_uniform_sampling", action="store_true", help="Generate non-uniform samples")

    args = parser.parse_args()
    uniform = True
    if args.non_uniform_sampling:        
        uniform = False
        
    if args.name == "circular-helix":
        x, y, z = get_helix(uniform=uniform)
        data = np.vstack((x, y, z)).T
        np.savetxt(sys.stdout, data)
    else:
        print(f"name {args.name}: is not available")
    
