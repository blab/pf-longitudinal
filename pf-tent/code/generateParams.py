'''
Generates matrices of params to simulate under
'''

import argparse
from scipy.stats import qmc
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assign colors based on ordering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--outputdir', required=True, help='Path to output directory')
    parser.add_argument('--lower', required=True,type=float, nargs='+', help='list of lower bound of params')
    parser.add_argument('--upper', required=True,type=float, nargs='+', help='list of upper bound of params')
    parser.add_argument('--ints', required=True,type=int, nargs='+', help='list of params that should be integers')
    args = parser.parse_args()

    dims = len(args.lower)
    sampler = qmc.LatinHypercube(d=dims)
    sample = sampler.random(n=10) # Edited

    l_bounds = args.lower
    u_bounds = args.upper

    scaled = qmc.scale(sample,l_bounds,u_bounds)

    for integer in args.ints:
        scaled[:,integer] = np.floor(scaled[:,integer])

    counter = 0
    for i in range(1,3): #Edited
        with open(args.outputdir+'params_' + str(i) + '.npy', 'wb') as f:
            arr = scaled[counter:counter+5,:] # Edited
            np.save(f,arr)
        counter += 5 # Edited
