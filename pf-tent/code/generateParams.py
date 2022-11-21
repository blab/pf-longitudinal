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
    args = parser.parse_args()

    sampler = qmc.LatinHypercube(d=9)
    sample = sampler.random(n=10000)

    l_bounds = [25,-4,0.1,100,4.5,2,1,10,1]
    u_bounds = [1000,-0.9,0.9,1001,6.5,51,3,250,101]

    scaled = qmc.scale(sample,l_bounds,u_bounds)

    scaled[:,3] = np.floor(scaled[:,3])
    scaled[:,5] = np.floor(scaled[:,5])
    scaled[:,8] = np.floor(scaled[:,8])

    counter = 0
    for i in range(1,101):
        with open(args.outputdir+'params_' + str(i) + '.npy', 'wb') as f:
            arr = scaled[counter:counter+100,:]
            np.save(f,arr)
        counter += 100
