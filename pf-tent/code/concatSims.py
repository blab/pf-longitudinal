'''
Combine param space simulation into one dataset.
'''

import argparse
import numpy as np

def concat_npy(data, axis=0):
    '''
    Concats list of numpy datasets into a matrix
    '''
    arrs = []
    for path in data:
        with open(path,'rb') as f:
            a = np.load(f)
        arrs.append(a)
    output = np.concatenate(arrs,axis=axis)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assign colors based on ordering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--in-params',nargs='+', required=True, help='Path to list of input numpy param files')
    parser.add_argument('--in-sims', nargs='+', required=True, help='Path to list of input numpy simulation outcome files')
    parser.add_argument('--out-params', required=True, help='Path to output param numpy file')
    parser.add_argument('--out-sims', required=True, help='Path to output outcome numpy file')
    args = parser.parse_args()

    params = concat_npy(args.in_params)
    sims = concat_npy(args.in_sims,axis=2)

    with open(args.out_params, 'wb') as f:
        np.save(f,params)

    with open(args.out_sims, 'wb') as f:
        np.savez_compressed(f,sims)
