'''
Performs power calculations for n_loci
'''

import argparse
import pandas as pd
import numpy as np
import powercalc as pc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Assign colors based on ordering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output', required=True, help="Path to output tsv")
    parser.add_argument('--years', required=True, type=int, help="Number of years to simulate")
    parser.add_argument('--experiments',type=int, default=1000, help="Number of iterations to run for each cohort")
    parser.add_argument('--measured', default=False, action="store_true")
    args = parser.parse_args()

    results = pd.DataFrame()
    intervals = [1,5,10,25,50,100]
    for n_loci in intervals:
        a = list(np.repeat(10,2+n_loci))
        w = [0,0]
        i_w = list(np.repeat(1/n_loci,n_loci))
        w.extend(i_w)
        print('n_loci: ' + str(n_loci))
        df = pc.power_calc_1st2nd(args.years,a,w,args.experiments,measured=args.measured)
        df['n_immloci'] = n_loci
        df['n_alleles'] = 10
        df['weight'] = w[-1]
        df['measured'] = args.measured
        df['n_exp'] = args.experiments
        df['years'] = args.years
        df['eir'] = 40
        df['allele_freq'] = 'uniform'
        results = results.append(df,ignore_index=True)
    with open(args.output, 'w') as file:
        results.to_csv(file,sep="\t",index=False)
