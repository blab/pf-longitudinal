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
    intervals = [2,20,40,100,250]
    for eir in intervals:
        a = list(np.repeat(10,7))
        w = [0,0]
        i_w = list(np.repeat(1/5, 5))
        w.extend(i_w)
        print('eir: ' + str(eir))
        df = pc.power_calc_1st2nd(args.years,a,w,args.experiments,measured=args.measured,eir=eir)
        df['n_immloci'] = 5
        df['n_alleles'] = 10
        df['weight'] = 1/5
        df['measured'] = args.measured
        df['n_exp'] = args.experiments
        df['years'] = args.years
        df['eir'] = eir
        df['allele_freq'] = 'uniform'
        results = results.append(df,ignore_index=True)
    with open(args.output, 'w') as file:
        results.to_csv(file,sep="\t",index=False)
