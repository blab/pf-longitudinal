'''
Performs power calculations for number of loci.
'''

import argparse
import pandas as pd
import numpy as np
import powercalc as pc
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Assign colors based on ordering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output', required=True, help="Path to output")
    parser.add_argument('--years', required=True, type=int, help="Number of years to simulate")
    parser.add_argument('--experiments',type=int, default=1000, help="Number of iterations to run for each cohort")
    parser.add_argument('--measured', default=False, action="store_true")
    args = parser.parse_args()

    results_df = pd.DataFrame()
    results_dict = {}
    intervals = [2,5,10,25,50,100]
    for n_loci in intervals:
        a = list(np.repeat(10,2+n_loci))
        w = [0,0]
        i_w = list(np.repeat(1/n_loci,n_loci))
        w.extend(i_w)
        print('n_loci: ' + str(n_loci))
        df, dic = pc.power_calc_1st2nd(args.years,a,w,args.experiments,measured=args.measured)
        df['n_immloci'] = n_loci
        df['weight'] = 1/n_loci
        results_df = results_df.append(df,ignore_index=True)
        results_dict[n_loci] = dic
    results_dict['variable'] = 'n_immLoci;weight'
    for d in [results_df, results_dict]:
        d['n_alleles'] = 10
        d['n_ctrlAlleles'] = 10
        d['measured'] = args.measured
        d['n_exp'] = args.experiments
        d['years'] = args.years
        d['eir'] = 90
        d['allele_freq'] = 2
        d['loci_importance'] = 'equal'
    with open(args.output+'.tsv', 'w') as file:
        results_df.to_csv(file,sep="\t",index=False)
    with open(args.output+'.json', 'w') as file:
        json.dump(results_dict,file)
