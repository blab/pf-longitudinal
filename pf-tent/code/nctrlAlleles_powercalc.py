'''
Performs power calculations for number of control alleles.
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
    intervals = [2,3,4,5,6,8]
    for n_ctrl in intervals:
        a = [n_ctrl]
        a.extend(list(np.repeat(10,6)))
        w = [0,0]
        i_w = list(np.repeat(1/5, 5))
        w.extend(i_w)
        print('n_ctrlAlleles: ' + str(n_ctrl))
        df, dic = pc.power_calc_1st2nd(args.years,a,w,args.experiments,measured=args.measured)
        df['n_ctrlAlleles'] = n_ctrl
        results_df = results_df.append(df,ignore_index=True)
        results_dict[n_ctrl] = dic
    results_dict['variable'] = 'n_ctrlAlleles'
    for d in [results_df, results_dict]:
        d['n_immloci'] = 5
        d['weight'] = 1/5
        d['n_alleles'] = 10
        d['measured'] = args.measured
        d['n_exp'] = args.experiments
        d['years'] = args.years
        d['eir'] = 40
        d['allele_freq'] = 'uniform'
        d['loci_importance'] = 'equal'
    with open(args.output+'.tsv', 'w') as file:
        results_df.to_csv(file,sep="\t",index=False)
    with open(args.output+'.json', 'w') as file:
        json.dump(results_dict,file)
