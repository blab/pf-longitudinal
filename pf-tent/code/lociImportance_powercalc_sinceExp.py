'''
Performs power calculations for transmission intensity
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
    intervals = [10,5,2,1,0.75,0.5,0.25]
    for importance in intervals:
        w_other, n_other = pc.get_nLoci(importance)
        a = list(np.repeat(10,2+n_other+1))
        w = [0,0]
        i_w = list(np.repeat(w_other, n_other))
        w.extend(i_w)
        w.append(0.2)
        print('loci_importance: ' + str(importance))
        rdict = pc.power_calc_sinceExp(args.years,a,w,args.experiments,measured=args.measured)
        df = pc.get_dataFrame(rdict,['pdensInitial','pdensAll','pdensMax','pdensArea'])
        df['loci_importance'] = importance
        df['n_immloci'] = n_other+1
        results_df = results_df.append(df,ignore_index=True)
        results_dict[importance] = rdict
    results_dict['variable'] = 'loci_importance;n_immloci'
    for d in [results_df, results_dict]:
        d['weight'] = 0.2
        d['n_alleles'] = 10
        d['n_ctrlAlleles'] = 10
        d['measured'] = args.measured
        d['n_exp'] = args.experiments
        d['years'] = args.years
        d['eir'] = 40
        d['allele_freq'] = 'uniform'
        d['sampling'] = 'complete'
    with open(args.output+'.tsv', 'w') as file:
        results_df.to_csv(file,sep="\t",index=False)
    with open(args.output+'.json', 'w') as file:
        json.dump(results_dict,file)
