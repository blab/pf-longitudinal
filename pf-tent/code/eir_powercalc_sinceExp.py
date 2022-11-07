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
    intervals = [2,20,40,100,250]
    for eir in intervals:
        a = list(np.repeat(10,7))
        w = [0,0]
        i_w = list(np.repeat(1/5, 5))
        w.extend(i_w)
        print('eir: ' + str(eir))
        rdict = pc.power_calc_sinceExp(args.years,a,w,args.experiments,measured=args.measured,eir=eir)
        df = pc.get_dataFrame(rdict,['pdensInitial','pdensAll','pdensMax','pdensArea'])
        df['eir'] = eir
        results_df = results_df.append(df,ignore_index=True)
        results_dict[eir] = rdict
    results_dict['variable'] = 'eir'
    for d in [results_df, results_dict]:
        d['n_immloci'] = 5
        d['n_alleles'] = 10
        d['n_ctrlAlleles'] = 10
        d['weight'] = 1/5
        d['measured'] = args.measured
        d['n_exp'] = args.experiments
        d['years'] = args.years
        d['allele_freq'] = 'uniform'
        d['loci_importance'] = 'equal'
        d['sampling'] = 'complete'
    with open(args.output+'.tsv', 'w') as file:
        results_df.to_csv(file,sep="\t",index=False)
    with open(args.output+'.json', 'w') as file:
        json.dump(results_dict,file)
