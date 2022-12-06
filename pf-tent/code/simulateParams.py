'''
Runs simulations for params.
'''

import argparse
import numpy as np
import pfTent as tent
import plotting as pt

def create_weight_alleles(loci, alleles):
    '''
    Returns weight & allele vectors.
    '''
    starter = np.ones(int(loci))
    a = alleles * starter
    a = a.astype(int)
    w = starter * (1/loci)
    return a, w

def get_visits(malaria,period,y):
    '''
    Returns passive & active visit dates in a list.
    '''
    start = np.random.randint(1,period)
    if len(malaria):
        if start > malaria[0]:
            start = malaria[0]
    active = np.arange(start,y*365,period,dtype=int)
    malaria = np.asarray(malaria,dtype=int)
    visits = np.union1d(active,malaria)
    return visits

def get_Parasitemia(pmatrix,visits):
    '''
    Returns prevalence & parasite density (if parasites) for every visit.
    Parasite density does not include Zeros.
    '''
    results = pmatrix[0,:,visits].sum(axis=1)
    pdens = results[results != 0]
    prev = np.count_nonzero(results)/len(results)
    return pdens, prev

def get_asymps(pmatrix,visits,malaria):
    '''
    Returns day of first asymptomatic parasitemia
    '''
    pdens = pmatrix[0,:,visits].sum(axis=1)
    locs = np.flatnonzero(pdens)
    pos = visits[locs]
    asymps = np.setdiff1d(pos, malaria)
    return asymps

def get_spacing(asymps, malaria):
    '''
    Returns time window between first asymptomatic case & last symptomatic case
    '''
    diff = malaria[-1] - asymps[0]
    spacing = max(diff,0)
    return spacing

def get_intermediate_density(visits,asymps, malaria, pmatrix):
    '''
    Returns mean measured parasite density during inbetween period. This mean does include zero values.
    Also returns % lamp only & % microscopy only. Sets microscopy cutoff at 90.
    '''
    between = visits[(visits >= asymps[0]) & (visits <= malaria[-1])]
    pdens = pmatrix[0,:,between].sum(axis=1)
    lamp = len(pdens[pdens>1])/len(pdens)
    micro = len(pdens[pdens>90])/len(pdens)
    mean = np.mean(pdens)
    return mean,lamp,micro

def get_yearly_cases(malaria,y):
    '''
    Returns number of cases for each year.
    '''
    starts = 365*np.arange(y)
    ends = 365*np.arange(1,y+1)
    malaria = np.asarray(malaria)
    cases = [len(malaria[(malaria < end) & (malaria>=start)]) for start,end in zip(starts,ends)]
    return cases

def get_yearly_prevs(y,visits,pmatrix):
    '''
    Returns number of cases for each year.
    '''
    starts = 365*np.arange(y)
    ends = 365*np.arange(1,y+1)
    times = [visits[(visits < end) & (visits>=start)] for start,end in zip(starts,ends)]
    n_visits = np.asarray([len(time) for time in times])
    pdens = [pmatrix[0,:,check].sum(axis=1) for check in times]
    micros = [len(pden[pden>90]) for pden in pdens]
    lamps = [len(pden[pden>1]) for pden in pdens]
    micro = np.asarray(micros)/n_visits
    lamp = np.asarray(lamps)/n_visits
    return micro,lamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assign colors based on ordering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', required=True, help='Path to input numpy file')
    parser.add_argument('--people', type=int, default=100, help='Number of people to simulate')
    parser.add_argument('--years', type=int, default=5, help='Number of years to simulate')
    parser.add_argument('--output', required=True, help='Path to output numpy file')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        params = np.load(f)

    n_outcomes = 15 + (3*args.years)
    outcomes = np.empty((n_outcomes,args.people,len(params)))
    for row in range(len(params)):
        t12 = params[row,0]
        b = params[row,1]
        xh = params[row,2]
        duration = params[row,3]
        maxP = params[row,4]
        alleles = params[row,5]
        power = params[row,6]
        beir = params[row,7]
        nloci = params[row,8]
        maxPv = params[row,9]
        limm = params[row,10]
        eir = limm*beir
        a,w = create_weight_alleles(nloci,alleles)
        all_parasites, all_immunity, all_strains, all_malaria, all_infections = tent.simulate_cohort(args.people,args.years,a,w,t12=t12,eir=eir,duration=duration,maxParasitemia=maxP,maxPv=maxPv,power=power,xh=xh,b=b)
        for person in range(args.people):
            pmatrix = all_parasites[person,...]
            smatrix = all_strains[person]
            malaria = all_malaria[person]
            visits = get_visits(malaria,30,args.years)

            Parasitemia, perPositivity = get_Parasitemia(pmatrix, visits)
            outcomes[0,person,row] = np.median(Parasitemia)
            outcomes[1,person,row] = np.mean(Parasitemia)

            infectionlengths = pt.check_infection_length(smatrix)
            outcomes[2,person,row] = np.median(infectionlengths)
            outcomes[3,person,row] = np.mean(infectionlengths)

            MOI = pt.check_moi(args.years,smatrix)
            outcomes[4,person,row] = np.median(MOI)
            outcomes[5,person,row] = np.mean(MOI)

            outcomes[6,person,row] = perPositivity
            outcomes[7,person,row] = len(malaria)

            pdensity = pmatrix[-1,:,:].sum(axis=0)
            #peaks = get_peaks(pdensity)
            #asymps = get_asymptomatics(peaks,malaria)
            asymps = get_asymps(pmatrix,visits,malaria)
            if len(asymps):
                outcomes[8,person,row] = asymps[0]
                if len(malaria):
                    spacing = get_spacing(asymps,malaria)
                    outcomes[10,person,row] = spacing
                    if spacing > 0:
                        mean,lamp,micro = get_intermediate_density(visits, asymps,malaria,pmatrix)
                        outcomes[11,person,row] = mean
                        outcomes[13,person,row] = micro
                        outcomes[14,person,row] = lamp
                    else:
                        outcomes[11,person,row] = None
                        outcomes[13,person,row] = None
                        outcomes[14,person,row] = None
                else:
                    outcomes[10,person,row] = None
                    outcomes[11,person,row] = None
                    outcomes[13,person,row] = None
                    outcomes[14,person,row] = None
            else:
                outcomes[8,person,row] = None
                outcomes[10,person,row] = None
                outcomes[11,person,row] = None
                outcomes[13,person,row] = None
                outcomes[14,person,row] = None
            if len(malaria):
                outcomes[9,person,row] = malaria[-1]
            else:
                outcomes[9,person,row] = None

            cases = get_yearly_cases(malaria,args.years)
            outcomes[15:15+args.years,person,row] = cases
            microY,lampY = get_yearly_prevs(args.years,visits,pmatrix)
            outcomes[15+args.years:15+(args.years*2),person,row] = microY
            outcomes[15+(args.years*2):,person,row] = lampY

        intermediate = np.median(outcomes[11,:,row])
        avg_cases = np.average(outcomes[7,:,row])
        avg_pdensity = np.average(outcomes[0,:,row])
        if (intermediate > 200) & (avg_cases < 20) & (avg_pdensity < 100):
            outcomes[12,:,row] = 1
        else:
            outcomes[12,:,row] = 0

    with open(args.output, 'wb') as f:
        np.save(f,outcomes)
