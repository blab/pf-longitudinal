'''
Runs simulations for params.
'''

import argparse
import numpy as np
import pfTent as tent

def create_weight_alleles(loci, alleles):
    '''
    Returns weight & allele vectors.
    '''
    starter = np.ones(int(loci))
    a = alleles * starter
    a = a.astype(int)
    w = starter * (1/loci)
    return a, w

def get_peaks(pdensity):
    '''
    Finds the time of all parasite density peaks
    '''
    lag = np.pad(pdensity,1,mode='constant')[:-2]
    sign = np.sign(pdensity-lag).astype(int)
    lead = np.pad(sign,(0,1), mode='constant')[1:]
    peak = np.where(sign > lead)[0]
    return peak

def get_asymptomatics(peaks, malaria):
    '''
    Returns peaks for asymptomatic cases
    '''
    asymps = []
    for peak in peaks:
        if peak not in malaria:
            asymps.append(peak)
    return asymps

def get_spacing(asymps, malaria):
    '''
    Returns time window between first asymptomatic case & last symptomatic case
    '''
    diff = malaria[-1] - asymps[0]
    return diff

def get_intermediate_density(peaks, asymps, malaria, pdensity):
    '''
    Returns mean peak parasite density between first asymptomatic & last symptomatic
    '''
    window = [peak for peak in peaks if peak >= asymps[0] and peak <= malaria[-1]]
    pdensities = [pdensity[time] for time in window]
    mean = np.average(pdensities)
    return mean

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

    outcomes = np.empty((13,args.people,len(params)))
    for row in range(len(params)):
        t12 = params[row,0]
        xh = params[row,1]
        b = params[row,2]
        duration = params[row,3]
        maxP = params[row,4]
        alleles = params[row,5]
        power = params[row,6]
        eir = params[row,7]
        nloci = params[row,8]
        a,w = create_weight_alleles(nloci,alleles)
        all_parasites, all_immunity, all_strains, all_malaria, all_infections = tent.simulate_cohort(args.people,args.years,a,w,t12=t12,eir=eir,duration=duration,maxParasitemia=maxP,power=power,xh=xh,b=b)
        for person in range(args.people):
            pmatrix = all_parasites[person,...]
            smatrix = all_strains[person]
            malaria = all_malaria[person]

            Parasitemia, perPositivity = tent.check_parasitemia(args.years,pmatrix)
            outcomes[0,person,row] = np.median(Parasitemia)
            outcomes[1,person,row] = np.mean(Parasitemia)

            infectionlengths = tent.check_infection_length(smatrix,args.years,malaria)
            outcomes[2,person,row] = np.median(infectionlengths)
            outcomes[3,person,row] = np.mean(infectionlengths)

            MOI = tent.check_moi(args.years,smatrix)
            outcomes[4,person,row] = np.median(MOI)
            outcomes[5,person,row] = np.mean(MOI)

            outcomes[6,person,row] = perPositivity
            outcomes[7,person,row] = len(malaria)

            pdensity = pmatrix[-1,:,:].sum(axis=0)
            peaks = get_peaks(pdensity)
            asymps = get_asymptomatics(peaks,malaria)
            if len(asymps):
                outcomes[8,person,row] = asymps[0]
                if len(malaria):
                    outcomes[10,person,row] = get_spacing(asymps, malaria)
                    outcomes[11,person,row] = get_intermediate_density(peaks,asymps,malaria,pdensity)
                else:
                    outcomes[10,person,row] = None
                    outcomes[11,person,row] = None
            else:
                outcomes[8,person,row] = None
                outcomes[10,person,row] = None
                outcomes[11,person,row] = None
            if len(malaria):
                outcomes[9,person,row] = malaria[-1]
            else:
                outcomes[9,person,row] = None
        intermediate = np.median(outcomes[11,:,row])
        avg_cases = np.average(outcomes[7,:,row])
        avg_pdensity = np.average(outcomes[0,:,row])
        if (intermediate > 200) & (avg_cases < 20) & (avg_pdensity < 100):
            outcomes[12,:,row] = 1
        else:
            outcomes[12,:,row] = 0

    with open(args.output, 'wb') as f:
        np.save(f,outcomes)
