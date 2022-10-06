'''
Code for power calculations on longitudinal trajectories
'''

import numpy as np
import scipy.stats as st
import pandas as pd
import pfTent as tent

def get_visits(malaria,period,y):
    '''
    Returns passive & active visit dates in a list.
    '''
    start = np.random.randint(1,period)
    if len(malaria) and start > malaria[0]:
        start = malaria[0]
    active = set(range(start,y*365,period))
    visits = list(active.union(set(malaria)))
    visits.sort()
    return visits

def get_infection_windows(loci, allele, pmatrix, visits=[],infectmatrix=[],smatrix=[],n_infections=2):
    '''
    Returns(2,n_infection) with time range for given n_infections at given loci & given allele.
    start = first day, end = last day ([,])
    If visits provided, start & end correspond to measured timepoints.
    If infectmatrix is provided, start & end correspond to true times.
    '''
    windows = np.zeros((2,n_infections),dtype=int) - 1

    if len(visits)>0:
        values = pmatrix[loci,allele,visits]
        positiveVisits = values.nonzero()[0]
        if len(positiveVisits):
            windows[0,0] = visits[positiveVisits[0]]
            shifted = np.roll(positiveVisits,1)
            test = positiveVisits-shifted
            new = np.where(test>1)[0]
            for i in range(0,n_infections):
                if i<len(new):
                    windows[1,i] = visits[positiveVisits[new[i]-1]]
                    if i-1 >= 0:
                        windows[0,i] = visits[positiveVisits[new[i-1]]]
                else:
                    windows[1,i-1] = visits[positiveVisits[-1]]

    elif len(infectmatrix)>0 and len(smatrix)>0:
        bites = np.where(infectmatrix[loci+1,:] == allele)[0] # bites are locations where you get an infection at that allele
        day = infectmatrix[0,bites[0]]
        windows[0,0] = day
        windows[1,0] = smatrix[bites[0],:].nonzero()[0][-1]
        counter = 0

        for i in range(1,len(bites)):
            new_day = infectmatrix[0,bites[i]]
            if new_day != day:
                if i-counter < n_infections:
                    windows[0,i-counter] = new_day
                    windows[1,i-counter] = smatrix[bites[i],:].nonzero()[0][-1]
                day = new_day
            else:
                counter += 1

    else:
        print("Must provide visits or infectmatrix & smatrix. If visits, will return measured time range of exposures. If infectmatrix & smatrix, will return true time range of exposures.")

    return windows

def get_peaks(pdensity):
    '''
    Finds the time of all parasite density peaks
    '''
    lag = np.pad(pdensity,1,mode='constant')[:-2]
    sign = np.sign(pdensity-lag).astype(int)
    lead = np.pad(sign,(0,1), mode='constant')[1:]
    peak = np.where(sign > lead)[0]
    if len(peak)==0:
        peak = np.argmax(pdensity)
    return peak

def get_max_pdensity(pmatrix,loci,allele,window,visits=[]):
    '''
    Returns maximum parasite density in a given time_window.
    '''
    if len(visits)>0:
        visits = np.asarray(visits)
        visited = visits[(visits >= window[0]) & (visits <= window[1])]
        pdensities = pmatrix[loci,allele,visited]
        maxima = max(pdensities)
    else:
        pdensities = pmatrix[loci,allele,window[0]:window[1]+1]
        peaktimes = get_peaks(pdensities)
        maxima = pdensities[peaktimes[0]]
    return maxima

def get_max_exp1_exp2(all_parasites, all_infections, all_strains, all_malaria,y,a,loci,period=28,measured=True):
    '''
    Returns arrays of maximum parasite density for exposure 1 & 2.
    If measured == True, this will be for measured maximums not true maximums.
    '''
    n_people = len(all_parasites)
    control = np.zeros((a[0]*n_people,2))
    test = np.zeros((a[loci]*n_people,2))
    n_control = []
    n_test = []

    ccount = 0
    tcount = 0
    for person in range(n_people):
        cstart = ccount
        tstart = tcount
        if measured==True:
            visits = get_visits(all_malaria[person],period,y)
        else:
            visits = []

        # Control loci
        for allele in range(a[0]): ## SPEED UP by doing get-infection-windows for all alleles all at once.
            windows = get_infection_windows(0,allele,all_parasites[person,...],visits=visits,infectmatrix=all_infections[person],smatrix=all_strains[person])
            if np.all(windows[:,0]>=0):
                control[ccount,0] = get_max_pdensity(all_parasites[person,...],0,allele,windows[:,0], visits=visits)
            if np.all(windows[:,1]>= 0): # WHY IS THIS NOT WORKING
                control[ccount,1] = get_max_pdensity(all_parasites[person,...],0,allele,windows[:,1], visits=visits)
            ccount += 1

        # Test loci
        for allele in range(a[loci]):
            windows = get_infection_windows(loci,allele,all_parasites[person,...],visits=visits,infectmatrix=all_infections[person],smatrix=all_strains[person])
            if np.all(windows[:,0] >=0):
                test[tcount,0] = get_max_pdensity(all_parasites[person,...],loci,allele,windows[:,0], visits=visits)
            if np.all(windows[:,1] >=0):
                test[tcount,1] = get_max_pdensity(all_parasites[person,...],loci,allele,windows[:,1], visits=visits)
            tcount+= 1

        person_ctrl = control[cstart:ccount,:]
        test_ctrl = control[tstart:tcount,:]
        n_control.append(len(person_ctrl[~(person_ctrl ==0).any(1)]))
        n_test.append(len(test_ctrl[~(test_ctrl==0).any(1)]))

    return control[~(control==0).any(1)],test[~(test==0).any(1)], n_control, n_test

def get_log(arr):
    new_arr = [np.log10(value) if value != 0 else -3 for value in arr]
    return np.asarray(new_arr)

def get_diff(control, test):
    '''
    Returns difference in parasite density for control & test from 1st to 2nd exposure.
    '''
    diff_control = get_log(control[:,1]) - get_log(control[:,0])
    diff_test = get_log(test[:,1]) - get_log(test[:,0])
    return diff_control, diff_test

def get_sens_spec(control, test, cutoff):
    '''
    Returns sensitivity & specificity of a test given some cutoff.
    '''
    test = np.asarray(test)
    control = np.asarray(control)
    tp = np.count_nonzero(test < cutoff)
    fp = np.count_nonzero(control < cutoff)
    tn = len(control) - fp
    fn = len(test) - tp
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    return sens, spec


def power_calc_1st2nd(y,a,w,experiments,eir=40,intervals=[10,50,100,500],cutoff=0.05,measured=True,power=0):
    '''
    Returns sensitivity & specificity for 10**x number of people.
    Test = Diff in parasite density at first vs. second exposure.
    '''
    people = []
    sensitivity = []
    specificity = []
    results = {}
    results['control'] = {}
    results['test'] = {}
    results['control']['diff_control'] = []
    results['control']['diff_test'] = []
    results['control']['exp2_control'] = []
    results['control']['exp2_test'] = []
    results['test']['diff_control'] = []
    results['test']['diff_test'] = []
    results['test']['exp2_control'] = []
    results['test']['exp2_test'] = []
    for n_people in intervals:
        print('n_people: ' + str(n_people))
        control_pvalue = []
        test_pvalue = []
        for experiment in range(experiments):
            all_parasites, all_immunity, all_strains, all_malaria, all_infections = tent.simulate_cohort(n_people,y,a,w,eir=eir,power=power)
            for l in [1,len(a)-1]:
                control,test, n_control, n_test = get_max_exp1_exp2(all_parasites,all_infections,all_strains,all_malaria,y,a,l,measured=measured)
                diff_control, diff_test = get_diff(control,test)
                try:
                    s,pvalue = st.mannwhitneyu(x=diff_control,y=diff_test)
                except ValueError:
                    pvalue = 1
                if l == 1:
                    control_pvalue.append(pvalue)
                    results['control']['diff_control'].extend(diff_control)
                    results['control']['diff_test'].extend(diff_test)
                    results['control']['exp2_control'].extend(n_control)
                    results['control']['exp2_test'].extend(n_test)
                else:
                    test_pvalue.append(pvalue)
                    results['test']['diff_control'].extend(diff_control)
                    results['test']['diff_test'].extend(diff_test)
                    results['test']['exp2_control'].extend(n_control)
                    results['test']['exp2_test'].extend(n_test)
        sens, spec = get_sens_spec(control_pvalue,test_pvalue,cutoff)
        sensitivity.append(sens)
        specificity.append(spec)
        people.append(n_people)

    df = pd.DataFrame({'n_people':people,'sensitivity':sensitivity, 'specificity':specificity})
    return df, results

def get_weights(imp,n_other=4):
    '''
    Returns weight of other loci & weight of test loci.
    '''
    wo = 1/(imp+n_other)
    wl = imp/(imp+n_other)
    return wo, wl

def get_weights(imp, wl=0.2):
    '''
    Returns n of other Loci if importance changes & constant weight.
    '''
    wo = wl/imp
    n_other = (1 - wl)/wo
    return wo, n_other
