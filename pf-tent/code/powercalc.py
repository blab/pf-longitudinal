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

def get_nLoci(imp, wl=0.2):
    '''
    Returns n of other Loci if importance changes & constant weight.
    '''
    wo = wl/imp
    n_other = (1 - wl)/wo
    return wo, n_other

def get_pdensity_area(pmatrix,locus,allele,start,end,visits=[]):
    '''
    Returns area of parasite density for a given start & end.
    If visits provided, returns measured values. Otherwise returns True values.
    '''
    if len(visits):
        our_visits = [visit for visit in visits if visit >= start if visit <= end]
        if len(our_visits) > 1:
            shifted = np.roll(our_visits,1)
            diff = our_visits-shifted
            area = 0
            for i,days in enumerate(diff[1:]):
                first = pmatrix[locus,allele,our_visits[i]]
                second = pmatrix[locus,allele,our_visits[i+1]]
                pdensDiff = max(first,second) - min(first,second)
                area += (min(first,second)*days) + (days*pdensDiff*0.5)
        else:
            area = pmatrix[locus,allele,our_visits][0]
    else:
        if start != end:
            area = pmatrix[locus,allele,start:end+1].sum()
        else:
            area = pmatrix[locus,allele,start]
    return area

def get_times_since(loci, allele, pmatrix, visits=[],infectmatrix=[],smatrix=[]):
    '''
    Returns start times for all infections after the first infection.
    Returns end times for all infections after the first infection.
    Returns the time since exposure for all infections after the first infection.
    If visits provided, start & end correspond to measured timepoints.
    If infectmatrix & smatrix are provided, start & end correspond to true times.
    '''
    if len(visits):
        values = pmatrix[loci,allele,visits]
        positiveVisits = values.nonzero()[0]
        if len(positiveVisits):
            shifted = np.roll(positiveVisits,1)
            test = positiveVisits-shifted
            new = np.where(test>1)[0]
            end_locs = np.append(new[1:]-1,len(positiveVisits)-1)
            starts = [visits[day] for day in positiveVisits[new]]
            ends = [visits[day] for day in positiveVisits[end_locs]]
            lastPos = [visits[day] for day in positiveVisits[new-1]]
        else:
            starts = []
            ends = []

    elif len(infectmatrix):
        bites = np.where(infectmatrix[loci+1,:] == allele)[0]
        all_starts, all_locs = np.unique(infectmatrix[0,bites],return_index=True)
        if len(all_starts)>1:
            starts = all_starts[1:]
            locs = all_locs[1:]
            lastPos = [pmatrix[loci,allele,:start].nonzero()[0][-1] for start in starts]
            ends = [smatrix[loc,:].nonzero()[0][-1] for loc in bites[locs]]
        else:
            starts = []
            ends = []
    else:
        print("Must provide visits or infectmatrix & smatrix. If visits, will return measured time range of exposures. If infectmatrix & smatrix, will return true time range of exposures.")

    if len(starts):
        times = [start - last for start,last in zip(starts,lastPos)]
    else:
        times = []
    return starts, ends, times

def getPdensityExpInitial(y,a,locus,all_parasites,all_malaria):
    '''
    Returns Pdensity & Time since exposure for a locus from a cohort of simulations.
    '''
    n_people = len(all_malaria)
    all_pdens = []
    all_times= []
    for person in range(n_people):
        visits = get_visits(all_malaria[person],28,y)
        for allele in range(a[locus]):
            starts,ends, times = get_times_since(locus,allele,all_parasites[person,...],visits=visits)
            if len(times):
                pdens = np.log10(all_parasites[person,locus,allele,starts])
                all_pdens.extend(pdens)
                all_times.extend(times)
    return all_pdens, all_times

def getPdensExpAll(y,a,locus, all_parasites,all_malaria):
    '''
    Returns parasite densities and distance since last exposure for all positive visits.
    '''
    n_people = len(all_malaria)
    all_times = []
    all_pdens = []

    for person in range(n_people):
        visits = get_visits(all_malaria[person],28,y)
        for allele in range(a[locus]):
            values = all_parasites[person,locus,allele,visits]
            positiveVisits = values.nonzero()[0]
            positiveDays = [visits[loc] for loc in positiveVisits]
            if len(positiveDays):
                shifted = np.roll(positiveDays,1)
                times = positiveDays[1:]-shifted[1:]
                pdens = np.log10(all_parasites[person,locus,allele,positiveDays[1:]])
                all_times.extend(times)
                all_pdens.extend(pdens)
    return all_pdens,all_times

def getPdensExpMax(y,a,locus, all_parasites,all_malaria=[],all_infections=[],all_strains=[]):
    '''
    Returns maximum parasite densities and times since last exposure.
    If provide all_malaria, will return measured values.
    If provide all_infections & all_strains, will return true values.
    '''
    n_people = len(all_parasites)
    all_times = []
    all_pdens = []
    for person in range(n_people):
        if len(all_malaria):
            visits = get_visits(all_malaria[person],28,y)
            infectmatrix=[]
            smatrix=[]
        elif len(all_infections) & len(all_strains):
            visits = []
            infectmatrix=all_infections[person]
            smatrix=all_strains[person]
        else:
            print("Must provide all_malaria or all_infections & all_strains. If all_malaria, values are measured. Otherwise, values are true.")

        for allele in range(a[locus]):
            starts, ends, times = get_times_since(locus,allele,all_parasites[person,...],visits=visits,infectmatrix=infectmatrix,smatrix=smatrix)
            for start,end,time in zip(starts,ends,times):
                maxima = get_max_pdensity(all_parasites[person,...],locus,allele,[start,end],visits=visits)
                all_times.append(time)
                all_pdens.append(np.log10(maxima))
    return all_pdens,all_times

def getPdensExpArea(y,a,locus,all_parasites,all_malaria=[],all_infections=[],all_strains=[]):
    '''
    Returns area of parasite densities and times since last exposure.
    If provide all_malaria, will return measured values.
    If provide all_infections & all_strains, will return true values.
    '''
    n_people = len(all_parasites)
    all_times = []
    all_pdens = []
    for person in range(n_people):
        if len(all_malaria):
            visits = get_visits(all_malaria[person],28,y)
            infectmatrix=[]
            smatrix=[]
        elif len(all_infections) & len(all_strains):
            visits = []
            infectmatrix=all_infections[person]
            smatrix=all_strains[person]
        else:
            print("Must provide all_malaria or all_infections & all_strains. If all_malaria, values are measured. Otherwise, values are true.")

        for allele in range(a[locus]):
            starts, ends, times = get_times_since(locus,allele,all_parasites[person,...],visits=visits,infectmatrix=infectmatrix,smatrix=smatrix)
            for start,end,time in zip(starts,ends,times):
                maxima = get_pdensity_area(all_parasites[person,...],locus,allele,start,end,visits=visits)
                all_times.append(time)
                all_pdens.append(np.log10(maxima))
    return all_pdens,all_times

def get_sens_spec_sinceExp(r2_ctrl, slope_ctrl, r2_test, slope_test, r2_cutoff):
    '''
    Returns sensitivity & specificity of a test given some cutoff.
    '''
    slope_ctrl = np.asarray(slope_ctrl)
    slope_test = np.asarray(slope_test)
    r2_ctrl = np.asarray(r2_ctrl)
    r2_test = np.asarray(r2_test)
    tp = np.count_nonzero(np.sign(slope_test)*r2_test > r2_cutoff)
    fp = np.count_nonzero(np.sign(slope_ctrl)*r2_ctrl > r2_cutoff)
    tn = len(r2_ctrl) - fp
    fn = len(r2_test) - tp
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    return sens, spec

def power_calc_sinceExp(y,a,w,experiments,eir=40,intervals=[10,50,100,500],r2_cutoff=0.01,measured=True,power=0):
    '''
    Returns sensitivity & specificity for 10**x number of people.
    Test = Diff in parasite density at first vs. second exposure.
    '''
    all_results = {}
    for n_people in intervals:
        print('n_people: ' + str(n_people))
        results = {}
        results['pdensAll'] = {}
        results['pdensInitial'] = {}
        results['pdensMax'] = {}
        results['pdensArea'] = {}
        for key in results.keys():
            results[key]['control'] = {}
            results[key]['test'] = {}
            for ltype in ['control','test']:
                results[key][ltype]['r2'] = []
                results[key][ltype]['slope'] = []
        for experiment in range(experiments):
            all_parasites, all_immunity, all_strains, all_malaria, all_infections = tent.simulate_cohort(n_people,y,a,w,eir=eir,power=power)
            for key, l in zip(['control','test'],[1,len(a)-1]):
                pdensInitial, timesInitial = getPdensityExpInitial(y,a,l,all_parasites,all_malaria)
                slopeInitial, interceptInitial, rInitial, pInitial, seInitial = st.linregress(timesInitial,pdensInitial)
                results['pdensInitial'][key]['r2'].append(rInitial**2)
                results['pdensInitial'][key]['slope'].append(slopeInitial)

                pdensAll, timesAll = getPdensExpAll(y,a,l, all_parasites,all_malaria)
                slopeAll, interceptAll, rAll, pAll, seAll = st.linregress(timesAll,pdensAll)
                results['pdensAll'][key]['r2'].append(rAll**2)
                results['pdensAll'][key]['slope'].append(slopeAll)

                pdensMax, timesMax = getPdensExpMax(y,a,l, all_parasites,all_malaria)
                slopeMax, interceptMax, rMax, pMax, seMax = st.linregress(timesMax,pdensMax)
                results['pdensMax'][key]['r2'].append(rMax**2)
                results['pdensMax'][key]['slope'].append(slopeMax)

                pdensArea, timesArea = getPdensExpArea(y,a,l, all_parasites,all_malaria)
                slopeArea, interceptArea, rArea, pArea, seArea = st.linregress(timesArea,pdensArea)
                results['pdensArea'][key]['r2'].append(rArea**2)
                results['pdensArea'][key]['slope'].append(slopeArea)

        for key in ['pdensInitial', 'pdensAll', 'pdensMax', 'pdensArea']:
            sens, spec = get_sens_spec_sinceExp(results[key]['control']['r2'], results[key]['control']['slope'], results[key]['test']['r2'], results[key]['test']['slope'],r2_cutoff)
            results[key]['sensitivity'] = sens
            results[key]['specificity'] = spec

        all_results[n_people] = results

    return all_results

def get_dataFrame(rdict,keys):
    '''
    Returns dataframes from keys
    '''
    people = []
    sensitivity = []
    specificity = []
    ktype = []
    for i, key in enumerate(keys):
        for n_people in rdict.keys():
            people.append(n_people)
            sensitivity.append(rdict[n_people][key]['sensitivity'])
            specificity.append(rdict[n_people][key]['specificity'])
            ktype.append(key)
    df = pd.DataFrame({'n_people':people,'sensitivity':sensitivity, 'specificity':specificity,'Type':ktype})
    return df
