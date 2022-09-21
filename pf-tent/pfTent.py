'''
Code for simulating longitudinal trajectories of individuals with _P.
falciparum_ infections.
Variables:
  y = years to simulate
  eir = 40, average annual entomological inoculation rate
  delta = 1/500, strain immunity loss rate

  duration = 500, max infection length
  meroz = 0.01, initial parasites/uL, not log scale
  timeToPeak = 10, time from first parasitemia to peak
  maxParasitemia = 6, maximum parasitemia on log scale

  pgone = -3, threshold for parasites being gonelog10 scale
  immune_thresh = 0.01, threshold at which you start gaining immunity

  w = vector for weighting immune impact, should all add to one. Length is len(a) + 1
  a = vector containing # of alleles at each loci.
'''

import numpy as np
import scipy.stats as st

def load_data():
    '''
    Loads data used in the model
    '''
    fever = np.load("data/fever.npy")
    breaks = np.load("data/breaks.npy")
    return fever, breaks

def simulate_bites(y,eir):
    '''
    Produces a vector with bite times up until year, y, is reached.

    Time between bites pulled from exponential distribution wih mean rate of k = eir/365.
    '''
    k = eir/365
    n = round(y*eir*2)
    spaces = st.expon.rvs(scale=(1/k), loc=0, size=n)
    times = np.cumsum(spaces)
    trimmed = times[times <= y*365]
    bites = np.ceil(trimmed).astype(int)
    return bites

def simulate_strains(n,a):
    '''
    n = number of strains to simulate
    a = vector whose length corresponds to number of loci.
    Each entry corresponds to the number of alleles at that loci.
    so a = [3,4,6] will simulate a strain with 3 loci. The first
    loci has 3 alleles; the second loci has 4 alleles, and the third
    loci has 6 alleles.

    Returns genotype as a L x n matrix, where L = the number of loci.
    '''
    length = len(a)
    M = np.empty((length,n),dtype=int)
    for i in range(n):
        floats = np.random.rand(length)
        genotype = np.ceil(floats*a)-1
        M[:,i] = genotype
    return M

def create_allele_matrix(a,y):
    '''
    Creates matrix to track parasitemia.
    '''
    length = len(a)
    width = max(a)
    days = 365*y
    M = np.zeros((length,width,days))
    return M

def get_infection_params(duration, meroz, timeToPeak, maxParasitemia):
    '''
    Generates the duration of infection from a normal distribution.
    Generates starting number of merozoites from a lognormal distribution.
    Generates time to peak from a normal distribution.
    Generates parasite max from a normal distribution.
    '''
    dur = np.rint(st.norm.rvs(loc=duration,scale=20))
    mz = st.lognorm.rvs(s=.5,scale=meroz)
    peaktime = np.rint(st.norm.rvs(loc=timeToPeak, scale=3))
    pmax = st.norm.rvs(loc=maxParasitemia,scale=0.25)
    params = np.array([dur, mz, peaktime, pmax])
    return params

def get_parasitemia(params,pgone):
    '''
    Returns a vector containing parasitemia values per day for an
    infection from params:
        pgone = threshold at which infection is over, defined on a log scale.
    '''
    dur = params[0].astype(int)
    mz = np.log10(params[1])
    peaktime = params[2].astype(int)
    pmax = params[3]
    gr = (pmax - mz)/peaktime
    dr = (pgone - pmax) / (dur - peaktime - 1)
    arr = np.zeros(dur)
    arr[0] = mz
    for i in np.arange(1,peaktime+1):
        arr[i] = arr[i-1] + gr
    for i in np.arange(peaktime+1,dur):
        arr[i] = arr[i-1] + dr
    return arr

def create_strain_matrix(n,y):
    '''
    Creates n x y*365 matrix to track strains presence per day.
    '''
    M = np.zeros((n,y*365),dtype=int)
    return M

def add_infection(p,pM,gtype,t,sM,s):
    '''
    Adds infection to strain & parasite matrices.
    inputs:
        p = parasite vector for infection
        pM = matrix tracking parasitemia
        gtype = genotype vector for infection
        t = time in days
        sM = matrix tracking tracking strains
        s = strain number
    '''
    dur = len(p)
    days = pM.shape[2]
    n_alleles = len(gtype)
    for i in np.arange(n_alleles):
        if t+dur >= days:
            dur = days-t
            p = p[:dur]
        pM[i,gtype[i],t:t+dur] += 10**p
        sM[s,t:t+dur] = 1

def sigmoid(x,param,xh=0.5,b=-1):
    '''
    Sigmoid modulates modulates an infection param based on immunity, x:
    '''
    c = np.tan(np.pi/2*xh)**b
    new_param = param/(c/np.tan(np.pi/2*x)**b+1)
    return new_param

def modulate_params(gtype, strain_imm, params, w):
    '''
    Changes all infection params according to immunity:
        gtype = genotype vector for infection
        strain_imm = strain_immunity at time of infection
        params = params vector to modulate
        w = vector modulating immunity effect at locus
    '''
    n_loci = len(gtype)
    cross = np.empty(n_loci)
    for i in np.arange(n_loci):
        allele = gtype[i]
        cross[i] = strain_imm[i,allele]

    M = np.zeros((4, n_loci+1))
    for i, p in enumerate(params):
        for j, v in enumerate(cross):
            if v == 0 | i == 2: # Immunity doesn't impact time to peak.
                M[i,j] = p*w[j]
            else:
                M[i,j] = sigmoid(v, p*w[j])
    modified = np.zeros(4)
    for i in np.arange(4):
        modified[i] = M[i,:].sum()
    modified[0] = np.rint(modified[0])
    modified[2] = np.rint(modified[2])
    return modified

def update_immunity(pM, iM, t, immune_thresh, delta,):
    '''
    If parasites present, gains immunity. If absent, loses immunity:
        pM = parasitemia matrix at each alleles across time.
        iM = matrix of strain immunity at each allele across time
        t = time
        immune_thresh = parasite density threshold at which gain immunity.
    '''
    # Strain immunity
    loci = pM.shape[0]
    n_alleles = pM.shape[1]
    for i in np.arange(loci):
        for j in np.arange(n_alleles):
            if pM[i,j,t] > immune_thresh:
                iM[i,j,t] = 1
            else:
                if t == 0:
                    iM[i,j,t] = 0
                else:
                    iM[i,j,t] = max(iM[i,j,t-1] - delta, 0)

def get_fever_threshold(t, eir,fever,breaks):
    '''
    Pulls fever threshold from model used in ["Quantification of anti-parasite and anti-disease immunity to malaria as a function of age and exposure"](https://elifesciences.org/articles/35832).
    '''
    age_loc = np.flatnonzero(breaks[:,0]>=t/365)[0]
    eir_loc = np.flatnonzero(breaks[:,2]>=eir)[0]
    pardens_loc = np.flatnonzero(fever[age_loc,:,eir_loc])[0]
    thresh = breaks[pardens_loc,1]
    return 10**thresh

def treat_as_needed(threshhold, pM, sM, t, m):
    '''
    Treat if parasitemia goes above certain threshold. Modifies parasite density
    matrix & strain matrix. Returns number of malaria cases thus far in
    individual's life:
        treatment_thresh = threshold for treatment
        pM = matrix tracking parasite density per allele across time.
        sM = matrix tracking presence of strains across time.
        t = time
        m = # of malaria cases that have occurred
    '''
    #threshhold = st.lognorm.rvs(s=0.4,scale=treatment_thresh) Can add back in if want stochasticity to treatment threshold
    if pM[0,:,t].sum(axis=0) > threshhold:
        pM[:,:,t+1:] = 0
        sM[:,t+1:] = 0
        m.append(t)
    return m

def simulate_person(y,a,w,fever,breaks, eir=40, delta=1/250,immune_thresh=0.01,duration = 500, meroz = .01, timeToPeak = 10, maxParasitemia = 6, pgone=-3):
    '''
    Runs simulation for one person.
    Returns:
    - matrix of parasitemia by allele across time
    - matrix of strains across time
    - matrix of immunity by allele across time
    - matrix of infection starts + gtype of infection by day. Size is
    (1+len(loci),n_infections). The first row is the day of the infection. The
    remaining rows are the gtype of the infection.
    '''
    malaria = []
    bites = simulate_bites(y,eir)
    n_bites = len(bites)
    strains = simulate_strains(n_bites,a)
    pmatrix = create_allele_matrix(a, y)
    smatrix = create_strain_matrix(n_bites,y)
    imatrix = create_allele_matrix(a,y)
    infections = {}
    infections["day"] = []
    infections["gtype"] = []

    counter = 0
    for t in range(365*y):
        update_immunity(pmatrix,imatrix,t,immune_thresh, delta)
        treatment_thresh = get_fever_threshold(t,eir,fever,breaks)
        malaria = treat_as_needed(treatment_thresh,pmatrix,smatrix,t,malaria)
        if t in bites:
            if not len(malaria) > 0 or t - malaria[-1] > 7:
                locs = np.where(bites == t)
                for i in locs[0]:
                    params = get_infection_params(duration, meroz, timeToPeak, maxParasitemia)
                    params = modulate_params(strains[:,i], imatrix[:,:,t], params, w)
                    if params[0] > 0 and params[3] > 0 and params[1] > 0.001 and params[0] > params[2] and params[2] > 0:
                        parasitemia = get_parasitemia(params, pgone)
                        add_infection(parasitemia,pmatrix,strains[:,i],t,smatrix,i)
                        infections["day"].append(t)
                        infections["gtype"].append(strains[:,i])

    n_infect = len(infections["day"])
    infectmatrix = np.zeros((1+len(a),n_infect),dtype=int)
    for i, day in enumerate(infections["day"]):
        infectmatrix[0,i] = day
        infectmatrix[1:,i] = infections["gtype"][i]
    smatrix = smatrix[~np.all(smatrix == 0, axis=1)]
    return pmatrix, smatrix, imatrix, malaria, infectmatrix


def simulate_cohort(n_people,y,a,w,delta=1/250,eir=40,immune_thresh=0.01,duration=500,meroz=0.1,timeToPeak=10,maxParasitemia=6,pgone=-3):
    '''
    Simulates an entire cohort of individuals.

    Returns n_people x loci x alleles x t matrices tracking parasite density & immunity at each allele.
    Returns dictionary containing strain matrices for each person.
    Returns dictionary containing lists of malaria episodes for each person.
    Returns dictionary containing infection matrices for each person.

    Input:
        y = years to simulate
        a = vector of len(loci) specifying number of alleles at each locus
        w = immune weighting for each locus
        delta = immunity waning rate
    '''
    # Create objects to record
    all_parasites = np.zeros((n_people, len(a), max(a), y*365))
    all_immunity = np.zeros((n_people, len(a), max(a), y*365))
    all_strains = {}
    all_malaria = {}
    all_infections = {}

    # Load dataset for fever threshhold
    fever, breaks = load_data()

    # Simulate people
    for person in range(n_people):
        pmatrix, smatrix, imatrix, malaria, infections = simulate_person(y,a,w,fever,breaks,eir=eir, delta=delta,immune_thresh=immune_thresh,duration=duration, meroz=meroz, timeToPeak=timeToPeak, maxParasitemia=maxParasitemia, pgone=pgone)
        all_parasites[person,:,:,:] = pmatrix
        all_immunity[person,:,:,:] = imatrix
        all_strains[person] = smatrix
        all_malaria[person] = malaria
        all_infections[person] = infections

    return all_parasites, all_immunity, all_strains, all_malaria, all_infections

def check_moi(y,sM):
    '''
    Returns MOI every 30 days:
        y = years
        sM = matrix tracking strains across time.
    '''
    mois = []
    for t in range(0,y*365,30):
        moi = sM[:,t].sum()
        mois.append(moi)
    return mois

def check_parasitemia(y,pM,detect_thresh=0.001):
    '''
    Returns parasite density every 30 days, and the percent of times (every 30
    days) over the course of the study that someone had parasites.
    '''
    pdensity = []
    ppositivity = []
    for t in range(0,y*365,30):
        p = pM[0,:,t].sum()
        pdensity.append(p)
        if p > detect_thresh:
            ppositivity.append(1)
        else:
            ppositivity.append(0)
    perpos = np.average(ppositivity)
    return pdensity, perpos

def check_infection_length(sM,y, malaria):
    lengths = []
    infections = len(sM)
    for i in range(infections):
        counter = 0
        for j in range(y*365):
            if sM[i,j] == 1:
                counter += 1
                if j in malaria:
                    counter = 0
        if counter > 0:
            lengths.append(counter)
    return lengths

def get_visits(malaria,period,y):
    '''
    Returns passive & active visit dates in a list.
    '''
    start = np.random.randint(1,period)
    if start > malaria[0]:
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

def get_max_exp1_exp2(all_parasites, all_infections, all_malaria,y,a,period=28,measured=True):
    '''
    Returns arrays of maximum parasite density for exposure 1 & 2.
    If measured == True, this will be for measured maximums not true maximums.
    '''
    n_people = len(all_parasites)
    test_1 = []
    test_2 = []
    control_1 = []
    control_2 = []

    for person in range(n_people):
        if measured==True:
            visits = get_visits(all_malaria[person],period,y)
        else:
            visits = []

        # Control loci
        counter = 0
        for allele in range(a[0]):
            windows = get_infection_windows(0,allele,all_parasites[person,...],visits=visits,infectmatrix=all_infections[person],smatrix=all_strains[person])
            if np.all(windows[:,0]>=0):
                control_1.append(get_max_pdensity(all_parasites[person,...],0,allele,windows[:,0], visits=visits))
            else:
                control_1.append(0)
            if np.all(windows[:,1]>= 0): # WHY IS THIS NOT WORKING
                control_2.append(get_max_pdensity(all_parasites[person,...],0,allele,windows[:,1], visits=visits))
            else:
                control_2.append(0)
            counter += 1

        # Test loci
        counter = 0
        test_loci = len(a)-1
        for allele in range(a[test_loci]):
            windows = get_infection_windows(test_loci,allele,all_parasites[person,...],visits=visits,infectmatrix=all_infections[person],smatrix=all_strains[person])
            if np.all(windows[:,0] >=0):
                test_1.append(get_max_pdensity(all_parasites[person,...],test_loci,allele,windows[:,0], visits=visits))
            else:
                test_1.append(0)
            if np.all(windows[:,1] >=0):
                test_2.append(get_max_pdensity(all_parasites[person,...],test_loci,allele,windows[:,1], visits=visits))
            else:
                test_2.append(0)
            counter += 1

    return control_1,control_2,test_1,test_2
