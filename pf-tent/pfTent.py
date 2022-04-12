'''
Code for simulating longitudinal trajectories of individuals with _P.
falciparum_ infections.
Variables:
  y = years to simulate
  k = 0.11, biting rate per day
  n = number of strains simulated

  alpha = 1/500, general immunity acquistion rate
  beta = 1/500, general immunity loss rate
  gamma = 1/50, strain immunity acquistion rate
  delta = 1/500, strain immunity loss rate

  duration = 500, max infection length
  meroz = 0.01, initial parasites/uL, not log scale
  timeToPeak = 10, time from first parasitemia to peak
  maxParasitemia = 6, maximum parasitemia on log scale

  xh = 0.5, midpoint location for sigmoid for immunity on params
  b = -1, slope of sigmoid for immunity on params

  treatment_thresh = 100000, average parasitemia threshhold at which people get treatment
  pgone = -3, threshold for parasites being gonelog10 scale
  immune_thresh = 10, threshold at which you start gaining immunity
  detect_thresh = 0.001, threshold for detecting infection. Same as pgone.

  w = vector for weighting immune impact, should all add to one. Length is len(a) + 1
  a = vector containing # of alleles at each loci.
'''

import numpy as np
import scipy.stats as st

def simulate_bites(y,k):
    '''
    Produces a vector with bite times up until year, y, is reached.

    Time between bites pulled from exponential distribution wih mean rate of k.
    '''
    n = round(y*k*365*2)
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

def create_gen_imm(y):
    '''
    creates vector for immunity of lenth y *365
    '''
    immunity = np.zeros(y*365)
    return immunity

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

def modulate_params(gtype, strain_imm, gen_imm, params, w):
    '''
    Changes all infection params according to immunity:
        gtype = genotype vector for infection
        strain_imm = strain_immunity at time of infection
        gen_imm = general immunity at time of infection.
        params = params vector to modulate
        w = vector modulating immunity effect at locus
    '''
    n_loci = len(gtype)
    cross = np.empty(n_loci)
    for i in np.arange(n_loci):
        allele = gtype[i]
        cross[i] = strain_imm[i,allele]
    imm = np.append(gen_imm, cross)

    M = np.zeros((4, n_loci+1))
    for i, p in enumerate(params):
        for j, v in enumerate(imm):
            if v == 0:
                M[i,j] = p*w[j]
            else:
                M[i,j] = sigmoid(v, p*w[j])
    modified = np.zeros(4)
    for i in np.arange(4):
        modified[i] = M[i,:].sum()
    modified[0] = np.rint(modified[0])
    modified[2] = np.rint(modified[2])
    return modified

def update_immunity(pM, iV, iM, t, immune_thresh, alpha, beta, gamma, delta,):
    '''
    If parasites present, gains immunity. If absent, loses immunity:
        pM = parasitemia matrix at each alleles across time.
        iV = vector of general immunity across time.
        iM = matrix of strain immunity at each allele across time
        t = time
        immune_thresh = parasite density threshold at which gain immunity.
    '''
    # General immunity
    if pM[0,:,t].sum(axis=0) > immune_thresh:
        if t == 0:
            print("Parasitemia was high enough at t=0, wowza, for gen imm")
            iV[t] = alpha
        else:
            iV[t] = min(iV[t-1] + alpha,1)
    else:
        if t == 0:
            iV[t] = 0
        else:
            iV[t] = max(iV[t-1] - beta, 0)

    # Strain immunity
    loci = pM.shape[0]
    n_alleles = pM.shape[1]
    for i in np.arange(loci):
        for j in np.arange(n_alleles):
            if pM[i,j,t] > immune_thresh:
                if t == 0:
                    print("Parasitemia was high enough at t=0, wowza, for strain imm")
                    iM[i,j,t] = gamma
                else:
                    iM[i,j,t] = min(iM[i,j,t-1] + gamma,1)
            else:
                if t == 0:
                    iM[i,j,t] = 0
                else:
                    iM[i,j,t] = max(iM[i,j,t-1] - delta, 0)

def treat_as_needed(treatment_thresh, pM, sM, t, m):
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
    threshhold = st.lognorm.rvs(s=0.4,scale=treatment_thresh)
    if pM[0,:,t].sum(axis=0) > threshhold:
        pM[:,:,t+1:] = 0
        sM[:,t+1:] = 0
        m.append(t)
    return m

def simulate_person(y,a,w, k=0.11, alpha=1/500, beta=1/500, gamma=1/50, delta=1/500,immune_thresh=10,treatment_thresh=100000,duration = 500, meroz = .01, timeToPeak = 10, maxParasitemia = 6, pgone=-3):
    '''
    Runs simulation for one person.
    Returns matrix of parasitemia by allele across time & matrix of strains
    across time.
    '''
    malaria = []
    bites = simulate_bites(y,k)
    n_bites = len(bites)
    strains = simulate_strains(n_bites,a)
    pmatrix = create_allele_matrix(a, y)
    smatrix = create_strain_matrix(n_bites,y)
    imatrix = create_allele_matrix(a,y)
    ivector = create_gen_imm(y)

    counter = 0
    for t in range(365*y):
        update_immunity(pmatrix,ivector,imatrix,t,immune_thresh, alpha, beta, gamma, delta)
        malaria = treat_as_needed(treatment_thresh,pmatrix,smatrix,t,malaria)
        if t in bites:
            locs = np.where(bites == t)
            for i in locs[0]:
                params = get_infection_params(duration, meroz, timeToPeak, maxParasitemia)
                params = modulate_params(strains[:,i], imatrix[:,:,t], ivector[t], params, w)
                if params[0] > 0 and params[3] > 0 and params[1] > 0.001 and params[0] > params[2] and params[2] > 0:
                    parasitemia = get_parasitemia(params, pgone)
                    add_infection(parasitemia,pmatrix,strains[:,i],t,smatrix,i)
    return pmatrix, smatrix, imatrix, ivector, malaria

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

def check_infection_length(sM,y):
    '''
    Returns infection length for all infections.
    '''
    lengths = []
    infections = len(sM)
    for i in range(infections):
        counter = 0
        for j in range(y*365):
            if sM[i,j] == 1:
                counter += 1
        if counter > 0:
            lengths.append(counter)
    return lengths
