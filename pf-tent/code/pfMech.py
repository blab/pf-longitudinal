'''
Code to simulate within-host Pf using growth rate model.
Variables:
 - y, years to simulate
 - eir, entomological inoculation rate to simulate under
 - a, vector whose length corresponds to number of loci. Each entry corresponds
 to number of alleles at that loci.
 - w, vector weighting immunity at each loci. Should sum to 1.
 - meroz = 0.8, scale for mz AKA rough median
 - mshape = 1, variance of lognorm distribution for starting number of merozoites
 - growthrate = 0.6, average growthrate
 - rscale = 0.1, variance in growthrate
 - tHalf = 100, half-life for immunity
 - rend = final growth rate at full immunity
 - xh = inflection point for % immunity change
 - b = intensity of immune effect
 - k = 500,000, maximum parasitemia
 - pgone = 0.001, parasite gone threshold

'''
import numpy as np
import scipy.stats as st

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

def simulate_genotypes(n,a):
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
        floats = np.random.rand(length) # Currently only simulates things from uniform distribution
        genotype = np.ceil(floats*a)-1 # Current only simulates things ffrom uniform distribution
        M[:,i] = genotype
    return M

def get_mz(meroz=0.8,mshape=1): ### Work here
    '''
    Generates starting number of merozoites from a lognormal distribution.
    Values from here: https://www.science.org/doi/10.1126/scitranslmed.aag2490
    '''
    mz = st.lognorm.rvs(s=mshape,scale=meroz)
    return mz

def get_r(growthrate=0.6,rscale=0.1):
    '''
    Generates r from normal distribution.
    Values from here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7198127/
    '''
    r = st.norm.rvs(loc = growthrate, scale=rscale)
    return r

def simulate_params(n,meroz,growthrate,mshape=mshape,rscale=rscale):
    '''
    Simulates mz and growth rate for strains.
    Returns 2 x n matrix.

    Inputs:
        n = number of strains
        meroz = average starting number of merozites
        growthrate = average starting growthrate
    '''
    M = np.zeros((2,n))
    for i in np.arange(n):
        M[0,i] = get_mz(meroz,mshape)
        M[1,i] = get_r(growthrate,rscale)
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

def create_strain_matrix(n,y):
    '''
    Creates n x y*365 matrix to track strains presence per day.
    '''
    M = np.zeros((n,y*365))
    return M

def add_infection(t,j,mz,gtype,sM,pM):
    '''
    Adds infection to strain & parasite matrices.

    Inputs:
        t = time in days
        j = bite number
        mz = start of merozites
        gtype = genotype of infection
        sM = matrix tracking parasitemia by strain
        pM = matrix tracking parasitemia by allele
    '''
    sM[j,t] = mz
    for locus, allele in enumerate(gtype): # Might be able to speed up with vectorization here
        pM[locus, allele,t] = mz

def update_immunity(t,locus,allele,pM,iM,tHalf): # Use with np.vectorize()
    '''
    Immunity exponentially decays if parasite not present.

    Inputs:
        t = time in days
        locus = locus number
        allele = allele number
        pM = matrix tracking parasitemia by allele
        iM = matarix tracking immunity by allele
        tHalf = half-life of immunity
    '''
    gamma = 0.693/tHalf
    if t >= 14:
        if pM[locus,allele,t-14] > 0:
            iM[locus,allele,t] = 1
        elif iM[locus,allele,t-1] != 0:
            iM[locus,allele,t] = iM[locus,allele,t-1]*np.exp(-gamma)

def get_strain_immunity(gtype,i2,w):
    '''
    Returns immunity to strain.
    gtype = genotype of strain
    i2 = immune matrix at t of interest.
    w = vector weighting immunity by allele
    '''
    n_loci = len(gtype)
    cross = np.empty(n_loci)
    for i in np.arange(n_loci): ## Could try to vectorize; might speed up.
        allele = gtype[i]
        cross[i] = i2[i,allele]
    imm = (cross*np.asarray(w)).sum()
    return imm

def modulate_growth_rate(imm,r0,rend=-0.05,xh=0.3, b=-2):
    '''
    Modulates growth rate based on immunity.
    rend = final growth rate at full immunity
    xh = inflection point for % immunity change
    b = intensity of immune effect
    '''
    if imm == 0:
        return r0
    else:
        c = np.tan(np.pi/2*xh)**b
        r = ((r0-rend)/(c/np.tan(np.pi/2*imm)**b+1)) + rend
    return r

def update_parasitemia(t,j,w,gM,iM,rV,sM,pM,k = 5*10**5,rend=-0.05,xh=0.3,b=-2,pgone=0.001): # vectorize across J
    '''
    Updates parasitemia for each strain and allele by bite number.

    Inputs:
        t = time in days
        j = bite number
        w = vector weighting immunity by allele
        gM = genotype matrix
        iM = immune matrix
        rV = vector tracking initial growth rate by bite number.
        sM = matrix tracking parasitemia by strain
        pM = matrix tracking parasitemia by allele
        k = carrying capacity
        rend = growth rate with 100% immunity
        xh = param scaling immunity's impact on growth rate
        b = param scaling immunity's impact on growth rate
        pgone = threshold below which infection goes to zero
    '''
    done = set()
    imm = get_strain_immunity(gM[:,j],iM[:,:,t-1],w)
    r = modulate_growth_rate(imm,rV[j],rend,xh,b)

    if sM[j,t] == 0:
        p0 = sM[j,t-1]
        if p0 != 0:
            if r > 0:
                p = k/(1+((k-p0)/p0)*np.exp(-r))
            else:
                p = p0*np.exp(r)
                if p < pgone:
                    p = 0
                    done.add(j)
        else:
            p = 0

        sM[j,t] = p

        for loci, allele in enumerate(gM[:,j]): # Might be able to speed up with vectorization here
            pM[loci, allele,t] += p
    return done

def get_fever_threshold(arr,t):
    '''
    Returns fever threshold at t
    '''
    thresh = arr[arr[:,0]>=(t/365),1][0]
    return thresh

def treat_malaria(t, threshhold, pM, sM, m,a):
    '''
    Treat if parasitemia goes above certain threshold. Modifies parasite density
    matrix & strain matrix. Returns updated list with days malaria has occured.
        threshold = threshold for treatment
        pM = matrix tracking parasite density by allele across time.
        sM = matrix tracking parasitemia by strain across time.
        m = list with days malaria has occured
    '''
    if pM[0,:,t-1].sum(axis=0) > threshhold:
        pM[:,:,t] = 0
        sM[:,t] = 0
        m.append(t)
        a = set()
    return m,a

def load_data():
    '''
    Loads data used in the model
    '''
    fever = np.load("data/fever.npy")
    breaks = np.load("data/breaks.npy")
    return fever, breaks

def get_fever_arr(eir,fever,breaks):
    '''
    Returns 40 x 2 array, where column 0 is age cutoffs, and column 1 is parasite density
    for fever threshold at that age.
    '''
    eir_loc = (breaks[:,2]>=eir).nonzero()[0][0]
    age_index,pardens_index = (fever[:,:,eir_loc]).nonzero()
    age_breaks, age_loc = np.unique(age_index,return_index=True)
    pdens = breaks[pardens_index[age_loc],1]
    age = breaks[age_breaks,0]
    arr = np.stack((age,10**pdens),axis=1)
    return arr

def simulate_person(y,eir,a,w,meroz=0.8,growthrate=1.2,mshape=1,rscale=0.1,tHalf=100,rend=-0.05,xh=0.3,b=-2,k=5*10**5,pgone=0.001):
    '''
    Runs simulation for one person. Returns matrix tracking parasitemia by allele,
    matrix tracking immunity by allele, matrix tracking parasitemia by strain, and
    a list containing the days of every malaria episode someone had.

    Inputs:
        y = years to simulate
        eir = entomological inoculation rate
        a = vector with number of allels per loci
        meroz = average starting number of merozites
        growthrate = average starting growthrate
        w = vector weighting immunity by allele
        k = carrying capacity
        rend = growth rate at 100% immmunity
        xh = param scaling immunity's impact on growth rate
        b = param scaling immunity's impact on growth rate
        pgone = threshold below which infection goes to zero
    '''
    # Simulates bites & strain characteristics
    bites = simulate_bites(y,eir)
    n = len(bites)
    gtypes = simulate_genotypes(n,a)
    params = simulate_params(n,meroz,growthrate,mshape=mshape,rscale=rscale)

    # Creates objects to record
    pM = create_allele_matrix(a,y)
    iM = create_allele_matrix(a,y)
    sM = create_strain_matrix(n,y)
    malaria = []
    active = set()

    # Runs forward time simulation
    ## Day 0:
    t = 0
    if t in bites: # Case where first bite is on day zero.
        locs = np.where(bites == t)
        for j in locs[0]:
            add_infection(t,j,params[0,j],gtypes[:,j],sM,pM)
            active.add(j)

    ## Day 1+
    for t in np.arange(1,y*365):
        if t in bites:
            if not len(malaria) > 0 or t - malaria[-1] > 14:
                locs = np.where(bites == t)
                for j in locs[0]:
                    add_infection(t,j,params[0,j],gtypes[:,j],sM,pM)
                    active.add(j)
        if active:
            for j in active:
                drop = update_parasitemia(t=t,j=j,w=w,gM=gtypes,iM=iM,rV=params[1,:],sM=sM,pM=pM,k=k,rend=rend,xh=xh,b=b,pgone=pgone)
                if drop:
                    active = active.difference(drop)
        thresh = get_fever_threshold(fever_arr, t)
        malaria,active = treat_malaria(t,thresh,pM,sM,malaria,active)
        for locus in np.arange(len(a)):
            for allele in np.arange(max(a)):
                update_immunity(t=t,locus=locus,allele=allele,pM=pM,iM=iM,tHalf=tHalf)
    return pM, iM, sM, malaria

def simulate_cohort(n_people,y,eir,a,w,meroz=0.8,growthrate=1.2,mshape=1,rscale=0.1,tHalf=100,rend=-0.05,xh=0.3,b=-2,k=5*10**5,pgone=0.001):
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

    # Load dataset for fever threshhold
    fever, breaks = load_data()
    fever_arr = get_fever_arr(eir,fever,breaks)

    # Simulate people
    for person in range(n_people):
        pmatrix, imatrix, smatrix, malaria = simulate_person(y,eir,a,w,meroz=meroz,growthrate=growthrate,mshape=mshape,rscale=rscale,tHalf=tHalf,rend=rend,xh=xh,b=b,k=k,pgone=pgone)
        all_parasites[person,:,:,:] = pmatrix
        all_immunity[person,:,:,:] = imatrix
        all_strains[person] = smatrix
        all_malaria[person] = malaria

    return all_parasites, all_immunity, all_strains, all_malaria
