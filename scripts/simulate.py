"""
Defines functions which implement  model of blood stage acquired malaria immunity
as described in [Pinkevych et al](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002729#s2).

simulate_cohort(y,k,N,n) is the main workhorse where y = years to simulate,
k=biting ate, n=number of strains, N=number of individuals to simulate.

These parameters are hardcoded in because I spent waaaay too long trying to make
them not hardcoded in to no avail:
Z = 0.005, parasite gone threshhold
r = 16, initial parasite multiplication rate
alpha = 0.000015, strain specific immunity acquisition rate
beta = 0.003, strain specific immunity loss rate
l = 0.00000006, general immunity acquisition rate
delta = 0.005, general immunity loss rate
epsilon = 0.056, initial number of infected RBCs in millions
"""

import numpy as np
from scipy.stats import expon
from scipy.integrate import solve_ivp

def simulate_bites(y,k,N,n):
    '''
    Produces N x b x 2 matrix with time between bites
    in M[:,:,0] & strain_choice in M[:,:,1].

    Time between bites pulled from exponential distribution with mean rate of k.
    Strains pulled from a uniform distribution.
    '''
    b = round(y*k*365*2)
    M = np.empty((N,b,2))
    strains = np.arange(n)
    for i in np.arange(N):
        biting_times = expon.rvs(scale=(1/k), loc=0, size=b)
        strain_choice = np.random.choice(strains,b)
        M[i,:,0] = biting_times
        M[i,:,1] = strain_choice
    return M

def volume(x):
    '''
    Returns blood volume at age x in days.
    '''
    if x < (22*365):
        return (0.00059*x + 0.3)
    else:
        return 5

def h(x,Z=0.005):
    '''
    Returns parasitemia below threshold, Z, as 1.
    '''
    if x < Z:
        return 1
    else:
        return 0

def drop(x,Z=0.005):
    '''
    Returns parasitemia below threshold, Z, as zero.
    Otherwise returns parasitemia
    '''
    if x <Z:
        return 0
    else:
        return x

def dP(x,y,z,r=16):
    '''
    dP/dT per strain
    '''
    value = 1-h(x)
    return (value*(np.log(r)/2) - z - y)*x

def dS(x,y,alpha=0.000015,beta=0.003):
    '''
    dS/dT per strain
    '''
    return (alpha*x) - (beta*h(x)*y)

def equations(t,state,n,l=0.00000006,delta=0.005):
    '''
    Returns ODEs to integrate
    '''
    p = state[0:n]
    s = state[n:(2*n)]
    G = state[-1]
    dY = np.empty((2*n)+1)
    dY[0:n] = np.vectorize(dP)(p,s,G)
    dY[n:(2*n)] = np.vectorize(dS)(p,s)
    dY[-1] = (l*np.sum(p)) - (delta*h(np.sum(p))*G)
    return dY

def get_bite_times(M,p,y):
    '''
    Returns bite times for 0 to y years from times between bites
    for person p.
    '''
    tUpdate = []
    time = 0
    for t in np.arange(M.shape[1]):
        time += M[p,t,0]
        if time > (y*365):
            break
        tUpdate.append(time)
    tUpdate.append(y*365)
    return tUpdate

def get_bite_strains(M,p):
    '''
    Returns strains associated with each bite for person p.
    '''
    strains = M[p,:,1]
    return strains.astype(int)

def simulate_person(years, n, tBites, bStrains,epsilon=0.056):
    '''
    Simulates infection per person.
    Returns strain specific parasitemia, immunity,
    & general immunity for y * 365 days in 2n+1 x days matrix.
    '''
    days = years*365
    results = np.empty((2*n+1,days))
    current_time = 0

    for i, t in enumerate(tBites):
        # first interval starts at zero
        if i == 0:
            y0 = np.zeros(2*n+1)
            t0 = 0

        record = np.arange(np.ceil(t0),np.ceil(t))
        length = len(record)
        times = np.append(record,t)

        output = solve_ivp(fun=lambda t, y: equations(t, y, n), t_span=(t0,t), y0=y0, t_eval=times)
        results[:,current_time:(current_time+length)] = output.y[:,:-1]

        if t < years*365:
            end_value = output.y[:,-1]
            t0 = output.t[-1]
            current_time += length

            # Check that strain parasitemia isn't below threshold
            y0 = np.vectorize(drop, otypes=[float])(end_value)

            # inoculate with strain
            strain = bStrains[i]
            y0[strain] += (epsilon/volume(t))

    return results

def simulate_cohort(y,k,N,n):
    '''
    Simulates infections for all people.
    Returns N x 2n+1 x y*365 matrix.
    '''
    M = simulate_bites(y,k,N,n)
    simulations = np.empty((N,2*n+1,y*365))

    for p in range(N):
        biteTimes = get_bite_times(M,p,y)
        biteStrains = get_bite_strains(M,p)
        simulations[p,:,:] = simulate_person(y,n,biteTimes,biteStrains)

    return simulations
