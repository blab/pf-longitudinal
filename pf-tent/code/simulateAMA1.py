import argparse
import pfMech as sim
import numpy as np
import pandas as pd

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

def grab_visits(cohort_malaria,period,y):
    cohort_visits = {person:get_visits(cohort_malaria[person],period,y) for person in cohort_malaria.keys()}
    return cohort_visits

def update_clone(clones, Type, d,locus, timer, counter,person,allparasites,date,abs_pdens):
    for clone in clones:
        d['person'].append(person)
        d['clone'].append(clone)
        d['lociType'].append(Type)
        cPdens = allparasites[person,locus,clone,date]
        d['pdens'].append(cPdens)
        d['relativeFreq'].append(cPdens/abs_pdens)
        d['age'].append(date)
        time = timer[clone]
        if time ==0:
            time = None
        d['timeSinceAllele'].append(time)
        d['nAllele'].append(counter[clone])
    return d

def infer_infect_AMA1(all_parasites,all_malaria,all_visits,ama1,testLocus,ctrlLocus,thresh=10):
    n_people,nLoci,nAlleles,days = all_parasites.shape
    testPop = len(np.where(np.sum(all_parasites[:,testLocus,:,:],axis=(0,2))>0,)[0])
    ctrlPop = len(np.where(np.sum(all_parasites[:,ctrlLocus,:,:],axis=(0,2))>0,)[0])

    idict = {
        'person':[],
        'pdens':[],
        'symptomatic':[],
        'infectNumber':[],
        'COIinfection':[],
        'clonesBefore':[],
        'malariaNumber':[],
        'age':[],
        'timeSince':[],
        'timeSinceSymp':[],
        'timeSinceTest':[],
        'timeSinceCtrl':[],
        'nTest':[],
        'nCtrl':[],
        'propTestSeen':[],
        'propCtrlSeen':[],
        'propTestPop':[],
        'propCtrlPop':[]
    }

    cdict = {
        'person':[],
        'clone':[],
        'lociType':[],
        'pdens':[],
        'relativeFreq':[],
        'age':[],
        'timeSinceAllele':[],
        'nAllele':[]
    }

    for person in range(n_people):
        current = set()
        currentTest = set()
        currentControl = set()
        clones = 0
        infects = 0
        malaria = 0
        timeInfect = None
        daySymp = None
        timeAllele = np.zeros(testPop,dtype=int)
        nAllele = np.zeros(testPop,dtype=int)
        timeControl = np.zeros(ctrlPop,dtype=int)
        nControl = np.zeros(ctrlPop,dtype=int)
        for date in all_visits[person]:
            abs_pdens = np.sum(all_parasites[person,0,:,date])
            if abs_pdens >=thresh: # LAMP sensitivity cutoff.
                found = set(np.where(all_parasites[person,ama1,:,date]/abs_pdens > 0.005)[0]) # Only detect things at 0.5% frequency
                test = np.where(all_parasites[person,testLocus,:,date]/abs_pdens > 0.005)[0] # Only detect things at 0.5% frequency
                control = np.where(all_parasites[person,ctrlLocus,:,date]/abs_pdens > 0.005)[0] # Only detect things at 0.5% frequency
                new = found - current
                newClones = len(new)

                if newClones:

                    infects += 1
                    idict['infectNumber'].append(infects)

                    idict['person'].append(person)
                    idict['pdens'].append(abs_pdens)
                    idict['malariaNumber'].append(malaria)
                    idict['clonesBefore'].append(clones)

                    if daySymp:
                        idict['timeSinceSymp'].append(date-daySymp)
                    else:
                        idict['timeSinceSymp'].append(daySymp)

                    if date in all_malaria[person]:
                        idict['symptomatic'].append(1)
                        malaria += 1
                        daySymp = date
                    else:
                        idict['symptomatic'].append(0)
                    idict['COIinfection'].append(newClones)
                    clones += newClones
                    idict['age'].append(date)

                    if timeInfect:
                        idict['timeSince'].append(date - timeInfect)
                    else:
                        idict['timeSince'].append(timeInfect)
                    timeInfect = date

                    lastDates = [timeAllele[newA] for newA in test if timeAllele[newA] != 0]
                    idict['timeSinceTest'].append(np.mean(lastDates))


                    lastDatesCtrl = [timeControl[newA] for newA in control if timeControl[newA] != 0]
                    idict['timeSinceCtrl'].append(np.mean(lastDatesCtrl))


                    idict['nTest'].append(np.mean(nAllele[test]))
                    propAllele = len([nAllele[testA] for testA in test if nAllele[testA]>0])/len(test)
                    idict['propTestSeen'].append(propAllele)
                    idict['propTestPop'].append(len(nAllele[nAllele>0])/testPop)


                    idict['nCtrl'].append(np.mean(nControl[control]))
                    propControl = len([nControl[controlA] for controlA in control if nControl[controlA]>0])/len(control)
                    idict['propCtrlSeen'].append(propControl)
                    idict['propCtrlPop'].append(len(nControl[nControl>0])/ctrlPop)

                    timeSinceAlleles = date - timeAllele
                    timeSinceControls = date - timeControl
                    cdict = update_clone(test, 'test', cdict,testLocus, timeSinceAlleles, nAllele,person,all_parasites,date,abs_pdens)
                    cdict = update_clone(control,'ctrl',cdict,ctrlLocus,timeSinceControls,nControl,person,all_parasites,date,abs_pdens)

                    for newA in control:
                        timeControl[newA] = date
                    for newA in test:
                        timeAllele[newA] = date
                    nControl[control] += 1
                    nAllele[test] += 1
                    current = new

    idf = pd.DataFrame(idict)
    cdf = pd.DataFrame(cdict)
    return idf,cdf

def get_sim_freq(all_bites,n_people,test_loci=10,ctrl_loci=2):
    results = {
        'clone':[],
        'person':[],
        'lociType':[],
        'simFreq':[]
    }
    popT = {clone:0 for clone in range(10)}
    popC = {clone:0 for clone in range(10)}
    total = 0
    for person in range(n_people):
        for lab,loc in zip(['test','ctrl'],[test_loci,ctrl_loci]):
            values = all_bites[person][loc,:]
            if lab == 'test':
                total += len(values)
            for clone in range(10):
                sim = len(values[values==clone])/len(values)
                if lab=='test':
                    popT[clone] += len(values[values==clone])
                else:
                    popC[clone] += len(values[values==clone])
                results['clone'].append(clone)
                results['person'].append(person)
                results['lociType'].append(lab)
                results['simFreq'].append(sim)

    df = pd.DataFrame(results)
    freqT = {clone:popT[clone]/total for clone in range(10)}
    freqC = {clone:popC[clone]/total for clone in range(10)}

    freqdf = pd.DataFrame({'clone':list(freqT.keys()) + list(freqC.keys()),'popSimFreq':[freqT[clone] for clone in freqT.keys()] + [freqC[clone] for clone in freqC.keys()], 'lociType': ['test']*len(freqT.keys()) + ['ctrl']*len(freqC.keys())})
    df = df.merge(freqdf, on = ['clone','lociType'])
    return df

def modify_df(infections,clones,all_bites,n_people,test,ctrl):

    infections['logPdens'] = np.log10(infections.pdens)
    clones['logPdens'] = np.log10(clones.pdens)
    infections['TestBefore'] = np.where(infections['nTest']<1,0,1)
    infections['CtrlBefore'] = np.where(infections['nCtrl']<1,0,1)
    clones['SeenBefore'] = np.where(clones['nAllele']>0,1,0)
    infections['deltaPdens'] = infections['logPdens'] - np.roll(infections['logPdens'],1)
    infections['deltaPdens'] = np.where(infections.infectNumber==1, np.nan,infections['deltaPdens'])
    clones['deltaPdens'] = clones.groupby(['person','clone','lociType'],group_keys=False)['logPdens'].apply(lambda x: x - np.roll(x,1))
    clones['deltaPdens'] = np.where(clones.nAllele==0,np.nan,clones['deltaPdens'])
    clones['deltaFreq'] = clones.groupby(['person','clone','lociType'],group_keys=False)['relativeFreq'].apply(lambda x: x - np.roll(x,1))
    clones['deltaFreq'] = np.where(clones.nAllele==0,np.nan,clones['deltaFreq'])
    clones['nInfections'] = clones.groupby(['lociType'])['lociType'].transform('size')
    clones['nClones'] = clones.groupby(['lociType','clone'])['clone'].transform('size')
    clones['popFreq'] = clones['nClones']/clones['nInfections']
    clones['nInfectionsPerson'] = clones.groupby(['lociType','person'])['lociType'].transform('size')
    clones['nClonesPerson'] = clones.groupby(['lociType','clone','person'])['clone'].transform('size')
    clones['personFreq'] = clones['nClonesPerson']/clones['nInfectionsPerson']

    merged = clones.merge(infections[['person','age','clonesBefore','infectNumber','COIinfection','symptomatic','timeSince','propTestPop','propCtrlPop','malariaNumber','timeSinceSymp']],on=['person','age'])

    simFreq = get_sim_freq(all_bites,n_people,test+1,ctrl+1)
    merged = merged.merge(simFreq,on=['person','clone','lociType'])
    return infections, merged

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assign colors based on ordering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n', type=int, default=1000, help='Number of people to simulate')
    parser.add_argument('--eir', type=float, default=100, help='EIR to simulate under')
    parser.add_argument('--a', nargs='+',default = [5,10,10,10,10,10,10,10,10,10,50,10], help='allele array')
    parser.add_argument('--w', nargs='+',default = [0,0,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.5], help='weight array')
    parser.add_argument('--test',type=int,default=11,help = 'test locus')
    parser.add_argument('--ctrl',type=int, default=1,help = 'ctrl locus')
    parser.add_argument('--ama1',type=int, default=10,help='ama1 locus')
    parser.add_argument('--y',default=8,type=int,help ='Number of years to simulate')
    parser.add_argument('--p',type=int,default=84, help = 'days between routine visits')
    parser.add_argument('--output-clone',type=str,help = 'path to output clone-level df')
    parser.add_argument('--output-infect',type=str,help='path to output infect-level df')

    args = parser.parse_args()


    fever = np.load("data/fever.npy")
    breaks = np.load("data/breaks.npy")
    fever_arr = sim.get_fever_arr(args.eir,fever,breaks)

    if args.n > 100:
        number = int(np.floor(args.n/100))
        r = args.n % 100
        if r != 0:
            people = np.concatenate((np.repeat(100,number), [r]))
        else:
            people = np.repeat(100,number)
    else:
        people = [args.n]

    all_infections = []
    all_clones = []
    counter = 0
    for i,n in enumerate(people):
        all_parasites, all_immunity, all_strains, all_malaria,all_bites = sim.simulate_cohort(n,args.y,args.eir,args.a,args.w)
        all_visits = grab_visits(all_malaria,args.p,args.y)
        infections,clones = infer_infect_AMA1(all_parasites,all_malaria,all_visits,args.ama1,args.test,args.ctrl)
        infections,clones = modify_df(infections,clones,all_bites,n,args.test,args.ctrl)

        del all_parasites
        del all_immunity
        del all_strains
        del all_malaria
        del all_bites

        infections['person'] = infections['person'] + counter
        clones['person'] = clones['person'] + counter
        all_infections.append(infections)
        all_clones.append(clones)

        counter += n

    infectDF = pd.concat(all_infections)
    infectDF.to_csv(args.output_infect,sep='\t',index=False)

    cloneDF = pd.concat(all_clones)
    cloneDF.to_csv(args.output_clone,sep='\t',index=False)
