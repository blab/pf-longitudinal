import argparse
import numpy as np
from simulate import simulate_cohort
import pandas as pd
import seaborn as sns
import matplotlib as mpl

def get_sim_stats(simulations, threshhold,n):
    '''
    Per person in simulation, returns
    infection lengths (strain specific),
    time between parasitemia,
    # of infections (strain specific), &
    # of continuous periods of parasitemia.
    '''

    infection_lengths = []
    between_infections = []
    n_infections = []
    g_infections = []
    N = len(simulations)
    for p in np.arange(N):
        infections = 0
        generic_infections = 0
        positive_times = np.where(simulations[p,:,:] > threshhold)[1]
        unique = np.unique(positive_times)
        for i in np.arange(len(unique)):
            if i==0:
                day1 = unique[i]
                generic_infections += 1
            else:
                time1 = unique[i]-day1
                if time1 > 1:
                    between_infections.append(time1-1)
                    generic_infections += 1
                day1 = unique[i]
        for strain in np.arange(n):
            infections_with_strain = 0
            locs = np.where(simulations[p,strain,:] < threshhold)[0]
            for v in np.arange(len(locs)):
                if v==0:
                    day = locs[v]
                else:
                    time = locs[v]-day
                    if time > 1:
                        infection_lengths.append(time)
                        infections_with_strain += 1
                    day = locs[v]
            infections += infections_with_strain
        n_infections.append(infections)
        g_infections.append(generic_infections)
    return infection_lengths, between_infections, n_infections, g_infections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Assign colors based on ordering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--strains', type=int, required=True, help="# of strains to simulate")
    parser.add_argument('--years', type=int, required=True, help="# of years to simulatee")
    parser.add_argument('--biting-rate', nargs=3, type=float, required=True, help="start, stop, and step over which to vary biting rate")
    parser.add_argument('--individuals', type=int, required=True, help="# of individuals to simulate")
    parser.add_argument('--threshhold', type=int, required=True, help="threshold of parasitemia detection")
    args = parser.parse_args()


    # Run simulations
    bite_rate_length = []
    length = []
    between = []
    bite_rate_between = []
    bite_rate_person = []
    infections = []
    parasitemia = []
    for k in np.arange(args.biting_rate[0], args.biting_rate[1], args.biting_rate[2]):
        simulated = simulate_cohort(args.years,k,args.individuals,args.strains)
        infection_lengths, between_infections, n_infections, n_parasitemia = get_sim_stats(simulated, args.threshhold, args.strains)

        tot_infections = len(infection_lengths)
        n_between = len(between_infections)
        n_persons = len(n_infections)

        bite_rate_length.extend([k]*tot_infections)
        bite_rate_between.extend([k]*n_between)
        bite_rate_person.extend([k]*n_persons)
        length.extend(infection_lengths)
        between.extend(between_infections)
        infections.extend(n_infections)
        parasitemia.extend(n_parasitemia)


    # Make dataframe
    sims_length = pd.DataFrame()
    sims_between = pd.DataFrame()
    sims_person = pd.DataFrame()
    sims_length["bite_rate"] = bite_rate_length
    sims_length["infection_length"] = length
    sims_between["bite_rate"] = bite_rate_between
    sims_between["between_infections_length"] = between
    sims_person["bite_rate"] = bite_rate_person
    sims_person["infection (strain specific)"] = infections
    sims_person["parasitemia detected"] = parasitemia
    sims_person_long = pd.melt(sims_person, id_vars="bite_rate", value_vars=["infection (strain specific)","parasitemia detected"], var_name='type')
    sims_person["bite_rate"] = bite_rate_person
    sims_person["infection (strain specific)"] = infections
    sims_person["parasitemia detected"] = parasitemia
    sims_person_long = pd.melt(sims_person, id_vars="bite_rate", value_vars=["infection (strain specific)","parasitemia detected"], var_name='type')

    # Plot
    l = sns.relplot(x="bite_rate", y="infection_length", data=sims_length, kind="line")
    (l.set_axis_labels("Average bites per day", "Length of infection")
      .tight_layout(w_pad=0))
    l.savefig("figures/infectionLength_biteRate.pdf")


    b = sns.relplot(x="bite_rate", y="between_infections_length", data=sims_between, kind="line")
    (b.set_axis_labels("Average bites per day", "Time between detectable parasitemia")
      .tight_layout(w_pad=0))
    b.savefig("figures/betweenInfections_biteRate.pdf")

    g = sns.relplot(x="bite_rate", y="value", data=sims_person_long, hue="type", kinde="line")
    (g.set_axis_labels("Average bites per day", "Number of episodes")
      .tight_layout(w_pad=0))
    g.savefig("figures/infections_biteRate.pdf")
