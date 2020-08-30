import sys
if '..' not in sys.path:
    sys.path.append('..')
    
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import os

from lib.mobilitysim import MobilitySimulator
from lib.town_data import generate_population, generate_sites, compute_distances
from lib.town_maps import MapIllustrator

import numpy as np
import pickle, math
import pandas as pd
import multiprocessing
from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import (MeasureList, Interval,
                          BetaMultiplierMeasureByType,
                          SocialDistancingForAllMeasure, 
                          SocialDistancingForPositiveMeasure,
                          SocialDistancingForPositiveMeasureHousehold,
                          SocialDistancingForSmartTracing,
                          ComplianceForAllMeasure,
                          ComplianceForEssentialWorkers)
from lib.inference import gen_initial_seeds, extract_seeds_from_summary
from lib.plot import Plotter
import matplotlib.pyplot as plt

# converting days to hours
TO_HOURS = 24.0
# Choose random seed
c = 0
# Set it
np.random.seed(c)
# Define prefix string used to save plots
runstr = f'run{c}_'

random_repeats = 20 # Set to at least 40 to obtain stable results

num_workers = multiprocessing.cpu_count() - 1

start_date = '2020-03-04'
end_date = '2020-06-19'
sim_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
max_time = TO_HOURS * sim_days # in hours


mob_settings = 'sf_type0_prop0.1_ds100.pk'
essential_to_total_pop_ratio = 0.1

    # See town-generator.ipynb for an example on how to create the settings
with open(mob_settings, 'rb') as fp:
    obj = pickle.load(fp)
mob = MobilitySimulator(**obj)
     
area = 'SF'
country = 'US'
new_cases_ = collect_data_from_df(country=country, area=area, datatype='new',
    start_date_string=start_date, end_date_string=end_date)
new_cases = np.ceil(
        (new_cases_ * mob.num_people_unscaled) /
        (mob.downsample * mob.region_population))
 
heuristic_seeds = True
# (a) define heuristically based on true cases and literature distribution estimates
if heuristic_seeds:
    initial_seeds = gen_initial_seeds(new_cases)

# (b) define based state of previous batch of simulations, 
# using the random rollout that best matched the true cases in terms of squared error
else:
    seed_summary_ = load_summary('summary_example.pk')
    seed_day_ = 7
    initial_seeeds = extract_seeds_from_summary(seed_summary_, seed_day_, new_cases) 
        
beta = 0.10 # this needs to be calibrated
inferred_params = {
    'betas' : {
        'education': beta,
        'social': beta,
        'bus_stop': beta,
        'office': beta,
        'supermarket': beta}, 
        'beta_household' : 0
}
    
def standard_testing(max_time):
    daily_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    standard_testing_params = {
        'testing_t_window'    : [0.0, max_time], # in hours
        'testing_frequency'   : 1 * TO_HOURS,     # in hours
        'test_reporting_lag'  : 2 * TO_HOURS,     # in hours (actual and self-report delay)
        'tests_per_batch'     : int(daily_increase.max()), # test capacity based on empirical positive tests
        'test_fpr'            : 0.0, # test false positive rate
        'test_fnr'            : 0.0, # test false negative rate
        'test_smart_delta'    : 3 * TO_HOURS, # in hours
        'test_smart_duration' : 7 * TO_HOURS, # in hours
        'test_smart_action'   : 'isolate', 
        'test_smart_num_contacts'   : 10, 
        'test_targets'        : 'isym',
        'test_queue_policy'   : 'fifo',
        'smart_tracing'       : None, 
    }
    return standard_testing_params
    
        
distributions = CovidDistributions(country=country)
    
    
    
def run(tparam, measure_list, t, local_seeds, dynamic_tracing=False):

   # add standard measure of positives staying isolated
    measure_list +=  [
        SocialDistancingForPositiveMeasure(
           t_window=Interval(0.0, t), p_stay_home=1.0),
        
        SocialDistancingForPositiveMeasureHousehold(
            t_window=Interval(0.0, t), p_isolate=1.0)
    ]
    measure_list = MeasureList(measure_list)

    # run simulations
    summary = launch_parallel_simulations(
        mob_settings, 
        distributions, 
        random_repeats, num_workers, 
        inferred_params, local_seeds, tparam, measure_list, 
        max_time=t, 
        num_people=mob.num_people, 
        num_sites=mob.num_sites, 
        site_loc=mob.site_loc, 
        home_loc=mob.home_loc,
        dynamic_tracing=dynamic_tracing,
        verbose=False)
    return summary
        
    
def save_summary(summary, filename):
    with open('summaries/' + filename, 'wb') as fp:
        pickle.dump(summary, fp)
    
def load_summary(filename):
    with open('summaries/' + filename, 'rb') as fp:
        summary = pickle.load(fp)
    return summary
    
    

"""
========================= END OF HEADER =========================

"""
    
num_essen = 0
num_non_essen = 0
work_site = []
for i in range(mob.num_people):
    if mob.essential_workers[i]:
        num_essen += 1
        work_site_temp = mob.essential_work_site[i]
        if work_site_temp not in work_site:
            if work_site_temp > -1 :
                work_site.append(work_site_temp)
        #print('essential worker no:',i,' working at site:',mob.essential_work_site[i])
    else:
        num_non_essen += 1

num_workers_at_work_site = [0]*len(work_site)
for i in range(mob.num_people):
    for j in range(len(work_site)):
        if mob.essential_work_site[i] == work_site[j]:
            num_workers_at_work_site[j] += 1
    
print('total population:',mob.num_people,', sites:', len(mob.site_type))
print('essential:',num_essen,', non_essential:',num_non_essen,', propotion:',num_essen/mob.num_people)
print('Sites by type: ',  [(np.array(mob.site_type) == i).sum() for i in range(len(mob.dur_mean_per_type))])
print('essential type:',mob.essential_type)
print('work_site:',work_site,len(work_site))
print('number of workers at each work site:',num_workers_at_work_site, np.array(num_workers_at_work_site).sum())
    
testing_params_SD_6 = standard_testing(max_time)
# time window considered for inspecting contacts
testing_params_SD_6['test_smart_delta'] = 24.0 * 3
testing_params_SD_6['test_smart_action'] = 'isolate'
testing_params_SD_6['test_targets'] = 'isym'
testing_params_SD_6['test_smart_num_contacts'] = 25
isolation_days_SD_6 = 7  # how many days selected people have to stay in isolation
duration_weeks_SD_6 = 12  # strategies tested for 12 weeks starting today

summaries_sf = dict()

    
p_compliance_temp = [1.0]    
p_compliance = [i * essential_to_total_pop_ratio for i in p_compliance_temp]
p_compliance_essential = p_compliance_temp

for policy in ['basic']:
    summaries_ = []
    testing_params_SD_6['smart_tracing'] = policy

    for p in p_compliance:

        m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
            ComplianceForAllMeasure(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p)
        ]
        res = run(testing_params_SD_6, m, max_time, initial_seeds)
        summaries_.append(res)

        print('all', p, ' done.')

        
    for p_essential in p_compliance_essential:

        m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
            ComplianceForEssentialWorkers(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p_essential)
        ]
        res = run(testing_params_SD_6, m, max_time, initial_seeds)
        summaries_.append(res)

        print('essential', p_essential, ' done.')

print('essential prop',essential_to_total_pop_ratio,' done.')        
summaries_sf[policy] = summaries_
save_summary(summaries_sf,'comp'+str(p_compliance_essential)+'_repeats'+str(random_repeats)+'_'+mob_settings)
