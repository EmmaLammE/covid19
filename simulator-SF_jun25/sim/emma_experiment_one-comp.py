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

# from lib.settings.town_settings_kaiserslautern import *
# from lib.settings.town_settings_ruedesheim import *
# from lib.settings.town_settings_tirschenreuth import *
# from lib.settings.town_settings_tubingen import *
from lib.settings.town_settings_sanfrancisco import *

# from lib.settings.town_settings_lausanne import *
# from lib.settings.town_settings_locarno import *
# from lib.settings.town_settings_lucerne import *
# from lib.settings.town_settings_jura import *

# Downsampling factor of population and sites
downsample = 200

# Country for different age groups
country = 'US' # 'GER', 'CH', 'US'

# Set the population generation mode.
# 3 options available: custom | random | heuristic
population_by = 'custom'

# Downsample population 
population_per_age_group = np.round(
    population_per_age_group * (town_population / (downsample * region_population))).astype('int').tolist()

print(f'Population per age group: {population_per_age_group}')

# change essential population ratio here
essential_to_total_ratio_list = [0.5]
summaries_ = []
summeries_sf = dict()
for essential_to_total_ratio in essential_to_total_ratio_list:
    num_essential_workers = np.floor(sum(population_per_age_group)*essential_to_total_ratio).astype('int').tolist()
    essential_type = 2# 0:edu, 1:social, 2:office, 3:supermarket
    
    essential_distribution = np.array([
    0,      # 0-4
    0,      # 5-14
    0.04,   # 15-19
    0.06,   # 20-24
    0.55,   # 25-44
    0.30,   # 45-59
    0.05,   # 60-79
    0])     # 80+

    num_essential_per_age_group = np.floor(num_essential_workers * essential_distribution).astype('int').tolist()
    essential_prop_per_age_group = np.divide((num_essential_per_age_group),(population_per_age_group))
    print('essential per age group',num_essential_per_age_group,'total',np.sum(num_essential_per_age_group))

    # This block sends queries to OpenStreetMap
    # Make sure you have a working internet connection
    # If an error occurs during execution, try executing again 
    # If the call times out or doesn't finish, try restarting your internet connection by e.g. restarting your computer
    site_files=[]
    for root,dirs,files in os.walk(sites_path):
        for f in files:
            if f.endswith(".txt") and f != 'buildings.txt':
                site_files.append(sites_path+f)

    site_loc, site_type, site_dict, density_site_loc = generate_sites(bbox=bbox, query_files=site_files,
                                    site_based_density_file=sites_path+'buildings.txt')

    if downsample > 1:
        np.random.seed(42)
        # downsample sites like populatoin
        idx = np.random.choice(len(site_loc), size=int(len(site_loc) / downsample), 
                               replace=False, p=np.ones(len(site_loc)) / len(site_loc))
    
        site_loc, site_type = np.array(site_loc)[idx].tolist(), np.array(site_type)[idx].tolist()


    if region_population == town_population:
        tile_level = 15
    else:
        tile_level = 16

    if population_by == 'custom':
        # generate population across tiles based on density input
        print('Tile level: ', tile_level)
        home_loc, people_age, home_tile, tile_loc, people_household, essential_workers, num_essential_workers, essential_work_site = generate_population(
            density_file=population_path, bbox=bbox,
            population_per_age_group=population_per_age_group, 
            household_info=household_info, tile_level=tile_level, seed=42,
            essential_prop_per_age_group=essential_prop_per_age_group,
            site_type = site_type, essential_type = essential_type)
    
    elif population_by == 'random':
        # generate population across tiles uniformly at random
        home_loc, people_age, home_tile, tile_loc, people_household , essential_workers, num_essential_workers, essential_work_site = generate_population(
            bbox=bbox, population_per_age_group=population_per_age_group,
            tile_level=16, seed=42,
            essential_prop_per_age_group=essential_prop_per_age_group)
    
    elif population_by == 'heuristic':
        # generate population across tiles proportional to buildings per tile
        home_loc, people_age, home_tile, tile_loc, people_household , essential_workers, num_essential_workers, essential_work_site = generate_population(
            bbox=bbox, density_site_loc=density_site_loc,
            population_per_age_group=population_per_age_group, tile_level=16, seed=42,
            essential_prop_per_age_group=essential_prop_per_age_group)
         
    essential_to_total_pop_ratio = num_essential_workers/sum(population_per_age_group)
    tile_site_dist = compute_distances(site_loc, tile_loc)


    if country == 'US':
        mob_rate_per_age_per_type = [
           [5, 0, 0, 0], # 0-5
           [5, 0, 0, 0], # 5-14
           [5, 0, 3.6, 0.22], # 15-19
           [1.48, 3.52, 3.6, 0.21], # 20-24
           [0, 5, 3.6, 0.27], # 25-44
           [0, 5, 3.6, 0.36], # 45-59
           [0, 0, 3.6, 0.35], # 60-79
           [0, 0, 3.6, 0.35]] # 80+
    else:
        print('we only have US data at this point...')
    
    # edu, social, office, supermarket
    dur_mean_per_type = [5.0, 5.0, 0.64, 0.4]
    variety_per_type = [1, 1, 10, 2]
    
    mob_rate_per_age_per_type = np.divide(np.array(mob_rate_per_age_per_type), (24.0 * 7))

    if essential_type == 0:
        essential_mob_rate_per_type = [5, 0, 3.6, 0.27]
        essential_dur_mean_per_type = [5, 0, 0.64, 0.4]
    elif essential_type == 1:
        essential_mob_rate_per_type = [0, 5, 3.6, 0.27]
        essential_dur_mean_per_type = [0, 5, 0.64, 0.4]
    elif essential_type == 2:
        essential_mob_rate_per_type = [0, 0, 5, 0.27]
        essential_dur_mean_per_type = [0, 0, 5, 0.4]
    elif essential_type == 3:
        essential_mob_rate_per_type = [0, 0, 3.6, 5]
        essential_dur_mean_per_type = [0, 5, 0.64, 5]
    
    essential_mob_rate_per_type = np.divide(np.array(essential_mob_rate_per_type), (24.0 * 7))

    # time horizon
    max_time = 17 * 24.0 # data availability
    delta  = 4.6438 # as set by distributions

    kwargs = dict(home_loc=home_loc, people_age=people_age,
        site_loc=site_loc, site_type=site_type, site_dict=site_dict,
        num_people_unscaled=town_population,
        region_population=region_population,
        downsample = downsample,
        mob_rate_per_age_per_type=mob_rate_per_age_per_type,
        dur_mean_per_type=dur_mean_per_type,
        variety_per_type=variety_per_type,
        daily_tests_unscaled=daily_tests_unscaled,
        delta=delta,
        home_tile=home_tile,
        tile_site_dist=tile_site_dist,
        people_household = people_household,
        essential_workers=essential_workers,
        essential_mob_rate_per_type=essential_mob_rate_per_type,
        essential_dur_mean_per_type = essential_dur_mean_per_type,
        essential_work_site = essential_work_site,
        essential_type=essential_type) # emma

    with open(f'sf_one-comp_essential-type{essential_type}_prop{essential_to_total_ratio}_ds{downsample}.pk', 'wb') as fp:
        pickle.dump(kwargs, fp)




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

    random_repeats = 10 # Set to at least 40 to obtain stable results

    num_workers = multiprocessing.cpu_count() - 1

    start_date = '2020-03-04'
    end_date = '2020-04-19'
    sim_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    max_time = TO_HOURS * sim_days # in hours


#     from lib.settings.town_settings_tubingen import *
    mob_settings = 'sf_one-comp_essential-type'+str(essential_type)+'_prop'+str(essential_to_total_ratio)+'_ds'+str(downsample)+'.pk'  

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
        
    beta = 0.40 # this needs to be calibrated
    inferred_params = {
        'betas' : {
            'education': beta,
            'social': beta,
            'bus_stop': beta,
            'office': beta,
            'supermarket': beta}, 
        'beta_household' : beta
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
    
    print('total population:',mob.num_people,', sites:', len(site_type))
    print('essential:',num_essen,', non_essential:',num_non_essen,', propotion:',num_essen/mob.num_people)
    print('essential proortion per age group:',essential_prop_per_age_group)
    print('Population (by Age): ', population_per_age_group)
    print('Sites by type: ',  [(np.array(site_type) == i).sum() for i in range(len(dur_mean_per_type))])
    print('essential type:',essential_type)
    print('work_site:',work_site,len(work_site))
    print('number of workers at work site:',num_workers_at_work_site, np.array(num_workers_at_work_site).sum())

    
    

    # ## Only simulate the future from here onwards
    #
    #

    # baseline (no measure for the future starting April 12)
    # future_baseline = run(standard_testing(max_time_future),
    #                       [], max_time_future, present_seeds)
    # save_summary(future_baseline, 'future_baseline_3.pk')

    # ### 4.3.5. Effects  of compliance on the efficacy of isolation for smart  tracing strategies

    testing_params_SD_6 = standard_testing(max_time)
    # time window considered for inspecting contacts
    testing_params_SD_6['test_smart_delta'] = 24.0 * 3
    testing_params_SD_6['test_smart_action'] = 'isolate'
    testing_params_SD_6['test_targets'] = 'isym'
    testing_params_SD_6['test_smart_num_contacts'] = 25
    isolation_days_SD_6 = 7  # how many days selected people have to stay in isolation
    duration_weeks_SD_6 = 12  # strategies tested for 12 weeks starting today

    summaries_sf = dict()

    
    p_compliance_temp = [0.0, 0.3, 0.6, 1.0]    
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

    print('essential prop',essential_to_total_ratio,' done.')        
summaries_sf[policy] = summaries_
save_summary(summaries_sf,'sf_comp'+str(p_compliance)+'_repeats'+str(random_repeats)+'_'+mob_settings)
