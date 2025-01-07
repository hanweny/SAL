import os
import sys
import time
import pickle as pkl
import numpy as np
from importlib import reload
from collections import defaultdict

import torch
import torch.nn as nn
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


import NN
import utils

from IPython import get_ipython
kernel = get_ipython()
kernel.run_line_magic('load_ext', 'rpy2.ipython')

import CompetingMethod
CompetingMethod.set_kernel(kernel)


num_variables = 20

result_dict = defaultdict(dict)

def main():
    _, n, T, match_mode, match_prob, homo, linear, IS_num, evolve_mode, sim = sys.argv
    n, T, imp_perc, evolve_mode, sim = int(n), int(T), int(IS_num) * 0.1, int(evolve_mode), int(sim)
    homo = (homo =='True')
    linear = (linear == 'True')
    hidden_layers = [1] if linear else [1]

    match_prob = None if match_prob == 'None' else float(match_prob)
    output_file = f"{sim}.pkl"

    print('\n\n', n, T, match_mode, match_prob, homo, linear, imp_perc, output_file, evolve_mode, sim, '\n\n')

    start = time.time()
    utils.define_global_variables(n, T, num_variables, int(np.ceil(T * imp_perc)))
    matching_prob = utils.generate_matching_probability(match_mode, match_prob, sim)
    print("Matching prob:  ", matching_prob)

    utils.generate_data(matching_prob=matching_prob, homo=homo, linear=linear, X_evolve_mode=evolve_mode, seed=sim, device=device)
    utils.train_test_split_data(seed=sim)
    print("Positive Assignment:  ", (utils.O==1).detach().cpu().numpy().mean(axis=0))
    
    CompetingMethod.set_utils(utils)
    CompetingMethod.update_global_variable()

    f_class = nn.Linear if homo else NN.TLinear

    learning_rate = 5e-3 # 2e-3
    num_epochs = max(800, 120*T) # 160*T

    ########## RUN BOWL and get matching estimates ############################################
    result_dict['BOWL'] = CompetingMethod.BOWL(seed=sim)
    print("BOWL Complete\n{}\n{}\n".format(result_dict['BOWL']['train'],result_dict['BOWL']['test']))

    # Get estimated matching probabilities from BOWl optimal estimations
    AC = torch.zeros_like(utils.A)
    for t in range(utils.num_measurements):
        AC[:,t] = torch.tensor(CompetingMethod.DynTxRegime_evolve(utils.XA0, t))
    est_matching_prob = torch.zeros(utils.num_measurements+1).to(utils.A.device)
    est_matches = torch.unique((utils.A == AC).sum(1).float(), return_counts=True)
    est_matching_prob[est_matches[0].int()] = est_matches[1] / utils.A.shape[0]
    avg_matches = (est_matching_prob * torch.arange(utils.num_measurements+1).to(utils.A.device)).sum()
    Rest = torch.quantile(utils.R, 1 - avg_matches / utils.num_measurements)
    
    ############################################################################################
    try:
        # Use estimated matching probabilities to evaluate KIPWE
        NN.reload(utils)
        KIPWE = NN.trainNN(
            NN.DTRModel, num_variables, hidden_layers, NN.KIPWELOSS, pi=None, Rest=utils.R.min(),
            sw=torch.tensor(est_matching_prob).to(device), theta=10,
            lr=learning_rate, epochs=num_epochs, weight_decay=1e-8, T_max=2000, small=1e-10, f_class=f_class,
            verbose = True, verboseFunc=NN.DTRVerbose, display_intvl=200, seed=48, device=device
        )
        result_dict['KIPWE'] = NN.DTREval2(KIPWE, seed=sim)

        # Use ground-true matching probabilities to evaluate KIPWE
        KIPWE = NN.trainNN(
            NN.DTRModel, num_variables, hidden_layers, NN.KIPWELOSS, pi=None, Rest=utils.R.min(),
            sw=torch.tensor(matching_prob).to(device), theta=10,
            lr=learning_rate, epochs=num_epochs, weight_decay=1e-8, T_max=2000, small=1e-10, f_class=f_class,
            verbose = True, verboseFunc=NN.DTRVerbose, display_intvl=200, seed=48, device=device
        )
        result_dict['KIPWE-true'] = NN.DTREval2(KIPWE, seed=sim)
        print("KIPWE Complete\n{}\n{}\n{}\n".format(result_dict['KIPWE']['train'],
                                                result_dict['KIPWE']['val'],
                                                result_dict['KIPWE']['test']))    
        end = time.time()
        print("Time elapsed:  {:.2f}s\n\n".format(end-start))
    except Exception as e:
        print("Error happened when running KIPWE: ", e)
    
    ############################################################################################
    try:
        NN.reload(utils)
        SAL = NN.trainNN(NN.DTRModel, num_variables, hidden_layers, NN.SALoss, pi=None, sw=None, Rest=Rest, theta=1,
            lr=learning_rate, epochs=num_epochs, weight_decay=1e-8, T_max=2000, small=1e-10, f_class=f_class,
            verbose = True, verboseFunc=NN.DTRVerbose, display_intvl=200, seed=48, device=device)
        result_dict['SAL'] = NN.DTREval2(SAL, seed=sim)
        print("SAL Complete\n{}\n{}\n{}\n".format(result_dict['SAL']['train'],
                                                result_dict['SAL']['val'],
                                                result_dict['SAL']['test']))    
        end = time.time()
        print("Time elapsed:  {:.2f}s\n\n".format(end-start))
    except Exception as e:
        print("Error happened when running SAL: ", e)

    ############################################################################################
    try:
        weightNN = NN.trainNN(NN.WeightNNModel, num_variables, [8, 1], NN.WeightLoss, theta=0,
                lr=1e-2, epochs=1200, weight_decay=1e-5, T_max=800, small=1e-8, f_class=nn.Linear,
                verbose = True, verboseFunc=NN.WeightVerbose, display_intvl=200, seed=48, device=device)
        sw_pred = weightNN.getWeights()
        si_pred = sw_pred.argsort()[0]
        result_dict['sw_pred'] = sw_pred
        overlap_num = len(set(si_pred[-utils.num_important_stages:]).intersection(set(utils.si_arr)))
        result_dict['overlap'] = overlap_num
        print("WeightNN Complete - Overlapp:  {}\n".format(overlap_num))
        
        SWL = NN.trainNN(
            NN.DTRModel, num_variables, hidden_layers, NN.SWLoss, pi=None, Rest=Rest, sw=torch.tensor(sw_pred*T).to(device), 
            lr=learning_rate, epochs=num_epochs, weight_decay=1e-8, T_max=2000, small=1e-10, f_class=f_class, theta=1,
            verbose = True, verboseFunc=NN.DTRVerbose, display_intvl=200, seed=48, device=device
        )
        result_dict['SWL'] = NN.DTREval2(SWL, seed=sim) 
        print("SWL Complete\n{}\n{}\n{}\n".format(result_dict['SWL']['train'],
                                                result_dict['SWL']['val'], 
                                                result_dict['SWL']['test']))
    except Exception as e:
        print("Error happened when running SWL: ", e)


    ############################################################################################
    # CompetingMethod
    ############################################################################################
    try:
        result_dict['Q-learning'] = CompetingMethod.QL_unshared(seed=sim)
        print("QL Complete\n{}\n{}\n".format(result_dict['Q-learning']['train'],result_dict['Q-learning']['test']))
    except Exception as e:
        print("Error happened when running QL: ", e)
    
    try:
        result_dict['RWL'] = CompetingMethod.RWL(seed=sim)
        print("RWL Complete\n{}\n{}\n".format(result_dict['RWL']['train'],result_dict['RWL']['test']))
    except Exception as e:
        print("Error happened when running RWL: ", e)

    try:
        result_dict['C-learning'] = CompetingMethod.CLearning(seed=sim)
        print("C-learning Complete\n{}\n{}\n".format(result_dict['C-learning']['train'],result_dict['C-learning']['test']))

        result_dict['C-learning-NP'] = CompetingMethod.CLearning(seed=sim, nonParam='TRUE')
        print("C-learning NP Complete\n{}\n{}\n".format(result_dict['C-learning-NP']['train'],result_dict['C-learning-NP']['test']))
    except Exception as e:
        print("Error happened when running RWL: ", e)

    try:
        result_dict['SOWL'] = CompetingMethod.SOWL(homo=False, seed=sim)
        print("SOWL Complete\n{}\n{}\n".format(result_dict['SOWL']['train'],result_dict['SOWL']['test']))
    except Exception as e:
        print("Error happened when running SOWL: ", e)

    try:
        result_dict['AIPW'] = CompetingMethod.AIPWClass(seed=sim)
        print("AIPW Complete\n{}\n{}\n".format(result_dict['AIPW']['train'],result_dict['AIPW']['test']))
    except Exception as e:
        print("Error happened when running AIPW: ", e)

    end = time.time()
    print("Iteration time taken:  {:.3f}s\n\n".format(end - start))
    pkl.dump(dict(result_dict), open(output_file, 'wb'))
    return

if __name__ == '__main__':
    main()