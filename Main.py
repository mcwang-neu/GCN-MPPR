import logging
import argparse
import os
import numpy as np
import torch
from tabulate import tabulate
from ppnp.pytorch import PPNP
from ppnp.pytorch.training import train_model
from ppnp.pytorch.earlystopping import stopping_args
from ppnp.pytorch.propagation import PPRExact, MPPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset
from cal_motif import cal_main

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_box(data, title):
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
    plt.rcParams["axes.unicode_minus"] = False

    sns.set_palette("husl")
    plt.figure(figsize=(4, 3))
    ax = sns.boxplot(data=data, palette="deep")
    ax.set_xticklabels(['Best\nBaseline', "$M_1$", '$M_2$', "$M_3$", '$M_4$',"$M_5$", '$M_6$',"$M_7$"])
    plt.title(title)
    plt.show()

def pickle_save(data, filename):
    import _pickle as pickle
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
def pickle_load(filename):
    import _pickle as pickle
    f =open(filename, 'rb')
    return pickle.load(f)

def ppnp(graph_name, per_seed):
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)

    graph = load_dataset(graph_name, 'ppnp\\dataset_for_paper')
    graph.standardize(select_lcc=True)
    test = True

    test_seeds = [
        2144199730, 794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407, 635625077, 3538425002, 960893189,
        497096336, 3940842554, 3594628340, 948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]
    val_seeds = [
        2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
        1920426428, 4272044734, 2092442742, 841404887, 2188879532,
        646784207, 1633698412, 2256863076, 374355442, 289680769,
        4281139389, 4263036964, 900418539, 119332950, 1628837138]

    if test:
        seeds = test_seeds
    else:
        seeds = val_seeds

    if graph_name == 'ms_academic':
        alpha = 0.2
    else:
        alpha = 0.1
    alpha = 0.1
    prop_ppnp = PPRExact(graph.adj_matrix, alpha=alpha)
    # prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=alpha, niter=10)

    model_args = {
        'hiddenunits': [64],
        'drop_prob': 0.5,
        'propagation': prop_ppnp}

    reg_lambda = 5e-3
    learning_rate = 0.01

    niter_per_seed = per_seed
    save_result = False
    print_interval = 100
    device = 'cuda'

    results = []
    niter_tot = niter_per_seed * len(seeds)
    i_tot = 0
    test_accs = []
    run_time = []
    logging.log(22, 0)
    for seed in seeds:
        idx_split_args['seed'] = seed
        for _ in range(niter_per_seed):
            i_tot += 1
            logging_string = f"Iteration {i_tot} of {niter_tot}"
            logging.log(22, '-' * len(logging_string) + logging_string + '-' * len(logging_string))
            _, result = train_model(
                graph_name, PPNP, graph, model_args, learning_rate, reg_lambda,
                idx_split_args, stopping_args, test, device, None, print_interval)
            results.append({})
            test_accs.append(result['valtest']['accuracy'])
            run_time.append(result['runtime'])
    pickle_save(test_accs, graph_name + '_ppnp.pkl')
    a = str(np.round(np.mean(test_accs), 4)) + '+-' + str(np.round(np.std(test_accs), 4))
    b = str(np.round(np.mean(run_time), 4)) + '+-' + str(np.round(np.std(run_time), 4))
    return ['PPNP', a, b]

idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500}
# idx_split_args = {'ntrain_per_class': 2000, 'nstopping': 20000, 'nknown': 120000}

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)

    graph_name = 'amazon_electronics_computers' # 'amazon_electronics_computers'
    cal_main(graph_name)
    graph = load_dataset(graph_name, 'ppnp\\dataset_for_paper')
    print(graph.num_nodes(), graph.num_edges())
    graph.standardize(select_lcc=True)

    result_ = []
    per_seed = 1
    a = ppnp(graph_name, per_seed)
    result_.append(a)
    test = True

    test_seeds = [
        2144199730, 794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407, 635625077, 3538425002, 960893189,
        497096336, 3940842554, 3594628340, 948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]
    val_seeds = [
        2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
        1920426428, 4272044734, 2092442742, 841404887, 2188879532,
        646784207, 1633698412, 2256863076, 374355442, 289680769,
        4281139389, 4263036964, 900418539, 119332950, 1628837138]

    if test:
        seeds = test_seeds
    else:
        seeds = val_seeds

    if graph_name == 'ms_academic':
        nknown = 5000
    else:
        nknown = 1500

    # idx_split_args = {'ntrain_per_class': 10, 'nstopping': 25, 'nknown': 200}

    if graph_name == 'ms_academic':
        alpha = 0.2
    else:
        alpha = 0.1
    alpha = 0.1
    flags = [0.5] # , 0.75, 1.0, 1.25, 1.5
    for ii in range(1, 8):
        motif_type = 'M' + str(ii)
        alpha_value = 0.08
        motif_vector = pickle_load('result/result_' + graph_name + '/' + motif_type +  '.pkl')
        prop_ppnp = MPPRExact(graph.adj_matrix, np.array(motif_vector), alpha=alpha, flag = flags)
        # prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=alpha, niter=10)

        model_args = {
            'hiddenunits': [64],
            'drop_prob': 0.5,
            'propagation': prop_ppnp}

        reg_lambda = 5e-3
        learning_rate = 0.01                    # 学习率

        niter_per_seed = per_seed
        save_result = False
        print_interval = 100
        device = 'cuda'

        results = []
        niter_tot = niter_per_seed * len(seeds)
        i_tot = 0
        test_accs = []
        run_times = []
        for seed in seeds:
            idx_split_args['seed'] = seed
            for _ in range(niter_per_seed):
                i_tot += 1
                logging_string = f"Iteration {i_tot} of {niter_tot}"
                logging.log(22,'-' * len(logging_string) + logging_string + '-' * len(logging_string))
                _, result = train_model(
                    graph_name, PPNP, graph, model_args, learning_rate, reg_lambda,
                    idx_split_args, stopping_args, test, device, None, print_interval)
                results.append({})
                test_accs.append(result['valtest']['accuracy'])
                run_times.append(result['runtime'])
        pickle_save(test_accs, graph_name +'_mppr_'+  motif_type + '.pkl')
        a = str(np.round(np.mean(test_accs), 4)) + '+-' + str(np.round(np.std(test_accs), 4))
        b = str(np.round(np.mean(run_times), 4)) + '+-' + str(np.round(np.std(run_times), 4))
        result_.append([motif_type, a, b])
    headers = ['模体', 'Accuracy', 'time']
    print(tabulate(result_, headers=headers, tablefmt='grid'))