import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from metrics import ranking
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../stats')
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()

    # list all json files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]
    dfs = []
    for file_name in file_names:
        model, dataset, temperature, correctness = file_name.split('_')[:4]
        df = pd.read_json(os.path.join(args.root_dir, file_name))
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.drop('mode', axis=1)
    # compute means and standard deviations across seeds
    means = df.groupby(['model', 'dataset', 'metric', 'temperature']).mean().drop('seed', axis=1)
    stds = df.groupby(['model', 'dataset', 'metric', 'temperature']).std().drop('seed', axis=1)

    # generate a latex table with means and stds as the subscript
    table = pd.DataFrame()
    for col in means.columns:
        table[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    table.index = means.index

    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                            'unnormalized_nll_greedy_erce', 'entropy_unnormalized_erce']
    df = df.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
    df['metric'] = df['model'] + '_' + df['dataset'] + '_' + df['metric'] + '_' + df['temperature'].astype(str) + '_' + df['seed'].astype(str)
    df['score'] = -df['score']
    df = df[df['indicator'].isin(uncertainty_indicators)]
    # change the names of the indicators to [r'$U_{\rm EigV}$', r'$U_{\rm Ecc}$', r'$U_{\rm Deg}$', r'$U_{\rm SE}$', r'$U_{\rm NLL}$']
    df['indicator'] = df['indicator'].replace({'ecc_u_agreement_erce': r'$U_{\rm Ecc}$', 'degree_u_agreement_erce': r'$U_{\rm Deg}$', 'spectral_u_agreement_erce': r'$U_{\rm EigV}$', 'unnormalized_nll_greedy_erce': r'$U_{\rm NLL}$', 'entropy_unnormalized_erce': r'$U_{\rm SE}$'})
    ranking.plot_cd_diagram(df, title=None, save_dir='../stats/plots/overall.png', col1='indicator', col2='metric', col3='score', alpha=args.alpha)
    
    # plot the critical diagram for each dataset
    for dataset in df['dataset'].unique():
        try:
            df_dataset = df[df['dataset'] == dataset]
            ranking.plot_cd_diagram(df_dataset, title=None, save_dir=f'../stats/plots/{dataset}.png', col1='indicator', col2='metric', col3='score', alpha=args.alpha)
        except:
            print('Could not plot critical diagram for', dataset)
    
    for model in df['model'].unique():
        try:
            df_model = df[df['model'] == model]
            ranking.plot_cd_diagram(df_model, title=None, save_dir=f'../stats/plots/{model}.png', col1='indicator', col2='metric', col3='score', alpha=args.alpha)
        except:
            print('Could not plot critical diagram for', model)