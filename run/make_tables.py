import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import pandas as pd
from metrics import ranking
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../stats')
    parser.add_argument('--alpha', type=float, default=0.05)
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

    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                            'unnormalized_nll_all_erce', 'entropy_unnormalized_erce', 'verbalized_erce']

    # first table: results with original setup
    df_tmp = df[df['model'].isin(['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'gpt-3.5-turbo'])]
    # rename "bert_similarity" to "bert"
    df_tmp['metric'] = df_tmp['metric'].replace({'bert_similarity': 'bert'})
    # rename "Llama-2-7b-chat-hf" to "Llama-2-chat"; and "Llama-2-7b-hf" to "Llama-2"
    df_tmp['model'] = df_tmp['model'].replace({'Llama-2-7b-chat-hf': 'Llama-2-chat', 'Llama-2-7b-hf': 'Llama-2', 'gpt-3.5-turbo': 'gpt-3.5'})
    # set temperature precision to 1 decimal
    df_tmp['temperature'] = df_tmp['temperature'].apply(lambda x: f'{x:.1f}')
    df_tmp = df_tmp[['model', 'dataset', 'metric', 'temperature', 'seed'] + uncertainty_indicators]
    # compute means and standard deviations across seeds
    means = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).mean().drop('seed', axis=1)
    stds = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).std().drop('seed', axis=1)
    means.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$', r'$C_{\rm Verbalized}$']
    stds.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$', r'$C_{\rm Verbalized}$']
    # generate a latex table with means and stds as the subscript
    table = pd.DataFrame()
    for col in means.columns:
        table[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    table.index = means.index
    # table = table.droplevel([2,3])
    # bold lowest mean in each row
    # table = table.style.apply(lambda x: ["font-weight: bold" if v == min(x) else "" for v in x], axis=1)
    print(table.to_latex())
    breakpoint()

    # first table: results with original setup
    df_tmp = df[df['model'].isin(['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'gpt-3.5-turbo'])]
    df_tmp = df_tmp[df_tmp['metric'] == 'rouge']
    # filter out rows whose model is llama-2 and temperature is 1.0
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-chat-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['dataset'] == 'meadow'))]
    # compute means and standard deviations across seeds
    means = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).mean().drop('seed', axis=1)[uncertainty_indicators]
    stds = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).std().drop('seed', axis=1)[uncertainty_indicators]
    means.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    stds.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    # generate a latex table with means and stds as the subscript
    table = pd.DataFrame()
    for col in means.columns:
        table[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    table.index = means.index
    table = table.droplevel([2,3])
    print(table.to_latex())

    # first table: results with original setup
    df_tmp = df[df['model'].isin(['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'gpt-3.5-turbo'])]
    df_tmp = df_tmp[df_tmp['metric'] == 'bert_similarity']
    # filter out rows whose model is llama-2 and temperature is 1.0
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-chat-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['dataset'] == 'meadow'))]
    # compute means and standard deviations across seeds
    means = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).mean().drop('seed', axis=1)[uncertainty_indicators]
    stds = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).std().drop('seed', axis=1)[uncertainty_indicators]
    means.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    stds.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    # generate a latex table with means and stds as the subscript
    table = pd.DataFrame()
    for col in means.columns:
        table[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    table.index = means.index
    table = table.droplevel([2,3])
    print(table.to_latex())

    # first table: results with original setup
    df_tmp = df[df['model'].isin(['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'gpt-3.5-turbo'])]
    df_tmp = df_tmp[df_tmp['metric'] == 'bert_similarity']
    # filter out rows whose model is llama-2 and temperature is 1.0
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-chat-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['model'] == 'Llama-2-7b-hf') & (df_tmp['temperature'] == 1.0))]
    df_tmp = df_tmp[~((df_tmp['dataset'] != 'meadow'))]
    # compute means and standard deviations across seeds
    means = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).mean().drop('seed', axis=1)[uncertainty_indicators]
    stds = df_tmp.groupby(['model', 'dataset', 'metric', 'temperature']).std().drop('seed', axis=1)[uncertainty_indicators]
    means.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    stds.columns = [r'$U_{\rm Ecc}$',  r'$U_{\rm Deg}$', r'$U_{\rm EigV}$', r'$U_{\rm NLL}$', r'$U_{\rm SE}$']
    # generate a latex table with means and stds as the subscript
    table = pd.DataFrame()
    for col in means.columns:
        table[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    table.index = means.index
    table = table.droplevel([2,3])
    print(table.to_latex())

    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                            'unnormalized_nll_greedy_erce', 'entropy_unnormalized_erce']
    df = df.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
    df['des'] = df['model'] + '_' + df['dataset'] + '_' + df['metric'] + '_' + df['temperature'].astype(str) + '_' + df['seed'].astype(str)
    df['score'] = -df['score']
    df = df[df['indicator'].isin(uncertainty_indicators)]
    # change the names of the indicators to [r'$U_{\rm EigV}$', r'$U_{\rm Ecc}$', r'$U_{\rm Deg}$', r'$U_{\rm SE}$', r'$U_{\rm NLL}$']
    df['indicator'] = df['indicator'].replace({'ecc_u_agreement_erce': r'$U_{\rm Ecc}$', 'degree_u_agreement_erce': r'$U_{\rm Deg}$', 'spectral_u_agreement_erce': r'$U_{\rm EigV}$', 'unnormalized_nll_greedy_erce': r'$U_{\rm NLL}$', 'entropy_unnormalized_erce': r'$U_{\rm SE}$'})
    ranking.plot_cd_diagram(df, title=None, save_dir='../stats/plots/overall.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
    
    df_triviaqa = df[df['dataset'] == 'triviaqa']
    df_triviaqa_llama2 = df_triviaqa[df_triviaqa['model'] == 'Llama-2-7b-chat-hf']
    df_triviaqa_llama2_06 = df_triviaqa_llama2[df_triviaqa_llama2['temperature'] < 1.0]
    df_triviaqa_llama2_06_rouge = df_triviaqa_llama2_06[df_triviaqa_llama2_06['metric'] == 'rouge']
    # group by indicator and compute the mean and std
    means = df_triviaqa_llama2_06_rouge[['indicator', 'score']].groupby('indicator').mean()
    ranking.plot_cd_diagram(df_triviaqa_llama2_06_rouge, title=None, save_dir='../stats/plots/triviaqa_llama2_rouge.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)

    df_triviaqa = df[df['dataset'] == 'triviaqa']
    df_triviaqa_gpt = df_triviaqa[df_triviaqa['model'] == 'gpt-3.5-turbo']
    df_triviaqa_gpt_10 = df_triviaqa_llama2[df_triviaqa_llama2['temperature'] == 1.0]
    df_triviaqa_gpt_10_rouge = df_triviaqa_llama2_06[df_triviaqa_llama2_06['metric'] == 'rouge']
    # group by indicator and compute the mean and std
    means = df_triviaqa_llama2_06_rouge[['indicator', 'score']].groupby('indicator').mean()
    ranking.plot_cd_diagram(df_triviaqa_llama2_06_rouge, title=None, save_dir='../stats/plots/triviaqa_gpt_rouge.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
    
    # plot the critical diagram for each dataset
    for dataset in df['dataset'].unique():
        try:
            df_dataset = df[df['dataset'] == dataset]
            ranking.plot_cd_diagram(df_dataset, title=None, save_dir=f'../stats/plots/{dataset}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
        except:
            print('Could not plot critical diagram for', dataset)
    
    for model in df['model'].unique():
        try:
            df_model = df[df['model'] == model]
            ranking.plot_cd_diagram(df_model, title=None, save_dir=f'../stats/plots/{model}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
        except:
            print('Could not plot critical diagram for', model)
