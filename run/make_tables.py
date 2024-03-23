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

    try: 
        path = os.path.join("../stats", "gpt_correctness")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 
    # first, plot cd with different metrics
    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                            'unnormalized_nll_all_erce', 'entropy_unnormalized_erce', 'verbalized_erce']
    df_gpt = df[df['model'] == 'gpt-3.5-turbo']
    df_gpt_triviaqa = df_gpt[df_gpt['dataset'] == 'triviaqa']
    df_gpt_triviaqa_10 = df_gpt_triviaqa[df_gpt_triviaqa['temperature'] == 1.0]
    df_tmp = df_gpt_triviaqa_10.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
    df_tmp['des'] = df_tmp['model'] + '_' + df_tmp['dataset'] + '_' + df_tmp['metric'] + '_' + df_tmp['temperature'].astype(str) + '_' + df_tmp['seed'].astype(str)
    scores_list = df_tmp['score'].tolist()
    scores_lists = []
    for item in scores_list:
        if isinstance(item, float):
            scores_lists.append(-item)
        else:
            scores_lists.append(-item[0])
    df_tmp['score'] = scores_lists
    df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
    df_tmp['indicator'] = df_tmp['indicator'].replace(
        {'ecc_u_agreement_erce': r'$U_{\rm Ecc}$', 
         'degree_u_agreement_erce': r'$U_{\rm Deg}$', 
         'spectral_u_agreement_erce': r'$U_{\rm EigV}$', 
         'unnormalized_nll_all_erce': r'$U_{\rm NLL}$', 
         'entropy_unnormalized_erce': r'$U_{\rm SE}$',
         'verbalized_erce': r'$C_{\rm Verb}$)'})
    # plot CD diagram
    for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
        df_tmp_metric = df_tmp[df_tmp['metric'] == metric]
        ranking.plot_cd_diagram(df_tmp_metric, title=None, save_dir=f'{path}/gpt-3.5-turbo_{metric}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
    
    # second: plot cd with different temperatures
    try: 
        path = os.path.join("../stats", "gpt_temperature")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 
    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                        'unnormalized_nll_all_erce', 'entropy_unnormalized_erce']
    df_gpt_rouge = df_gpt[df_gpt['metric'] == 'rouge']
    df_gpt_rouge_triviaqa = df_gpt_rouge[df_gpt_rouge['dataset'] == 'triviaqa']
    df_tmp = df_gpt_rouge_triviaqa.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
    df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
    for temperature in [0.5, 1.0, 1.5]:
        for indicator in uncertainty_indicators:
            df_tmp.loc[(df_tmp['temperature'] == temperature) & (df_tmp['indicator'] == indicator), 'seed'] = list(range(20))
    df_tmp['des'] = df_tmp['model'] + '_' + df_tmp['dataset'] + '_' + df_tmp['metric'] + '_' + df_tmp['temperature'].astype(str) + '_' + df_tmp['seed'].astype(str)
    scores_list = df_tmp['score'].tolist()
    scores_lists = []
    for item in scores_list:
        if isinstance(item, float):
            scores_lists.append(-item)
        else:
            scores_lists.append(-item[0])
    df_tmp['score'] = scores_lists
    df_tmp['indicator'] = df_tmp['indicator'].replace(
        {'ecc_u_agreement_erce': r'$U_{\rm Ecc}$', 
         'degree_u_agreement_erce': r'$U_{\rm Deg}$', 
         'spectral_u_agreement_erce': r'$U_{\rm EigV}$', 
         'unnormalized_nll_all_erce': r'$U_{\rm NLL}$', 
         'entropy_unnormalized_erce': r'$U_{\rm SE}$'})
    # plot CD diagram
    for temperature in [0.5, 1.0, 1.5]:
        df_tmp_temperature = df_tmp[df_tmp['temperature'] == temperature]
        ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/gpt-3.5-turbo_{temperature}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)

    try: 
        path = os.path.join("../stats", "llama_temperature")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 
    # third: plot cd with llama-2-chat, triviaqa, rouge
    uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                        'unnormalized_nll_all_erce', 'entropy_unnormalized_erce']
    df_llama = df[df['model'] == 'Llama-2-7b-chat-hf']
    df_llama_triviaqa = df_llama[df_llama['dataset'] == 'triviaqa']
    df_llama_triviaqa_rouge = df_llama_triviaqa[df_llama_triviaqa['metric'] == 'rouge']
    df_tmp = df_llama_triviaqa_rouge.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
    df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
    df_tmp['des'] = df_tmp['model'] + '_' + df_tmp['dataset'] + '_' + df_tmp['metric'] + '_' + df_tmp['temperature'].astype(str) + '_' + df_tmp['seed'].astype(str)
    scores_list = df_tmp['score'].tolist()
    scores_lists = []
    for item in scores_list:
        if isinstance(item, float):
            scores_lists.append(-item)
        else:
            scores_lists.append(-item[0])
    df_tmp['score'] = scores_lists
    df_tmp['indicator'] = df_tmp['indicator'].replace(
        {'ecc_u_agreement_erce': r'$U_{\rm Ecc}$', 
         'degree_u_agreement_erce': r'$U_{\rm Deg}$', 
         'spectral_u_agreement_erce': r'$U_{\rm EigV}$', 
         'unnormalized_nll_all_erce': r'$U_{\rm NLL}$', 
         'entropy_unnormalized_erce': r'$U_{\rm SE}$'})
    for temperature in [0.6, 1.0]:
        if temperature == 0.6:
            df_tmp_temperature = df_tmp[df_tmp['temperature'] < 1.0]
        else:
            df_tmp_temperature = df_tmp[df_tmp['temperature'] == 1.0]
        ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/llama-2-chat_{temperature}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
    
    print('Plot finished')