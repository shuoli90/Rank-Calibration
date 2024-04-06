import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import pandas as pd
from metrics import ranking
import seaborn as sns
import matplotlib.pyplot as plt
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../submission/evaluation_stats')
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
    # generate a latex table with means and stds as the subscript using the index of means
    for col in means.columns:
        means[col] = [f'{means[col][mode]:.3f}$_{{\pm {stds[col][mode]:.3f}}}$' for mode in means.index]
    # bold the lowest mean in each row
    means = means.style.apply(lambda x: ['font-weight: bold' if v == min(x) else '' for v in x], axis=1)
    print(means.to_latex())

    plt.rcParams.update({'font.size': 40})
    for dataset in ['triviaqa', 'squad', 'nq-open', 'meadow']:
        try: 
            path = os.path.join(f"{args.root_dir}", f"gpt_correctness_{dataset}")
            os.makedirs(path, exist_ok = True) 
        except OSError as error: 
            print("Directory '%s' can not be created" % path) 
        # first, plot cd with different metrics
        uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                                'unnormalized_nll_all_erce', 'entropy_unnormalized_erce', 'verbalized_erce']
        df_gpt = df[df['model'] == 'gpt-3.5-turbo']
        df_gpt_dataset = df_gpt[df_gpt['dataset'] == dataset]
        df_gpt_dataset_10 = df_gpt_dataset[df_gpt_dataset['temperature'] == 1.0]
        df_tmp = df_gpt_dataset_10.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
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
            'verbalized_erce': r'$C_{\rm Verb}$'})
        # plot CD diagram
        for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
            df_tmp_metric = df_tmp[df_tmp['metric'] == metric]
            try:
                ranking.plot_cd_diagram(df_tmp_metric, title=None, save_dir=f'{path}/gpt-3.5-turbo_{metric}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')
        
        try:
            ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/gpt-3.5-turbo_metric.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
        except:
            print('No significant difference')

        # Customization
        sns.set_theme(style='whitegrid')
        df_tmp['score'] = -df_tmp['score']
        df_tmp['metric'] = df_tmp['metric'].replace(
                {'bert_similarity': 'BERT',
                 'rouge': 'Rouge-L',
                 'rouge1': 'Rouge-1',
                 'meteor': 'METEOR'})
        plt.figure(figsize=(14, 5))
        # Create and display the plot
        sns.boxplot(x="metric",
            y="score",
            hue="indicator",
            data=df_tmp,
            width=0.6,
            linewidth=0.6,
            showmeans=False,
            fliersize=1,
            )  

        # # Add labels to the axes
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Correctness Score", fontsize=20)
        plt.ylabel("RCE", fontsize=20)
        # place legend outside to the right
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/gpt-3.5-turbo_box.pdf')

        # first, plot cd with different metrics
        uncertainty_indicators = ['ecc_u_agreement_auroc', 'degree_u_agreement_auroc', 'spectral_u_agreement_auroc',
                                'unnormalized_nll_all_auroc', 'entropy_unnormalized_auroc', 'verbalized_auroc']
        df_gpt = df[df['model'] == 'gpt-3.5-turbo']
        df_gpt_dataset = df_gpt[df_gpt['dataset'] == dataset]
        df_gpt_dataset_10 = df_gpt_dataset[df_gpt_dataset['temperature'] == 1.0]
        df_tmp = df_gpt_dataset_10.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
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
            {'ecc_u_agreement_auroc': r'$U_{\rm Ecc}$', 
            'degree_u_agreement_auroc': r'$U_{\rm Deg}$', 
            'spectral_u_agreement_auroc': r'$U_{\rm EigV}$', 
            'unnormalized_nll_all_auroc': r'$U_{\rm NLL}$', 
            'entropy_unnormalized_auroc': r'$U_{\rm SE}$',
            'verbalized_erce': r'$C_{\rm Verb}$'})
        # plot CD diagram
        for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
            df_tmp_metric = df_tmp[df_tmp['metric'] == metric]
            try:
                ranking.plot_cd_diagram(df_tmp_metric, title=None, save_dir=f'{path}/gpt-3.5-turbo_{metric}_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')
        
        try:
            ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/gpt-3.5-turbo_metric_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
        except:
            print('No significant difference')

        # Customization
        sns.set_theme(style='whitegrid')
        df_tmp['score'] = -df_tmp['score']
        df_tmp['metric'] = df_tmp['metric'].replace(
                {'bert_similarity': 'BERT',
                 'rouge': 'Rouge-L',
                 'rouge1': 'Rouge-1',
                 'meteor': 'METEOR'})
        plt.figure(figsize=(14, 5))
        # Create and display the plot
        sns.boxplot(x="metric",
            y="score",
            hue="indicator",
            data=df_tmp,
            width=0.6,
            linewidth=0.6,
            showmeans=False,
            fliersize=1,
            )  

        # # Add labels to the axes
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Correctness Score", fontsize=20)
        plt.ylabel("RCE", fontsize=20)
        # place legend outside to the right
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/gpt-3.5-turbo_box_auroc.pdf')

    # second: plot cd with different temperatures
    for dataset in ['triviaqa']:
        for metric in ['rouge', 'rouge1', 'meteor', 'bert_similarity']:
            try: 
                path = os.path.join("{args.root_dir}", f"gpt_temperature_{dataset}_{metric}")
                os.makedirs(path, exist_ok = True) 
            except OSError as error: 
                print("Directory '%s' can not be created" % path) 
            uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                                'unnormalized_nll_all_erce', 'entropy_unnormalized_erce']
            df_gpt_metric = df_gpt[df_gpt['metric'] == metric]
            df_gpt_metric_dataset = df_gpt_metric[df_gpt_metric['dataset'] == dataset]
            df_tmp = df_gpt_metric_dataset.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score').copy()
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
                try:
                    ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/gpt-3.5-turbo_{temperature}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            try:
                ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/gpt-3.5-turbo_temperature.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')

                # Customization
            sns.set_theme(style='whitegrid')
            df_tmp['score'] = -df_tmp['score']
            plt.figure(figsize=(14, 5))
            # Create and display the plot
            sns.boxplot(x="temperature",
                y="score",
                hue="indicator",
                data=df_tmp,
                width=0.6,
                linewidth=0.6,
                showmeans=False,
                fliersize=1,
                )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/gpt-3.5-turbo_{metric}_box.pdf')

            uncertainty_indicators = ['ecc_u_agreement_auroc', 'degree_u_agreement_auroc', 'spectral_u_agreement_auroc',
                                'unnormalized_nll_all_auroc', 'entropy_unnormalized_auroc']
            df_gpt_metric = df_gpt[df_gpt['metric'] == metric]
            df_gpt_metric_dataset = df_gpt_metric[df_gpt_metric['dataset'] == dataset]
            df_tmp = df_gpt_metric_dataset.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score').copy()
            df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
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
                {'ecc_u_agreement_auroc': r'$U_{\rm Ecc}$', 
                'degree_u_agreement_auroc': r'$U_{\rm Deg}$', 
                'spectral_u_agreement_auroc': r'$U_{\rm EigV}$', 
                'unnormalized_nll_all_auroc': r'$U_{\rm NLL}$', 
                'entropy_unnormalized_auroc': r'$U_{\rm SE}$'})
            # plot CD diagram
            for temperature in [0.5, 1.0, 1.5]:
                df_tmp_temperature = df_tmp[df_tmp['temperature'] == temperature]
                try:
                    ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/gpt-3.5-turbo_{temperature}_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            try:
                ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/gpt-3.5-turbo_temperature_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')

                # Customization
            sns.set_theme(style='whitegrid')
            df_tmp['score'] = -df_tmp['score']
            plt.figure(figsize=(14, 5))
            # Create and display the plot
            sns.boxplot(x="temperature",
                y="score",
                hue="indicator",
                data=df_tmp,
                width=0.6,
                linewidth=0.6,
                showmeans=False,
                fliersize=1,
                )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/gpt-3.5-turbo_{metric}_box_auroc.pdf')
            
    # third, plot cd with opensource model
    for dataset in ['triviaqa', 'squad', 'nq-open']:
        for metric in ['rouge', 'rouge1', 'meteor', 'bert_similarity']:
            try: 
                path = os.path.join("{args.root_dir}", f"llama_temperature_{dataset}_{metric}")
                os.makedirs(path, exist_ok = True) 
            except OSError as error: 
                print("Directory '%s' can not be created" % path) 
            # third: plot cd with llama-2-chat, triviaqa, rouge
            uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                                'unnormalized_nll_all_erce', 'entropy_unnormalized_erce']
            df_llama = df[df['model'] == 'Llama-2-7b-chat-hf']
            df_llama_dataset = df_llama[df_llama['dataset'] == dataset]
            df_llama_dataset_metric = df_llama_dataset[df_llama_dataset['metric'] == metric]
            df_tmp = df_llama_dataset_metric.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
            df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
            for temperature in [0.6, 1.0]:
                for indicator in uncertainty_indicators:
                    if temperature == 0.6:
                        df_tmp.loc[(df_tmp['temperature'] < 1.0) & (df_tmp['indicator'] == indicator), 'seed'] = list(range(20))
                    else:
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
            for temperature in [0.6, 1.0]:
                if temperature == 0.6:
                    df_tmp_temperature = df_tmp[df_tmp['temperature'] < 1.0]
                else:
                    df_tmp_temperature = df_tmp[df_tmp['temperature'] == 1.0]
                try:
                    ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/llama-2-chat_{temperature}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            # set temperature precision to 1 decimal
            df_tmp['score'] = -df_tmp['score']
            df_tmp['temperature'] = df_tmp['temperature'].apply(lambda x: f'{x:.1f}')
            sns.set_theme(style='whitegrid')
            plt.figure(figsize=(14, 5))
            # Create and display the plot
            sns.boxplot(x="temperature",
                        y="score",
                        hue="indicator",
                        data=df_tmp,
                        width=0.6,
                        linewidth=0.6,
                        showmeans=False,
                        fliersize=1,
                        )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/llama_{metric}_box.pdf')
            
            try:
                ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/llama_temperature.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')

            uncertainty_indicators = ['ecc_u_agreement_auroc', 'degree_u_agreement_auroc', 'spectral_u_agreement_auroc',
                                'unnormalized_nll_all_auroc', 'entropy_unnormalized_auroc']
            df_llama = df[df['model'] == 'Llama-2-7b-chat-hf']
            df_llama_dataset = df_llama[df_llama['dataset'] == dataset]
            df_llama_dataset_metric = df_llama_dataset[df_llama_dataset['metric'] == metric]
            df_tmp = df_llama_dataset_metric.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
            df_tmp = df_tmp[df_tmp['indicator'].isin(uncertainty_indicators)]
            for temperature in [0.6, 1.0]:
                for indicator in uncertainty_indicators:
                    if temperature == 0.6:
                        df_tmp.loc[(df_tmp['temperature'] < 1.0) & (df_tmp['indicator'] == indicator), 'seed'] = list(range(20))
                    else:
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
                {'ecc_u_agreement_auroc': r'$U_{\rm Ecc}$', 
                'degree_u_agreement_auroc': r'$U_{\rm Deg}$', 
                'spectral_u_agreement_auroc': r'$U_{\rm EigV}$', 
                'unnormalized_nll_all_auroc': r'$U_{\rm NLL}$', 
                'entropy_unnormalized_auroc': r'$U_{\rm SE}$'})
            for temperature in [0.6, 1.0]:
                if temperature == 0.6:
                    df_tmp_temperature = df_tmp[df_tmp['temperature'] < 1.0]
                else:
                    df_tmp_temperature = df_tmp[df_tmp['temperature'] == 1.0]
                try:
                    ranking.plot_cd_diagram(df_tmp_temperature, title=None, save_dir=f'{path}/llama-2-chat_{temperature}_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            # set temperature precision to 1 decimal
            df_tmp['score'] = -df_tmp['score']
            df_tmp['temperature'] = df_tmp['temperature'].apply(lambda x: f'{x:.1f}')
            sns.set_theme(style='whitegrid')
            plt.figure(figsize=(14, 5))
            # Create and display the plot
            sns.boxplot(x="temperature",
                        y="score",
                        hue="indicator",
                        data=df_tmp,
                        width=0.6,
                        linewidth=0.6,
                        showmeans=False,
                        fliersize=1,
                        )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/llama_{metric}_box_auroc.pdf')

    for temperature in [0.6, 1.0]:
        for dataset in ['triviaqa', 'squad', 'nq-open']:
            try: 
                path = os.path.join("{args.root_dir}", f"llama_correctness_{temperature}_{dataset}")
                os.makedirs(path, exist_ok = True) 
            except OSError as error: 
                print("Directory '%s' can not be created" % path) 
            # first, plot cd with different metrics
            uncertainty_indicators = ['ecc_u_agreement_erce', 'degree_u_agreement_erce', 'spectral_u_agreement_erce',
                                    'unnormalized_nll_all_erce', 'entropy_unnormalized_erce']
            df_gpt = df[df['model'] == 'Llama-2-7b-chat-hf']
            df_gpt_dataset = df_gpt[df_gpt['dataset'] == dataset]
            if temperature == 0.6:
                df_gpt_dataset_temperature = df_gpt_dataset[df_gpt_dataset['temperature'] < 1.0]
            else:
                df_gpt_dataset_temperature = df_gpt_dataset[df_gpt_dataset['temperature'] == 1.0]
            df_tmp = df_gpt_dataset_temperature.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
            for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
                for indicator in uncertainty_indicators:
                    df_tmp.loc[(df_tmp['indicator'] == indicator) & (df_tmp['metric'] == metric), 'seed'] = list(range(20))
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
                'entropy_unnormalized_erce': r'$U_{\rm SE}$'})
            # plot CD diagram
            for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
                df_tmp_metric = df_tmp[df_tmp['metric'] == metric]
                try:
                    ranking.plot_cd_diagram(df_tmp_metric, title=None, save_dir=f'{path}/llama_{metric}.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            try:
                ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/llama_metric.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')
            
            df_tmp['metric'] = df_tmp['metric'].replace(
                {'bert_similarity': 'BERT',
                 'rouge': 'Rouge-L',
                 'rouge1': 'Rouge-1',
                 'meteor': 'METEOR'})
            # set temperature precision to 1 decimal
            df_tmp['temperature'] = df_tmp['temperature'].apply(lambda x: f'{x:.1f}')
            df_tmp['score'] = -df_tmp['score']
            plt.figure(figsize=(14, 5))
            sns.set_theme(style='whitegrid')
            # Create and display the plot
            sns.boxplot(x="metric",
                        y="score",
                        hue="indicator",
                        data=df_tmp,
                        width=0.6,
                        linewidth=0.6,
                        showmeans=False,
                        fliersize=1,
                        )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/llama_box.pdf')

            # first, plot cd with different metrics
            uncertainty_indicators = ['ecc_u_agreement_auroc', 'degree_u_agreement_auroc', 'spectral_u_agreement_auroc',
                                    'unnormalized_nll_all_auroc', 'entropy_unnormalized_auroc']
            df_gpt = df[df['model'] == 'Llama-2-7b-chat-hf']
            df_gpt_dataset = df_gpt[df_gpt['dataset'] == dataset]
            if temperature == 0.6:
                df_gpt_dataset_temperature = df_gpt_dataset[df_gpt_dataset['temperature'] < 1.0]
            else:
                df_gpt_dataset_temperature = df_gpt_dataset[df_gpt_dataset['temperature'] == 1.0]
            df_tmp = df_gpt_dataset_temperature.melt(id_vars=['model', 'dataset', 'metric', 'temperature', 'seed'], var_name='indicator', value_name='score')
            for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
                for indicator in uncertainty_indicators:
                    df_tmp.loc[(df_tmp['indicator'] == indicator) & (df_tmp['metric'] == metric), 'seed'] = list(range(20))
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
                {'ecc_u_agreement_auroc': r'$U_{\rm Ecc}$', 
                'degree_u_agreement_auroc': r'$U_{\rm Deg}$', 
                'spectral_u_agreement_auroc': r'$U_{\rm EigV}$', 
                'unnormalized_nll_all_auroc': r'$U_{\rm NLL}$', 
                'entropy_unnormalized_auroc': r'$U_{\rm SE}$'})
            # plot CD diagram
            for metric in ['rouge', 'bert_similarity', 'meteor', 'rouge1']:
                df_tmp_metric = df_tmp[df_tmp['metric'] == metric]
                try:
                    ranking.plot_cd_diagram(df_tmp_metric, title=None, save_dir=f'{path}/llama_{metric}_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
                except:
                    print('No significant difference')
            
            try:
                ranking.plot_cd_diagram(df_tmp, title=None, save_dir=f'{path}/llama_metric_auroc.pdf', col1='indicator', col2='des', col3='score', alpha=args.alpha)
            except:
                print('No significant difference')
            
            df_tmp['metric'] = df_tmp['metric'].replace(
                {'bert_similarity': 'BERT',
                 'rouge': 'Rouge-L',
                 'rouge1': 'Rouge-1',
                 'meteor': 'METEOR'})
            # set temperature precision to 1 decimal
            df_tmp['temperature'] = df_tmp['temperature'].apply(lambda x: f'{x:.1f}')
            df_tmp['score'] = -df_tmp['score']
            plt.figure(figsize=(14, 5))
            sns.set_theme(style='whitegrid')
            # Create and display the plot
            sns.boxplot(x="metric",
                        y="score",
                        hue="indicator",
                        data=df_tmp,
                        width=0.6,
                        linewidth=0.6,
                        showmeans=False,
                        fliersize=1,
                        )  

            # # Add labels to the axes
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Correctness Score", fontsize=20)
            plt.ylabel("RCE", fontsize=20)
            # place legend outside to the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/llama_box_auroc.pdf')
    print('Plot finished')