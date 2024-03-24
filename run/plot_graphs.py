import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import json
from utils import make_plots
from metrics import correctness
import matplotlib.patches as patches
epsilon = 1e-3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../tmp')
    parser.add_argument('--correctness', type=str, default='rouge')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--mode', type=str, default='rougeL')
    args = parser.parse_args()

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]
    model = args.model.split('/')[-1]
    collected_file = '_'.join([model, args.dataset, str(args.temperature)]) + '.json'
    collected = json.load(open(os.path.join('../collected', collected_file)))

    # compute the correctness score
    if os.path.exists(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'):
        scores = json.load(open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json'))
    else:
        if args.correctness == 'rouge1':
            args.correctness = 'rouge'
            args.mode = 'rouge1'

         # setup generation correctness score
        if args.correctness in ['rouge', 'bleu', 'meteor']:
            SCORE = correctness.Score(metric_name=args.correctness, mode=args.mode)
        elif args.correctness in ['bert_similarity']:
            SCORE = correctness.BertSimilarity()
        elif args.correctness in ['chatgpt']:
            SCORE = correctness.ChatgptCorrectness()
        else:
            raise ValueError(f"Correctness metric {args.correctness} not found")
    
        from multiprocessing import Manager
        if args.correctness in ['rouge', 'bleu', 'meteor']:
            def compute_scores(col, base, scores):
                for idx, collected_row in tqdm(enumerate(col), total=len(col)):
                    score_tmp = {}
                    # prompt = collected_row['prompt'].split('\n')[-2].strip()
                    reference = collected_row['references']
                    generations = collected_row['generated']
                    if args.correctness in ['rouge', 'bleu', 'meteor']:
                        s_unnormalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                        s_normalized = [SCORE(references=[reference], predictions=[generation]) for generation in generations]
                    score_tmp['id'] = idx + base
                    score_tmp['normalized_score'] = s_normalized
                    score_tmp['unnormalized_score'] = s_unnormalized
                    scores.append(score_tmp)
            manager = Manager()
            scores = manager.list()
            processes = []
            def split(a, n):
                k, m = divmod(len(a), n)
                return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
            # collecteds = split(collected, 10)
            collecteds = split(scores, 10)
            start = 0
            for col in collecteds:
                p = multiprocessing.Process(target=compute_scores, args=(col, start, scores))
                start += len(col)
                processes.append(p)
                p.start()
            for process in processes:
                process.join()
            scores = [score for score in scores]
        elif args.correctness in ['bert_similarity']:
            scores = []
            for idx, collected_row in tqdm(enumerate(collected), total=len(collected)):
            # for idx, collected_row in tqdm(enumerate(scores), total=len(scores)):
                score_tmp = {}
                # question = collected_row['prompt'].split('\n')[-2].strip()
                reference = collected_row['references'] if isinstance(collected_row['references'], list) else [collected_row['references']]
                generations = collected_row['generated']
                scores_tmp = SCORE(prompt="", references=reference, predictions=generations).tolist()
                score_tmp['id'] = idx
                score_tmp['normalized_score'] = scores_tmp
                score_tmp['unnormalized_score'] = scores_tmp
                scores.append(score_tmp)

                if idx % 10 == 0:
                    with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
                        json.dump(scores, f)
        elif args.correctness in ['chatgpt']:
            scores = []
            for idx, collected_row in tqdm(enumerate(collected), total=len(collected)):
            # for idx, collected_row in tqdm(enumerate(scores), total=len(scores)):
                score_tmp = {}
                question = collected_row['prompt'].split('\n')[-2].strip()
                reference = collected_row['references'][0]
                generations = collected_row['generated']
                scores_tmp = SCORE(prompt=question, reference=reference, generateds=generations)
                score_tmp['id'] = idx
                score_tmp['normalized_score'] = scores_tmp
                score_tmp['unnormalized_score'] = scores_tmp
                scores.append(score_tmp)

                if idx % 10 == 0:
                    with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
                        json.dump(scores, f)
        if args.correctness == 'rouge':
            with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.mode}.json', 'w') as f:
                json.dump(scores, f)
        else:
            with open(f'../tmp/{model}_{args.dataset}_{args.temperature}_{args.correctness}.json', 'w') as f:
                json.dump(scores, f)
        exit(0)
    scores = pd.DataFrame(scores).dropna(axis=0)
    
    model = args.model.split('/')[-1]
    dataset = args.dataset
    file_names = []
    for method in ['whitebox', 'blackbox', 'verbalized']:
        if method == 'whitebox':
            affinity_mode = 'none'
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'whitebox']) + '.json'
            file_names.append(file_name)
        elif method == 'verbalized':
            try:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), 'disagreement', 'verbalized']) + '.json'
                file_names.append(file_name)
            except:
                continue
        else:
            for affinity_mode in ['disagreement', 'agreement']:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'blackbox']) + '.json'
                file_names.append(file_name)
    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(open(os.path.join(args.root_dir, file_names[1])))
    data_blackbox_agreement = json.load(open(os.path.join(args.root_dir, file_names[2])))
    if model == 'gpt-3.5-turbo':
        data_verbalized = json.load(open(os.path.join(args.root_dir, file_names[3])))
        results = []
        for row_verbalized in data_verbalized:
            result = {'model':model, 'dataset':dataset, 'metric':args.correctness}
            idx = row_verbalized['idx']
            row_whitebox = data_whitebox[idx]
            row_blackbox_disagreement = data_blackbox_disagreement[idx]
            row_blackbox_agreement = data_blackbox_agreement[idx]
            
            # result['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
            # result['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
            # result['ecc_u_disagreement'] = row_blackbox_disagreement['ecc_u']
            # result['degree_u_disagreement'] = row_blackbox_disagreement['degree_u']
            # result['spectral_u_disagreement'] = row_blackbox_disagreement['spectral_u']

            result[r'$U_{\rm EigV}$'] = [row_blackbox_agreement['spectral_u']] * 10
            result[r'$U_{\rm Ecc}$'] = [row_blackbox_agreement['ecc_u']] * 10
            result[r'$U_{\rm Deg}$'] = [row_blackbox_agreement['degree_u']] * 10

            # result['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
            # result['degree_c_agreement'] = row_blackbox_agreement['degree_c']
            # result['ecc_u_agreement'] = row_blackbox_agreement['ecc_u']
            # result['degree_u_agreement'] = row_blackbox_agreement['degree_u']
            # result['spectral_u_agreement'] = row_blackbox_agreement['spectral_u']

            # result['entropy_normalized'] = row_whitebox['entropy_normalized']
            # result['entropy_unnormalized'] = row_whitebox['entropy_unnormalized']
            result['normalized_nll_all'] = row_whitebox['normalized_nll']
            result['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']
            # result['normalized_nll_greedy'] = np.min(row_whitebox['normalized_nll'])
            # result['unnormalized_nll_greedy'] = np.min(row_whitebox['unnormalized_nll'])
            result[r'$U_{\rm SE}$'] = [row_whitebox['entropy_unnormalized']] * 10
            unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
            result[r'$U_{\rm NLL}$'] =result['unnormalized_nll_all']
            result[r'$C_{\rm Verb}$'] = row_verbalized['verbalized']

            # select scores with the same index
            score = scores[scores['id'] == idx]
            result['normalized_score_all'] = score.iloc[0]['normalized_score']
            result['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
            normalized_min_index = np.argmin(result['normalized_nll_all'])
            unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
            result['normalized_score_greedy'] = result['normalized_score_all'][normalized_min_index]
            result['unnormalized_score_greedy'] = result['unnormalized_score_all'][unnormalized_min_index]
            results.append(result)
        df = pd.DataFrame(results).dropna(axis=0)
    else:
        results = []
        for idx, (row_whitebox, row_blackbox_disagreement, row_blackbox_agreement) in tqdm(enumerate(zip(data_whitebox, data_blackbox_disagreement, data_blackbox_agreement)), total=len(data_whitebox)):
            result = {'model':model, 'dataset':dataset, 'metric':args.correctness}
            
            # result['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
            # result['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
            # result['ecc_u_disagreement'] = row_blackbox_disagreement['ecc_u']
            # result['degree_u_disagreement'] = row_blackbox_disagreement['degree_u']
            # result['spectral_u_disagreement'] = row_blackbox_disagreement['spectral_u']

            result[r'$U_{\rm EigV}$'] = [row_blackbox_agreement['spectral_u']] * 10
            result[r'$U_{\rm Ecc}$'] = [row_blackbox_agreement['ecc_u']] * 10
            result[r'$U_{\rm Deg}$'] = [row_blackbox_agreement['degree_u']] * 10

            # result['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
            # result['degree_c_agreement'] = row_blackbox_agreement['degree_c']
            # result['ecc_u_agreement'] = row_blackbox_agreement['ecc_u']
            # result['degree_u_agreement'] = row_blackbox_agreement['degree_u']
            # result['spectral_u_agreement'] = row_blackbox_agreement['spectral_u']

            # result['entropy_normalized'] = row_whitebox['entropy_normalized']
            # result['entropy_unnormalized'] = row_whitebox['entropy_unnormalized']
            result['normalized_nll_all'] = row_whitebox['normalized_nll']
            result['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']
            # result['normalized_nll_greedy'] = np.min(row_whitebox['normalized_nll'])
            # result['unnormalized_nll_greedy'] = np.min(row_whitebox['unnormalized_nll'])
            result[r'$U_{\rm SE}$'] = [row_whitebox['entropy_unnormalized']] * 10
            unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
            result[r'$U_{\rm NLL}$'] =result['unnormalized_nll_all']

            # select scores with the same index
            score = scores[scores['id'] == idx]
            result['normalized_score_all'] = score.iloc[0]['normalized_score']
            result['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
            normalized_min_index = np.argmin(result['normalized_nll_all'])
            unnormalized_min_index = np.argmin(result['unnormalized_nll_all'])
            result['normalized_score_greedy'] = result['normalized_score_all'][normalized_min_index]
            result['unnormalized_score_greedy'] = result['unnormalized_score_all'][unnormalized_min_index]
            results.append(result)
        df = pd.DataFrame(results).dropna(axis=0)

    if model == 'gpt-3.5-turbo':
        uncertainty_indicators = [r'$U_{\rm EigV}$', r'$U_{\rm Ecc}$', r'$U_{\rm Deg}$', r'$U_{\rm SE}$', r'$U_{\rm NLL}$', r'$C_{\rm Verb}$']
        uncertainty_indicators_print = ['EigV', 'Ecc', 'Deg', 'SE', 'NLL', 'Verb']
    else:
        uncertainty_indicators = [r'$U_{\rm EigV}$', r'$U_{\rm Ecc}$', r'$U_{\rm Deg}$', r'$U_{\rm SE}$', r'$U_{\rm NLL}$']
        uncertainty_indicators_print = ['EigV', 'Ecc', 'Deg', 'SE', 'NLL']
    # indicators = uncertainty_indicators + confidence_indicators

    try: 
        path = os.path.join(args.root_dir, f"{model}_{dataset}_{args.correctness}_{args.temperature}")
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 

    # change plot font size
    plt.rcParams.update({'font.size': 30})

    plt.rcParams.update({'font.size': 30})

    correctness_scores = np.stack(df['normalized_score_all'])
    fig, ax = plt.subplots(figsize=(10, 10))
    for indicator in uncertainty_indicators:
        confidence = np.stack(df[indicator]) if 'Verb' in indicator else -np.stack(df[indicator])
        min_val = np.max(np.min(correctness_scores, axis=0))
        max_val = np.min(np.max(correctness_scores, axis=0))
        thresholds = np.linspace(min_val+epsilon, max_val-epsilon, 10)
        ax = make_plots.AUROC_vs_Correctness_average(correctness_scores, confidence, thresholds, ax=ax, label=indicator)
    ax.grid()
    plt.tight_layout()
    ax.figure.savefig(f'{path}/auroc.pdf')

    fig, ax = plt.subplots(figsize=(10, 10))
    correctness_scores = np.stack(df['normalized_score_all'])
    for indicator in uncertainty_indicators:
        confidence = np.stack(df[indicator]) if 'Verb' in indicator else -np.stack(df[indicator])
        thresholds = np.linspace(np.min(correctness_scores)+epsilon, np.max(correctness_scores)-epsilon, 10)
        ax = make_plots.AUPRC_vs_Correctness_average(correctness_scores, confidence, thresholds, ax=ax, label=indicator)
    ax.grid()
    plt.tight_layout()
    ax.figure.savefig(f'{path}/auprc.pdf')

    # def flatten(x):
    #     x = np.stack(x).flatten()
    #     return x
    
    fig, ax = plt.subplots(figsize=(10, 10))
    spacing = 0.3  # spacing between hat groups
    width = 0.7
    # plot the value range of the uncertainty measures
    for idx, indicator in enumerate(uncertainty_indicators):
        confidence = -np.stack(df[indicator]) if 'Verb' in indicator else np.stack(df[indicator])
        uncertainty = confidence.flatten()
        uncertainty_max = np.max(uncertainty)
        uncertainty_min = np.min(uncertainty)
        ax.bar(idx, uncertainty_max-uncertainty_min, width, bottom=uncertainty_min, label=indicator)
    plt.grid()
    plt.xticks(range(len(uncertainty_indicators)), uncertainty_indicators)
    plt.xlabel('Uncertainty/Confidence Measures')
    plt.ylabel('Output Ranges')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{path}/uncertainty.pdf')
    
    # plot the histogram of correctness score
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(correctness_scores.flatten(), bins=20, color='dodgerblue', edgecolor='dodgerblue')
    # ax.set_title('Correctness score distribution')
    ax.set_xlabel(r'Correctness A')
    ax.set_ylabel('Frequency')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{path}/correctness.pdf')

    # plt.rcParams.update({'font.size': 30})
    # correctness_scores = np.stack(df['normalized_score_all']).flatten()
    # for indicator, print_name in zip(uncertainty_indicators, uncertainty_indicators_print):
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     confidence = -np.stack(df[indicator]) if 'Verb' in indicator else np.stack(df[indicator])
    #     uncertainty = confidence.flatten()
    #     # uncertainty = df[indicator].to_numpy() if 'Verb' not in indicator else -df[indicator].to_numpy()
    #     ax = make_plots.indication_diagram(confidence='Verb' in indicator, correctness=correctness_scores, uncertainties=uncertainty, fig=fig, ax=ax, num_bins=20)
    #     # ax.set_title(f'{indicator} distribution')
    #     ax.legend(loc='upper right', frameon=False, fontsize=30)
    #     ax.set_xlabel(f'Percentage of {indicator} (%)', fontsize=30)
    #     ax.set_ylabel('Percentage of Regressed Correctness (%)', fontsize=30)
    #     plt.grid()
    #     # plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f'{path}/{print_name}.pdf')

    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax = make_plots.variability_diagram(confidence='Verb' in indicator, correctness=correctness_scores, uncertainties=uncertainty, fig=fig, ax=ax, num_bins=20)
    #     # ax.set_title(f'{indicator} distribution')
    #     ax.legend(loc='upper right', frameon=False, fontsize=30)
    #     ax.set_xlabel(f'Percentage of {indicator} (%)', fontsize=30)
    #     ax.set_ylabel('Variability of Conditioned Correctness', fontsize=30)
    #     plt.grid()
    #     # plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f'{path}/{print_name}_variability.pdf')