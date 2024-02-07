import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import blackbox
from models import opensource

if __name__ == '__main__':
    # generations = ["Once upon a time: I am confident", "Once upon a time: I am probably", "Once upon a time: I am very unlikely",  'Hello world!', 'Good morning', 'How are you?']
    generations = ['Leukemia', 'Low Blood Pressure', 'Cervical cancer', 'Cancer in 1953 at 41', 'Breast cancer', 'Tuberculosis', 'Cancer', 'Leukaemia',
                   'Cancer (in 1953 at age 41)', 'Throat cancer']
    jaccard = blackbox.jaccard_similarity([generations])

    nlimodel = opensource.NLIModel(device='cuda')
    sc = blackbox.SemanticConsistency(nlimodel)
    sc_mat = sc.similarity_mat("", [generations])
    # sc_mat = sc.similarity_mat("What claimed the life of singer Kathleen Ferrier?", [generations])

    # ecc = blackbox.Eccentricity(
    #     eigv_threshold=0.5, 
    #     affinity_mode='disagreement_w', 
    #     temperature=1.0,)
    # _u, _c = ecc.compute_scores(sc_mat)
    # print(_u, _c)

    # deg = blackbox.Degree(
    #     affinity_mode='disagreement_w', 
    #     temperature=1.0)
    # _u, _c = deg.compute_scores(sc_mat)
    # print(_u, _c)

    eigv = blackbox.SpectralEigv(
        affinity_mode='agreement',
        temperature=1.0,
        adjust=False)
    result = eigv.compute_scores(sc_mat)

    print(result)
    breakpoint()