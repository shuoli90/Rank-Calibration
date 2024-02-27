import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from indicators import blackbox
from models import opensource, gpt
from utils import text_processing

if __name__ == '__main__':
    # prompt = "Once upon a time:"
    demos = ['90%: confident', '75%: probably', '10%: very unlikely']
    demons = blackbox.demo_perturb(demos)
    
    prompt = 'Answer the following question shortly according to the context: Who was the man behind The Chipmunks?[SEP] Context: A struggling songwriter named Dave Seville finds success when he comes across a trio of singing chipmunks: ... Title: Alvin and the Chipmunks (2007) ...[SEP] Answer: '

    generations = ['Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville was the man behind The Chipmunks', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Ross Bagdasarian Sr', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'Dave Seville', 'The man behind The Chipmunks was Dave Seville', 'Ross Bagdasarian Sr', 'The man behind The Chipmunks was Dave Seville', 'Dave Seville', 'Ross Bagdasarian, Sr']
    generations = ['biden', 'biden', 'trump', 'trump', 'reagan'] * 6
    generations = ['biden'] * 29 + ['trump']
    sim = blackbox.jaccard_similarity([generations])
    
    ecc = blackbox.Eccentricity(device=1)
    ecc_u, ecc_c, sim_mats = ecc.compute_scores([prompt], [generations])

    degree = blackbox.Degree(device=1, semantic_model=ecc.sm)
    degree_u, degree_c, sim_mats = degree.compute_scores([prompt], [generations], batch_sim_mats=sim_mats)

    spectral = blackbox.SpectralEigv(device=1, temperature=1.0, semantic_model=ecc.sm)
    spectral_u, _, sim_mats = spectral.compute_scores([prompt], [generations], sim_mats=sim_mats)