import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import correctness

if __name__ == '__main__':
    # A = correctness.Score(score_name='exact_match')
    A = correctness.Score(score_name='exact_match', mode='exact_match')
    print(A(references=['hello'], predictions=['hello world']))