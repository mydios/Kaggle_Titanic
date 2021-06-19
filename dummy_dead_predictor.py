import numpy as np
import pandas as pd
from create_submission import create_submission

def dummyDeadPredictor(X):
    return np.zeros(len(X), dtype=np.int)

if __name__ == '__main__':
    df = pd.read_csv('titanic/test.csv')
    preds = dummyDeadPredictor(df)
    create_submission(preds, 'dummyDead')
