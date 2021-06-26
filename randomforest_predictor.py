import numpy as np
import pandas as pd
from create_submission import create_submission
from preprocessing import preprocess

from sklearn.ensemble import RandomForestClassifier

def train_model(X, Y):
    rf = RandomForestClassifier(max_depth=3, min_weight_fraction_leaf=0.1)
    rf.fit(X, Y)
    return rf

if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, mean_age, mean_fare = preprocess(df)

    Y = df['Survived'].to_numpy()
    X = df.drop(columns=['Survived']).to_numpy()
    rf = train_model(X, Y)

    X_test = pd.read_csv('titanic/test.csv')
    X_test = preprocess(X_test, mean_age, mean_fare)[0]
    X_test = X_test.to_numpy()
    preds = rf.predict(X_test)

    create_submission(preds, 'rf')