import numpy as np
import pandas as pd
from create_submission import create_submission
from preprocessing import preprocess

from sklearn.tree import DecisionTreeClassifier

def train_model(X, Y):
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X, Y)
    return dt

if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, age_predictor, mean_fare = preprocess(df)

    Y = df['Survived'].to_numpy()
    X = df.drop(columns=['Survived']).to_numpy()
    dt = train_model(X, Y)

    X_test = pd.read_csv('titanic/test.csv')
    X_test = preprocess(X_test, age_predictor, mean_fare)[0]
    X_test = X_test.to_numpy()
    preds = dt.predict(X_test)

    create_submission(preds, 'dt')