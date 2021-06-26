import numpy as np
import pandas as pd
import math

from sklearn.tree import DecisionTreeRegressor

def preprocess(df, dt=None, mean_fare=None):
    drop = []

    #drop columns with majority missing values
    drop.append('Cabin')

    #drop irrelevant columns
    drop.append('Name')
    drop.append('Ticket')
    drop.append('PassengerId')

    #Fix missing Fare value by imputing mean fare of known rows
    if (mean_fare is None):
        mean_fare = np.round(np.mean(df['Fare'][np.invert(np.isnan(df['Fare']))]))
    def fix_fare(f):
        if math.isnan(f):
            return mean_fare
        return f
    df['Fare'] = df['Fare'].apply(fix_fare)

    #Fix missing Embarked value by imputing with most common value S
    def fix_embarked(e):
        try: 
            math.isnan(e)
            return 'S'
        except:
            return e
    df['Embarked'] = df['Embarked'].apply(fix_embarked)

    #make categorical columns one-hot
    df['Sex'] = df['Sex'] == 'female'
    df = df.join(pd.get_dummies(df.Embarked, prefix='Embarked'))
    drop.append('Embarked')
    df = df.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
    drop.append('Pclass')


    df = df.drop(columns=drop)


    #Fix missing ages by predicting it from other rows
    if dt is None:
        dt = DecisionTreeRegressor(max_depth=5)
        X = df.dropna().drop(columns=['Survived', 'Age']).to_numpy()
        Y = df['Age'].dropna().to_numpy()
        dt.fit(X, Y)
    a = np.zeros(len(df))
    p = []
    for i in range(len(a)):
        age = df['Age'][i]
        if math.isnan(age):
            try:
                row = df.drop(columns=['Survived', 'Age']).iloc[i].to_numpy().reshape(1,-1)
            except:
                row = df.drop(columns=['Age']).iloc[i].to_numpy().reshape(1,-1)
            age = np.sum(dt.predict(row))
            p.append(age)
        a[i] = age
    df['Age'] = a

    print(p)

    return df, dt, mean_fare


if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, dt, mean_fare= preprocess(df)
    print(df.head(50))