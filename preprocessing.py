import numpy as np
import pandas as pd
import math

from sklearn.ensemble import RandomForestRegressor

from joblib import load

def preprocess_basic(df, mean_fare=None):
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


    #Create feature about title
    df['Title'] = df['Name']
    def title_mapper(s):
        if ('Dr.' in s) or ('Master.' in s) or ('Miss.' in s) or ('Mr.' in s) or ('Mrs.' in s) or ('Rev.' in s):
            return s.split(', ')[1].split(' ')[0]
        elif ('Lady.' in s) or ('Sir.' in s) or ('Lady.' in s) or ('Dona.' in s) or ('Jonkheer.' in s) or ('Countess' in s):
            return 'Noble.'
        elif ('Col.' in s) or ('Major.' in s) or ('Capt.' in s):
            return 'Military.'
        elif ('Ms.' in s) or ('Mlle.' in s):
            return 'Miss.'
        elif ('Mme.' in s):
            return 'Mrs.'
        elif ('Don.' in s):
            return 'Mr.'
        else:
            return 'Mr.'
    df['Title'] = df['Title'].apply(title_mapper)

    #Create feature about family size
    df['Family'] = df['Parch'] + df['SibSp']


    #make categorical columns one-hot
    df['Sex'] = df['Sex'] == 'female'
    df = df.join(pd.get_dummies(df.Embarked, prefix='Embarked'))
    drop.append('Embarked')
    df = df.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
    drop.append('Pclass')
    df = df.join(pd.get_dummies(df.Title, prefix='Title'))
    drop.append('Title')


    df = df.drop(columns=drop)

    return df, mean_fare


def preprocess(df, mean_fare=None):
    df, mean_fare = preprocess_basic(df, mean_fare)

    #Fix missing ages by predicting it from other rows
    age_predictor = load('age_predictor.joblib')

    for i in range(len(df)):
        if math.isnan(df['Age'].iloc[i]):
            try:
                row = df.drop(columns=['Age', 'Survived']).iloc[i].to_numpy()
            except:
                row = df.drop(columns=['Age']).iloc[i].to_numpy()
            age = np.sum(age_predictor.predict([row]))
            df.loc[i, 'Age'] = age

    return df, mean_fare


if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, mean_fare= preprocess(df)
    print(df['Age'].head(20))