import numpy as np
import pandas as pd
import math

from sklearn.ensemble import RandomForestRegressor

from joblib import load

def preprocess_basic(df, mean_fare=None):
    drop = []

    #drop columns with majority missing values
    #drop.append('Cabin')

    #drop irrelevant columns
    drop.append('Name')
    drop.append('Ticket')
    drop.append('PassengerId')

    #Fix missing Cabin values by grouping them all in an extra class
    #Group them by sections (A, B, C, D, E, F...)
    def fix_cabin(c):
        try:
            if math.isnan(c):
                return 'M'
        except:
            pass
        if 'T' in c:
            return 'A'
        else:
            return c[0]
        
    df['Cabin'] = df['Cabin'].apply(fix_cabin)

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

    #Creatue feature about married women
    df['Is_married'] = df['Name'].apply(lambda s: ('Mrs.' in s))

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
    df['Family'] = df['Parch'] + df['SibSp'] + 1

    
    #Create features about fare cost
    df['Fare_low1'] = df['Fare']<7
    df['Fare_low2'] = df['Fare']<7.5
    df['Fare_low3'] = df['Fare']<13
    df['Fare_low4'] = df['Fare']<25
    df['Fare_low5'] = df['Fare']<30
    df['Fare_low6'] = df['Fare']<50
    df['Fare_low7'] = df['Fare']<75

    #Create feature about average cost per ticket of a family
    df['Estimated_cost_per_ticket'] = df['Fare']/(df['Family'])


    #make categorical columns one-hot
    df['Sex'] = df['Sex'] == 'female'
    df = df.join(pd.get_dummies(df.Embarked, prefix='Embarked'))
    drop.append('Embarked')
    df = df.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
    drop.append('Pclass')
    df = df.join(pd.get_dummies(df.Title, prefix='Title'))
    drop.append('Title')
    df = df.join(pd.get_dummies(df.Cabin, prefix='Cabin'))
    drop.append('Cabin')

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
    """
    #impute age by taking the median age of the corresponding price class
    m1 = df.groupby(['Pclass_1']).median()['Age'][1]
    m2 = df.groupby(['Pclass_2']).median()['Age'][1]
    m3 = df.groupby(['Pclass_3']).median()['Age'][1]
    for i in range(len(df)):
        if math.isnan(df['Age'].iloc[i]):
            age = None
            if df['Pclass_1'].iloc[i] == 1:
                age = m1
            elif df['Pclass_2'].iloc[i] == 1:
                age = m2
            else:
                age = m3
            df.loc[i, 'Age'] = age
    """
    return df, mean_fare


if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, mean_fare= preprocess(df)
    print(df['Age'].head(20))