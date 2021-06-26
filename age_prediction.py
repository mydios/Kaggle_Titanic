import pandas as pd 
import numpy as np
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from joblib import dump

df = pd.read_csv('titanic/train.csv')

drop = []

#drop columns with majority missing values
drop.append('Cabin')

#drop irrelevant columns
drop.append('Name')
drop.append('Ticket')
drop.append('PassengerId')

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

drop.append('Survived')
df = df.drop(columns=drop)

df = df.dropna()

Y = df['Age'].to_numpy()
X = df.drop(columns=['Age']).to_numpy()

X_train = X[0:600:]
X_test = X[600::]

Y_train = Y[0:600:]
Y_test = Y[600::]

predictor = RandomForestRegressor(criterion='mae')

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=350, stop=800, num=10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in range(5, len(X[0]))]
# Minimum number of samples required to split a node
min_samples_split = [0.001 + i*0.001 for i in range(30)]
# Minimum number of samples required at each leaf node
min_samples_leaf = min_samples_split
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=predictor, param_distributions=random_grid,
                                n_iter=50, cv=2, verbose=1, random_state=42, n_jobs=-1, refit=True, scoring='neg_mean_absolute_error')

rf_random.fit(X_train, Y_train)

"""
print(rf_random.best_score_)
print(rf_random.best_params_)

pred = rf_random.best_estimator_.predict(X_test).reshape(-1,)

print(np.mean(np.abs(Y_test-pred)))

print(np.abs(Y_test-pred))
"""

ap = rf_random.best_estimator_
ap.fit(X, Y)

dump(ap, 'age_predictor.joblib')





