import pandas as pd 
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from joblib import dump

from preprocessing import preprocess_basic

df = pd.read_csv('titanic/train.csv')

#perform basic preprocessing steps
df = preprocess_basic(df)[0]

#survived labels cannot be used since these are not available in the test set
drop = [] 
drop.append('Survived')
df = df.drop(columns=drop)

#rows with missing age are useless to train and evaluate the age predictor
df = df.dropna()

Y = df['Age'].to_numpy()
X = df.drop(columns=['Age']).to_numpy()

X_train = X[0:600:]
X_test = X[600::]

Y_train = Y[0:600:]
Y_test = Y[600::]

predictor = RandomForestRegressor(criterion='mae')

# Number of trees in random forest
n_estimators = [400]
# Maximum number of levels in tree
max_depth = [int(x) for x in range(2, 10)]
# Minimum number of samples required to split a node
min_samples_split = [0.001 + i*0.0002 for i in range(20)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [0.001 + i*0.0002 for i in range(20)]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=predictor, param_distributions=random_grid,
                                n_iter=25, cv=5, verbose=1, random_state=42, n_jobs=-1, refit=True, scoring='neg_mean_absolute_error')

rf_random.fit(X_train, Y_train)


print('Best score: ', rf_random.best_score_)
print('Best parameters: ', rf_random.best_params_)
print('')
pred = rf_random.best_estimator_.predict(X_test).reshape(-1,)

print('Mean absolute age error on test set: ', np.mean(np.abs(Y_test-pred)))
print('Rounded absolute age errors on test set')
print(np.round(np.abs(Y_test-pred)))


ap = rf_random.best_estimator_
ap.fit(X, Y)

dump(ap, 'age_predictor.joblib')




