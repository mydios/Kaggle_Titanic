import numpy as np
import pandas as pd
from create_submission import create_submission
from preprocessing import preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA


def train_model(X, Y):
    rf = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [800]
    # Maximum number of levels in tree
    max_depth = list(range(2, 12))
    # Minimum number of samples required to split a node
    min_samples_split = [0.001 + i*0.0002 for i in range(60)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [0.001 + i*0.0002 for i in range(60)]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=30, cv=5, verbose=1, random_state=42, n_jobs=-1, refit=True)

    rf_random.fit(X, Y)
    print(rf_random.best_score_)
    print(rf_random.best_params_)
    return rf_random.best_estimator_


if __name__ == '__main__':
    df = pd.read_csv('titanic/train.csv')
    df, mean_fare = preprocess(df)

    r = PCA(n_components=5)

    Y = df['Survived'].to_numpy()
    X = df.drop(columns=['Survived'])

    X = r.fit_transform(X)
    print('#components: ', len(r.explained_variance_ratio_))
    model = train_model(X, Y)

    X_test = pd.read_csv('titanic/test.csv')
    X_test = preprocess(X_test, mean_fare)[0]
    X_test = r.transform(X_test)

    preds = model.predict(X_test)

    create_submission(preds, 'rfANDpca_tuned')
