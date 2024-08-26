import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

import joblib
from supervised_utils import *

sns.set_style('whitegrid')


# funzione che ottiene l'importanza delle features di un dato modello
def get_feature_importances(model, cols):
    importances = model.feature_importances_

    importance_df = pd.DataFrame({'feature': cols, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    return importance_df


# funzione che plotta l'importanza delle features di un modello
def plot_feature_importances(importance_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    plt.show()

# funzione che ottiene le features più importanti con RFECV
def get_best_features_with_rfecv(model, cols, X_train, y_train, retain=1, task='regression'):
    metric = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'

    rfecv = RFECV(estimator=model, step=1, min_features_to_select=retain,
                  cv=5, scoring=metric, n_jobs=-1)
    rfecv.fit(X_train, y_train)

    best_features = cols[rfecv.support_]

    return best_features

