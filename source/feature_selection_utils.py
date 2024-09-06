import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import RFECV

from supervised_utils import *

sns.set_style('whitegrid')


# funzione che ottiene l'importanza delle features di un dato modello
def get_feature_importances(model, cols_list):
    # ottengo l'importanza delle features
    importances = model.feature_importances_

    importance_df = pd.DataFrame({'feature': cols_list, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    # plotto l'importanza delle features
    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    plt.show()

    return importance_df

# funzione che ottiene le features più importanti con RFECV
def get_best_features_with_rfecv(model, X, y, cv, retain=2, task='regression'):
    # definisco la metrica
    metric = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'

    rfecv = RFECV(estimator=model, step=1, min_features_to_select=retain,
                  cv=cv, scoring=metric, n_jobs=-1)
    rfecv.fit(X, y)

    # ottengo le features selezionate
    selected_features = list(X.columns[rfecv.support_])
    ranking = rfecv.ranking_[rfecv.support_]

    best_features_df = pd.DataFrame({'feature': selected_features, 'ranking': ranking})
    best_features_df = best_features_df.sort_values(by='ranking', ascending=True)

    # plotto l'importanza delle features
    plt.figure(figsize=(8, 5))
    sns.barplot(x='ranking', y='feature', data=best_features_df)
    plt.title('Best Features got with RFECV')
    plt.xlabel('Ranking')
    plt.ylabel('Features')

    plt.show()

    return best_features_df


# funzione che, dato un modello, ottiene le features più importanti con RFECV e lo riaddestra
def study_model_with_best_features(model_name, df, cols, folds=5, retain=None,
                                   grid_params=None, resample=False, task='regression'):
    # preparo i dati
    X_train, X_test, y_train, y_test = prepare_data(df, cols['target'], cols['drop'], cols['dummies'], cols['labels'],
                                                    cols['round'], cols['clipping'], cols['standardize'], cols['minmax'],
                                                    resample=resample, task=task)
    model_path = f'models/{model_name}.joblib'

    model = joblib.load(model_path)
    model_class = model.__class__
    model_params = model.get_params()

    # ottengo l'importanza delle features
    if retain == 'all':
        get_feature_importances(model, list(X_train.columns))

    new_model = model_class(**model_params)
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    # ottengo le features più importanti con RFECV
    best_features_df = get_best_features_with_rfecv(new_model, X_train, y_train, cv, retain, task)


    # tuning del modello
    if grid_params:
        best_features = list(best_features_df['feature'])
        X_train = X_train[best_features]
        X_test = X_test[best_features]

        metric = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'

        new_best_model = tune_model(new_model, model_name, X_train, y_train, cv=cv,
                                    grid_params=grid_params, grid_metrics=[metric],
                                    ylabel=metric)
        
        # riaddestro il nuovo modello
        new_best_model.fit(X_train, y_train)
        y_pred = new_best_model.predict(X_test)

        joblib.dump(new_best_model, f'models/{model_name}-best.joblib')

        print('Test score:')
        if task == 'regression':
            print(f'MSE: {mean_squared_error(y_test, y_pred)}')
        else:
            print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    
    
    # ritorno le features più importanti
    if retain != 'all':
        return best_features