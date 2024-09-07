import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score

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

def manual_recursive_feature_elimination(model, X, y, retain=2, task='regression', verbose=False):
    # Definisco la metrica
    if task == 'regression':
        score_func = mean_squared_error
    elif task == 'classification':
        score_func = accuracy_score
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    features = list(X.columns)  # Lista delle feature iniziali
    retained = len(features)  # Numero di feature da mantenere

    # Suddivido in training set e validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=36)

    while retained > retain:
        feature_importances = []

        # Addestro il modello con tutte le feature attuali
        model.fit(X_train[features], y_train)
        original_score = score_func(y_val, model.predict(X_val[features]))

        # Itero su ogni feature per calcolarne l'importanza
        for feature in features[:]:  # Uso una copia della lista features
            feature_subset = [f for f in features if f != feature]

            # Valido il modello con il subset delle features
            X_train_subset = X_train[feature_subset]
            X_val_subset = X_val[feature_subset]
            model.fit(X_train_subset, y_train)
            score = score_func(y_val, model.predict(X_val_subset))

            # Calcolo l'importanza della feature
            importance = abs(original_score - score)
            feature_importances.append((feature, importance))

        # Trovo la feature con l'impatto minore e la rimuovo
        worst_feature, worst_importance = min(feature_importances, key=lambda x: x[1])

        if verbose:
            print(f"Removing {worst_feature} with importance {worst_importance}")

        # Rimuovo la feature con l'importanza minore
        features.remove(worst_feature)

        # Aggiorno il numero di feature rimanenti
        retained = len(features)

    best_features_df = pd.DataFrame({'feature': features})
    return best_features_df

"""
# funzione che ottiene le features più importanti con tramite recursive feature elimination
def manual_recursive_feature_elimination(model, X, y, retain=2, task='regression', verbose=False):
    # definisco la metrica
    if task == 'regression':
        def score_func(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)
    elif task == 'classification':
        def score_func(y_true, y_pred):
            return accuracy_score(y_true, y_pred)


    features = list(X.columns)
    feature_indices = list(range(X.shape[1]))
    retained = X.shape[1]

    # suddivido in training set e validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=36)

    while retained > retain:
        feature_importances = []
        model.fit(X_train, y_train)
        original_score = score_func(y_val, model.predict(X_val))

        for j in feature_indices:
            feature_subset = [f for f in features if f != features[j]]

            # valido il modello con il subset delle features
            X_train_subset = X_train[feature_subset]
            X_val_subset = X_val[feature_subset]
            model.fit(X_train_subset, y_train)
            score = score_func(y_val, model.predict(X_val_subset))

            importance = abs(abs(original_score) - abs(score))
            feature_importances.append((features[j], importance))

        # trovo la feature con l'impatto minore e la rimuovo
        worst_feature, worst_importance = min(feature_importances, key=lambda x: x[1])

        if verbose:
            print(f'Removing {worst_feature} with importance {worst_importance}')

        feature_indices.remove(features.index(worst_feature))
        features.remove(worst_feature)

        retained = len(features)
    
    best_features_df = pd.DataFrame({'feature': features})
    return best_features_df
"""


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
        return

    new_model = model_class(**model_params)
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    # ottengo le features più importanti con RFECV
    best_features_df = manual_recursive_feature_elimination(new_model, X_train, y_train, retain, task, verbose=True)

    if not grid_params:
        return best_features_df
    else:
        # faccio tuning del modello con le features più importanti
        best_features = list(best_features_df['Top k features'])
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
    