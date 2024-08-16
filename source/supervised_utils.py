import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

sns.set_style('whitegrid')


# funzione che prepara i dati per il modello
def prepare_data(df, target_col, drop_cols=[],
                dummies_cols=[], labels_cols=[],
                standardize_cols=[], log_standardize_cols=[],
                minmax_standardize_cols=[], seed=42):
    X = df.drop(columns=[target_col] + drop_cols, axis=1)
    y = df[target_col]

    if dummies_cols:
        X = pd.get_dummies(X, columns=dummies_cols)

    if labels_cols:
        encoder = LabelEncoder()
        for col in labels_cols:
            X[col] = encoder.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)

    if standardize_cols:
        scaler = StandardScaler()
        X_train[standardize_cols] = np.round(scaler.fit_transform(X_train[standardize_cols]), 2)
        X_test[standardize_cols] = np.round(scaler.transform(X_test[standardize_cols]), 2)

    if log_standardize_cols:
        scaler = StandardScaler()
        X_train[log_standardize_cols] = np.round(scaler.fit_transform(np.log(X_train[log_standardize_cols] + 1)), 2)
        X_test[log_standardize_cols] = np.round(scaler.transform(np.log(X_test[log_standardize_cols] + 1)), 2)
    
    if minmax_standardize_cols:
        scaler = MinMaxScaler()
        X_train[minmax_standardize_cols] = np.round(scaler.fit_transform(X_train[minmax_standardize_cols]), 2)
        X_test[minmax_standardize_cols] = np.round(scaler.transform(X_test[minmax_standardize_cols]), 2)
    
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


# funzione che visualizza in un grafico i risultati della cross validation
def plot_cv_results(param_range, scores,
                    xlabel, ylabel, title=''):
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, scores['train'], label='Train score', linestyle='dashed', linewidth=2)
    plt.plot(param_range, scores['val'], label='Validation score', linewidth=2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


# funzione che trova il miglior modello
def find_best_model(model, X, y, param, param_range, cv=5, metric='neg_mean_squared_error'):
    best_model = None
    best_param = None

    regression = metric.startswith('neg')

    best_score = np.inf if regression else -np.inf
    train_scores, test_scores = [], []

    for val in param_range:
        current_model = model
        current_model.set_params(**{param: val})

        cv_results = cross_validate(current_model, X, y, scoring=metric, cv=cv, return_train_score=True)

        train_score = -cv_results['train_score'].mean() if regression else cv_results['train_score'].mean()
        test_score = -cv_results['test_score'].mean() if regression else cv_results['test_score'].mean()

        train_scores.append(train_score)
        test_scores.append(test_score)

        if (regression and test_score < best_score) or (not regression and test_score > best_score):
            best_model = current_model
            best_param = val
            best_score = test_score
    
    scores = {
        'train': train_scores,
        'val': test_scores
    }

    return best_model, best_param, best_score, scores


# funzione che implementa una pipeline per il tuning
def tune_model(model, model_name, X, y, cv=5,
               grid_params={}, grid_metrics=[],
               verbose=True, plot=True, ylabel=''):
    best_model = model

    regression = grid_metrics[0].startswith('neg')
    grid_metric_name = grid_metrics[0].replace('neg_', '') if regression else grid_metrics[0]
    do_cv = ('max_depth' in grid_params.keys()) or ('n_estimators' in grid_params.keys())

    if grid_params:
        grid_search = GridSearchCV(model, grid_params, cv=cv,
                                   scoring={m: m for m in grid_metrics},
                                   refit=grid_metrics[0],
                                   n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_

        if verbose:
            grid_score = -grid_search.best_score_ if regression else grid_search.best_score_
            print(f'Results after GridSearchCV:')
            print(f'Best parameters: {grid_search.best_params_}')
            print(f'Best score: {{\'{grid_metric_name}\': {grid_score}}}')
    
    if do_cv and grid_params:
        cv_params = get_cv_params(grid_search.best_params_)
        for param in cv_params.keys():
            current_model = best_model

            best_model, best_param, best_score, scores = find_best_model(current_model, X, y,
                                                                         param, cv_params[param],
                                                                         cv=cv, metric=grid_metrics[0])

            if verbose:
                print(f'Results after {param} tuning:')
                print(f'Best parameters: {{\'{param}\': {best_param}}}')
                print(f'Best score: {{\'{grid_metric_name}\': {best_score}}}')

            if plot:
                plot_cv_results(cv_params[param], scores, param, ylabel, title=f'{model_name} - {param}')
    
    return best_model


# funzione che restituisce i range di valori per il tuning
def get_cv_params(grid_best_params):
    params_dict = {}

    if 'max_depth' in grid_best_params.keys():
        params_dict['max_depth'] = [v for v in range(grid_best_params['max_depth'] - 5, grid_best_params['max_depth'] + 6)]
    
    if 'n_estimators' in grid_best_params.keys():
        params_dict['n_estimators'] = [v for v in range(grid_best_params['n_estimators'] - 50, grid_best_params['n_estimators'] + 51, 10)]
    
    return params_dict


# funzione che esegue tuning e test dei modelli per il task di regressione
def tune_and_test_models_for_regression(df, cols, cv=5):

    X_train, X_test, y_train, y_test = prepare_data(df, cols['target_col'], cols['drop_cols'],
                                                    cols['dummies_cols'], cols['labels_cols'],
                                                    cols['standardize_cols'], cols['log_standardize_cols'],
                                                    cols['minmax_standardize_cols'])

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(),
        'Gradient Boosting Regression': GradientBoostingRegressor()
    }

    grid_params = {
        'Linear Regression': {},

        'Ridge Regression': {
            'alpha': [.001, .01, .05, .1, .5, 1]
        },

        'Decision Tree Regression': {
            'criterion': ['squared_error', 'friedman_mse'], 
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        },

        'Random Forest Regression': {
            'criterion': ['squared_error', 'friedman_mse'],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },

        'Gradient Boosting Regression': {
            'loss': ['squared_error', 'huber'],
            'learning_rate': [.001, .01, .05, .1, .5],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }

    for model_name, model in models.items():
        print(f'\nTraining and tuning [{model_name}]...')

        best_model = tune_model(model, model_name, X_train, y_train, cv=cv,
                                grid_params=grid_params[model_name], grid_metrics=['neg_mean_squared_error'], ylabel='MSE')
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        print(f'\n\nTest score:')
        print(f'MSE: {mean_squared_error(y_test, y_pred)}')
        print(f'MAE: {mean_absolute_error(y_test, y_pred)}')




