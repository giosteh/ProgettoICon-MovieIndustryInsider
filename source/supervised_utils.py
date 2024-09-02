import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold

from imblearn.over_sampling import SMOTE
import joblib

sns.set_style('whitegrid')


# funzione che prepara i dati per il training
def prepare_data(df, target_col, drop_cols=[], dummies_cols=[], labels_cols=[],
                 round_cols=[], clipping_cols=[], standardize_cols=[], minmax_cols=[],
                 task='regression', resample=False, seed=42):
    # separazione tra features e target
    X = df.drop(columns=[target_col] + drop_cols, axis=1)
    y = df[target_col]


    # encoding
    if dummies_cols:
        X = pd.get_dummies(X, columns=dummies_cols, drop_first=True)
        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)

    if labels_cols:
        encoder = LabelEncoder()
        for col in labels_cols:
            X[col] = encoder.fit_transform(X[col])
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)


    # clipping degli outliers
    if clipping_cols:
        for col in clipping_cols:
            lower_bound = X_train[col].quantile(.02)
            upper_bound = X_train[col].quantile(.98)
            X_train[col] = X_train[col].clip(lower_bound, upper_bound)
            X_test[col] = X_test[col].clip(lower_bound, upper_bound)


    # resampling con SMOTE
    if task == 'classification' and resample:
        smote = SMOTE(sampling_strategy='not majority', random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    

    # normalizzazione
    if standardize_cols:
        scaler = StandardScaler()
        X_train[standardize_cols] = np.round(scaler.fit_transform(X_train[standardize_cols]), 3)
        X_test[standardize_cols] = np.round(scaler.transform(X_test[standardize_cols]), 3)

    if minmax_cols:
        scaler = MinMaxScaler()
        X_train[minmax_cols] = np.round(scaler.fit_transform(X_train[minmax_cols]), 3)
        X_test[minmax_cols] = np.round(scaler.transform(X_test[minmax_cols]), 3)
    
    if round_cols:
        X_train[round_cols] = np.round(X_train[round_cols], 3)
        X_test[round_cols] = np.round(X_test[round_cols], 3)


    # gestione della variabile target y
    if task == 'regression':
        y_train = np.round(y_train, 3)
        y_test = np.round(y_test, 3)
    else:
        encoder = OneHotEncoder()
        y_train = encoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()
        y_test = encoder.transform(y_test.values.reshape(-1, 1)).toarray()
    
    return X_train, X_test, y_train, y_test


# funzione che visualizza in un grafico i risultati della cross validation
def plot_cv_results(param_range, scores, xlabel, ylabel, title):
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, scores['train'], label='Train score', linestyle='dashed', linewidth=2)
    plt.plot(param_range, scores['val'], label='Validation score', linewidth=2.3)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


# funzione che trova il miglior modello
def find_best_model(model, X, y, param, param_range, cv, metric='neg_mean_squared_error'):
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
    
    scores = {'train': train_scores, 'val': test_scores}

    return best_model, best_param, best_score, scores


# funzione che implementa una pipeline per il tuning
def tune_model(model, model_name, X, y, cv,
               grid_params={}, grid_metrics=[],
               verbose=True, plot=True, ylabel=''):
    best_model = model

    regression = grid_metrics[0].startswith('neg')
    grid_metric_name = grid_metrics[0].replace('neg_', '') if regression else grid_metrics[0]
    do_cv = ('max_depth' in grid_params.keys()) or ('n_estimators' in grid_params.keys())

    if grid_params:
        grid_search = GridSearchCV(model, grid_params, cv=cv,
                                   scoring={m: m for m in grid_metrics},
                                   refit=grid_metrics[0], n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_

        if verbose:
            grid_score = -grid_search.best_score_ if regression else grid_search.best_score_
            print(f'Results after GridSearchCV:')
            print(f'Best parameters: {grid_search.best_params_}')
            print(f'Best score: {{\'{grid_metric_name}\': {grid_score:.4f}}}\n')
    
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
                print(f'Best score: {{\'{grid_metric_name}\': {best_score:.4f}}}\n')

            if plot:
                plot_cv_results(cv_params[param], scores, param, ylabel, title=f'{model_name} - {param}')
    
    return best_model


# funzione che restituisce i range di valori per il tuning
def get_cv_params(grid_best_params):
    params_dict = {}

    if 'max_depth' in grid_best_params.keys():
        param_range = []
        if grid_best_params['max_depth'] <= 10:
            param_range = [v for v in range(1, 21)]
        else:
            param_range = [v for v in range(grid_best_params['max_depth'] - 10, grid_best_params['max_depth'] + 11)]

        params_dict['max_depth'] = param_range
    
    if 'n_estimators' in grid_best_params.keys():
        param_range = [v for v in range(40, 321, 40)]
        params_dict['n_estimators'] = param_range

    return params_dict


# funzione che esegue tuning e test dei modelli per il task di regressione
def tune_and_test_models_for_regression(df, cols, folds=5, seed=42, session_name=''):

    X_train, X_test, y_train, y_test = prepare_data(df, cols['target'], cols['drop'], cols['dummies'], cols['labels'], cols['round'],
                                                    cols['clipping'], cols['standardize'], cols['minmax'], task='regression')
    
    models = {
        'Ridge_Regressor': Ridge(),
        'Decision_Tree_Regressor': DecisionTreeRegressor(),
        'Random_Forest_Regressor': RandomForestRegressor(),
        'Gradient_Boosting_Regressor': GradientBoostingRegressor()
    }

    grid_params = {
        'Ridge_Regressor': {
            'alpha': [.05, .1, .5, 1, 2, 5]
        },

        'Decision_Tree_Regressor': {
            'criterion': ['squared_error', 'friedman_mse'], 
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [2, 4, 8, 10, 12]
        },

        'Random_Forest_Regressor': {
            'criterion': ['squared_error'],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 8, 10]
        },

        'Gradient_Boosting_Regressor': {
            'loss': ['squared_error'],
            'learning_rate': [.01, .05],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 8, 10]
        }
    }

    # preparo lo splitting per il tuning
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)


    # ciclo di tuning e test
    for model_name, model in models.items():
        print('-' * 80)
        print(f'\nTraining and tuning [{model_name}]...\n')

        best_model = tune_model(model, model_name, X_train, y_train, cv=cv,
                                grid_params=grid_params[model_name],
                                grid_metrics=['neg_mean_squared_error'],
                                ylabel='MSE')
        
        # il miglior modello viene riaddestrato, testato e salvato
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        joblib.dump(best_model, f'models/{model_name}-{session_name}.joblib')

        print(f'\nTest score for [{model_name}]:')
        print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')
        print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')
        print('\n\n')


# funzione che esegue tuning e test dei modelli per il task di classificazione
def tune_and_test_models_for_classification(df, cols, folds=5, seed=42, session_name=''):

    X_train, X_test, y_train, y_test = prepare_data(df, cols['target'], cols['drop'], cols['dummies'], cols['labels'], cols['round'],
                                                    cols['clipping'], cols['standardize'], cols['minmax'], task='classification')
    
    models = {
        'Logistic_Classifier': LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'),
        'Decision_Tree_Classifier': DecisionTreeClassifier(),
        'Random_Forest_Classifier': RandomForestClassifier(),
        'Gradient_Boosting_Classifier': GradientBoostingClassifier()
    }

    grid_params = {
        'Logistic_Classifier': {
            'C': [.05, .1, .5, 1, 2, 5],
            'penalty': ['l1', 'l2', 'none']
        },

        'Decision_Tree_Classifier': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [2, 4, 8, 10, 12]
        },

        'Random_Forest_Classifier': {
            'criterion': ['gini'],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 8, 10]
        },

        'Gradient_Boosting_Classifier': {
            'loss': ['log_loss'],
            'learning_rate': [.01, .05],
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4, 8, 10]
        }
    }

    # preparo lo splitting per il tuning
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)


    # ciclo di tuning e test
    for model_name, model in models.items():
        print('-' * 80)
        print(f'\nTraining and tuning [{model_name}]...\n')

        best_model = tune_model(model, model_name, X_train, y_train, cv=cv,
                                grid_params=grid_params[model_name],
                                grid_metrics=['accuracy'],
                                ylabel='Accuracy')
        
        # il miglior modello viene riaddestrato, testato e salvato
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        joblib.dump(best_model, f'models/{model_name}-{session_name}.joblib')

        print(f'\nTest score for [{model_name}]:')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}\n')

        # visualizza il report
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Classe 0', 'Classe 1', 'Classe 2'])
        report_df = pd.DataFrame(report).transpose()

        plt.figure(figsize=(8, 5))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
        plt.title('Classification Report')
        plt.show()
        print('\n\n')
