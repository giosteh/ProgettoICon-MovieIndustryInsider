import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold, StratifiedKFold

from imblearn.over_sampling import SMOTE
import joblib

sns.set_style("darkgrid")



def print_info(df):
    """
    Stampa le informazioni sul dataframe.
    """
    info = [(col, str(type)) for col, type in df.dtypes.items()]
    print(f"# cols: {len(df.columns)} | # rows: {len(df)}\n")
    print(tabulate(info, headers=["Column", "Type"], tablefmt="pretty"))


def prepare_data(df, cols, task="regression", resample=False, seed=36):
    """
    Prepara i dati per il training di un modello.
    """
    X = df.drop(columns=[cols["target"]] + cols["drop"], axis=1)
    y = df[cols["target"]].to_numpy()
    y = y.ravel()

    # encoding
    if cols["dummies"]:
        X = pd.get_dummies(X, columns=cols["dummies"])
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    if cols["labels"]:
        encoder = LabelEncoder()
        for col in cols["labels"]:
            X[col] = encoder.fit_transform(X[col])
    
    float_cols = X.select_dtypes(include="float").columns
    X[float_cols] = np.round(X[float_cols], 4)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)

    # resampling con SMOTE
    if task == "classification" and resample:
        smote = SMOTE(sampling_strategy="not majority", random_state=seed)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # normalizzazione
    if cols["standardize"]:
        scaler = StandardScaler()
        X_train[cols["standardize"]] = np.round(scaler.fit_transform(X_train[cols["standardize"]]), 4)
        X_test[cols["standardize"]] = np.round(scaler.transform(X_test[cols["standardize"]]), 4)

    if cols["minmax"]:
        scaler = MinMaxScaler()
        X_train[cols["minmax"]] = np.round(scaler.fit_transform(X_train[cols["minmax"]]), 4)
        X_test[cols["minmax"]] = np.round(scaler.transform(X_test[cols["minmax"]]), 4)

    # gestione della variabile target y
    if task == "regression":
        y_train = np.round(y_train, 4)
        y_test = np.round(y_test, 4)
    elif task == "classification":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
    
    return X_train, X_test, y_train, y_test


def plot_cv_results(param_range, scores, xlabel, ylabel, title):
    """
    Visualizza in un grafico i risultati della cross validation.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(param_range, scores["train"], label="Train Score", color="dodgerblue", linestyle="dashed", linewidth=2.3)
    plt.plot(param_range, scores["val"], label="Val Score", color="crimson", linewidth=2.3)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


def find_best_model(model, X, y, param, param_range, cv, metric="neg_mean_squared_error"):
    """
    Trova il miglior modello tramite la cross validation.
    """
    best_model = None
    best_param = None

    regression = metric.startswith("neg_")
    best_score = np.inf if regression else -np.inf
    score_sign = -1 if regression else 1

    train_scores, test_scores = [], []

    for val in param_range:
        current_model = model
        current_model.set_params(**{param: val})

        cv_results = cross_validate(current_model, X, y, scoring=metric, cv=cv, return_train_score=True)

        train_score = score_sign * cv_results["train_score"].mean()
        test_score = score_sign * cv_results["test_score"].mean()
        train_scores.append(train_score)
        test_scores.append(test_score)

        if (regression and test_score < best_score) or (not regression and test_score > best_score):
            best_model = current_model
            best_param = val
            best_score = test_score
    
    scores = {"train": train_scores, "val": test_scores}

    return best_model, best_param, best_score, scores


def tune_model(model, model_name, X, y, cv, grid_params={}, grid_metrics=[],
               ylabel="", verbose=True, plot=True):
    """
    Implementa una ricerca dei migliori parametri per un modello.
    """
    best_model = model

    regression = grid_metrics[0].startswith("neg")
    grid_metric_name = grid_metrics[0].replace("neg_", "") if regression else grid_metrics[0]
    do_cv = ("max_depth" in grid_params.keys()) or ("n_estimators" in grid_params.keys())

    if grid_params:
        grid_search = GridSearchCV(model, grid_params, cv=cv,
                                   scoring={m: m for m in grid_metrics},
                                   refit=grid_metrics[0], n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        if verbose:
            grid_score = -grid_search.best_score_ if regression else grid_search.best_score_
            print(f"Results after GridSearchCV:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {{'{grid_metric_name}': {grid_score:.4f}}}\n")
    
    # CV supplementare
    if do_cv and grid_params:
        cv_params = get_cv_params(grid_search.best_params_)
        for param in cv_params.keys():
            current_model = best_model

            best_model, best_param, best_score, scores = find_best_model(current_model, X, y,
                                                                         param, cv_params[param],
                                                                         cv=cv, metric=grid_metrics[0])
            if verbose:
                print(f"Results after {param} tuning:")
                print(f"Best parameters: {{'{param}': {best_param}}}")
                print(f"Best score: {{'{grid_metric_name}': {best_score:.4f}}}\n")

            if plot:
                plot_cv_results(cv_params[param], scores, param, ylabel, title=f"{model_name} / {param}")
    
    return best_model


def get_cv_params(grid_best_params):
    """
    Restituisce i range dei valori da testare per i parametri max_depth e n_estimators.
    """
    params_dict = {}

    # range sui parametri max_depth e n_estimators
    if "n_estimators" in grid_best_params.keys():
        params_dict["n_estimators"] = list(range(50, 401, 50))

    if "max_depth" in grid_best_params.keys():
        params_start = grid_best_params["max_depth"] - 10
        if params_start < 2:
            params_start = 2
        params_dict["max_depth"] = list(range(params_start, params_start + 21, 2))
    
    return params_dict


# Modelli per la regressione
MODELS_REG = {
    "Ridge_Regressor": Ridge(),
    "Decision_Tree_Regressor": DecisionTreeRegressor(),
    "Random_Forest_Regressor": RandomForestRegressor(),
    "XGBoost_Regressor": XGBRegressor()
}

GRID_PARAMS_REG = {
    "Ridge_Regressor": {
        "alpha": [.05, .1, .5, 1, 2, 5]
    },

    "Decision_Tree_Regressor": {
        "criterion": ["squared_error", "friedman_mse"], 
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [2, 4, 8, 10, 12]
    },

    "Random_Forest_Regressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 8, 10]
    },

    "XGBoost_Regressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [.01, .1, .2],
        "min_child_weight": [1, 3, 5],
        "subsample": [.6, .8, 1]
    }
}

# Modelli per la classificazione
MODELS_CLS = {
    "Logistic_Regression": LogisticRegression(),
    "Decision_Tree_Classifier": DecisionTreeClassifier(),
    "Random_Forest_Classifier": RandomForestClassifier(),
    "XGBoost_Classifier": XGBClassifier()
}

GRID_PARAMS_CLS = {
    "Logistic_Regression_Classifier": {
        "solver": ["saga"],
        "max_iter": [10000, 20000],
        "penalty": ["l1", "l2"]
    },

    "Decision_Tree_Classifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [2, 4, 8, 10, 12]
    },

    "Random_Forest_Classifier": {
        "criterion": ["gini", "entropy"],
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 8, 10]
    },

    "XGBoost_Classifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [.01, .1, .2],
        "min_child_weight": [1, 3, 5],
        "subsample": [.6, .8, 1]
    }
}


def tune_and_test_models(df, cols, task="regression", models=None, grid_params=None,
                         folds=5, resample=False, seed=42, session_name=""):
    """
    Esegue tuning e test dei modelli per i task di regressione e classificazione.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, cols, task=task, resample=resample)
    # preparazione della CV
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    metric, metric_name = "neg_mean_squared_error", "MSE"
    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        metric, metric_name = "accuracy", "Accuracy"
    
    if not models:
        models = MODELS_REG if task == "regression" else MODELS_CLS
    if not grid_params:
        grid_params = GRID_PARAMS_REG if task == "regression" else GRID_PARAMS_CLS

    # ciclo di tuning per i modelli
    for model_name, model in models.items():
        name = " ".join(model_name.split("_"))
        print("***\n")
        print(f"TUNING & TRAINING <{name}>...\n")

        best_model = tune_model(model, name, X_train, y_train, cv, grid_params[model_name], [metric], metric_name)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        model_path = f"models/{model_name}-{session_name}.joblib"
        joblib.dump(best_model, model_path)

        # test del modello
        print("> TESTING...")
        if task == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = root_mean_squared_error(y_test, y_pred)**2
            print(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\n")
        else:
            acc = accuracy_score(y_test, y_pred) * 100
            print(f"Accuracy: {acc:.2f}%\n")

            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.drop(columns=["support"])
            plt.figure(figsize=(7, 5))
            sns.heatmap(report_df.iloc[:-3, :], annot=True, cmap="Blues", fmt=".2f")
            plt.title(f"Classification Report for {name}")
            plt.show()


def train_and_test_naive_bayes(df, cols, mode="multinomial", resample=False, seed=42, session_name=""):
    """
    Esegue training e test del modello naive bayes.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, cols, task="classification", resample=resample, seed=seed)

    print("***\n")
    model = MultinomialNB() if mode == "multinomial" else GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_path = f"models/naive_bayes-{session_name}.joblib"
    joblib.dump(model, model_path)

    print("> TESTING...")
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy: {acc:.2f}%\n")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=["support"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(report_df.iloc[:-3, :], annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.show()
