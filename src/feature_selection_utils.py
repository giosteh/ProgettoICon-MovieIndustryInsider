import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from supervised_utils import *

sns.set_style("darkgrid")



def plot_mutual_information(df, cols, task="regression"):
    """
    Visualizza in un grafico la mutual information delle features.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, cols, task)
    X = pd.concat([X_train, X_test])
    y = np.concatenate([y_train, y_test], axis=0)
    cols_list = list(X.columns)

    if task == "regression":
        mi = mutual_info_regression(X, y)
    else:
        mi = mutual_info_classif(X, y)

    mi_df = pd.DataFrame({"feature": cols_list, "importance": mi})
    mi_df = mi_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=mi_df)
    plt.title("Mutual Information")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    return mi_df


def plot_feature_importances(df, cols, model, task="regression"):
    """
    Visualizza in un grafico l'importanza delle features.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, cols, task)
    X = pd.concat([X_train, X_test])
    y = np.concatenate([y_train, y_test], axis=0)
    cols_list = list(X.columns)

    # ottengo l'importanza delle features
    if hasattr(model, "get_score"):
        importances_dict = model.get_score(importance_type="gain")
        importances = np.array([importances_dict.get(col, 0) for col in cols_list])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = model.coef_

    importances_df = pd.DataFrame({"feature": cols_list, "importance": importances})
    importances_df = importances_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=importances_df)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    return importances_df


def manual_forward_selection(df, cols, model, k_features=10, task="regression"):
    """
    Implementa l'algoritmo di forward selection.
    """
    metric = "neg_mean_squared_error"
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    if task == "classification":
        metric = "accuracy"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # preparo il dataset
    X_train, X_test, y_train, y_test = prepare_data(df, cols, task)
    X = pd.concat([X_train, X_test])
    y = np.concatenate([y_train, y_test], axis=0)

    features = []
    remaining_features = list(X.columns)

    for _ in range(k_features):
        best_feature = None
        best_score = -np.inf

        for feature in remaining_features:
            current_features = features + [feature]
            X_subset = X[current_features]

            score = cross_val_score(model, X_subset, y, cv=cv, scoring=metric).mean()

            if score > best_score:
                best_feature = feature
                best_score = score

        features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"+ Added {best_feature}")

    return pd.DataFrame({"feature": features})
