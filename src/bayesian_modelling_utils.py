from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def visualize_bayesian_model(bayesian_model, title="Visualization of Bayesian Model"):
    """
    Visualizza in un grafico la struttura del modello bayesiano.
    """
    G = bayesian_model.to_networkx()
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=12)
    plt.title(title)
    plt.show()


def build_bayesian_model(edges):
    """
    Crea il modello bayesiano a partire da una lista di archi.
    """
    model = BayesianModel(edges)
    assert model.check_model()

    return model


def process_df(df, drop_cols=[]):
    """
    Prepara il dataframe per l'apprendimento del modello bayesiano.
    """
    df = df.drop(drop_cols, axis=1)

    # processing delle variabili categoriche
    categorical_cols = list(df.select_dtypes("object").columns)
    if categorical_cols:
        for col in categorical_cols:
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(df[[col]])
    
    # processing delle variabili numeriche
    float_cols = list(df.select_dtypes("float").columns)
    int_cols = list(df.select_dtypes("int").columns)
    numerical_cols = float_cols + int_cols
    if numerical_cols:
        for col in numerical_cols:
            df[f"{col}_binned"] = pd.cut(df[col], bins=3, labels=["L", "M", "H"])

    return df


def learn_bayesian_model(df):
    """
    Apprende il modello bayesiano sul dataset.
    """
    processed_df = process_df(df)

    # apprendimento della struttura
    hc = HillClimbSearch(processed_df, scoring_method=BicScore(processed_df))
    model_structure = hc.estimate()
    model = build_bayesian_model(model_structure.edges())

    # apprendimento delle CPD
    model.fit(processed_df, estimator=MaximumLikelihoodEstimator)

    return model


def sensitivity_analysis(model, target, variables, baseline_evidence=None):
    """
    Esegue una analisi di sensibilità della variabile target rispetto a una lista di variabili.
    """
    inference = VariableElimination(model)
    if not baseline_evidence:
        baseline_evidence = {}
    
    results = {}
    # itero sulle variabili
    for var in variables:
        var_values = model.get_cpds(var).state_names[var]
        var_results = {}

        for value in var_values:
            evidence = baseline_evidence.copy()
            evidence[var] = value

            query_result = inference.query(variables=[target], evidence=evidence)
            var_results[value] = query_result.values

        results[var] = var_results

    for var, var_results in results.items():
        print(f"Sensibilità rispetto alla variabile {var}:")
        for value, prob in var_results.items():
            print(f"- {value} | Target prob: {prob:.4f}")
    
    return results
