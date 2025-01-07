from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate



def visualize_bayesian_model(bayesian_model):
    """
    Visualizza il modello bayesiano come grafo.
    """
    def assign_layers(G):
        """
        Assegna un livello ad ogni nodo del modello.
        """
        roots = [n for n in G.nodes() if not list(G.predecessors(n))]
        layers = {}

        for root in roots:
            for node, succs in nx.bfs_successors(G, root):
                if node not in layers:
                    layers[node] = 0
                for succ in succs:
                    layers[succ] = max(layers.get(succ, 0), layers[node] + 1)
        return layers
    # re-mapping delle etichette dei nodi
    node_mapping = {node: node.replace("_", " ").replace("cat", "").strip() for node in bayesian_model.nodes()}
    G = nx.relabel_nodes(bayesian_model, node_mapping)
    pos = nx.planar_layout(G, scale=10)
    
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color="powderblue", node_size=1160,
            arrowstyle="->", arrowsize=11, font_size=8.5, font_color="midnightblue")
    plt.title("Bayesian network graph")
    plt.show()

def process_df(df, drop_cols=[], only_drop=True):
    """
    Prepara il dataframe per l'apprendimento del modello bayesiano.
    """
    df = df.drop(columns=drop_cols, axis=1)
    if only_drop:
        return df
    
    # processing delle variabili numeriche
    float_cols = list(df.select_dtypes("float").columns)
    int_cols = list(df.select_dtypes("int").columns)
    numerical_cols = float_cols + int_cols
    if numerical_cols:
        for col in numerical_cols:
            df[f"{col}_binned"] = pd.cut(df[col], bins=3, labels=["low", "mid", "high"])

    return df

def learn_model_cpds(model, df, process=False):
    """
    Crea il modello bayesiano a partire da una lista di archi.
    """
    processed_df = df if not process else process_df(df)
    model.fit(processed_df, estimator=MaximumLikelihoodEstimator)

    return model

def learn_bayesian_model(df, process=False):
    """
    Apprende il modello bayesiano sul dataset.
    """
    processed_df = df if not process else process_df(df)

    # apprendimento della struttura della rete
    hc = HillClimbSearch(processed_df)
    model_structure = hc.estimate(scoring_method=BicScore(processed_df))

    # apprendimento delle CPD
    model = learn_model_cpds(BayesianNetwork(model_structure.edges()), processed_df, process)

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
            # inferenza sulla target
            query_results = inference.query(variables=[target], evidence=evidence)

            target_values = query_results.state_names[target]
            target_probs = query_results.values
            var_results[value] = list(zip(target_values, target_probs))

        results[var] = var_results

    for var, var_results in results.items():
        print(f"\nSensibilità di {target.upper()} rispetto alla variabile [{var.upper()}]")
        for value, probs in var_results.items():
            print(f"* Dato {var.upper()} = '{value}':")
            # stampa le probabilità condizionate della target
            table = [(f"'{target_val}'", f"{target_prob*100:.2f}%") for target_val, target_prob in probs]
            print(tabulate(table, headers=[target.upper(), "Prob. (%)"], tablefmt="fancy_grid"))
