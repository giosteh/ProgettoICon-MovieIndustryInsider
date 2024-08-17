import pandas as pd
import numpy as np
from pyswip import Prolog

# funzione che salva su un file i fatti
def save_to_file(strings, filename):
    with open(filename, 'a') as f:
        f.write('\n'.join(strings))

def create_kb():
    save_to_file([":-style_check(-discontiguous).\n"], 'facts.pl')

    # fatti per i film
    df = pd.read_csv('../dataset/movies_v2.csv')

    facts = []
    for row in df.itertuples():
        facts.append(
            f'movie({row.id}).\n'
            f'title({row.id}, "{row.title}").\n'
            f'rating({row.id}, "{row.rating}").\n'
            f'genre({row.id}, "{row.genre}").\n'
            f'year({row.id}, {row.year}).\n'
            f'country({row.id}, "{row.country}").\n'
            f'company({row.id}, "{row.company}").\n'
            f'runtime({row.id}, {row.runtime}).\n'
            f'budget({row.id}, {row.budget}).\n'
            f'gross({row.id}, {row.gross}).\n'
            f'score({row.id}, {row.score}).\n'
            f'votes({row.id}, {row.votes}).\n'
            f'directed_by({row.id}, "{row.director}").\n'
            f'star({row.id}, "{row.star}").'
        )

    save_to_file(facts, 'facts.pl')

    # seleziono tutti i valori unici per la colonna 'director'
    df_directors = df['director'].unique()
    # seleziono tutti i valori unici per la colonna 'star'
    df_actors = df['star'].unique()

    facts = []
    # fatti per i registi
    for director in df_directors:
        facts.append(f'director("{director}").')

    # fatti per gli attori
    for actor in df_actors:
        facts.append(f'actor("{actor}").')

    save_to_file(facts, 'facts.pl')

# funzione che esegue una query safe su una kb
def query_kb(prolog_kb, query):
    result = prolog_kb.query(query)
    if result:
        return list(result)[0]['X']
    return None

# funzione che deriva il nuovo dataframe dei movies dalla kb
def derive_movies_data(df, prolog_kb, binning=False):
    new_data = []

    for movie_id in df['id']:
        features = {}

        features['id'] = movie_id
        features['title'] = query_kb(prolog_kb, f'title({movie_id}, X).')
        features['country'] = query_kb(prolog_kb, f'country({movie_id}, X).')
        features['company'] = query_kb(prolog_kb, f'company({movie_id}, X).')
        features['rating'] = query_kb(prolog_kb, f'rating({movie_id}, X).')
        
        features['genre'] = query_kb(prolog_kb, f'genre_regrouped({movie_id}, X).')
        features['score'] = query_kb(prolog_kb, f'score({movie_id}, X).')

        features['director'] = query_kb(prolog_kb, f'directed_by({movie_id}, X).')
        features['star'] = query_kb(prolog_kb, f'star({movie_id}, X).')

        if not binning:
            features['votes'] = query_kb(prolog_kb, f'votes({movie_id}, X).')
            features['runtime'] = query_kb(prolog_kb, f'runtime({movie_id}, X).')
            features['budget'] = query_kb(prolog_kb, f'budget({movie_id}, X).')
            features['gross'] = query_kb(prolog_kb, f'gross({movie_id}, X).')
            features['profit_index'] = query_kb(prolog_kb, f'movie_profit_index({movie_id}, X).')
            features['success_index'] = query_kb(prolog_kb, f'movie_success_index({movie_id}, X).')
            features['cult_index'] = query_kb(prolog_kb, f'movie_cult_index({movie_id}, X).')
        else:
            features['votes'] = query_kb(prolog_kb, f'movie_log_votes_binned({movie_id}, X).')
            features['runtime'] = query_kb(prolog_kb, f'movie_runtime_binned({movie_id}, X).')
            features['budget'] = query_kb(prolog_kb, f'movie_log_budget_binned({movie_id}, X).')
            features['gross'] = query_kb(prolog_kb, f'movie_log_gross_binned({movie_id}, X).')
            features['profit_index'] = query_kb(prolog_kb, f'movie_profit_index_binned({movie_id}, X).')
            features['success_index'] = query_kb(prolog_kb, f'movie_success_index_binned({movie_id}, X).')
            features['cult_index'] = query_kb(prolog_kb, f'movie_cult_index_binned({movie_id}, X).')
        new_data.append(features)
    
    return pd.DataFrame(new_data)

# funzione che deriva il nuovo dataframe dei registi dalla kb
def derive_directors_data(df, prolog_kb):
    new_data = []

    for director in df:
        features = {}

        features['director'] = director
        features['director_num_movies'] = query_kb(prolog_kb, f'director_num_movies("{director}", X).')
        features['director_profit_mean'] = query_kb(prolog_kb, f'avg_profit_by_director("{director}", X).')
        features['director_profit_std'] = query_kb(prolog_kb, f'std_dev_profit_by_director("{director}", X).')
        features['director_score_mean'] = query_kb(prolog_kb, f'avg_score_by_director("{director}", X).')
        features['director_score_std'] = query_kb(prolog_kb, f'std_dev_score_by_director("{director}", X).')

        new_data.append(features)
    
    return pd.DataFrame(new_data)

# funzione che deriva il nuovo dataframe degli attori dalla kb
def derive_actors_data(df, prolog_kb):
    new_data = []

    for actor in df:
        features = {}

        features['actor'] = actor
        features['actor_num_movies'] = query_kb(prolog_kb, f'star_num_movies("{actor}", X).')
        features['actor_profit_mean'] = query_kb(prolog_kb, f'avg_profit_by_star("{actor}", X).')
        features['actor_profit_std'] = query_kb(prolog_kb, f'std_dev_profit_by_star("{actor}", X).')
        features['actor_score_mean'] = query_kb(prolog_kb, f'avg_score_by_star("{actor}", X).')
        features['actor_score_std'] = query_kb(prolog_kb, f'std_dev_score_by_star("{actor}", X).')

        new_data.append(features)

    return pd.DataFrame(new_data)

