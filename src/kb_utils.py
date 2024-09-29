import pandas as pd



def save_to_file(strings, filename):
    """
    Salva su un file i fatti.
    """
    with open(filename, "a") as f:
        f.write("\n".join(strings))


def create_kb():
    """
    Crea la kb a partire dal dataset.
    """
    df_movies = pd.read_csv("../data/movies_adj.csv")
    save_to_file([":-style_check(-discontiguous).\n\n"], "kb/facts.pl")

    # fatti per i `movies`
    facts = []
    for row in df_movies.itertuples():
        facts.append(
            f"movie({row.id}).\n"
            f"title({row.id}, \"{row.title}\").\n"
            f"rating({row.id}, \"{row.rating}\").\n"
            f"genre({row.id}, \"{row.genre}\").\n"
            f"year({row.id}, {row.year}).\n"
            f"score({row.id}, {row.score}).\n"
            f"votes({row.id}, {row.votes}).\n"
            f"runtime({row.id}, {row.runtime}).\n"
            f"budget({row.id}, {row.budget}).\n"
            f"gross({row.id}, {row.gross}).\n"
            f"directed_by({row.id}, \"{row.director}\").\n"
            f"starring({row.id}, \"{row.star}\")."
        )
    save_to_file(facts, "kb/facts.pl")

    # fatti per gli `artists`
    df_artists = pd.read_csv("../data/artists.csv")
    for row in df_artists.itertuples():
        facts.append(
            f"professional(\"{row.primaryName}\").\n"
            f"birth_year(\"{row.primaryName}\", {row.birthYear}).\n"
            f"death_year(\"{row.primaryName}\", {row.deathYear}).\n"
            f"known_for(\"{row.primaryName}\", \"{row.knownForTitle}\").\n"
            f"primary_profession(\"{row.primaryName}\", \"{row.primaryProfession}\").\n"
            f"secondary_profession(\"{row.primaryName}\", \"{row.secondaryProfession}\")."
        )
    save_to_file(facts, "kb/facts.pl")


def query_kb(prolog_kb, query):
    """
    Esegui una query safe sulla kb.
    """
    result = prolog_kb.query(query)
    if result:
        return list(result)[0]["X"]
    return None


def derive_movies_data(df, kb):
    """
    Deriva il nuovo dataframe dei movies dalla kb.
    """
    new_data = []

    for row in df.itertuples():
        features = {}

        movie_id = row.id
        director = row.director
        star = row.star

        # `movie` features
        features["id"] = movie_id
        features["title"] = query_kb(kb, f"title({movie_id}, X).")
        features["rating"] = query_kb(kb, f"rating_regrouped({movie_id}, X).")
        features["genre"] = query_kb(kb, f"genre_regrouped({movie_id}, X).")
        features["age"] = query_kb(kb, f"age({movie_id}, X).")
        features["runtime"] = query_kb(kb, f"runtime_category({movie_id}, X).")
        features["popularity"] = query_kb(kb, f"popularity_category({movie_id}, X).")
        features["score"] = query_kb(kb, f"score({movie_id}, X).")
        features["is_acclaimed"] = query_kb(kb, f"is_acclaimed({movie_id}).")
        features["is_panned"] = query_kb(kb, f"is_panned({movie_id}).")
        features["budget_efficiency"] = query_kb(kb, f"budget_efficiency_category({movie_id}, X).")
        features["is_blockbuster"] = query_kb(kb, f"is_blockbuster({movie_id}).")
        features["is_indie"] = query_kb(kb, f"is_indie({movie_id}).")

        # `director` features
        features["director_age_in_movie"] = query_kb(kb, f"age_in_movie(\"{director}\", {movie_id}, X).")
        features["director_experience"] = query_kb(kb, f"director_experience(\"{director}\", {movie_id}, X).")
        features["director_is_acclaimed"] = query_kb(kb, f"director_is_acclaimed(\"{director}\", {movie_id}).")
        features["director_is_panned"] = query_kb(kb, f"director_is_panned(\"{director}\", {movie_id}).")
        features["director_budget_efficiency"] = query_kb(kb, f"director_budget_efficiency(\"{director}\", {movie_id}, X).")

        # `star` features
        features["star_age_in_movie"] = query_kb(kb, f"age_in_movie(\"{star}\", {movie_id}, X).")
        features["star_experience"] = query_kb(kb, f"star_experience(\"{star}\", {movie_id}, X).")
        features["star_is_acclaimed"] = query_kb(kb, f"star_is_acclaimed(\"{star}\", {movie_id}).")
        features["star_is_panned"] = query_kb(kb, f"star_is_panned(\"{star}\", {movie_id}).")
        features["star_budget_efficiency"] = query_kb(kb, f"star_budget_efficiency(\"{star}\", {movie_id}, X).")

        new_data.append(features)
    
    return pd.DataFrame(new_data)
