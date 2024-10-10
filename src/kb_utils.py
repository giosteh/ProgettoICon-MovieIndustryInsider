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


def derive_movies_data_for_reg(df, kb):
    """
    Deriva il nuovo dataframe dei movies dalla kb per la regressione.
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

        features["age"] = query_kb(kb, f"age({movie_id}, X).")
        features["rating_cat"] = query_kb(kb, f"rating_regrouped({movie_id}, X).")
        features["genre_cat"] = query_kb(kb, f"genre_regrouped({movie_id}, X).")
        features["runtime"] = query_kb(kb, f"runtime({movie_id}, X).")
        features["popularity"] = query_kb(kb, f"votes({movie_id}, X).")
        features["score"] = query_kb(kb, f"score({movie_id}, X).")
        features["budget"] = query_kb(kb, f"budget({movie_id}, X).")
        features["budget_efficiency"] = query_kb(kb, f"budget_efficiency({movie_id}, X).")

        # `director` features
        features["director_age"] = query_kb(kb, f"age_in_movie(\"{director}\", {movie_id}, X).")
        features["director_experience"] = query_kb(kb, f"director_experience(\"{director}\", {movie_id}, X).")
        features["director_is_acclaimed"] = query_kb(kb, f"director_is_acclaimed(\"{director}\", {movie_id}).")
        features["director_is_panned"] = query_kb(kb, f"director_is_panned(\"{director}\", {movie_id}).")
        features["director_efficiency"] = query_kb(kb, f"director_budget_efficiency(\"{director}\", {movie_id}, X).")

        # `star` features
        features["star_age"] = query_kb(kb, f"age_in_movie(\"{star}\", {movie_id}, X).")
        features["star_experience"] = query_kb(kb, f"star_experience(\"{star}\", {movie_id}, X).")
        features["star_is_acclaimed"] = query_kb(kb, f"star_is_acclaimed(\"{star}\", {movie_id}).")
        features["star_is_panned"] = query_kb(kb, f"star_is_panned(\"{star}\", {movie_id}).")
        features["star_efficiency"] = query_kb(kb, f"star_budget_efficiency(\"{star}\", {movie_id}, X).")

        new_data.append(features)

    new_df = pd.DataFrame(new_data)
    return new_df


def derive_movies_data_for_cls(df, kb):
    """
    Deriva il nuovo dataframe dei movies dalla kb per la classificazione.
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

        features["age"] = query_kb(kb, f"age({movie_id}, X).")
        features["rating_cat"] = query_kb(kb, f"rating_regrouped({movie_id}, X).")
        features["genre_cat"] = query_kb(kb, f"genre_regrouped({movie_id}, X).")
        features["runtime"] = query_kb(kb, f"runtime({movie_id}, X).")
        features["popularity"] = query_kb(kb, f"votes({movie_id}, X).")
        features["score"] = query_kb(kb, f"score({movie_id}, X).")
        features["budget"] = query_kb(kb, f"budget({movie_id}, X).")
        features["budget_efficiency_cat"] = query_kb(kb, f"budget_efficiency_category({movie_id}, X).")

        # `director` features
        features["director_age"] = query_kb(kb, f"age_in_movie(\"{director}\", {movie_id}, X).")
        features["director_experience"] = query_kb(kb, f"director_experience(\"{director}\", {movie_id}, X).")
        features["director_is_acclaimed"] = query_kb(kb, f"director_is_acclaimed(\"{director}\", {movie_id}).")
        features["director_is_panned"] = query_kb(kb, f"director_is_panned(\"{director}\", {movie_id}).")
        features["director_efficiency"] = query_kb(kb, f"director_budget_efficiency(\"{director}\", {movie_id}, X).")

        # `star` features
        features["star_age"] = query_kb(kb, f"age_in_movie(\"{star}\", {movie_id}, X).")
        features["star_experience"] = query_kb(kb, f"star_experience(\"{star}\", {movie_id}, X).")
        features["star_is_acclaimed"] = query_kb(kb, f"star_is_acclaimed(\"{star}\", {movie_id}).")
        features["star_is_panned"] = query_kb(kb, f"star_is_panned(\"{star}\", {movie_id}).")
        features["star_efficiency"] = query_kb(kb, f"star_budget_efficiency(\"{star}\", {movie_id}, X).")

        new_data.append(features)

    new_df = pd.DataFrame(new_data)
    return new_df


def derive_movies_data_for_nb(df, kb):
    """
    Deriva il nuovo dataframe dei movies dalla kb per la classificazione con Naive Bayes.
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

        features["age_cat"] = query_kb(kb, f"age_category({movie_id}, X).")
        features["rating_cat"] = query_kb(kb, f"rating_regrouped({movie_id}, X).")
        features["genre_cat"] = query_kb(kb, f"genre_regrouped({movie_id}, X).")
        features["runtime_cat"] = query_kb(kb, f"runtime_category({movie_id}, X).")
        features["popularity_cat"] = query_kb(kb, f"popularity_category({movie_id}, X).")
        features["score_cat"] = query_kb(kb, f"score_category({movie_id}, X).")
        features["budget_cat"] = query_kb(kb, f"budget_category({movie_id}, X).")
        features["budget_efficiency_cat"] = query_kb(kb, f"budget_efficiency_category({movie_id}, X).")

        # `director` features
        features["director_age_cat"] = query_kb(kb, f"age_in_movie_category(\"{director}\", {movie_id}, X).")
        features["director_experience_cat"] = query_kb(kb, f"director_experience_category(\"{director}\", {movie_id}, X).")
        features["director_is_acclaimed"] = query_kb(kb, f"director_is_acclaimed(\"{director}\", {movie_id}).")
        features["director_is_panned"] = query_kb(kb, f"director_is_panned(\"{director}\", {movie_id}).")
        features["director_efficiency_cat"] = query_kb(kb, f"director_efficiency_category(\"{director}\", {movie_id}, X).")

        # `star` features
        features["star_age_cat"] = query_kb(kb, f"age_in_movie_category(\"{star}\", {movie_id}, X).")
        features["star_experience_cat"] = query_kb(kb, f"star_experience_category(\"{star}\", {movie_id}, X).")
        features["star_is_acclaimed"] = query_kb(kb, f"star_is_acclaimed(\"{star}\", {movie_id}).")
        features["star_is_panned"] = query_kb(kb, f"star_is_panned(\"{star}\", {movie_id}).")
        features["star_efficiency_cat"] = query_kb(kb, f"star_efficiency_category(\"{star}\", {movie_id}, X).")

        new_data.append(features)
    
    new_df = pd.DataFrame(new_data)
    return new_df
