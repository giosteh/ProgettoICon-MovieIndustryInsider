% clausola per il conteggio dei film per ogni attore
actor_num_movies(Actor, N) :-
    findall(Movie, starring(Movie, Actor), Movies),
    length(Movies, Count),
    N is Count - 1.

% clausola per il conteggio dei film per ogni regista
director_num_movies(Director, N) :-
    findall(Movie, directed_by(Movie, Director), Movies),
    length(Movies, Count),
    N is Count - 1.

% clausola sum_of_squares
sum_of_squares([], _, _, 0).
sum_of_squares([Value|Rest], Total, Avg, SumSq) :-
    SumSqRest is SumSq + (Value - Avg)^2,
    sum_of_squares(Rest, Total, Avg, SumSqRest).


% clausola per la media di profit per ogni attore
actor_profit_mean(ExcludedMovie, Actor, AvgProfit) :-
    findall(Profit, 
            (starring(Movie, Actor), Movie \= ExcludedMovie, movie_profit_index(Movie, Profit)), 
            Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    (N > 0 -> AvgProfit is Total / N ; AvgProfit is 0).

% clausola per la deviazione standard di profit per ogni attore
actor_profit_std(ExcludedMovie, Actor, StdDev) :-
    findall(Profit, 
            (starring(Movie, Actor), Movie \= ExcludedMovie, movie_profit_index(Movie, Profit)), 
            Profits),
    length(Profits, N),
    (N > 1 -> 
        sum_list(Profits, Total),
        AvgProfit is Total / N,
        sum_of_squares(Profits, Total, AvgProfit, SumSq),
        StdDev is sqrt(SumSq / (N - 1))
    ;
        StdDev is 0).

% clausola per la media di score per ogni attore
actor_score_mean(ExcludedMovie, Actor, AvgScore) :-
    findall(Score,
            (starring(Movie, Actor), Movie \= ExcludedMovie, score(Movie, Score)),
            Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    (N > 0 -> AvgScore is Total / N ; AvgScore is 0).

% clausola per la deviazione standard di score per ogni attore
actor_score_std(ExcludedMovie, Actor, StdDev) :-
    findall(Score,
            (starring(Movie, Actor), Movie \= ExcludedMovie, score(Movie, Score)),
            Scores),
    length(Scores, N),
    (N > 1 ->
        sum_list(Scores, Total),
        AvgScore is Total / N,
        sum_of_squares(Scores, Total, AvgScore, SumSq),
        StdDev is sqrt(SumSq / (N - 1))
    ;
        StdDev is 0).

% clausola per la media di profit per ogni regista
director_profit_mean(ExcludedMovie, Director, AvgProfit) :-
    findall(Profit, 
            (directed_by(Movie, Director), Movie \= ExcludedMovie, movie_profit_index(Movie, Profit)), 
            Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    (N > 0 -> AvgProfit is Total / N ; AvgProfit is 0).

% clausola per la deviazione standard di profit per ogni regista
director_profit_std(ExcludedMovie, Director, StdDev) :-
    findall(Profit, 
            (directed_by(Movie, Director), Movie \= ExcludedMovie, movie_profit_index(Movie, Profit)), 
            Profits),
    length(Profits, N),
    (N > 1 -> 
        sum_list(Profits, Total),
        AvgProfit is Total / N,
        sum_of_squares(Profits, Total, AvgProfit, SumSq),
        StdDev is sqrt(SumSq / (N - 1))
    ;
        StdDev is 0).

% clausola per la media di score per ogni regista
director_score_mean(ExcludedMovie, Director, AvgScore) :-
    findall(Score,
            (directed_by(Movie, Director), Movie \= ExcludedMovie, score(Movie, Score)),
            Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    (N > 0 -> AvgScore is Total / N ; AvgScore is 0).

% clausola per la deviazione standard di score per ogni regista
director_score_std(ExcludedMovie, Director, StdDev) :-
    findall(Score,
            (directed_by(Movie, Director), Movie \= ExcludedMovie, score(Movie, Score)),
            Scores),
    length(Scores, N),
    (N > 1 ->
        sum_list(Scores, Total),
        AvgScore is Total / N,
        sum_of_squares(Scores, Total, AvgScore, SumSq),
        StdDev is sqrt(SumSq / (N - 1))
    ;
        StdDev is 0).


% clausola per calcolare età di registi e attori
professional_age(Movie, Professional, Age) :-
    year(Movie, MovieYear),
    birth_year(Professional, BirthYear),
    Age is MovieYear - BirthYear.


% clausola per il movie_profit_index
movie_profit_index(Movie, Index) :-
    gross(Movie, Gross),
    budget(Movie, Budget),
    Budget > 0,
    Index is (Gross - Budget) / Budget.

% clausola per il movie_cult_index
movie_cult_index(Movie, Index) :-
    votes(Movie, Votes),
    score(Movie, Score),
    Index is Score * log(Votes).

% clausola per la movie_age
movie_age(Movie, Age) :-
    year(Movie, Year),
    Age is 2024 - Year.


% clausole per rimappare il genere
genre_regrouped(Movie, "Other") :-
    genre(Movie, Genre),
    member(Genre, ["Fantasy", "Mystery", "Thriller", "Sci-Fi", "Family", "Romance", "Western"]).

genre_regrouped(Movie, Genre) :-
    genre(Movie, Genre),
    \+ member(Genre, ["Fantasy", "Mystery", "Thriller", "Sci-Fi", "Family", "Romance", "Western"]).

% clausole per rimappare il rating
rating_regrouped(Movie, "Other") :-
    rating(Movie, Rating),
    member(Rating, ["NC-17", "TV-MA", "Approved", "X"]).

rating_regrouped(Movie, Rating) :-
    rating(Movie, Rating),
    \+ member(Rating, ["NC-17", "TV-MA", "Approved", "X"]).


% clausole per il binning di runtime
movie_runtime_binned(Movie, "short") :-
    runtime(Movie, Runtime),
    Runtime < 90.

movie_runtime_binned(Movie, "mid") :-
    runtime(Movie, Runtime),
    Runtime >= 90,
    Runtime < 135.

movie_runtime_binned(Movie, "long") :-
    runtime(Movie, Runtime),
    Runtime >= 135.

% clausole per il binning di age
movie_age_binned(Movie, "new") :-
    movie_age(Movie, Age),
    Age <= 15.

movie_age_binned(Movie, "old") :-
    movie_age(Movie, Age),
    Age > 15,
    Age <= 30.

movie_age_binned(Movie, "very-old") :-
    movie_age(Movie, Age),
    Age > 30.


% clausola per la media di score dei film
movie_score_mean(Movie, AvgScore) :-
    findall(Score, score(Movie, Score), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    AvgScore is Total / N.

% clausola per la deviazione standard di score dei film
movie_score_std(Movie, StdDev) :-
    movie_score_mean(Movie, AvgScore),
    findall(Score, score(Movie, Score), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    sum_of_squares(Scores, Total, AvgScore, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).

% clausole per il binning di score
movie_score_binned(Movie, "low") :-
    score(Movie, Score),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Score < AvgScore - StdDev.

movie_score_binned(Movie, "mid-low") :-
    score(Movie, Score),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Score >= AvgScore - StdDev,
    Score < AvgScore.

movie_score_binned(Movie, "mid-high") :-
    score(Movie, Score),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Score >= AvgScore,
    Score < AvgScore + StdDev.

movie_score_binned(Movie, "high") :-
    score(Movie, Score),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Score >= AvgScore + StdDev.

% clausole per il binning di profit_index
movie_profit_index_binned(Movie, "not-profitable") :-
    movie_profit_index(Movie, Value),
    Value < 1.

movie_profit_index_binned(Movie, "profitable") :-
    movie_profit_index(Movie, Value),
    Value >= 1, Value < 3.

movie_profit_index_binned(Movie, "very-profitable") :-
    movie_profit_index(Movie, Value),
    Value >= 3.


% clausola per i quartili di cult_index
quartile_limits_cult_index(Q1, Q2, Q3, Q4) :-
    findall(Value, movie_cult_index(_, Value), CultValues),
    sort(CultValues, SortedValues),
    length(SortedValues, N),

    IndexQ1 is N // 4,
    IndexQ2 is N // 2,
    IndexQ3 is 3 * N // 4,
    IndexQ4 is N,

    nth1(IndexQ1, SortedValues, Q1),
    nth1(IndexQ2, SortedValues, Q2),
    nth1(IndexQ3, SortedValues, Q3),
    nth1(IndexQ4, SortedValues, Q4).

% clausole per il binning di cult_index
movie_cult_index_binned(Movie, "low") :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(Q1, _, _, _),
    Value <= Q1.

movie_cult_index_binned(Movie, "mid-low") :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(Q1, Q2, _, _),
    Value > Q1, Value <= Q2.

movie_cult_index_binned(Movie, "mid-high") :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(_, Q2, Q3, _),
    Value > Q2, Value <= Q3.

movie_cult_index_binned(Movie, "high") :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(_, _, Q3, Q4),
    Value > Q3, Value <= Q4.

% clausola per i quartili di budget
quartile_limits_budget(Q1, Q2, Q3, Q4) :-
    findall(Value, budget(Movie, Value), BudgetValues),
    sort(BudgetValues, SortedValues),
    length(SortedValues, N),

    IndexQ1 is N // 4,
    IndexQ2 is N // 2,
    IndexQ3 is 3 * N // 4,
    IndexQ4 is N,

    nth1(IndexQ1, SortedValues, Q1),
    nth1(IndexQ2, SortedValues, Q2),
    nth1(IndexQ3, SortedValues, Q3),
    nth1(IndexQ4, SortedValues, Q4).

% clausole per il binning di budget
movie_budget_binned(Movie, "low") :-
    budget(Movie, Value),
    quartile_limits_budget(Q1, _, _, _),
    Value <= Q1.

movie_budget_binned(Movie, "mid-low") :-
    budget(Movie, Value),
    quartile_limits_budget(Q1, Q2, _, _),
    Value > Q1, Value <= Q2.

movie_budget_binned(Movie, "mid-high") :-
    budget(Movie, Value),
    quartile_limits_budget(_, Q2, Q3, _),
    Value > Q2, Value <= Q3.

movie_budget_binned(Movie, "high") :-
    budget(Movie, Value),
    quartile_limits_budget(_, _, Q3, Q4),
    Value > Q3, Value <= Q4.

% clausola per i quartili di gross
quartile_limits_gross(Q1, Q2, Q3, Q4) :-
    findall(Value, gross(Movie, Value), GrossValues),
    sort(GrossValues, SortedValues),
    length(SortedValues, N),

    IndexQ1 is N // 4,
    IndexQ2 is N // 2,
    IndexQ3 is 3 * N // 4,
    IndexQ4 is N,

    nth1(IndexQ1, SortedValues, Q1),
    nth1(IndexQ2, SortedValues, Q2),
    nth1(IndexQ3, SortedValues, Q3),
    nth1(IndexQ4, SortedValues, Q4).

% clausole per il binning di gross
movie_gross_binned(Movie, "low") :-
    gross(Movie, Value),
    quartile_limits_gross(Q1, _, _, _),
    Value <= Q1.

movie_gross_binned(Movie, "mid-low") :-
    gross(Movie, Value),
    quartile_limits_gross(Q1, Q2, _, _),
    Value > Q1, Value <= Q2.

movie_gross_binned(Movie, "mid-high") :-
    gross(Movie, Value),
    quartile_limits_gross(_, Q2, Q3, _),
    Value > Q2, Value <= Q3.

movie_gross_binned(Movie, "high") :-
    gross(Movie, Value),
    quartile_limits_gross(_, _, Q3, Q4),
    Value > Q3, Value <= Q4.

% clausola per i quartili di votes
quartile_limits_votes(Q1, Q2, Q3, Q4) :-
    findall(Value, votes(Movie, Value), VotesValues),
    sort(VotesValues, SortedValues),
    length(SortedValues, N),

    IndexQ1 is N // 4,
    IndexQ2 is N // 2,
    IndexQ3 is 3 * N // 4,
    IndexQ4 is N,

    nth1(IndexQ1, SortedValues, Q1),
    nth1(IndexQ2, SortedValues, Q2),
    nth1(IndexQ3, SortedValues, Q3),
    nth1(IndexQ4, SortedValues, Q4).

% clausole per il binning di votes
movie_votes_binned(Movie, "low") :-
    votes(Movie, Value),
    quartile_limits_votes(Q1, _, _, _),
    Value <= Q1.

movie_votes_binned(Movie, "mid-low") :-
    votes(Movie, Value),
    quartile_limits_votes(Q1, Q2, _, _),
    Value > Q1, Value <= Q2.

movie_votes_binned(Movie, "mid-high") :-
    votes(Movie, Value),
    quartile_limits_votes(_, Q2, Q3, _),
    Value > Q2, Value <= Q3.

movie_votes_binned(Movie, "high") :-
    votes(Movie, Value),
    quartile_limits_votes(_, _, Q3, Q4),
    Value > Q3, Value <= Q4.


% clausola per il binning della media di profit_index per registi
director_profit_mean_binned(Movie, Director, "not-profitable") :-
    director_profit_mean(Movie, Director, Value),
    Value < 1.

director_profit_mean_binned(Movie, Director, "profitable") :-
    director_profit_mean(Movie, Director, Value),
    Value >= 1, Value < 3.

director_profit_mean_binned(Movie, Director, "very-profitable") :-
    director_profit_mean(Movie, Director, Value),
    Value >= 3.

% clausola per il binning della media di profit_index per attori
actor_profit_mean_binned(Movie, Actor, "not-profitable") :-
    actor_profit_mean(Movie, Actor, Value),
    Value < 1.

actor_profit_mean_binned(Movie, Actor, "profitable") :-
    actor_profit_mean(Movie, Actor, Value),
    Value >= 1, Value < 3.

actor_profit_mean_binned(Movie, Actor, "very-profitable") :-
    actor_profit_mean(Movie, Actor, Value),
    Value >= 3.

% clausola per il binning della deviazione standard di profit_index per registi
director_profit_std_binned(Movie, Director, "low") :-
    director_profit_std(Movie, Director, Value),
    Value < 1.

director_profit_std_binned(Movie, Director, "mid") :-
    director_profit_std(Movie, Director, Value),
    Value >= 1, Value < 3.

director_profit_std_binned(Movie, Director, "high") :-
    director_profit_std(Movie, Director, Value),
    Value >= 3.

% clausola per il binning della deviazione standard di profit_index per attori
actor_profit_std_binned(Movie, Actor, "low") :-
    actor_profit_std(Movie, Actor, Value),
    Value < 1.

actor_profit_std_binned(Movie, Actor, "mid") :-
    actor_profit_std(Movie, Actor, Value),
    Value >= 1, Value < 3.

actor_profit_std_binned(Movie, Actor, "high") :-
    actor_profit_std(Movie, Actor, Value),
    Value >= 3.

% causola per il binning della media di score per registi
director_score_mean_binned(Movie, Director, "low") :-
    director_score_mean(Movie, Director, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value < AvgScore - StdDev.

director_score_mean_binned(Movie, Director, "mid-low") :-
    director_score_mean(Movie, Director, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore - StdDev,
    Value < AvgScore.

director_score_mean_binned(Movie, Director, "mid-high") :-
    director_score_mean(Movie, Director, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore,
    Value < AvgScore + StdDev.

director_score_mean_binned(Movie, Director, "high") :-
    director_score_mean(Movie, Director, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore + StdDev.

% clausola per il binning della media di score per attori
actor_score_mean_binned(Movie, Actor, "low") :-
    actor_score_mean(Movie, Actor, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value < AvgScore - StdDev.

actor_score_mean_binned(Movie, Actor, "mid-low") :-
    actor_score_mean(Movie, Actor, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore - StdDev,
    Value < AvgScore.

actor_score_mean_binned(Movie, Actor, "mid-high") :-
    actor_score_mean(Movie, Actor, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore,
    Value < AvgScore + StdDev.

actor_score_mean_binned(Movie, Actor, "high") :-
    actor_score_mean(Movie, Actor, Value),
    movie_score_mean(Movie, AvgScore),
    movie_score_std(Movie, StdDev),
    Value >= AvgScore + StdDev.

% clausola per il binning della deviazione standard di score per registi
director_score_std_binned(Movie, Director, "low") :-
    director_score_std(Movie, Director, Value),
    Value < 0.35.

director_score_std_binned(Movie, Director, "mid") :-
    director_score_std(Movie, Director, Value),
    Value >= 0.35, Value < 0.7.

director_score_std_binned(Movie, Director, "high") :-
    director_score_std(Movie, Director, Value),
    Value >= 0.7.

% clausola per il binning della deviazione standard di score per attori
actor_score_std_binned(Movie, Actor, "low") :-
    actor_score_std(Movie, Actor, Value),
    Value < 0.35.

actor_score_std_binned(Movie, Actor, "mid") :-
    actor_score_std(Movie, Actor, Value),
    Value >= 0.35, Value < 0.7.

actor_score_std_binned(Movie, Actor, "high") :-
    actor_score_std(Movie, Actor, Value),
    Value >= 0.7.

% clausola per il binning di num_movies per registi
director_num_movies_binned(Director, "low") :-
    director_num_movies(Director, Value),
    Value < 3.

director_num_movies_binned(Director, "mid") :-
    director_num_movies(Director, Value),
    Value >= 3, Value < 6.

director_num_movies_binned(Director, "high") :-
    director_num_movies(Director, Value),
    Value >= 6.

% clausola per il binning di num_movies per attori
actor_num_movies_binned(Actor, "low") :-
    actor_num_movies(Actor, Value),
    Value < 3.

actor_num_movies_binned(Actor, "mid") :-
    actor_num_movies(Actor, Value),
    Value >= 3, Value < 6.

actor_num_movies_binned(Actor, "high") :-
    actor_num_movies(Actor, Value),
    Value >= 6.

% clausola per il binning di professional_age
professional_age_binned(Movie, Professional, "young") :-
    professional_age(Movie, Professional, Value),
    Value < 30.

professional_age_binned(Movie, Professional, "adult") :-
    professional_age(Movie, Professional, Value),
    Value >= 30, Value < 60.

professional_age_binned(Movie, Professional, "old") :-
    professional_age(Movie, Professional, Value),
    Value >= 60.
