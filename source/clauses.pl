% clausola per il conteggio dei film per ogni attore
num_movies_per_star(Star, Count) :-
    findall(Movie, star(Movie, Star), Movies),
    length(Movies, Count).

% clausola per il conteggio dei film per ogni regista
num_movies_per_director(Director, Count) :-
    findall(Movie, directed_by(Movie, Director), Movies),
    length(Movies, Count).

% clausola sum_of_squares
sum_of_squares([], _, _, 0).
sum_of_squares([Value|Rest], Total, Avg, SumSq) :-
    SumSqRest is SumSq + (Value - Avg)^2,
    sum_of_squares(Rest, Total, Avg, SumSqRest).

% clausola per la media di profit per ogni attore
avg_profit_by_star(Star, AvgProfit) :-
    findall(Profit, (star(Movie, Star), movie_profit_index(Movie, Profit)), Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    AvgProfit is Total / N.

% clausola per la deviazione standard di profit per ogni attore
std_dev_profit_by_star(Star, StdDev) :-
    avg_profit_by_star(Star, AvgProfit),
    findall(Profit, (star(Movie, Star), movie_profit_index(Movie, Profit)), Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    sum_of_squares(Profits, Total, AvgProfit, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).

% clausola per la media di score per ogni attore
avg_score_by_star(Star, AvgScore) :-
    findall(Score, (star(Movie, Star), score(Movie, Score)), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    AvgScore is Total / N.

% clausola per la deviazione standard di score per ogni attore
std_dev_score_by_star(Star, StdDev) :-
    avg_score_by_star(Star, AvgScore),
    findall(Score, (star(Movie, Star), score(Movie, Score)), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    sum_of_squares(Scores, Total, AvgScore, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).

% clausola per la media di profit per ogni regista
avg_profit_by_director(Director, AvgProfit) :-
    findall(Profit, (directed_by(Movie, Director), movie_profit_index(Movie, Profit)), Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    AvgProfit is Total / N.

% clausola per la deviazione standard di profit per ogni regista
std_dev_profit_by_director(Director, StdDev) :-
    avg_profit_by_director(Director, AvgProfit),
    findall(Profit, (directed_by(Movie, Director), movie_profit_index(Movie, Profit)), Profits),
    length(Profits, N),
    sum_list(Profits, Total),
    sum_of_squares(Profits, Total, AvgProfit, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).

% clausola per la media di score per ogni regista
avg_score_by_director(Director, AvgScore) :-
    findall(Score, (directed_by(Movie, Director), score(Movie, Score)), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    AvgScore is Total / N.

% clausola per la deviazione standard di score per ogni regista
std_dev_score_by_director(Director, StdDev) :-
    avg_score_by_director(Director, AvgScore),
    findall(Score, (directed_by(Movie, Director), score(Movie, Score)), Scores),
    length(Scores, N),
    sum_list(Scores, Total),
    sum_of_squares(Scores, Total, AvgScore, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).

% clausola per il director_score_index
director_score_index(Director, Index) :-
    std_dev_score_by_director(Director, StdDev),
    avg_score_by_director(Director, AvgScore),
    Index is AvgScore / StdDev.

% clausola per il star_score_index
star_score_index(Star, Index) :-
    std_dev_score_by_star(Star, StdDev),
    avg_score_by_star(Star, AvgScore),
    Index is AvgScore / StdDev.

% clausola per il director_profit_index
director_profit_index(Director, Index) :-
    std_dev_profit_by_director(Director, StdDev),
    avg_profit_by_director(Director, AvgProfit),
    Index is AvgProfit / StdDev.

% clausola per il star_profit_index
star_profit_index(Star, Index) :-
    std_dev_profit_by_star(Star, StdDev),
    avg_profit_by_star(Star, AvgProfit),
    Index is AvgProfit / StdDev.

% clausola per il movie_profit_index
movie_profit_index(Movie, Index) :-
    gross(Movie, Gross),
    budget(Movie, Budget),
    Budget > 0,
    Index is (Gross - Budget) / Budget.

% clausola per il movie_success_index
movie_success_index(Movie, Index) :-
    movie_profit_index(Movie, ProfitIndex),
    gross(Movie, Gross),
    Index is ProfitIndex * log(Gross).

% clausola per il movie_cult_index
movie_cult_index(Movie, Index) :-
    votes(Movie, Votes),
    score(Movie, Score),
    Index is Score * log(Votes).

% clausola per la movie_age
movie_age(Movie, Age) :-
    year(Movie, Year),
    Age is 2024 - Year.

% clausola per rimappare il genere
genre_regrouped(Movie, "Others") :-
    genre(Movie, Genre),
    member(Genre, ["Mystery", "Thriller", "Sci-Fy", "Family", "Romance", "Western"]).

genre_regrouped(Movie, Genre) :-
    genre(Movie, Genre),
    \+ member(Genre, ["Mystery", "Thriller", "Sci-Fy", "Family", "Romance", "Western"]).

% clausola per la media di score dei film
avg_score(AvgScore) :-
    findall(Score, score(Movie, Score), Scores),
    length(Scores, N),
    N > 0,
    sum_list(Scores, Total),
    AvgScore is Total / N.

% clausola per la deviazione standard di score dei film
std_dev_score(StdDev) :-
    avg_score(AvgScore),
    findall(Score, score(Movie, Score), Scores),
    length(Scores, N),
    N > 1,
    sum_list(Scores, Total),
    sum_of_squares(Scores, Total, AvgScore, SumSq),
    Variance is SumSq / (N - 1),
    StdDev is sqrt(Variance).
