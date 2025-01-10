% CLAUSOLE PER LA KB.


% ARTISTS

% clausola per calcolare la age di un artista in un film
age_in_movie(Artist, UntilMovie, Age) :-
    year(UntilMovie, YearUntilMovie),
    birth_year(Artist, BirthYear),
    Age is YearUntilMovie - BirthYear.


% clausola per determinare la categoria di age di un artista di un film
age_in_movie_category(Artist, UntilMovie, "young") :-
    age_in_movie(Artist, UntilMovie, Age),
    Age < 30.

age_in_movie_category(Artist, UntilMovie, "adult") :-
    age_in_movie(Artist, UntilMovie, Age),
    Age >= 30, Age < 60.

age_in_movie_category(Artist, UntilMovie, "elderly") :-
    age_in_movie(Artist, UntilMovie, Age),
    Age >= 60.


% clausola per contare i film fatti da un regista fino a un certo anno
director_experience(Director, UntilMovie, N) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie),
            Movies),
    length(Movies, N).

% clausola per contare i film fatti da un attore fino a un certo anno
star_experience(Actor, UntilMovie, N) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie),
            Movies),
    length(Movies, N).


% clausola per determinare la categoria di esperienza di un regista
director_experience_category(Director, UntilMovie, "low") :-
    director_experience(Director, UntilMovie, N),
    N < 3.

director_experience_category(Director, UntilMovie, "mid") :-
    director_experience(Director, UntilMovie, N),
    N >= 3, N < 6.

director_experience_category(Director, UntilMovie, "high") :-
    director_experience(Director, UntilMovie, N),
    N >= 6.


% clausola per determinare la categoria di esperienza di un attore
star_experience_category(Actor, UntilMovie, "low") :-
    star_experience(Actor, UntilMovie, N),
    N < 3.

star_experience_category(Actor, UntilMovie, "mid") :-
    star_experience(Actor, UntilMovie, N),
    N >= 3, N < 6.

star_experience_category(Actor, UntilMovie, "high") :-
    star_experience(Actor, UntilMovie, N),
    N >= 6.


% clausola per determinare se un regista ha almeno un film acclamato fino a un certo anno
director_is_acclaimed(Director, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, is_acclaimed(Movie)),
            Movies),
    length(Movies, N),
    N > 0.

% clausola per determinare se un regista ha almeno un film stroncato fino a un certo anno
director_is_panned(Director, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, is_panned(Movie)),
            Movies),
    length(Movies, N),
    N > 0.


% clausola per determinare se un attore ha almeno un film acclamato fino a un certo anno
star_is_acclaimed(Actor, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, is_acclaimed(Movie)),
            Movies),
    length(Movies, N),
    N > 0.

% clausola per determinare se un attore ha almeno un film stroncato fino a un certo anno
star_is_panned(Actor, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, is_panned(Movie)),
            Movies),
    length(Movies, N),
    N > 0.


% clausola per calcolare la media di budget efficiency per un regista fino a un certo anno
director_budget_efficiency(Director, UntilMovie, AvgEff) :-
    year(UntilMovie, YearUntilMovie),
    findall(Eff, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, budget_efficiency(Movie, Eff)),
            Effs),
    length(Effs, N),
    sum_list(Effs, Total),
    (N > 0 -> AvgEff is Total / N ; AvgEff is 0).

% clausola per calcolare la media di budget efficiency per un attore fino a un certo anno
star_budget_efficiency(Actor, UntilMovie, AvgEff) :-
    year(UntilMovie, YearUntilMovie),
    findall(Eff, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, budget_efficiency(Movie, Eff)),
            Effs),
    length(Effs, N),
    sum_list(Effs, Total),
    (N > 0 -> AvgEff is Total / N ; AvgEff is 0).


% clausola per determinare la categoria di budget efficiency di un regista fino a un certo anno
director_budget_efficiency_category(Director, UntilMovie, "none") :-
    director_budget_efficiency(Director, UntilMovie, AvgEff),
    AvgEff =:= 0.

director_efficiency_category(Director, UntilMovie, "low") :-
    director_budget_efficiency(Director, UntilMovie, AvgEff),
    AvgEff > 0, AvgEff < 1.

director_efficiency_category(Director, UntilMovie, "mid") :-
    director_budget_efficiency(Director, UntilMovie, AvgEff),
    AvgEff >= 1, AvgEff < 3.

director_efficiency_category(Director, UntilMovie, "high") :-
    director_budget_efficiency(Director, UntilMovie, AvgEff),
    AvgEff >= 3.


% clausola per determinare la categoria di budget efficiency di un attore fino a un certo anno
star_budget_efficiency_category(Actor, UntilMovie, "none") :-
    star_budget_efficiency(Actor, UntilMovie, AvgEff),
    AvgEff =:= 0.

star_efficiency_category(Actor, UntilMovie, "low") :-
    star_budget_efficiency(Actor, UntilMovie, AvgEff),
    AvgEff > 0, AvgEff < 1.

star_efficiency_category(Actor, UntilMovie, "mid") :-
    star_budget_efficiency(Actor, UntilMovie, AvgEff),
    AvgEff >= 1, AvgEff < 3.

star_efficiency_category(Actor, UntilMovie, "high") :-
    star_budget_efficiency(Actor, UntilMovie, AvgEff),
    AvgEff >= 3.


% MOVIES

% clausola per calcolare la age di un film
age(Movie, Age) :-
    year(Movie, Year),
    Age is 2024 - Year.


% clausola per determinare la categoria di age di un film
age_category(Movie, "recent") :-
    age(Movie, Age),
    Age < 10.

age_category(Movie, "old") :-
    age(Movie, Age),
    Age >= 10, Age < 25.

age_category(Movie, "very-old") :-
    age(Movie, Age),
    Age >= 25.


% clausola per determinare la categoria di runtime di un film
runtime_category(Movie, "short") :-
    runtime(Movie, X),
    X < 90.

runtime_category(Movie, "medium") :-
    runtime(Movie, X),
    X >= 90, X < 120.

runtime_category(Movie, "long") :-
    runtime(Movie, X),
    X >= 120.


% clausola per calcolare la budget efficiency di un film
budget_efficiency(Movie, Eff) :-
    budget(Movie, Budget),
    gross(Movie, Gross),
    Budget \= 0,
    Eff is Gross / Budget.

% clausola per determinare la categoria di budget efficiency di un film
budget_efficiency_category(Movie, "low") :-
    budget_efficiency(Movie, X),
    X < 1.

budget_efficiency_category(Movie, "mid") :-
    budget_efficiency(Movie, X),
    X >= 1, X < 3.

budget_efficiency_category(Movie, "high") :-
    budget_efficiency(Movie, X),
    X >= 3.


% clausola per determinare la categoria di budget di un film
budget_category(Movie, "low") :-
    budget(Movie, X),
    X < 10000000.

budget_category(Movie, "mid") :-
    budget(Movie, X),
    X >= 10000000, X < 30000000.

budget_category(Movie, "high") :-
    budget(Movie, X),
    X >= 30000000, X < 60000000.

budget_category(Movie, "very-high") :-
    budget(Movie, X),
    X >= 60000000.


% clausola per determinare la categoria di popolarità di un film
popularity_category(Movie, "low") :-
    votes(Movie, X),
    X < 10000.

popularity_category(Movie, "mid") :-
    votes(Movie, X),
    X >= 10000, X < 50000.

popularity_category(Movie, "high") :-
    votes(Movie, X),
    X >= 50000, X < 100000.

popularity_category(Movie, "very-high") :-
    votes(Movie, X),
    X >= 100000.


% clausola per determinare la categoria di score di un film
score_category(Movie, "very-low") :-
    score(Movie, X),
    score_mean(Mean),
    score_std(Std),
    X < Mean - Std.

score_category(Movie, "low") :-
    score(Movie, X),
    score_mean(Mean),
    score_std(Std),
    X >= Mean - Std, X < Mean.

score_category(Movie, "high") :-
    score(Movie, X),
    score_mean(Mean),
    score_std(Std),
    X >= Mean, X < Mean + Std.

score_category(Movie, "very-high") :-
    score(Movie, X),
    score_mean(Mean),
    score_std(Std),
    X >= Mean + Std.


% clausola per calcolare la media di score dei film
score_mean(Mean) :-
    findall(Score, score(_, Score), Scores),
    sum_list(Scores, Total),
    length(Scores, N),
    N > 0,
    Mean is Total / N.


sum_of_squares([], _, 0).
sum_of_squares([Value|Rest], Avg, SumSq) :-
    sum_of_squares(Rest, Avg, RestSumSq),
    SumSq is RestSumSq + (Value - Avg)^2.


% clausola per calcolare la deviazione standard di score dei film
score_std(Std) :-
    score_mean(Mean),
    findall(Score, score(_, Score), Scores),
    sum_of_squares(Scores, Mean, SumSq),
    length(Scores, N),
    N > 1,
    Std is sqrt(SumSq / (N - 1)).


% clausola per determinare se un film è stato acclamato dal pubblico
is_acclaimed(Movie) :-
    score(Movie, Score),
    score_mean(Mean),
    score_std(Std),
    Score > Mean + Std.


% clausola per determinare se un film è stato stroncato dal pubblico
is_panned(Movie) :-
    score(Movie, Score),
    score_mean(Mean),
    score_std(Std),
    Score < Mean - Std.


% clausola per rimappare il genere di un film
genre_regrouped(Movie, "Other") :-
    genre(Movie, Genre),
    member(Genre, ["Fantasy", "Mystery", "Thriller", "Sci-Fi", "Family", "Romance", "Western"]).

genre_regrouped(Movie, Genre) :-
    genre(Movie, Genre),
    \+ member(Genre, ["Fantasy", "Mystery", "Thriller", "Sci-Fi", "Family", "Romance", "Western"]).

% clausola per rimappare il rating di un film
rating_regrouped(Movie, "Other") :-
    rating(Movie, Rating),
    member(Rating, ["NC-17", "TV-MA", "Approved", "X"]).

rating_regrouped(Movie, Rating) :-
    rating(Movie, Rating),
    \+ member(Rating, ["NC-17", "TV-MA", "Approved", "X"]).
