% CLAUSOLE PER LA KB.


% ARTISTS

% clausola per il calcolo della age per ogni artista
age_in_movie(Artist, UntilMovie, Age) :-
    year(UntilMovie, YearUntilMovie),
    birth_year(Artist, BirthYear),

    Age is YearUntilMovie - BirthYear.


% clausola per contare i film per ogni regista
director_experience(Director, UntilMovie, N) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie),
            Movies),
    
    length(Movies, N).

% clausola per contare i film per ogni attore
star_experience(Actor, UntilMovie, N) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie),
            Movies),
    
    length(Movies, N).


% clausola per determinare se un regista ha almeno un film acclamato
director_is_acclaimed(Director, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, is_acclaimed(Movie)),
            Movies),
    length(Movies, N),
    N > 0.

% clausola per determinare se un regista ha almeno un film stroncato
director_is_panned(Director, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, is_panned(Movie)),
            Movies),
    length(Movies, N),
    N > 0.


% clausola per determinare se un attore ha almeno un film acclamato
star_is_acclaimed(Actor, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, is_acclaimed(Movie)),
            Movies),
    length(Movies, N),
    N > 0.

% clausola per determinare se un attore ha almeno un film stroncato
star_is_panned(Actor, UntilMovie) :-
    year(UntilMovie, YearUntilMovie),
    findall(Movie, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, is_panned(Movie)),
            Movies),
    length(Movies, N),
    N > 0.


% clausola per calcolare la media di budget_efficiency per un regista
director_budget_efficiency(Director, UntilMovie, AvgEff) :-
    year(UntilMovie, YearUntilMovie),
    findall(Eff, 
            (directed_by(Movie, Director), year(Movie, Year), Year < YearUntilMovie, budget_efficiency(Movie, Eff)),
            Effs),
    length(Effs, N),
    sum_list(Effs, Total),
    (N > 0 -> AvgEff is Total / N ; AvgEff is 0).

% clausola per calcolare la media di budget_efficiency per un attore
star_budget_efficiency(Actor, UntilMovie, AvgEff) :-
    year(UntilMovie, YearUntilMovie),
    findall(Eff, 
            (starring(Movie, Actor), year(Movie, Year), Year < YearUntilMovie, budget_efficiency(Movie, Eff)),
            Effs),
    length(Effs, N),
    sum_list(Effs, Total),
    (N > 0 -> AvgEff is Total / N ; AvgEff is 0).


% MOVIES

% clausola per il calcolo della age di un film
age(Movie, Age) :-
    year(Movie, Year),
    Age is 2024 - Year.


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


% clausola per il calcolo della budget efficiency
budget_efficiency(Movie, Eff) :-
    budget(Movie, Budget),
    gross(Movie, Gross),
    Budget \= 0,

    Eff is Gross / Budget.

budget_efficiency_category(Movie, "low") :-
    budget_efficiency(Movie, X),
    X < 1.

budget_efficiency_category(Movie, "mid") :-
    budget_efficiency(Movie, X),
    X >= 1, X < 3.

budget_efficiency_category(Movie, "high") :-
    budget_efficiency(Movie, X),
    X >= 3.


% clausola per determinare se un film è un blockbuster
is_blockbuster(Movie) :-
    budget(Movie, Budget),

    Budget > 100000000.


% clausola per determinare se un film è indipendente
is_indie(Movie) :-
    budget(Movie, Budget),

    Budget < 10000000.


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


% clausola per il calcolo della media di score dei film
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


% clausola per il calcolo della deviazione standard di score dei film
score_std(Std) :-
    score_mean(Mean),
    findall(Score, score(_, Score), Scores),
    sum_of_squares(Scores, Mean, SumSq),
    length(Scores, N),
    N > 1,

    Std is sqrt(SumSq / (N - 1)).


% clausola per determinare se un film è acclamato dal pubblico
is_acclaimed(Movie) :-
    score(Movie, Score),
    score_mean(Mean),
    score_std(Std),

    Score > Mean + Std.


% clausola per determinare se un film è stroncato dal pubblico
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
