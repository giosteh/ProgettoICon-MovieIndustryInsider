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

% clausola per fare il binning di movie_age
movie_age_binned(Movie, Bin) :-
    movie_age(Movie, Age),
    (Age < 10 -> Bin = '0-10';
    Age < 20 -> Bin = '10-20';
    Age < 30 -> Bin = '20-30';
    Age < 40 -> Bin = '30-40';
    Age < 50 -> Bin = '40-50').

sum_logs([], 0).
sum_logs([Value|Rest], Sum) :-
    LogValue is log(Value),
    sum_logs(Rest, SumRest),
    Sum is SumRest + LogValue.

% clausola per la media del numero di voti di scala logaritmica
avg_log_votes(AvgLogVotes) :-
    findall(Votes, (votes(Movie, Votes)), VotesList),
    length(VotesList, N),
    N > 0,
    sum_logs(VotesList, Total),
    AvgLogVotes is Total / N.

% clausola per fare la deviazione standard del numero di voti di scala logaritmica
std_dev_log_votes(StdDevLogVotes) :-
    avg_log_votes(AvgLogVotes),
    findall(Votes, (votes(Movie, Votes)), VotesList),
    length(VotesList, N),
    N > 1,
    sum_logs(VotesList, Total),
    sum_of_squares(VotesList, Total, AvgLogVotes, SumSq),
    Variance is SumSq / (N - 1),
    StdDevLogVotes is sqrt(Variance).

% clausola per fare il binning standard del numero di voti
movie_log_votes_binned(Movie, Bin) :-
    votes(Movie, Votes),
    avg_log_votes(AvgLogVotes),
    std_dev_log_votes(StdDevLogVotes),
    (Votes < AvgLogVotes - 2 * StdDevLogVotes -> Bin = 'low';
    Votes > AvgLogVotes + 2 * StdDevLogVotes -> Bin = 'high';
    Votes < AvgLogVotes - StdDevLogVotes -> Bin = 'mid-low';
    Votes > AvgLogVotes + StdDevLogVotes -> Bin = 'mid-high';
    Bin = 'mid').

% clausola per la media del budget dei film in scala logaritmica
avg_log_budget(AvgLogBudget) :-
    findall(Budget, (budget(Movie, Budget)), BudgetList),
    length(BudgetList, N),
    N > 0,
    sum_logs(BudgetList, Total),
    AvgLogBudget is Total / N.

% clausola per la deviazione standard del budget dei film in scala logaritmica
std_dev_log_budget(StdDevLogBudget) :-
    avg_log_budget(AvgLogBudget),
    findall(Budget, (budget(Movie, Budget)), BudgetList),
    length(BudgetList, N),
    N > 1,
    sum_logs(BudgetList, Total),
    sum_of_squares(BudgetList, Total, AvgLogBudget, SumSq),
    Variance is SumSq / (N - 1),
    StdDevLogBudget is sqrt(Variance).

% clausola per fare il binning standard del budget
movie_log_budget_binned(Movie, Bin) :-
    budget(Movie, Budget),
    avg_log_budget(AvgLogBudget),
    std_dev_log_budget(StdDevLogBudget),
    (Budget < AvgLogBudget - 2 * StdDevLogBudget -> Bin = 'low';
    Budget > AvgLogBudget + 2 * StdDevLogBudget -> Bin = 'high';
    Budget < AvgLogBudget - StdDevLogBudget -> Bin = 'mid-low';
    Budget > AvgLogBudget + StdDevLogBudget -> Bin = 'mid-high';
    Bin = 'mid').

% clausola per la media del gross dei film in scala logaritmica
avg_log_gross(AvgLogGross) :-
    findall(Gross, (gross(Movie, Gross)), GrossList),
    length(GrossList, N),
    N > 0,
    sum_logs(GrossList, Total),
    AvgLogGross is Total / N.

% clausola per la deviazione standard del gross dei film in scala logaritmica
std_dev_log_gross(StdDevLogGross) :-
    avg_log_gross(AvgLogGross),
    findall(Gross, (gross(Movie, Gross)), GrossList),
    length(GrossList, N),
    N > 1,
    sum_logs(GrossList, Total),
    sum_of_squares(GrossList, Total, AvgLogGross, SumSq),
    Variance is SumSq / (N - 1),
    StdDevLogGross is sqrt(Variance).

% clausola per fare il binning standard del gross
movie_log_gross_binned(Movie, Bin) :-
    gross(Movie, Gross),
    avg_log_gross(AvgLogGross),
    std_dev_log_gross(StdDevLogGross),
    (Gross < AvgLogGross - 2 * StdDevLogGross -> Bin = 'low';
    Gross > AvgLogGross + 2 * StdDevLogGross -> Bin = 'high';
    Gross < AvgLogGross - StdDevLogGross -> Bin = 'mid-low';
    Gross > AvgLogGross + StdDevLogGross -> Bin = 'mid-high';
    Bin = 'mid').

% clausola per la media del profit_index dei film
avg_profit_index(AvgProfitIndex) :-
    findall(ProfitIndex, (movie_profit_index(Movie, ProfitIndex)), ProfitIndexList),
    length(ProfitIndexList, N),
    N > 0,
    sum_list(ProfitIndexList, Total),
    AvgProfitIndex is Total / N.

% clausola per la deviazione standard del profit_index dei film
std_dev_profit_index(StdDevProfitIndex) :-
    avg_profit_index(AvgProfitIndex),
    findall(ProfitIndex, (movie_profit_index(Movie, ProfitIndex)), ProfitIndexList),
    length(ProfitIndexList, N),
    N > 1,
    sum_list(ProfitIndexList, Total),
    sum_of_squares(ProfitIndexList, Total, AvgProfitIndex, SumSq),
    Variance is SumSq / (N - 1),
    StdDevProfitIndex is sqrt(Variance).

% clausola per il binning standard del profit_index usando media e deviazione standard
movie_profit_index_binned(Movie, Bin) :-
    profit_index(Movie, ProfitIndex),
    avg_profit_index(AvgProfitIndex),
    std_dev_profit_index(StdDevProfitIndex),
    (ProfitIndex < AvgProfitIndex - StdDevProfitIndex -> Bin = 'low';
    ProfitIndex > AvgProfitIndex + StdDevProfitIndex -> Bin = 'high';
    Bin = 'mid').

% clausola per la media del success_index di ogni film
avg_success_index(AvgSuccessIndex) :-
    findall(SuccessIndex, (success_index(Movie, SuccessIndex)), SuccessIndexList),
    length(SuccessIndexList, N),
    N > 0,
    sum_list(SuccessIndexList, Total),
    AvgSuccessIndex is Total / N.

% clausola per la deviazione standard del success_index di ogni film
std_dev_success_index(StdDevSuccessIndex) :-
    avg_success_index(AvgSuccessIndex),
    findall(SuccessIndex, (success_index(Movie, SuccessIndex)), SuccessIndexList),
    length(SuccessIndexList, N),
    N > 1,
    sum_list(SuccessIndexList, Total),
    sum_of_squares(SuccessIndexList, Total, AvgSuccessIndex, SumSq),
    Variance is SumSq / (N - 1),
    StdDevSuccessIndex is sqrt(Variance).

% clausola per il binning standard del success_index usando media e deviazione standard
movie_success_index_binned(Movie, Bin) :-
    success_index(Movie, SuccessIndex),
    avg_success_index(AvgSuccessIndex),
    std_dev_success_index(StdDevSuccessIndex),
    (SuccessIndex < 2 * AvgSuccessIndex - StdDevSuccessIndex -> Bin = 'low';
    SuccessIndex > 2 * AvgSuccessIndex + StdDevSuccessIndex -> Bin = 'high';
    SuccessIndex < 2 * AvgSuccessIndex - StdDevSuccessIndex -> Bin = 'mid-low';
    SuccessIndex > 2 * AvgSuccessIndex + StdDevSuccessIndex -> Bin = 'mid-high';
    Bin = 'mid').

% clausola per la media di cult_index dei film
avg_cult_index(AvgCultIndex) :-
    findall(CultIndex, (cult_index(Movie, CultIndex)), CultIndexList),
    length(CultIndexList, N),
    N > 0,
    sum_list(CultIndexList, Total),
    AvgCultIndex is Total / N.

% clausola per la deviazione standard di cult_index dei film
std_dev_cult_index(StdDevCultIndex) :-
    avg_cult_index(AvgCultIndex),
    findall(CultIndex, (cult_index(Movie, CultIndex)), CultIndexList),
    length(CultIndexList, N),
    N > 1,
    sum_list(CultIndexList, Total),
    sum_of_squares(CultIndexList, Total, AvgCultIndex, SumSq),
    Variance is SumSq / (N - 1),
    StdDevCultIndex is sqrt(Variance).

% clausola per il binning standard del cult_index usando media e deviazione standard
movie_cult_index_binned(Movie, Bin) :-
    cult_index(Movie, CultIndex),
    avg_cult_index(AvgCultIndex),
    std_dev_cult_index(StdDevCultIndex),
    (CultIndex < AvgCultIndex - 2 * StdDevCultIndex -> Bin = 'low';
    CultIndex > AvgCultIndex + 2 * StdDevCultIndex -> Bin = 'high';
    CultIndex < AvgCultIndex - StdDevCultIndex -> Bin = 'mid-low';
    CultIndex > AvgCultIndex + StdDevCultIndex -> Bin = 'mid-high';
    Bin = 'mid').

% clausola per il binninbg di runtime per ogni film
movie_runtime_binned(Movie, Bin) :-
    runtime(Movie, Runtime),
    (Runtime < 90 -> Bin = 'short';
    Runtime > 150 -> Bin = 'long';
    Bin = 'mid').

quartile_limits_profit_index(Q1_Limit, Q2_Limit, Q3_Limit, Q4_Limit) :-
    findall(Value, movie_profit_index(_, Value), ProfitValues),

    sort(ProfitValues, SortedValues),

    length(SortedValues, Len),

    Q1_Index is Len // 4,
    Q2_Index is Len // 2,
    Q3_Index is 3 * Len // 4,
    Q4_Index is Len,

    nth1(Q1_Index, SortedValues, Q1_Limit),
    nth1(Q2_Index, SortedValues, Q2_Limit),
    nth1(Q3_Index, SortedValues, Q3_Limit),
    nth1(Q4_Index, SortedValues, Q4_Limit).

quartile_limits_cult_index(Q1_Limit, Q2_Limit, Q3_Limit, Q4_Limit) :-
    findall(Value, movie_cult_index(_, Value), CultValues),

    sort(CultValues, SortedValues),

    length(SortedValues, Len),

    Q1_Index is Len // 4,
    Q2_Index is Len // 2,
    Q3_Index is 3 * Len // 4,
    Q4_Index is Len,

    nth1(Q1_Index, SortedValues, Q1_Limit),
    nth1(Q2_Index, SortedValues, Q2_Limit),
    nth1(Q3_Index, SortedValues, Q3_Limit),
    nth1(Q4_Index, SortedValues, Q4_Limit).

quartile_limits_gross_log(Q1_Limit, Q2_Limit, Q3_Limit, Q4_Limit) :-
    findall(Value, movie_gross_log(_, Value), GrossValues),

    sort(GrossValues, SortedValues),

    length(SortedValues, Len),

    Q1_Index is Len // 4,
    Q2_Index is Len // 2,
    Q3_Index is 3 * Len // 4,
    Q4_Index is Len,

    nth1(Q1_Index, SortedValues, Q1_Limit),
    nth1(Q2_Index, SortedValues, Q2_Limit),
    nth1(Q3_Index, SortedValues, Q3_Limit),
    nth1(Q4_Index, SortedValues, Q4_Limit).

quartile_limits_budget_log(Q1_Limit, Q2_Limit, Q3_Limit, Q4_Limit) :-
    findall(Value, movie_budget_log(_, Value), BudgetValues),

    sort(BudgetValues, SortedValues),

    length(SortedValues, Len),

    Q1_Index is Len // 4,
    Q2_Index is Len // 2,
    Q3_Index is 3 * Len // 4,
    Q4_Index is Len,

    nth1(Q1_Index, SortedValues, Q1_Limit),
    nth1(Q2_Index, SortedValues, Q2_Limit),
    nth1(Q3_Index, SortedValues, Q3_Limit),
    nth1(Q4_Index, SortedValues, Q4_Limit).

quartile_limits_votes_log(Q1_Limit, Q2_Limit, Q3_Limit, Q4_Limit) :-
    findall(Value, movie_votes_log(_, Value), VotesValues),

    sort(VotesValues, SortedValues),

    length(SortedValues, Len),

    Q1_Index is Len // 4,
    Q2_Index is Len // 2,
    Q3_Index is 3 * Len // 4,
    Q4_Index is Len,

    nth1(Q1_Index, SortedValues, Q1_Limit),
    nth1(Q2_Index, SortedValues, Q2_Limit),
    nth1(Q3_Index, SortedValues, Q3_Limit),
    nth1(Q4_Index, SortedValues, Q4_Limit).

profit_bin(Movie, 'low') :-
    movie_profit_index(Movie, Value),
    quartile_limits(Q1_Limit, _, _, _),
    Value < Q1_Limit.

profit_bin(Movie, 'mid-low') :-
    movie_profit_index(Movie, Value),
    quartile_limits(Q1_Limit, Q2_Limit, _, _),
    Value >= Q1_Limit, Value < Q2_Limit.

profit_bin(Movie, 'mid-high') :-
    movie_profit_index(Movie, Value),
    quartile_limits(_, Q2_Limit, Q3_Limit, _),
    Value >= Q2_Limit, Value < Q3_Limit.

profit_bin(Movie, 'high') :-
    movie_profit_index(Movie, Value),
    quartile_limits(_, _, Q3_Limit, Q4_Limit),
    Value >= Q3_Limit, Value =< Q4_Limit.

cult_bin(Movie, 'low') :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(Q1_Limit, _, _, _),
    Value < Q1_Limit.

cult_bin(Movie, 'mid-low') :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(Q1_Limit, Q2_Limit, _, _),
    Value >= Q1_Limit, Value < Q2_Limit.

cult_bin(Movie, 'mid-high') :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(_, Q2_Limit, Q3_Limit, _),
    Value >= Q2_Limit, Value < Q3_Limit.

cult_bin(Movie, 'high') :-
    movie_cult_index(Movie, Value),
    quartile_limits_cult_index(_, _, Q3_Limit, Q4_Limit),
    Value >= Q3_Limit, Value =< Q4_Limit.

gross_bin(Movie, 'low') :-
    movie_gross_log(Movie, Value),
    quartile_limits_gross_log(Q1_Limit, _, _, _),
    Value < Q1_Limit.

gross_bin(Movie, 'mid-low') :-
    movie_gross_log(Movie, Value),
    quartile_limits_gross_log(Q1_Limit, Q2_Limit, _, _),
    Value >= Q1_Limit, Value < Q2_Limit.

gross_bin(Movie, 'mid-high') :-
    movie_gross_log(Movie, Value),
    quartile_limits_gross_log(_, Q2_Limit, Q3_Limit, _),
    Value >= Q2_Limit, Value < Q3_Limit.

gross_bin(Movie, 'high') :-
    movie_gross_log(Movie, Value),
    quartile_limits_gross_log(_, _, Q3_Limit, Q4_Limit),
    Value >= Q3_Limit, Value =< Q4_Limit.

budget_bin(Movie, 'low') :-
    movie_budget_log(Movie, Value),
    quartile_limits_budget_log(Q1_Limit, _, _, _),
    Value < Q1_Limit.

budget_bin(Movie, 'mid-low') :-
    movie_budget_log(Movie, Value),
    quartile_limits_budget_log(Q1_Limit, Q2_Limit, _, _),
    Value >= Q1_Limit, Value < Q2_Limit.

budget_bin(Movie, 'mid-high') :-
    movie_budget_log(Movie, Value),
    quartile_limits_budget_log(_, Q2_Limit, Q3_Limit, _),
    Value >= Q2_Limit, Value < Q3_Limit.

budget_bin(Movie, 'high') :-
    movie_budget_log(Movie, Value),
    quartile_limits_budget_log(_, _, Q3_Limit, Q4_Limit),
    Value >= Q3_Limit, Value =< Q4_Limit.

votes_bin(Movie, 'low') :-
    movie_votes_log(Movie, Value),
    quartile_limits_votes_log(Q1_Limit, _, _, _),
    Value < Q1_Limit.

votes_bin(Movie, 'mid-low') :-
    movie_votes_log(Movie, Value),
    quartile_limits_votes_log(Q1_Limit, Q2_Limit, _, _),
    Value >= Q1_Limit, Value < Q2_Limit.

votes_bin(Movie, 'mid-high') :-
    movie_votes_log(Movie, Value),
    quartile_limits_votes_log(_, Q2_Limit, Q3_Limit, _),
    Value >= Q2_Limit, Value < Q3_Limit.

votes_bin(Movie, 'high') :-
    movie_votes_log(Movie, Value),
    quartile_limits_votes_log(_, _, Q3_Limit, Q4_Limit),
    Value >= Q3_Limit, Value =< Q4_Limit.
