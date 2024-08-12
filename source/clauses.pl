num_films_by_star(Star, Count) :-
	findall(Movie, (star(Movie, Star), Movies),
	length(Movies, Count)).


num_films_by_director(Director, Count) :-
	findall(Movie, (director(Movie, Director), Movies),
	length(Movies, Count)).


average_profit_by_star(Star, AvgProfit) :-
	findall(Profit, (star(Movie, Star), film_profit_index(Movie, Profit)), Profits),
	length(Profits, N),
	sum_list(Profits, Total),
	AvgProfit is Total / N.


average_score_by_star(Star, AvgScore) :-
	findall(Score, (star(Movie, Star), score(Movie, Score)), Scores),
	length(Scores, N),
	sum_list(Scores, Total),
	AvgScore is Total / N.


std_dev_profit_by_star(Star, StdDev) :-
	average_profit_by_star(Star, AvgProfit),
	findall(Profit, (star(Movie, Star), film_profit_index(Movie, Profit)), Profits),
	length(Profits, N),
	sum_list(Profits, Total),
	sum_of_squares(Profits, Total, AvgProfit, SumSq),
	Variance is SumSq / (N - 1),
	StdDev is sqrt(Variance).


std_dev_score_by_star(Star, StdDev) :-
	average_score_by_star(Star, AvgScore),
	findall(Score, (star(Movie, Star), score(Movie, Score)), Scores),
	length(Scores, N),
	sum_list(Scores, Total),
	sum_of_squares(Scores, Total, AvgScore, SumSq),
	Variance is SumSq / (N - 1),
	StdDev is sqrt(Variance).


sum_of_squares([], _, _, 0).
	sum_of_squares([Value|Rest], Total, Avg, SumSq) :-
	SumSqRest is SumSq + (Value - Avg)^2,
	sum_of_squares(Rest, Total, Avg, SumSqRest).


average_profit_by_director(Director, AvgProfit) :-
	findall(Profit, (director(Movie, Director), film_profit_index(Movie, Profit)), Profits),
	length(Profits, N),
	sum_list(Profits, Total),
	AvgProfit is Total / N.


average_score_by_director(Director, AvgScore) :-
	findall(Score, (director(Movie, Director), score(Movie, Score)), Scores),
	length(Scores, N),
	sum_list(Scores, Total),
	AvgScore is Total / N.


std_dev_profit_by_director(Director, StdDev) :-
	average_profit_by_director(Director, AvgProfit),
	findall(Profit, (director(Movie, Director), film_profit_index(Movie, Profit)), Profits),
	length(Profits, N),
	sum_list(Profits, Total),
	sum_of_squares(Profits, Total, AvgProfit, SumSq),
	Variance is SumSq / (N - 1),
	StdDev is sqrt(Variance).


std_dev_score_by_director(Director, StdDev) :-
	average_score_by_director(Director, AvgScore),
	findall(Score, (director(Movie, Director), score(Movie, Score)), Scores),
	length(Scores, N),
	sum_list(Scores, Total),
	sum_of_squares(Scores, Total, AvgScore, SumSq),
	Variance is SumSq / (N - 1),
	StdDev is sqrt(Variance).


director_profit_index(Director, Ratio) :-
	std_dev_profit_by_director(Director, StdDev),
	average_profit_by_director(Director, AvgProfit),
	Ratio is AvgProfit / StdDev.

director_score_index(Director, Ratio) :-
	std_dev_score_by_director(Director, StdDev),
	average_score_by_director(Director, AvgScore),
	Ratio is AvgScore / StdDev.

star_profit_index(Star, Ratio) :-
	std_dev_profit_by_star(Star, StdDev),
	average_profit_by_star(Star, AvgProfit),
	Ratio is AvgProfit / StdDev.

star_score_index(Star, Ratio) :-
	std_dev_score_by_star(Star, StdDev),
	average_score_by_star(Star, AvgScore),
	Ratio is AvgScore / StdDev.

film_profit_index(Movie, Ratio) :-
	gross(Movie, Gross),
	budget(Movie, Budget),
	Budget > 0,
	Ratio is (Gross / Budget).

film_success_index(Movie, Ratio) :-
	film_profit_index(Movie, ProfitIndex),
	gross(Movie, Gross),
	Ratio is ProfitIndex * log(Gross).

film_cult_index(Movie, Ratio) :-
	votes(Movie, Votes),
	score(Movie, Score),
	Ratio is Score * log(Votes).

avg_score(AvgScore) :-
	findall(Score, score(movie(_), Score), Scores),
	length(Scores, N),
	N > 0,
	sum_list(Scores, TotalScore),
	AvgScore is TotalScore / N.

std_dev_score(StdDev) :-
	avg_score(AvgScore),
	findall(Score, score(movie(_), Score), Scores),
	length(Scores, N),
	N > 1,
	sum_list(Scores, TotalScore),
	sum_of_squares(Scores, TotalScore, AvgScore, SumSq),
	Variance is SumSq / (N - 1),
	StdDev is sqrt(Variance).

film_quality(Movie, 'bassa') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(Movie, Score),
	Score =< AvgScore - StdDev.

film_quality(Movie, 'media') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(Movie, Score),
	Score > AvgScore - StdDev,
	Score =< AvgScore + StdDev.

film_quality(Movie, 'alta') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(Movie, Score),
	Score > AvgScore + StdDev.

remap_genre(Movie, 'Others') :-
	genre(Movie, Genre),
	member(Genre, ['Mystery', 'Thriller', 'Sci-Fy', 'Family', 'Romance', 'Western']).

remap_genre(Movie, Genre) :-
	genre(Movie, Genre),
	\+ member(Genre, ['Mystery', 'Thriller', 'Sci-Fy', 'Family', 'Romance', 'Western']).
