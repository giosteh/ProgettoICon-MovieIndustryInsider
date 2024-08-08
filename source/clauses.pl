num_films_artist(Artist, Count) :-
	findall(Movie, (star(movie(Movie), Artist) ; director(movie(Movie), Artist) ; writer(movie(Movie), Artist)), Movies),
	length(Movies, Count).

num_films_directed(Director, Count) :-
	findall(Movie, director(movie(Movie), Director), Movies),
	length(Movies, Count).

num_films_starred(Star, Count) :-
	findall(Movie, star(movie(Movie), Star), Movies),
	length(Movies, Count).

num_films_written(Writer, Count) :-
	findall(Movie, writer(movie(Movie), Writer), Movies),
	length(Movies, Count).

avg_budget_directed(Director, AvgBudget) :-
	findall(Budget, (director(movie(Movie), Director), budget(movie(Movie), Budget)), Budgets),
	length(Budgets, Count),
	sum_list(Budgets, TotalBudget),
	AvgBudget is TotalBudget / Count.

avg_gross_directed(Director, AvgGross) :-
	findall(Gross, (director(movie(Movie), Director), gross(movie(Movie), Gross)), Grosses),
	length(Grosses, Count),
	sum_list(Grosses, TotalGross),
	AvgGross is TotalGross / Count.

gross_to_budget_ratio(Movie, Ratio) :-
	gross(movie(Movie), Gross),
	budget(movie(Movie), Budget),
	Budget > 0, % Evita la divisione per zero
	Ratio is (Gross / Budget).

film_success(Movie, Success) :-
	max_gross(MaxGross),
	MaxGross > 0,
	gross(movie(Movie), Gross),
	budget(movie(Movie), Budget),
	Budget > 0,
	gross_to_budget_ratio(Movie, Ratio),
	NormalizedGross is Gross / MaxGross,
	Success is 0.7 * NormalizedGross + 0.3 * Ratio.

max_gross(MaxGross) :-
	findall(Gross, gross(movie(_), Gross), Grosses),
	max_list(Grosses, MaxGross).

film_success_category(Movie, 'basso') :-
	film_success_normalized(Movie, Success),
	Success =< 0.4.

film_success_category(Movie, 'medio') :-
	film_success_normalized(Movie, Success),
	Success > 0.4,
	Success =< 0.7.

film_success_category(Movie, 'alto') :-
	film_success_normalized(Movie, Success),
	Success > 0.7.

avg_weighted_score(AvgScore) :-
	findall(Score, weighted_score(movie(_), Score), Scores),
	length(Scores, Count),
	Count > 0, % Assicurati che ci siano score da calcolare
	sum_list(Scores, TotalScore),
	AvgScore is TotalScore / Count.

std_dev_weighted_score(StdDev) :-
	avg_score(AvgScore),
	findall(Score, weighted_score(movie(_), Score), Scores),
	length(Scores, Count),
	Count > 1,
	sum_list(Scores, TotalScore),
	sum_of_squares(Scores, TotalScore, AvgScore, SumSq),
	Variance is SumSq / (Count - 1),
	StdDev is sqrt(Variance).

sum_of_squares([], _, _, 0).
	sum_of_squares([Score|Rest], TotalScore, AvgScore, SumSq) :-
	SumSqRest is SumSq + (Score - AvgScore)^2,
	sum_of_squares(Rest, TotalScore, AvgScore, SumSqRest).

max_votes(MaxVotes) :-
	findall(Votes, votes(movie(_), Votes), VotesList),-
	max_list(VotesList, MaxVotes).

film_quality(Movie, 'bassa') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(movie(Movie), Score),
	Score =< AvgScore - StdDev.

film_quality(Movie, 'media') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(movie(Movie), Score),
	Score > AvgScore - StdDev,
	Score =< AvgScore + StdDev.

film_quality(Movie, 'alta') :-
	avg_score(AvgScore),
	std_dev_score(StdDev),
	score(movie(Movie), Score),
	Score > AvgScore + StdDev.

avg_score(AvgScore) :-
	findall(Score, score(movie(_), Score), Scores),
	length(Scores, Count),
	Count > 0,
	sum_list(Scores, TotalScore),
	AvgScore is TotalScore / Count.

std_dev_score(StdDev) :-
	avg_score(AvgScore),
	findall(Score, score(movie(_), Score), Scores),
	length(Scores, Count),
	Count > 1,
	sum_list(Scores, TotalScore),
	sum_of_squares(Scores, TotalScore, AvgScore, SumSq),
	Variance is SumSq / (Count - 1),
	StdDev is sqrt(Variance).

film_quality_category(Movie, 'bassa') :-
	avg_weighted_score(AvgScore),
	std_dev_weighted_score(StdDev),
	score(movie(Movie), Score),
	Score =< AvgScore - StdDev.

film_quality_category(Movie, 'media') :-
	avg_weighted_score(AvgScore),
	std_dev_weighted_score(StdDev),
	score(movie(Movie), Score),
	Score > AvgScore - StdDev,
	Score =< AvgScore + StdDev.

film_quality_category(Movie, 'alta') :-
	avg_weighted_score(AvgScore),
	std_dev_weighted_score(StdDev),
	score(movie(Movie), Score),
	Score > AvgScore + StdDev.

avg_votes(AvgVotes) :-
	findall(Votes, votes(movie(_), Votes), VotesList),
	length(VotesList, Count),
	Count > 0,
	sum_list(VotesList, TotalVotes),
	AvgVotes is TotalVotes / Count.

std_dev_votes(StdDevVotes) :-
	avg_votes(AvgVotes),
	findall(Votes, votes(movie(_), Votes), VotesList),
	length(VotesList, Count),
	Count > 1,
	sum_list(VotesList, TotalVotes),
	sum_of_squares(VotesList, AvgVotes, SumSq),
	Variance is SumSq / (Count - 1),
	StdDevVotes is sqrt(Variance).

vote_category(Movie, 'poco') :-
	avg_votes(AvgVotes),
	std_dev_votes(StdDevVotes),
	votes(movie(Movie), Votes),
	Votes =< AvgVotes - StdDevVotes.

vote_category(Movie, 'medio') :-
	avg_votes(AvgVotes),
	std_dev_votes(StdDevVotes),
	votes(movie(Movie), Votes),
	Votes > AvgVotes - StdDevVotes,
	Votes =< AvgVotes + StdDevVotes.

vote_category(Movie, 'molto') :-
	avg_votes(AvgVotes),
	std_dev_votes(StdDevVotes),
	votes(movie(Movie), Votes),
	Votes > AvgVotes + StdDevVotes.

weighted_score(Movie, AdjustedScore) :-
	% Ottieni lo score iniziale del film
	score(movie(Movie), Score),
	% Ottieni la categoria di voto del film
	vote_category(Movie, VoteCategory),
	% Ottieni la qualita del film
	film_quality(Movie, Quality),
	% Aggiusta lo score in base alla categoria di voto e alla qualità
	(
		VoteCategory = 'molto',
		(
			Quality = 'alta' -> AdjustedScore is min(10, Score + 0.3) ; % Alza leggermente lo score se qualità alta
			Quality = 'bassa' -> AdjustedScore is max(0, Score - 0.3) ; % Abbassa leggermente lo score se qualità bassa
			Quality = 'media' -> AdjustedScore is Score % Lascia invariato lo score se qualità media
		);
		VoteCategory = 'poco',
		(
			Quality = 'alta' -> AdjustedScore is max(0, Score - 0.3) ; % Abbassa leggermente lo score se qualità alta
			Quality = 'bassa' -> AdjustedScore is min(10, Score + 0.3) ; % Alza leggermente lo score se qualità bassa
			Quality = 'media' -> AdjustedScore is Score % Lascia invariato lo score se qualità media
		);
		VoteCategory = 'medio',
	% Lascia invariato lo score se il voto è medio
		AdjustedScore is Score
	)
