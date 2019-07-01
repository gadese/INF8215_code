ask(X) :- 
    format('~w? ', [X]), 
    read(Reply), 
    Reply = 'yes'.
    
ask(X, Y) :- 
    format('~w ~w? ', [X, Y]), 
    read(Reply), 
    Reply = 'yes'.
    
% RULES
person(X) :- ask('Are they an artist'), artist(X).
person(X) :- ask('Are they a politician'), politician(X).
person(X) :- ask('Are they a religious figure'), religious_figure(X).
person(X) :- ask('Are they a fictional character'), fictitious_character(X).
person(X) :- ask('Are they a car racer'), car_racer(X).

artist(X) :- ask('Are they an author'), author(X).
artist(X) :- ask('Are they a singer'), singer(X).
artist(X) :- ask('Are they in cinema'), in_cinema(X).
artist(X) :- ask('Do they make video games'), video_game_maker(X).
artist(X) :- ask('Are they a street artist'), drawer(X).

author(X) :- author_from(X, Y), country(Y), ask('Are they from', Y).
singer(X) :- ask('Are they female'), singer_of_gender(X, female).
singer(X) :- singer_of_gender(X, male).
in_cinema(X) :- ask('Are they an actor/actress'), acting(X).
in_cinema(X) :- ask('Are they a movie writer'), movie_writer(X).
acting(X) :- ask('Are they female'), acting_of_gender(X, female).
acting(X) :- acting_of_gender(X, male).

politician(X) :- ask('Were they born pre XXe century'), politician_born(X, preXX).
politician(X) :- politician_born(X, postXX).
politician_born(X, Z) :- governed(X, Y, Z), country(Y), ask('Did they govern', Y).

religious_figure(X) :- ask('Are they a pope'), pope(X).
religious_figure(X) :- ask('Are they in bible'), in_bible(X).
in_bible(X) :- ask('Are they in the new testament'), new_testament_figure(X).
in_bible(X) :- old_testament_figure(X).

fictitious_character(X) :- ask('Are they a video game character'), video_game_character(X).
fictitious_character(X) :- ask('Are they a movie character'), movie_character(X).
video_game_character(X) :- ask('Are they female'), female_video_game_character(X).
video_game_character(X) :- male_video_game_character(X).

car_racer(X) :- car_racer_from(X, Y), country(Y), ask('Are they from', Y).

% PEOPLE LIST
author_from(j_k_rowling, united_kingdom).
author_from(victor_hugo, france).
singer_of_gender(lady_gaga, female).
singer_of_gender(michael_jackson, male).
acting_of_gender(jennifer_lawrence, female).
acting_of_gender(denzel_washington, male).
movie_writer(quentin_tarantino).
video_game_maker(hideo_kojima).
drawer(bansky).
governed(richard_nixon, usa, postXX).
governed(dwight_d_eisenhower, usa, preXX).
governed(joseph_staline, ussr, preXX).
governed(mikhail_gorbatchev, ussr, postXX).
governed(cleopatra, egypt, preXX).
pope(pope_francis).
new_testament_figure(jesus).
old_testament_figure(moses).
male_video_game_character(mario).
female_video_game_character(lara_croft).
movie_character(james_bond).
car_racer_from(ayrton_senna, brazil).
car_racer_from(fernando_alonso, spain).
country(united_kingdom).
country(france).
country(usa).
country(ussr).
country(egypt).
country(brazil).
country(spain).