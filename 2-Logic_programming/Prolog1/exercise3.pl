cours(inf1005c).
cours(inf1500).
cours(inf1010).
cours(log1000).
cours(inf1600).
cours(inf2010).
cours(log2410).
cours(log2810).
cours(mth1007).
cours(log2990).
cours(inf2705).
cours(inf2205).
cours(inf1900).

prerequis_direct(inf1005c,inf1010).
prerequis_direct(inf1005c,log1000).
prerequis_direct(inf1005c,inf1600).

prerequis_direct(inf1500,inf1600).

prerequis_direct(inf1010,inf2010).
prerequis_direct(inf1010,log2410).

prerequis_direct(log1000,log2410).

prerequis_direct(inf2010,inf2705).
prerequis_direct(mth1007,inf2705).

corequis_direct(log2990,inf2705).
corequis_direct(log2810,inf2010).

corequis_direct(log1000,inf1900).
corequis_direct(inf1600,inf1900).
corequis_direct(inf2205,inf1900).


corequis(X, Y):-
    corequis_direct(X,Y) ; 
    corequis_direct(Y,X),
    X \== Y.


corequis(X,Y):-
    corequis_direct(X,Z), corequis_direct(Z,Y);
    corequis_direct(X,Z), corequis_direct(Y,Z);
    corequis_direct(Z,X), corequis_direct(Z,Y);
    corequis_direct(Z,X), corequis_direct(Y,Z),
    X \== Y.

prealable(X,Y):-
    prerequis_direct(X,Y).

prealable(X,Y):-
    prerequis_direct(X,Z),
    prealable(Z,Y).

prealable(X,Y):-
    corequis(X, Y),
    X \== Y.

prealable(X,Y):-
    corequis(X,Z),
    prerequis_direct(Z,Y),
    X \== Y.

prealable(X,Y):-
    corequis(X,Z),
    prerequis_direct(Z,A),
    corequis(A,Y),
    X \== Y.
    
completeRequirementsFor(X):-
    findall(Y, prealable(Y,X), ResultSet),
    sort(ResultSet, Result),
    print(Result).


