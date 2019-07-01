include "globals.mzn";

int: nbTeams;
array[int, int] of int: pv;
set of int: WIDTH = 1..(nbTeams);
set of int: HEIGHT = 1..(nbTeams-1);
array[HEIGHT, WIDTH] of var 1..nbTeams: roundRobin;

predicate col_diff(int: col) =
  all_different(row in HEIGHT)(roundRobin[row,col]);
  
predicate row_diff(int: row) = 
  all_different(col in WIDTH)(roundRobin[row,col]);
  
predicate coherent_match(int: team, int:row) =
  team = roundRobin[row,roundRobin[row,team]];

predicate coherent_round(int: row) =
  forall(team in WIDTH)(coherent_match(team,row));
  
predicate not_four_successive(int: col, int: row) = 
  0 != pv[roundRobin[row,col],col]+
       pv[roundRobin[row+1,col],col]+
       pv[roundRobin[row+2,col],col]+
       pv[roundRobin[row+3,col],col] /\
  4 != pv[roundRobin[row,col],col]+
       pv[roundRobin[row+1,col],col]+
       pv[roundRobin[row+2,col],col]+
       pv[roundRobin[row+3,col],col];
  
predicate matches_home_away(int: team) = 
  forall(row in 1..nbTeams-4)(not_four_successive(team, row));
  
predicate not_against_yourself(int: team) = 
  forall(row in HEIGHT)(team != roundRobin[row,team]);
  
constraint forall(col in WIDTH)
  (col_diff(col));
  
constraint forall(row in HEIGHT)
  (row_diff(row));

constraint forall(row in HEIGHT)
  (coherent_round(row));
  
constraint forall(col in WIDTH)
  (not_against_yourself(col));
  
constraint forall(col in WIDTH)
  (matches_home_away(col));
  
% contrainte pour briser la symetrie (break symmetry)
constraint (roundRobin[1, 1] < roundRobin[nbTeams-1, 1]);

solve satisfy;

output ["roundRobin = ", show(roundRobin), "\n"];