include "globals.mzn";

% Colors
var 1..5: red;
var 1..5: green;
var 1..5: white;
var 1..5: yellow;
var 1..5: blue;

% Nationalities
var 1..5: norway;
var 1..5: spain;
var 1..5: japan;
var 1..5: england;
var 1..5: ukraine;

% Animals
var 1..5: horse;
var 1..5: dog;
var 1..5: snails;
var 1..5: fox;
var 1..5: zebra;

% Drinks
var 1..5: tea;
var 1..5: coffee;
var 1..5: milk;
var 1..5: juice;
var 1..5: water;

% Jobs
var 1..5: sculptor;
var 1..5: diplomat;
var 1..5: violonist;
var 1..5: physician;
var 1..5: acrobat;

% Constraints
constraint alldifferent([england, spain, japan, norway, ukraine]);
constraint alldifferent([dog, snails, zebra, horse, fox]);
constraint alldifferent([diplomat, acrobat, violonist, physician, sculptor]);
constraint alldifferent([red, blue, green, white, yellow]);
constraint alldifferent([tea, coffee, milk, juice, water]);

constraint england = red;
constraint spain = dog;
constraint green = coffee;
constraint ukraine = tea;
constraint green = white + 1;
constraint sculptor = snails;
constraint diplomat = yellow;
constraint milk = 3;
constraint norway = 1;
constraint abs(physician - fox) = 1;
constraint abs(diplomat - horse) = 1;
constraint violonist = juice;
constraint japan = acrobat;
constraint abs(norway - blue) = 1;

solve satisfy;
