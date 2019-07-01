ask(X):- 
    format('~w? ', [X]), 
    read(Reply), 
    Reply = 'yes'.

% rules
object(X):-
    ask('Is the object used for food'), kitchen(X).
object(X):-
    ask('Is the object used for cleaning'), cleaning(X).
object(X):-
    ask('Is the object something you carry on yourself (except electronics)'), personnal(X).
object(X):-
    ask('Is the object an electronics'), electronics(X).
object(X):-
    ask('Is the object found in the office'), office(X).
object(X):-
    ask('Is the object a decoration'), decoration(X).
object(X):-
    ask('Is the object a musical instrument'), music_instruments(X).
object(X):-
    ask('Is the object found in the bathroom'), bathroom(X).
object(X):-
    ask('Is the object found in the bedroom'), bedroom(X).

kitchen(X):-
    ask('Is the object an appliance'), appliance(X).
kitchen(X):-
    ask('Is the object part of kitchenware'), kitchenware(X).
appliance(X):-
    ask('Is the object used to cook food'), cook_food(X).
appliance(X):-
    ask('Is the object used to warm up liquids'), heat_liquids(X).
kitchenware(X):-
    ask('Is the object used for cooking'), cooking(X).
kitchenware(X):-
    ask('Is the object used for eating'), eating(X).
cook_food(X):-
    ask('Is the object used to bake a cake'), cake(X).
cook_food(X):-
    ask('Is the object used to toast bread'), toast(X).
heat_liquids(X):-
    ask('Is the object used to boil water'), boil(X).
heat_liquids(X):-
    ask('Is the object used to make hot beverages'), hot(X).
eating(X):-
    ask('Is the object made of tines'), fork(X).
eating(X):-
    ask('Is the object used to place food'), plate(X).

cleaning(X):-
    ask('Is the object used to get rid of dust'), dust(X).
cleaning(X):-
    ask('Is the object used within an appliance'), appliance_cleaner(X).
dust(X):-
    ask('Is the object used for brushing'), brush(X).
dust(X):-
    ask('Does the object use an air pump'), air_pump(X).

personnal(X):-
    ask('Is the object used to store stuff'), stuff(X).
personnal(X):-
    ask('Is the object used to lock something'), house(X).
stuff(X):-
    ask('Can the object fit in your pocket'), pocket(X).
stuff(X):-
    ask('Is the object a big bag'), bag(X).

office(X):-
    ask('Is the object used to write on'), paper(X).
office(X):-
    ask('Is the object a piece of furniture'), furniture(X).

decoration(X):-
    ask('Can the object emit light'), light(X).
decoration(X):-
    ask('Is the object a plant'), plant(X).

electronics(X):-
    ask('Is the object used to call people'), call_people(X).
electronics(X):-
    ask('Is the object used to go on the Internet at home'), internet(X).

% kb
cake(oven).
toast(toaster).
boil(range).
hot(coffee_machine).
fork(fork).
plate(plate).
cooking(pan).
appliance_cleaner(dishwashing_detergent).
brush(broom).
air_pump(vacuum).
house(key).
pocket(wallet).
bag(backpack).
paper(paper).
furniture(desk).
light(lamp).
plant(cactus).
internet(computer).
call_people(phone).
music_instruments(piano).
bathroom(shampoo).
bedroom(bed).