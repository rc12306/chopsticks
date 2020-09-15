# chopsticks

This Python programmes learns the winning strategy for the game chopsticks through reinforcement learning by Monte Carlo Tree Search. 

MCTS does not require an evaluation metric for the nodes and does not require a full tree search which makes trees with large branching factors no longer intractable. 
Tic tac toe is usually used as the default game to illustrate a simple Monte Carlo Tree Search. In this case, however, another simple game, chopsticks will be used instead. 
The rules of the games are as follows:

> * Each player begins with one finger raised on each hand. After the first player, turns proceed clockwise.
> * On a player's turn, they must either attack or split, but not both.
> * To attack, a player uses one of their live hands to strike an opponent's live hand. The number of fingers on the opponent's struck hand will increase by the number of fingers on the hand used to strike.
> * To split, a player strikes their own two hands together, and transfers raised fingers from one hand to the other as desired. A move is not allowed to simply reverse one's own hands (e.g., going [2, 3] → [3, 2] is prohibited) and they must not kill one of their own hands (e.g., going [1, 1] → [0, 2] is prohibited).
> * If any hand of any player reaches exactly five fingers, then the hand is killed, and this is indicated by raising zero fingers (i.e. a closed fist).
> * A player may revive their own dead hand using a split, as long as they abide by the rules for splitting. However, players may not revive opponents' hands using an attack. Therefore, a player with two dead hands can no longer play and is eliminated from the game.
> * If any hand of any player reaches more than five fingers, then five fingers are subtracted from that hand. For instance, if a 4-finger hand strikes a 2-finger hand, for a total of 6 fingers, then 5 fingers are automatically subtracted, leaving 1 finger. Under alternate rules, when a hand reaches 5 fingers and above it is considered a "dead hand".
> * A player wins once all opponents are eliminated (by each having two dead hands at once).
> * A player can kill their own hand.

*Taken from [Wikipedia](https://en.wikipedia.org/wiki/Chopsticks_(hand_game))*

## MCTS
## State Representation
## Interaction Modes
### Playing against computer
### Retrieving best strategy for all moves