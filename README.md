# Chopsticks

This Python programmes learns the winning strategy for the game chopsticks through reinforcement learning by Monte Carlo Tree Search (MCTS). 

MCTS does not require an evaluation metric for the nodes and does not require a full tree search which makes trees with large branching factors no longer intractable. 
Tic tac toe is often used as the default game to illustrate a simple Monte Carlo Tree Search. 
In this case, however, we will be using another simple game, chopsticks. 
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

## Terms Used
* Actioner: the player who has the current turn i.e. the player making the move
* Receiver: the player who does not have the current turn i.e. the player waiting for opponent to make a move 

## MCTS
## State Representation
The state of the game is represented by 4 numbers and a letter ('C' or 'P'): the letter 'C' represents that the computer's turn while 'P' represents player's turn. The numbers represents the number of fingers on the players' hands. The first two represent the hand configuration for the actioner while the last two represent the hand configuration for the receiver. Since the state is not dependent on the side of the hand i.e. 2 fingers on the left hand and 3 fingers on the right hand is the same state as 2 fingers on the right hand and 3 fingers on the left hand, the pair of numbers representing the hand configuration is ordered numerically by convention. Below are some examples to illustrate this more clearly:

* `1322C`: Computer has 1 and 3 fingers; Player has 2 fingers on each hand; Computer to make a move
* `0133P`: Player has 1 finger on one hand, other hand is dead; Computer has 3 fingers on each hand; Player to make a move
* `0003C`: Computer has no fingers left; Player has 3 fingers on one hand, other hand is dead; Game over since computer has no fingers left by default

## Hands
As mentioned above, the state of the game is represented by a binary variable representing whose turn it is and the state of the two players' hands. 
The class `Hands` represents the state of a player's hand. It contains two numbers representing the number of fingers on the two hands.
Since the side of a hand (left or right) does not matter in representing a `Hands` state, the two hands are distinguished by comparing its numeric value
i.e. add 2 to the hand with fewer fingers (`self.smaller`) instead of add 2 to the left hand
The class has a function `addFingers` function which enables players to make moves. 
Following the game rules, if the number of fingers on a hand exceed 4, it is killed i.e. number of fingers become 0.

## Node
The class `Node` is used to represent a Monte Carlo Tree node. A `Node` instance contains the following information:
* Total number of games played: number of times node has been visited
* Number of games won: number of visits of which the outcome was a win *(Note: A draw is considered as 0.5)*
* Game state: whose turn it is as well as the computer's and player's `Hands` state i.e. how many fingers on each hand for both players
* Children nodes: nodes that have game states which can be reached from the current node according to game rules e.g. `1111C` to `1112P` and `1211P`

A `Node` instance can also run a round of MCTS. To instantiate the nodes, use the function `create_nodes`. This gives a dictionary of all possible nodes. Note that these nodes are still untrained i.e. no MCTS has been carried out yet.

To get trained nodes instead, use the function `get_trained_nodes`. This function uses `create_nodes` to instantiate the nodes and then trains the nodes by running MCTS 1000 times for each node using the `run_MCTS` function. The details of the MCTS implementation are described in more details below:

*With reference to [Wikipedia](https://en.wikipedia.org/wiki/Chopsticks_(hand_game))*
* Selection: starting from the root `Node` instance **R**, select successive child nodes using the Upper Confidence Bound 1 (UCB1) policy until a leaf `Node` instance ***L** is reached
    * Root node refers to the current game state
    * Leaf node is any node where the game ends/has a potential child from which no simulation (playout) has yet been initiated
* Expansion: if **L** does not represent a game state where the game is over, choose a node **C** from the children of **L** randomly
* Simulation: randomly choose moves from the game state in **C** until an end state (win/draw/lose) is reached i.e. a random playout/rollout 
* Backpropagation: depending on the outcome of the end state, update the nodes on the path from **C** to **R** accordingly
    * If computer **wins**, add 1 to `total_num_games` and `num_games_won` both for nodes on the path
    * If computer **draws**, add 1 to `total_num_games` and 0.5 to `num_games_won` for nodes on the path
    * If computer **loses**, add 1 to `total_num_games` for nodes on the path

## Interaction Modes
### Playing against computer
To play against the computer, use the function `play()`. The default is for the computer to start first. To start, set the argument `computer_starts=False` i.e.
use the function `play(computer_starts=False)` instead. 

You can also choose whether to let the computer train on the spot while the game is being played or train the nodes before hand by passing the trained nodes into the function (`play(trained_nodes=nodes)`).The computer is trained on the spot by default. 
Training beforehand can take some time as opposed to training on the spot since we have to train starting from all possible game states.

### Retrieving best strategy for all moves
To get the best moves to take for each possible state, use the function `getBestStrategy()`. A table will be printed with the column representing the state of the actioner 
and the row representing the state of the receiver. The cell value shows the best move to play for the actioner, together with the worst possible outcome among all possible playouts from that state. To get the worst possible outcome, we parse through the possible paths. As such, if you are insterested in getting the strategy only, the argument `show_move_only` can be set to `True` for faster runtime.

### View possible outcomes
To view all possible outcomes given the trained computer nodes and a starting game state, use the function `showAllOutcomes(TRAINED_NODES, STARTING_STATE)`.
The outcomes assume that the computer makes the best move, but the player (opponent) is allowed to make any move. 
A list of all possible oucomes (end game states) will be shown and you can pick a specific outcome to see all paths leading to that particular end game state.
If you choose the starting state `1111C`, you should see that all outcomes is a winning state which means that the computer will always win.
As such, this means that the person who starts a chopsticks game is guaranteed to win.