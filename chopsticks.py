import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import textwrap

class Hands:
    def __init__(self, a, b):
        self.__setfinger(a,b)

    def __eq__(self, value):
        return self.smaller == value.smaller and self.larger == value.larger

    def __setfinger(self, a, b):
        a = 0 if a > 4 else a
        b = 0 if b > 4 else b
        if a < b:
            self.smaller = a
            self.larger = b
        else:
            self.smaller = b
            self.larger = a
        
    def addFingers(self, numFingersToAdd, addToSmaller):
        if addToSmaller:
            self.__setfinger(self.smaller + numFingersToAdd, self.larger)
        else:
            self.__setfinger(self.smaller, self.larger + numFingersToAdd)
    
    def alive(self):
        # game over when both hands are 0
        return self.larger != 0

    def __str__(self):
        return "Hands ({}, {})".format(self.smaller, self.larger)

class Node:
    def __init__(self, own_hands, opponent_hands, num_games_won, total_num_games, my_turn):
        self.my_turn = my_turn
        self.own_hands = own_hands
        self.opponent_hands = opponent_hands
        self.num_games_won = num_games_won
        self.total_num_games = total_num_games
        self.children = None
        player = "C" if my_turn else "P"
        self.state_name = str(own_hands.smaller)+str(own_hands.larger)+str(opponent_hands.smaller)+str(opponent_hands.larger)+player


    def __eq__(self, value):
        if self.my_turn != value.my_turn:
            return False
        if self.own_hands != value.own_hands or self.opponent_hands != value.opponent_hands:
            return False
        # if self.num_games_won != value.num_games_won or self.total_num_games != value.total_num_games:
        #     return False
        return True 

    def __str__(self):
        turn = "COMPUTER'S TURN" if self.my_turn else "PLAYER'S TURN" 
        return "{} \t Computer: {} \t Player: {} \t Total number of visits: {} \t Number of games won:{}".format(turn, self.own_hands, self.opponent_hands, self.total_num_games, self.num_games_won)

    @property
    def score(self):
        if self.total_num_games == 0:
            return -1
        return self.num_games_won/self.total_num_games
    
    # Actioner refers to the hands of the player making the move
    @property
    def actioner(self):
        if self.my_turn:
            return self.own_hands
        else:
            return self.opponent_hands

    # Receiver refers to the hands of the player that iw waiting for the opponent to make a move
    @property
    def receiver(self):
        if self.my_turn:
            return self.opponent_hands
        else:
            return self.own_hands

    def updateScore(self, points_won):
        self.total_num_games += 1
        self.num_games_won += points_won
        # if self.own_hands == Hands(0, 1) and self.opponent_hands.larger == 4 and not self.my_turn:
        #     print(self)

    def gameOver(self):
        return not self.own_hands.alive() or not self.opponent_hands.alive()

    @staticmethod
    def getStateName(actioner, receiver, my_turn):
        if my_turn:
            # computer is actioner
            return str(actioner.smaller)+str(actioner.larger)+str(receiver.smaller)+str(receiver.larger)+"C"
        else:
            # computer is receiver
            return str(receiver.smaller)+str(receiver.larger)+str(actioner.smaller)+str(actioner.larger)+"P"

    def getChildrenNodes(self, nodes):
        if self.gameOver():
            return []
        '''
            ============== legal moves ==============
            Given (A, B) representing number of fingers where a <= b:
            1. Add A to B
            2. Add B to A
            3. Add A to opponent's A
            4. Add A to opponent's B
            5. Add B to opponent's A
            6. Add B to opponent's B
            7. Transfer some from B to A (but without reversal/killing own hand)
        '''
        legal_moves = []
        # Move 1: Add A to B
        # print("Actioner:", self.actioner, "Receiver:", self.receiver)
        player = "C" if not self.my_turn else "P"
        if self.actioner.smaller != 0 and self.actioner.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            actioner_new_hands.addFingers(self.actioner.smaller, False)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            # print("Add A to B", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 2: Add B to A
        if self.actioner.smaller != 0 and self.actioner.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            actioner_new_hands.addFingers(self.actioner.larger, True)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            # print("Add B to A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 3: Add A to opponent's A
        if self.actioner.smaller != 0 and self.receiver.smaller != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.smaller, True)
            # print("Add A to opponent's A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 4: Add A to opponent's B
        if self.actioner.smaller != 0 and self.receiver.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.smaller, False)
            # print("Add A to opponent's B", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 5: Add B to opponent's A
        if self.actioner.larger != 0 and self.receiver.smaller != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.larger, True)
            # print("Add B to opponent's A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 6: Add B to opponent's B
        if self.actioner.larger != 0 and self.receiver.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.larger, False)
            # print("Add B to opponent's B", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # Move 7: Transfer some from B to A (but without reversal/killing own hand)
        # if self.actioner.smaller != 0 and self.actioner.larger != 0:
        receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
        for i in range(1, self.actioner.larger):
            A = self.actioner.smaller + i
            B = self.actioner.larger - i
            if not (A == self.actioner.larger and B == self.actioner.smaller) and A <= 4: # no reversal and no killing
                actioner_new_hands = Hands(A, B)
                # print("Transfer some from B to A (but without reversal/killing own hand)", actioner_new_hands, receiver_new_hands)
                legal_moves.append(nodes[Node.getStateName(receiver_new_hands, actioner_new_hands, not self.my_turn)])
        # getting unique states 
        set_moves = []
        for move in legal_moves:
            if move not in set_moves:
                set_moves.append(move)
        return set_moves

    # a leaf node is a node which has child nodes that are unexplored
    def __isLeafNode(self):
        if self.children is None or self.gameOver():    # if none of the children has been explored or node is terminal node
            return True
        # check if all children have been explored
        for child in self.children:
            if child.total_num_games == 0:
                return True
        return False
    
    def run_MCTS(self, policy, t, param, nodes):
        # selection
        chosen_node = self
        path = []
        expanded_nodes = []
        cycle = False
        while not chosen_node.__isLeafNode() and not cycle:
            chosen_node_index = policy(chosen_node.children, t, param, chosen_node.my_turn)
            # if chosen_node.own_hands == Hands(1, 3) and chosen_node.opponent_hands == Hands(0, 4) and chosen_node.my_turn:
            #     print("!!!!!!!", chosen_node)
            #     for child in chosen_node.children:
            #         print(child)
            #     Q = np.array([child.score if self.my_turn else -child.score for child in chosen_node.children])
            #     N = np.array([child.total_num_games if child.total_num_games != 0 else  0.001 for child in chosen_node.children]).astype(float)   # np.reciprocal does not work on int => change to float; 0.001 to prevent division by 0
            #     score = Q + 1.5 * np.sqrt(np.math.log(t) * np.reciprocal(N))
            #     print(Q, N,  np.reciprocal(N), score, chosen_node.my_turn, chosen_node_index)
            path.append(chosen_node_index)
            if chosen_node.children[chosen_node_index].state_name in expanded_nodes:
                cycle = True
            expanded_nodes.append(chosen_node.children[chosen_node_index].state_name)
            chosen_node = chosen_node.children[chosen_node_index]
        points_won = 0
        if cycle:
            # draw
            points_won = 0.5
        else:
            # expansion
            if not chosen_node.gameOver():
                if chosen_node.children is None:
                    chosen_node.children = chosen_node.getChildrenNodes(nodes)
                chosen_node_index = policy(chosen_node.children, t, param, chosen_node.my_turn)
                path.append(chosen_node_index)
                chosen_node = chosen_node.children[chosen_node_index]
            # simulation
            playout_node = chosen_node
            while not playout_node.gameOver():
                # randomly select action 
                children_nodes = playout_node.getChildrenNodes(nodes)
                playout_node = children_nodes[random.randrange(len(children_nodes))]
            # win
            if playout_node.own_hands.alive():
                points_won = 1
        # backpropagation
        chosen_node = self
        chosen_node.updateScore(points_won)
        for node_index in path:
            chosen_node = chosen_node.children[node_index]
            chosen_node.updateScore(points_won)

    def getHandsState(self): 
        return "Computer: {} \t Player: {}".format(self.own_hands, self.opponent_hands)

    def getBestMove(self, score_based=False):
        if self.children is None:
            return None
        if score_based:
            scores = np.array([child.score for child in self.children])   # total_num_games instead of score since score may be noisy e.g. 98/100 games won vs 1/1 games won
        else:
            scores = np.array([child.total_num_games for child in self.children])   # total_num_games instead of score since score may be noisy e.g. 98/100 games won vs 1/1 games won
        return self.children[np.argmax(scores)]
        
    def move(self, new_state):
        # new_state: [own_smaller, own_larger, opponent_smaller, opponent_larger]
        assert self.children is not None
        if len(new_state) != 4:
            return None
        for child in self.children:
            if child.own_hands.smaller == int(new_state[0]) and child.own_hands.larger == int(new_state[1]) and child.opponent_hands.smaller == int(new_state[2]) and child.opponent_hands.larger == int(new_state[3]):
                return child
        return None

def UCB(children, t, c, max_agent):
    Q = np.array([child.score if max_agent else -child.score for child in children])
    N = np.array([child.total_num_games if child.total_num_games != 0 else  0.001 for child in children]).astype(float)   # np.reciprocal does not work on int => change to float; 0.001 to prevent division by 0
    # print(Q, N, np.math.log(t) * np.reciprocal(N))
    score = Q + c * np.sqrt(np.math.log(t) * np.reciprocal(N))
    '''
    score takes into account:
     1. EXPLORATION: the less unexplored the higher the score for both max and min players
     2. EXPLOITATION: current running average - the higher the average the higher the score for max player, but the lower the score for min player)
    '''
    return np.argmax(score) 

def create_nodes():
    '''
        ============== 15 possible finger combinations ==============
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)
        (1, 1), (1, 2), (1, 3), (1, 4)
        (2, 2), (2, 3), (2, 4)
        (3, 3), (3, 4)
        (4, 4)
    '''
    nodes = {}
    for i in range(5):
        for j in range(i, 5):
                for k in range(5):
                    for l in range(k, 5):
                        nodes[str(i) + str(j) + str(k) + str(l) + "C"] = Node(Hands(i, j), Hands(k, l), 0, 0, True)
                        nodes[str(i) + str(j) + str(k) + str(l) + "P"] = Node(Hands(i, j), Hands(k, l), 0, 0, False)
    return nodes
                        
# def run(numIterations, policy, param, computer_starts=True):
#     nodes = create_nodes()
#     t = 1
#     reward = np.zeros(numIterations + 1)
#     root_node = Node(Hands(1, 1), Hands(1, 1), 0, 0, computer_starts)
#     for i in range(numIterations):
#         # choose action and get reward
#         root_node.run_MCTS(policy, t, param, nodes)
#         reward[t] = root_node.score
#         t += 1
#     return reward 

def play(trained_nodes=None, computer_starts=True):
    # if trained nodes not passed in, create nodes
    if trained_nodes is None:
        nodes = create_nodes()
    else:
        nodes = trained_nodes
    current_node = nodes["1111C"] if computer_starts else nodes["1111P"]
    # play
    print("Starting state:", current_node.getHandsState())
    if computer_starts:
        print("Computer starts first..........")
    else:
        print("Player starts first..........")
        current_node.children = current_node.getChildrenNodes(nodes)
    player_turn = not computer_starts
    t = 1
    while not current_node.gameOver():
        if player_turn:
            move = input("Enter move:")
            given_move = current_node.move(move.split())
            while given_move is None:
                move = input("Illegal move, please try again:")
                given_move = current_node.move(move.split())
            current_node = given_move
            print("After your move:", current_node.getHandsState())
        else:
            '''
            We can either train on the spot or feed in trained nodes
            We train the nodes on the spot if trained nodes are not passed in
            '''
            if trained_nodes is None:
                for i in range(5000):
                    # choose action and get reward
                    current_node.run_MCTS(UCB, t, 1.5, nodes)
                    t += 1
            for child in current_node.children:
                print(child.getHandsState(), child.num_games_won, child.total_num_games)
            current_node = current_node.getBestMove(score_based=True)
            print("After computer's move:", current_node.getHandsState())
        player_turn = not player_turn
    winner = "COMPUTER" if current_node.own_hands.alive() else "PLAYER"
    print("=========== GAME OVER: {} WINS ===========".format(winner))

# def collect_points():
#     T = 1000
#     avg_ucb = np.zeros(T + 1)
#     avg_ucb_2 = np.zeros(T + 1)
#     avg_ucb_3 = np.zeros(T + 1)

#     for i in range(20):
        
#         ucb = run(T, UCB, 1)
#         ucb_2 = run(T, UCB, 2)
#         ucb_3 = run(T, UCB, 3)
#         avg_ucb += (ucb - avg_ucb) / (i + 1)
#         avg_ucb_2 += (ucb_2 - avg_ucb_2) / (i + 1)
#         avg_ucb_3 += (ucb_3 - avg_ucb_3) / (i + 1)

#     t = np.arange(T + 1)
#     plt.plot(t, avg_ucb, 'r-', label='UCB (1)')
#     plt.plot(t, avg_ucb_2, 'g-', label='UCB (2)')
#     plt.plot(t, avg_ucb_3, 'b-', label='UCB (3)')
#     plt.legend()
#     plt.show()

# recursive function to help trace all possible paths
def getAllOutcomes(starting_node_name, nodes, current_path):
    paths = []
    starting_node = nodes[starting_node_name]
    # if state is end state: won. lost or draw
    if starting_node.gameOver() or starting_node_name in current_path:
        current_path.append(starting_node_name)
        paths.append(current_path)
        # if current_path[0] == "1301C" and not starting_node.gameOver():
        #     print("DRAW", starting_node_name, current_path)
        #     for child in nodes["1301C"].getChildrenNodes(nodes):
        #         print(child)
        return paths
    # if computer starts first
    current_path.append(starting_node_name)
    if starting_node.my_turn:
        # given a state, make best move and draw path to new state
        my_move = starting_node.getBestMove(score_based=True)
        assert my_move is not None
        current_path.append(my_move.state_name)
        # if my move ends in game ending, end current_path
        if my_move.gameOver():
            paths.append(current_path)
            return paths
    else:
        my_move = starting_node
    # continue expanding path
    for child in my_move.getChildrenNodes(nodes):
        # draw lines to all possible oppononent moves
        # if current_path[0] != "0101C":
            # print("!!!", current_path[0])
            # print(current_path)
        paths += getAllOutcomes(child.state_name, nodes, list(current_path))
    return paths
    
def getStrategy(nodes, show_move_only=False):
    # collate results
    print("Preparing results...")
    values = [[] for i in range(15)]
    actioner_labels = []
    receiver_labels = []
    count_actioner = 0
    for i in range(5):
        for j in range(i, 5):
                actioner_labels.append(str(i) + ", "  + str(j))
                receiver_labels.append(str(i) + ", "  + str(j))
                for k in range(5):
                    for l in range(k, 5):
                        best_move = nodes[str(i) + str(j) + str(k) + str(l) + "C"].getBestMove(score_based=True)
                        # values[count_actioner].append(str(nodes[str(i) + str(j) + str(k) + str(l) + "C"]))
                        if best_move is None:
                            values[count_actioner].append("-")
                        else:
                            if show_move_only: 
                                result = str(best_move.own_hands.smaller)+str(best_move.own_hands.larger)+str(best_move.opponent_hands.smaller)+str(best_move.opponent_hands.larger)
                            else:
                                all_paths = getAllOutcomes(str(i) + str(j) + str(k) + str(l) + "C", nodes, [])
                                # W: win, D: draw, L: lose
                                worst_outcome = "W"
                                for path in all_paths:
                                    last_node = nodes[path[-1]]
                                    if not last_node.gameOver():
                                        worst_outcome = "D"
                                    elif  last_node.own_hands == Hands(0, 0):
                                        worst_outcome = "L"
                                        break
                                result = worst_outcome + ":" + str(best_move.own_hands.smaller)+str(best_move.own_hands.larger)+str(best_move.opponent_hands.smaller)+str(best_move.opponent_hands.larger)
                            values[count_actioner].append(result)
                count_actioner += 1
    # display results
    row_format = "{:>10}" * (len(receiver_labels) + 1)
    print("{:=^{length}}".format(" Best Moves ", length=(10*(len(receiver_labels)+1))))
    print(textwrap.TextWrapper(width=(10*(len(receiver_labels)+1))).fill(text=
        "Column refers to the state of the hands of the actioner i.e. the person making the move while "
        "the row refers to the state of the hands of the receiver i.e. the person waiting for the opponent to make a move. "
        "The value inside the table represents the worst possible outcome if moves are made correctly and the best move to make "
        "e.g.: W:0401 means that the actioner is guaranteed to win if the best moves are made, and the best move the actioner "
        "can take is to move to the state 0401"
    ))
    print(row_format.format("", *receiver_labels))
    for actioner, row in zip(actioner_labels, values):
        print(row_format.format(actioner, *row))

        
def showAllOutcomes(nodes, starting_node="1111C"):
    
    # collate results and get labels
    print("Preparing results...")
    all_paths = getAllOutcomes(starting_node, nodes, [])
    end_points = [path[-1] for path in all_paths]
    end_points = []
    for path in all_paths:
        last_node = nodes[path[-1]]
        if not last_node.gameOver():
            end_points.append((last_node.state_name, "DRAW"))
        elif last_node.own_hands == Hands(0, 0):
            end_points.append((last_node.state_name, "LOSE"))
        else:
            end_points.append((last_node.state_name, "WIN"))
    # display results
    print("Displaying results...")
    outcomes = list(set(end_points))
    print("Results: {} paths; {} possible outcomes".format(len(all_paths), len(outcomes)))
    print("================= ALL POSSIBLE OUTCOMES =================")
    for index, state in enumerate(outcomes):
        print("{}. {}".format(index + 1, state))
    # current_set_all = set(all_paths[0])
    # current_set_1212 = None
    # current_set_2211 = None
    # current_set_1311 = None
    # for path in all_paths:
    #     current_set_all = current_set_all.intersection(path)
    #     if "1212C" in path:
    #         if current_set_1212 is None:
    #             current_set_1212 = set(path)
    #         else:
    #             current_set_1212 = current_set_1212.intersection(path)
    #     if "1122C" in path:
    #         if current_set_2211 is None:
    #             current_set_2211 = set(path)
    #         else:
    #             current_set_2211 = current_set_2211.intersection(path)
    #     if "1311C" in path:
    #         if current_set_1311 is None:
    #             current_set_1311 = set(path)
    #         else:
    #             current_set_1311 = current_set_1311.intersection(path)

    # print(current_set_all)
    # print(current_set_1212)
    # print(current_set_2211)
    # print(current_set_1311)
    while True:
        index = input("Enter the index of the state to view all paths leading to that state (or 'exit' to quit):")
        if index == "exit":
            break
        final_state, _ = outcomes[int(index)-1]
        for path in all_paths:
            last_state = path[-1]
            if final_state == last_state:
                print(path)

def get_trained_nodes():
    nodes = create_nodes()
    # train
    print("Training...")
    t = 1
    for key, value in nodes.items():
        if key[4] == "C" and not value.gameOver():
            current_node = value
            for i in range(1000):
                current_node.run_MCTS(UCB, t, 1.5, nodes)
                t += 1
    return nodes

# collect_points()

nodes = get_trained_nodes()
# play(nodes)
# play()
# getStrategy(nodes)
showAllOutcomes(nodes, "1111C")
# play(nodes)

# def test():
#     nodes = create_nodes()
#     for key, value in nodes.items():
#         # train
#         if key[4] == "C" and not value.gameOver():
#             print(value)
#             current_node = value
#             t = 1
#             for i in range(5000):
#                 # choose action and get reward
#                 # print(current_node.children)
#                 current_node.run_MCTS(UCB, t, 1.5, nodes)
#                 t += 1
#             for child in current_node.children:
#                 print(child.getHandsState
# (), child.num_games_won, child.total_num_games)
#                 # num_games = []
#                 # for child in current_node.children:
#                 #     if child.total_num_games != 0:
#                 #         num_games.append(child.total_num_games)
#                 #     elif current_node.my_turn:
#                 #         num_games.append(0.001)
#                 #     else:
#                 #         num_games.append(float('inf'))
#                 # Q = np.array([child.score for child in current_node.children])
#                 # N = np.array(num_games).astype(float)
#                 # score = Q + 15 * np.sqrt(np.math.log(t) * np.reciprocal(N))
#                 # print(Q, N,  np.reciprocal(N), score)
# test()

# nodes = create_nodes()
# test_node = Node(Hands(2, 3), Hands(0, 1), 0, 0, False)
# child_nodes = test_node.getChildrenNodes(nodes)
# print("Parent Node:", test_node)
# for child in child_nodes:
#     print(child)