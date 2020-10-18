import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx

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
    def __init__(self, actioner, receiver, num_games_won, total_num_games, my_turn):
        self.my_turn = my_turn
        self.actioner = actioner
        self.receiver = receiver
        self.num_games_won = num_games_won
        self.total_num_games = total_num_games
        self.children = None
        player = "C" if my_turn else "P"
        self.state_name = str(actioner.smaller)+str(actioner.larger)+str(receiver.smaller)+str(receiver.larger)+player


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
    
    @actioner.setter
    def actioner(self, new_hands):
        if self.my_turn:
            self.own_hands = new_hands
        else:
            self.opponent_hands = new_hands

    # Receiver refers to the hands of the player that iw waiting for the opponent to make a move
    @property
    def receiver(self):
        if self.my_turn:
            return self.opponent_hands
        else:
            return self.own_hands
    
    @receiver.setter
    def receiver(self, new_hands):
        if self.my_turn:
            self.opponent_hands = new_hands
        else:
            self.own_hands = new_hands

    def updateScore(self, points_won):
        self.total_num_games += 1
        self.num_games_won += points_won
        # if self.own_hands == Hands(0, 1) and self.opponent_hands.larger == 4 and not self.my_turn:
        #     print(self)

    def gameOver(self):
        return not self.own_hands.alive() or not self.opponent_hands.alive()

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
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 2: Add B to A
        if self.actioner.smaller != 0 and self.actioner.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            actioner_new_hands.addFingers(self.actioner.larger, True)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            # print("Add B to A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 3: Add A to opponent's A
        if self.actioner.smaller != 0 and self.receiver.smaller != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.smaller, True)
            # print("Add A to opponent's A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 4: Add A to opponent's B
        if self.actioner.smaller != 0 and self.receiver.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.smaller, False)
            # print("Add A to opponent's B", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 5: Add B to opponent's A
        if self.actioner.larger != 0 and self.receiver.smaller != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.larger, True)
            # print("Add B to opponent's A", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 6: Add B to opponent's B
        if self.actioner.larger != 0 and self.receiver.larger != 0:
            actioner_new_hands = Hands(self.actioner.smaller, self.actioner.larger)
            receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
            receiver_new_hands.addFingers(self.actioner.larger, False)
            # print("Add B to opponent's B", actioner_new_hands, receiver_new_hands)
            legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
        # Move 7: Transfer some from B to A (but without reversal/killing own hand)
        # if self.actioner.smaller != 0 and self.actioner.larger != 0:
        receiver_new_hands = Hands(self.receiver.smaller, self.receiver.larger)
        for i in range(1, self.actioner.larger):
            A = self.actioner.smaller + i
            B = self.actioner.larger - i
            if not (A == self.actioner.larger and B == self.actioner.smaller) and A <= 4: # no reversal and no killing
                actioner_new_hands = Hands(A, B)
                # print("Transfer some from B to A (but without reversal/killing own hand)", actioner_new_hands, receiver_new_hands)
                legal_moves.append(nodes[str(receiver_new_hands.smaller)+str(receiver_new_hands.larger)+str(actioner_new_hands.smaller)+str(actioner_new_hands.larger)+player])
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
        if not cycle:
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
        # backpropagation
        points_won = 0
        # if draw
        if cycle:
            points_won = 0.5
        # else if won
        elif playout_node.own_hands.alive():
            points_won = 1
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
        for child in self.children:
            print(child.getHandsState())
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
                        
def run(numIterations, policy, param, computer_starts=True):
    nodes = create_nodes()
    t = 1
    reward = np.zeros(numIterations + 1)
    root_node = Node(Hands(1, 1), Hands(1, 1), 0, 0, computer_starts)
    for i in range(numIterations):
        # choose action and get reward
        root_node.run_MCTS(policy, t, param, nodes)
        reward[t] = root_node.score
        t += 1
    return reward 

def play(computer_starts=True):
    nodes = create_nodes()
    current_node = nodes["1111C"] if computer_starts else nodes["1111P"]
    # play
    print("Starting state:", current_node.getHandsState())
    if computer_starts:
        print("Computer starts first..........")
    else:
        print("Player starts first..........")
        current_node.children = current_node.getChildrenNodes(nodes)
    player_turn = not computer_starts
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
            # train
            t = 1
            for i in range(1000):
                # choose action and get reward
                current_node.run_MCTS(UCB, t, 1.5, nodes)
                t += 1
            for child in current_node.children:
                print(child.getHandsState(), child.num_games_won, child.total_num_games)
                # num_games = []
                # for child in current_node.children:
                #     if child.total_num_games != 0:
                #         num_games.append(child.total_num_games)
                #     elif current_node.my_turn:
                #         num_games.append(0.001)
                #     else:
                #         num_games.append(float('inf'))
                # Q = np.array([child.score for child in current_node.children])
                # N = np.array(num_games).astype(float)
                # score = Q + 15 * np.sqrt(np.math.log(t) * np.reciprocal(N))
                # print(Q, N,  np.reciprocal(N), score)
            current_node = current_node.getBestMove()
            print("After computer's move:", current_node.getHandsState())
        player_turn = not player_turn
    winner = "COMPUTER" if current_node.own_hands.alive() else "PLAYER"
    print("=========== GAME OVER: {} WINS ===========".format(winner))

def results():
    T = 1000
    avg_ucb = np.zeros(T + 1)
    avg_ucb_2 = np.zeros(T + 1)
    avg_ucb_3 = np.zeros(T + 1)

    for i in range(20):
        
        ucb = run(T, UCB, 1)
        ucb_2 = run(T, UCB, 2)
        ucb_3 = run(T, UCB, 3)
        avg_ucb += (ucb - avg_ucb) / (i + 1)
        avg_ucb_2 += (ucb_2 - avg_ucb_2) / (i + 1)
        avg_ucb_3 += (ucb_3 - avg_ucb_3) / (i + 1)

    t = np.arange(T + 1)
    plt.plot(t, avg_ucb, 'r-', label='UCB (1)')
    plt.plot(t, avg_ucb_2, 'g-', label='UCB (2)')
    plt.plot(t, avg_ucb_3, 'b-', label='UCB (3)')
    plt.legend()
    plt.show()

def getAllOutcomes(nodes, starting_node_name):
    # receusrive function to help trace all possible paths
    def drawPath(node, nodes, current_path, paths, end_points):
        if node.gameOver():
            return []
        # if computer starts first
        if node.my_turn:
            # given a state, make best move and draw path to new state
            current_path.append(node.state_name)
            my_move = node.getBestMove(score_based=True)
            assert my_move is not None
            current_path.append(my_move.state_name)
        else:
            my_move = node
        # if move ends in game ending, put as end point
        if my_move.gameOver():
            end_points.append((my_move.state_name, "WIN"))
            paths.append(current_path)
        # else continue expanding path
        else:
            for child in my_move.getChildrenNodes(nodes):
                # draw lines to all possible oppononent moves
                updated_path = list(current_path)
                # if opponent move has led to game ending, put as end point
                if child.gameOver():
                    updated_path.append(child.state_name)
                    end_points.append((child.state_name, "LOSE"))
                    paths.append(updated_path)
                # expand path from child state if it has not been explored before
                elif child.state_name not in current_path:
                    drawPath(child, nodes, updated_path, paths, end_points)
                # else child state has been reached before, we have reached a draw
                else:
                    updated_path.append(child.state_name)
                    end_points.append((child.state_name, "DRAW"))
                    paths.append(updated_path)
    
    # collate results and get labels
    end_points = []
    all_paths = []
    starting_node = nodes[starting_node_name]
    drawPath(starting_node, nodes, [], all_paths, end_points)
    return end_points

def getStrategy(nodes, show_move=True):
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
                            if show_move:
                                values[count_actioner].append(str(best_move.own_hands.smaller)+str(best_move.own_hands.larger)+str(best_move.opponent_hands.smaller)+str(best_move.opponent_hands.larger))
                            else: 
                                outcomes = getAllOutcomes(nodes, nodes[str(i) + str(j) + str(k) + str(l) + "C"].state_name)
                                result = "W:" + str(best_move.own_hands.smaller)+str(best_move.own_hands.larger)+str(best_move.opponent_hands.smaller)+str(best_move.opponent_hands.larger) if len(outcomes) > 0 else "-"
                                for path, outcome in outcomes:
                                    if outcome == "LOSE":
                                        result = "-"
                                        break
                                    elif outcome == "DRAW" and result[0] == "W":
                                        result = "D:" + str(best_move.own_hands.smaller)+str(best_move.own_hands.larger)+str(best_move.opponent_hands.smaller)+str(best_move.opponent_hands.larger)
                                values[count_actioner].append(result)
                count_actioner += 1
    # display results
    row_format ="{:>10}" * (len(receiver_labels) + 1)
    print(row_format.format("", *receiver_labels))
    for actioner, row in zip(actioner_labels, values):
        print(row_format.format(actioner, *row))

def showWinningPath(nodes):
    # gets coordinate of the node in the display graph
    def getCoordinate(node):
        own_coord = node.own_hands.smaller * (5 + (5 - node.own_hands.smaller + 1))/2 + (node.own_hands.larger - node.own_hands.smaller)
        opponent_coord = node.opponent_hands.smaller * (5 + (5 - node.opponent_hands.smaller + 1))/2 + (node.opponent_hands.larger - node.opponent_hands.smaller)
        return (own_coord,opponent_coord)

    # receusrive function to help trace all possible paths
    def drawPath(node, nodes, current_path, paths, end_points):
        assert not node.gameOver(), "No path to draw -- game already over"
        # if computer starts first
        if node.my_turn:
            # given a state, make best move and draw path to new state
            current_path.append(node.state_name)
            my_move = node.getBestMove(score_based=True)
            assert my_move is not None
            current_path.append(my_move.state_name)
        else:
            my_move = node
        # if move ends in game ending, put as end point
        if my_move.gameOver():
            end_points.append((my_move.state_name, "WIN"))
            paths.append(current_path)
        # else continue expanding path
        else:
            for child in my_move.getChildrenNodes(nodes):
                # draw lines to all possible oppononent moves
                updated_path = list(current_path)
                # if opponent move has led to game ending, put as end point
                if child.gameOver():
                    updated_path.append(child.state_name)
                    end_points.append((child.state_name, "LOSE"))
                    paths.append(updated_path)
                # expand path from child state if it has not been explored before
                elif child.state_name not in current_path:
                    drawPath(child, nodes, updated_path, paths, end_points)
                # else child state has been reached before, we have reached a draw
                else:
                    updated_path.append(child.state_name)
                    end_points.append((child.state_name, "DRAW"))
                    paths.append(updated_path)
    
    # collate results and get labels
    print("Preparing results...")
    actioner_labels = []
    receiver_labels = []
    end_points = []
    all_paths = []
    starting_node = nodes["1111C"]
    drawPath(starting_node, nodes, [], all_paths, end_points)
    for i in range(5):
        for j in range(i, 5):
            actioner_labels.append(str(i) + ", "  + str(j))
            receiver_labels.append(str(i) + ", "  + str(j))
    # display results
    print("Displaying results...")
    fig, ax = plt.subplots()
    win_x = []
    for path in all_paths:
        my_turn = starting_node.my_turn
        tail = getCoordinate(nodes[path[0]])
        for i in range(1, len(path)):
            head = getCoordinate(nodes[path[i]])
            if my_turn:
                ax.add_patch(patches.FancyArrow(tail[0], tail[1], head[0] - tail[0], head[1] - tail[1], length_includes_head=True, head_width=0.2, color="orange"))
            else:
                ax.add_patch(patches.FancyArrow(tail[0], tail[1], head[0] - tail[0], head[1] - tail[1], length_includes_head=True, head_width=0.2, color="gray"))
            tail = head
            my_turn = not my_turn
    win_y = []
    draw_x = []
    draw_y = []
    lose_x = []
    lose_y = []
    for state, outcome in end_points:
        (x, y) = getCoordinate(nodes[state])
        if outcome == "WIN":
            win_x.append(x)
            win_y.append(y)
        elif outcome == "LOSE":
            lose_x.append(x)
            lose_y.append(y)
        elif outcome == "DRAW":
            draw_x.append(x)
            draw_y.append(y)
    plt.scatter(win_x, win_y, color='g')
    plt.scatter(draw_x, draw_y, color='b')
    plt.scatter(lose_x, lose_y, color='r')
    ax.set_xticks(np.arange(len(actioner_labels)))
    ax.set_yticks(np.arange(len(receiver_labels)))
    ax.set_xticklabels(actioner_labels)
    ax.set_yticklabels(receiver_labels)
    plt.show()

def showWinningPath2(nodes):    
    def getCoordinate(node):
        own_coord = node.own_hands.smaller * (5 + (5 - node.own_hands.smaller + 1))/2 + (node.own_hands.larger - node.own_hands.smaller)
        opponent_coord = node.opponent_hands.smaller * (5 + (5 - node.opponent_hands.smaller + 1))/2 + (node.opponent_hands.larger - node.opponent_hands.smaller)
        return (own_coord,opponent_coord)
    
    def drawPath(node, nodes, current_path, paths, end_points):
        # given a state, make best move and draw path to new state
        assert not node.gameOver(), "No path to draw -- game already over"
        best_move = node.getBestMove()
        assert best_move is not None
        current_path.append(getCoordinate(node))
        # if move ends in game ending, put as end point
        if best_move.gameOver():
            end_points.append((getCoordinate(best_move), "WIN"))
            paths.append(current_path)
        # else continue expanding path
        else:
            for child in best_move.getChildrenNodes(nodes):
                # draw lines to all possible oppononent moves
                updated_path = list(current_path)
                updated_path.append(getCoordinate(child))
                # if opponent move has led to game ending, put as end point
                if child.gameOver():
                    end_points.append((getCoordinate(child), "LOSE"))
                    paths.append(updated_path)
                # expand path from child state if it has not been explored before
                elif getCoordinate(child) not in current_path:
                    drawPath(child, nodes, updated_path, paths, end_points)
                # else child state has been reached before, we have reached a draw
                else:
                    end_points.append((getCoordinate(child), "DRAW"))
                    paths.append(updated_path)

    # collate results and get labels
    print("Preparing results...")
    actioner_labels = []
    receiver_labels = []
    end_points = []
    all_paths = []
    starting_node = nodes["1111C"]
    drawPath(starting_node, nodes, [], all_paths, end_points)
    for i in range(5):
        for j in range(i, 5):
                actioner_labels.append(str(i) + ", "  + str(j))
                receiver_labels.append(str(i) + ", "  + str(j))
    # display results
    print("Displaying results...")
    fig, ax = plt.subplots()
    colours_available = list(mcolors.CSS4_COLORS.values())
    colour_index = 0
    for path in all_paths:
        tail = path[0]
        colour = colours_available[colour_index]
        for i in range(1, len(path)):
            head = path[i]
            ax.add_patch(patches.FancyArrow(tail[0], tail[1], head[0] - tail[0], head[1] - tail[1], length_includes_head=True, head_width=0.2, color=colour))
            tail = head
        colour_index = (colour_index + 1) % len(colours_available)
    win_x = []
    win_y = []
    draw_x = []
    draw_y = []
    lose_x = []
    lose_y = []
    for point, state in end_points:
        if state == "WIN":
            win_x.append(point[0])
            win_y.append(point[1])
        elif state == "LOSE":
            lose_x.append(point[0])
            lose_y.append(point[1])
        elif state == "DRAW":
            draw_x.append(point[0])
            draw_y.append(point[1])
    plt.scatter(win_x, win_y, color='g')
    plt.scatter(draw_x, draw_y, color='b')
    plt.scatter(lose_x, lose_y, color='r')
    ax.set_xticks(np.arange(len(actioner_labels)))
    ax.set_yticks(np.arange(len(receiver_labels)))
    ax.set_xticklabels(actioner_labels)
    ax.set_yticklabels(receiver_labels)
    plt.show()

def showWinningPath3(nodes):
    
    def getCoordinate(node):
        player = "C" if node.my_turn else "P"
        return str(node.own_hands.smaller) + str(node.own_hands.larger) + str(node.opponent_hands.smaller) + str(node.opponent_hands.larger) + player

    def drawPath(node, nodes, current_path, paths, end_points):
        assert not node.gameOver(), "No path to draw -- game already over"
        # if computer starts first
        if node.my_turn:
            # given a state, make best move and draw path to new state
            current_path.append(getCoordinate(node))
            my_move = node.getBestMove(score_based=True)
            assert my_move is not None
            current_path.append(getCoordinate(my_move))
        else:
            my_move = node
        # if move ends in game ending, put as end point
        if my_move.gameOver():
            end_points.append((getCoordinate(my_move), "WIN"))
            paths.append(current_path)
        # else continue expanding path
        else:
            for child in my_move.getChildrenNodes(nodes):
                # draw lines to all possible oppononent moves
                updated_path = list(current_path)
                # if opponent move has led to game ending, put as end point
                if child.gameOver():
                    updated_path.append(getCoordinate(child))
                    end_points.append((getCoordinate(child), "LOSE"))
                    paths.append(updated_path)
                # expand path from child state if it has not been explored before
                elif getCoordinate(child) not in current_path:
                    drawPath(child, nodes, updated_path, paths, end_points)
                # else child state has been reached before, we have reached a draw
                else:
                    updated_path.append(getCoordinate(child))
                    end_points.append((getCoordinate(child), "DRAW"))
                    paths.append(updated_path)
    
    # collate results and get labels
    print("Preparing results...")
    end_points = []
    all_paths = []
    starting_node = nodes["1111C"]
    drawPath(starting_node, nodes, [], all_paths, end_points)
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
        index = input("Enter the index of the state to view all paths leading to that state (or 'exit' to exit):")
        if index == "exit":
            break
        final_state, outcome = outcomes[int(index)-1]
        for path in all_paths:
            last_state = path[len(path)-1]
            if outcome == "DRAW" and final_state == last_state:
                print(path)
            elif outcome == "WIN" and final_state == last_state:
                print(path)
            elif outcome == "LOSE" and final_state == last_state:
                print(path)


# def showGraph(nodes):
#     ''' Taken from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209'''
#     def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5 ):
#         '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

#         G: the graph
#         root: the root node of current branch
#         width: horizontal space allocated for this branch - avoids overlap with other branches
#         vert_gap: gap between levels of hierarchy
#         vert_loc: vertical location of root
#         xcenter: horizontal location of root
#         '''

#         def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, 
#                     pos = None, parent = None, parsed = [] ):
#             if(root not in parsed):
#                 parsed.append(root)
#                 if pos == None:
#                     pos = {root:(xcenter,vert_loc)}
#                 else:
#                     pos[root] = (xcenter, vert_loc)
#                 neighbors = list(G.neighbors(root))
#                 if parent != None:
#                     neighbors.remove(parent)
#                 if len(neighbors)!=0:
#                     dx = width/len(neighbors) 
#                     nextx = xcenter - width/2 - dx/2
#                     for neighbor in neighbors:
#                         nextx += dx
#                         pos = h_recur(G,neighbor, width = dx, vert_gap = vert_gap, 
#                                             vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos, 
#                                             parent = root, parsed = parsed)
#             return pos

#         return h_recur(G, root, width, vert_gap, vert_loc, xcenter)
#     # receusrive function to help trace all possible paths
#     def drawPath(node, nodes, current_path, paths, end_points):
#         assert not node.gameOver(), "No path to draw -- game already over"
#         # if computer starts first
#         if node.my_turn:
#             # given a state, make best move and draw path to new state
#             current_path.append(node.state_name)
#             my_move = node.getBestMove(score_based=True)
#             assert my_move is not None
#             current_path.append(my_move.state_name)
#         else:
#             my_move = node
#         # if move ends in game ending, put as end point
#         if my_move.gameOver():
#             end_points.append((my_move.state_name, "WIN"))
#             paths.append(current_path)
#         # else continue expanding path
#         else:
#             for child in my_move.getChildrenNodes(nodes):
#                 # draw lines to all possible oppononent moves
#                 updated_path = list(current_path)
#                 # if opponent move has led to game ending, put as end point
#                 if child.gameOver():
#                     updated_path.append(child.state_name)
#                     end_points.append((child.state_name, "LOSE"))
#                     paths.append(updated_path)
#                 # expand path from child state if it has not been explored before
#                 elif child.state_name not in current_path:
#                     drawPath(child, nodes, updated_path, paths, end_points)
#                 # else child state has been reached before, we have reached a draw
#                 else:
#                     # updated_path.append(child.state_name)
#                     end_points.append((child.state_name, "DRAW"))
#                     paths.append(updated_path)
    
#     # collate results and get labels
#     print("Preparing results...")
#     actioner_labels = []
#     receiver_labels = []
#     end_points = []
#     all_paths = []
#     starting_node = nodes["1111C"]
#     drawPath(starting_node, nodes, [], all_paths, end_points)
#     for i in range(5):
#         for j in range(i, 5):
#             actioner_labels.append(str(i) + ", "  + str(j))
#             receiver_labels.append(str(i) + ", "  + str(j))
#     # display results
#     print("Displaying results...")
#     fig, ax = plt.subplots()
#     win_x = []
#     G = nx.Graph()
#     for path in all_paths:
#         my_turn = starting_node.my_turn
#         tail_node = nodes[path[0]]
#         tail = str(tail_node.own_hands.smaller)+str(tail_node.own_hands.larger)+str(tail_node.opponent_hands.smaller)+str(tail_node.opponent_hands.larger)
#         for i in range(1, len(path)): 
#             head_node = nodes[path[i]]
#             head = str(head_node.own_hands.smaller)+str(head_node.own_hands.larger)+str(head_node.opponent_hands.smaller)+str(head_node.opponent_hands.larger)
#             G.add_node(tail)
#             G.add_node(head)
#             if my_turn:
#                 G.add_edge(tail, head, color="orange")
#             else:
#                 G.add_edge(tail, head, color="gray")
#             tail = head
#             my_turn = not my_turn
#     # win_y = []
#     # draw_x = []
#     # draw_y = []
#     # lose_x = []
#     # lose_y = []
#     # for state, outcome in end_points:
#     #     (x, y) = getCoordinate(nodes[state])
#     #     if outcome == "WIN":
#     #         win_x.append(x)
#     #         win_y.append(y)
#     #     elif outcome == "LOSE":
#     #         lose_x.append(x)
#     #         lose_y.append(y)
#     #     elif outcome == "DRAW":
#     #         draw_x.append(x)
#     #         draw_y.append(y)

#     # nx.draw_circular(DG, with_labels=True)
#     # nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, prog="dot"), with_labels=True, node_size=1000)
#     pos = hierarchy_pos(G, "1111", vert_gap=2.5)    
#     nx.draw(G, pos=pos, with_labels=True)
#     plt.show()


# play()
# results()
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
# showWinningPath(nodes)

# getStrategy(nodes, show_move=False)
# showWinningPath3(nodes)
showGraph(nodes)

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