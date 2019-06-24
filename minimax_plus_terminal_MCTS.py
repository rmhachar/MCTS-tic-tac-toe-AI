import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time

class policy(object):
    def __init__(self):
        self.tree = {}
        pass


class minimax_plus_terminal_MCTS(object):
    def __init__(self, n_iterations=50, depth=15, exploration_constant=5.0, tree = None, win_mark=3, game_board=None, player=None):
        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0

        self.leaf_node_id = None

        n_rows = len(game_board)
        self.n_rows = n_rows
        self.win_mark = win_mark

        if tree == None:
            self.tree = self._set_tictactoe(game_board, player)
        else:
            self.tree = tree

        """
        open_spaces = 9 - np.count_nonzero(game_board)
        if open_spaces > 6:
            self.n_iterations = n_iterations
        elif 7 > open_spaces and open_spaces > 3:
            self.n_iterations = 20000
        else:
            self.n_iterations = 5000
        """

    def _set_tictactoe(self, game_board, player):
        root_id = (0,)
        tree = {root_id: {'state': game_board,
                          'player': player,
                          'child': [],
                          'parent': None,
                          'n': 0,
                          'w': 0,
                          'q': None}}
        return tree

    def selection(self):
        '''
        select leaf node which have maximum uct value
        in:
        - tree
        out:
        - leaf node id (node to expand)
        - depth (depth of node root=0)
        '''
        leaf_node_found = False
        leaf_node_id = (0,) # root node id
        # print('-------- selection ----------')

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['child'])

            # TEST PRINTS
            # print("TREE: \n", self.tree)
            # print("TREE[(0,)]: \n", self.tree[(0,)])
            # print("LEN TREE: ", len(self.tree))

            # Start at current node
            # current_node = [(0,)]
            # for child in self.tree[current_node]['child']:
            #   for child in self
            #       for child in self.tree[current_node]['child']:
            #           ...
            # 
            # 
            # 

            # 



            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                maximum_uct_value = -100.0
                for i in range(n_child):
                    action = self.tree[node_id]['child'][i]

                    child_id = node_id + (action,)
                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']
                    total_n = self.total_n
                    # parent_id = self.tree[node_id]['parent']
                    # if parent_id == None:
                    #     total_n = 1
                    # else:
                    #     total_n = self.tree[parent_id]['n']

                    if n == 0:
                        n = 1e-4
                    exploitation_value = w / n
                    exploration_value  = np.sqrt(np.log(total_n)/n)
                    uct_value = exploitation_value + self.exploration_constant * exploration_value

                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(leaf_node_id) # as node_id records selected action set
        # print('leaf node found: ', leaf_node_found)
        # print('n_child: ', n_child)
        # print('selected leaf node: ')
        # print(self.tree[leaf_node_id])
        return leaf_node_id, depth

    def expansion(self, leaf_node_id):
        '''
        create all possible outcomes from leaf node
        in: tree, leaf_node
        out: expanded tree (self.tree),
             randomly selected child node id (child_node_id)
        '''
        leaf_state = self.tree[leaf_node_id]['state']
        winner = self._is_terminal(leaf_state)
        possible_actions = self._get_valid_actions(leaf_state)

        child_node_id = leaf_node_id # default value
        if winner is None:
            '''
            when leaf state is not terminal state
            '''
            childs = []
            for action_set in possible_actions:
                action, action_idx = action_set
                state = deepcopy(self.tree[leaf_node_id]['state'])
                current_player = self.tree[leaf_node_id]['player']

                if current_player == 'o':
                    next_turn = 'x'
                    state[action] = 1
                else:
                    next_turn = 'o'
                    state[action] = -1

                child_id = leaf_node_id + (action_idx, )
                childs.append(child_id)

                if next_turn == 'x':
                    minimax_turn = 'max'
                else:
                    minimax_turn = 'min'

                self.tree[child_id] = {'state': state,
                                       'player': next_turn,
                                       'child': [],
                                       'parent': leaf_node_id,
                                       'minimax_turn': minimax_turn,
                                       'n': 0, 'w': 0, 'q':0}
                self.tree[leaf_node_id]['child'].append(action_idx)
            rand_idx = np.random.randint(low=0, high=len(childs), size=1)
            # print(rand_idx)
            # print('childs: ', childs)
            child_node_id = childs[rand_idx[0]]
        return child_node_id

    def _is_terminal(self, leaf_state):
        '''
        check terminal
        in: game state
        out: who wins? ('o', 'x', 'draw', None)
             (None = game not ended)
        '''
        def __who_wins(sums, win_mark):
            if np.any(sums == win_mark):
                return 'o'
            if np.any(sums == -win_mark):
                return 'x'
            return None

        def __is_terminal_in_conv(leaf_state, win_mark):
            # check row/col
            for axis in range(2):
                sums = np.sum(leaf_state, axis=axis)
                result = __who_wins(sums, win_mark)
                if result is not None:
                    return result
            # check diagonal
            for order in [-1,1]:
                diags_sum = np.sum(np.diag(leaf_state[::order]))
                result = __who_wins(diags_sum, win_mark)
                if result is not None:
                    return result
            return None

        win_mark = self.win_mark
        n_rows_board = len(self.tree[(0,)]['state'])
        window_size = win_mark
        window_positions = range(n_rows_board - win_mark + 1)

        for row in window_positions:
            for col in window_positions:
                window = leaf_state[row:row+window_size, col:col+window_size]
                winner = __is_terminal_in_conv(window, win_mark)
                if winner is not None:
                    return winner

        if not np.any(leaf_state == 0):
            '''
            no more action i can do
            '''
            return 'draw'
        return None

    def _get_valid_actions(self, leaf_state):
        '''
        return all possible action in current leaf state
        in:
        - leaf_state
        out:
        - set of possible actions ((row,col), action_idx)
        '''
        actions = []
        count = 0
        state_size = len(leaf_state)

        for i in range(state_size):
            for j in range(state_size):
                if leaf_state[i][j] == 0:
                    actions.append([(i, j), count])
                count += 1
        return actions

    def simulation(self, child_node_id):
        '''
        simulate game from child node's state until it reaches the resulting state of the game.
        in:
        - child node id (randomly selected child node id from `expansion`)
        out:
        - winner ('o', 'x', 'draw')
        '''
        self.total_n += 1
        state = deepcopy(self.tree[child_node_id]['state'])
        previous_player = deepcopy(self.tree[child_node_id]['player'])
        anybody_win = False

        """
        if previous_player == 'o':
            previous_player = 'x'
        else:
            previous_player = 'o'
        """

        # print("State: \n", state)

        while not anybody_win:
            winner = self._is_terminal(state)
            if winner is not None:
                anybody_win = True
                possible_scores = {
                    'o': -1,
                    'x': 1,
                    'draw': 0
                }
                self.tree[child_node_id]['minimax_score'] = possible_scores[winner]
            else:
                possible_actions = self._get_valid_actions(state)
                # randomly choose action for simulation (= random rollout policy)
                rand_idx = np.random.randint(low=0, high=len(possible_actions), size=1)[0]
                action, _ = possible_actions[rand_idx]

                if previous_player == 'o':
                    current_player = 'x'
                    state[action] = -1
                else:
                    current_player = 'o'
                    state[action] = 1

                # print("State in while: \n ", state)

                previous_player = current_player
        return winner

    def backprop(self, child_node_id, winner):
        player = deepcopy(self.tree[(0,)]['player'])

        if winner == 'draw':
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        finish_backprob = False
        node_id = child_node_id
        while not finish_backprob:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
            parent_id = self.tree[node_id]['parent']
            if parent_id == (0,):
                self.tree[parent_id]['n'] += 1
                self.tree[parent_id]['w'] += reward
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
                finish_backprob = True
            else:
                node_id = parent_id

    def solve(self):

        if self.tree[(0,)]['state'][1][1] == 0:
            print("DEFAULTING TO CENTER")
            best_action = 4
            best_q = 0
            depth_searched = 0
        else:                
            
            for i in range(self.n_iterations):
                leaf_node_id, depth_searched = self.selection() # Selection
                child_node_id = self.expansion(leaf_node_id) # Expansion
                """
                # TREE
                print("SELF TREE: ", self.tree[(0,)])
                temp_state = deepcopy(self.tree[(0,)]['state'])
                possible_actions = self._get_valid_actions(temp_state)
                for possible_action in possible_actions:
                    print("POSSIBLE ACTION: ", possible_action)
                    temp_state = deepcopy(self.tree[(0,)]['state'])
                    action, _ = possible_action
                    temp_state[action] = 1
                    print("Temp State: \n", temp_state)
                    winner = self._is_terminal(temp_state)
                    if winner is not None:
                        print('winner identified')
                        best_action = action
                        print('best action: ', action)
                        best_q = 0
                        depth_searched = 0
                        return best_action, best_q, depth_searched
                """
                winner = self.simulation(child_node_id) # Simulation
                self.backprop(child_node_id, winner) # Back
                if depth_searched > self.depth:
                    break

            # Check Terminality (See if MCTS can win)
            temp_state = deepcopy(self.tree[(0,)]['state'])
            possible_actions = self._get_valid_actions(temp_state)
            
            for i in range(0, len(self.tree[(0,)]['child'])):
                temp_state = deepcopy(self.tree[(0,)]['state'])
                action, _ = possible_actions[i]
                temp_state[action] = -1
                winner = self._is_terminal(temp_state)
                if winner is not None:
                    best_action = self.tree[(0,)]['child'][i]
                    print('best action: ', action)
                    best_q = 0
                    depth_searched = 0
                    return best_action, best_q, depth_searched

            # Check Terminality (See if user can win)
            temp_state = deepcopy(self.tree[(0,)]['state'])
            possible_actions = self._get_valid_actions(temp_state)
            for i in range(0, len(self.tree[(0,)]['child'])):
                temp_state = deepcopy(self.tree[(0,)]['state'])
                action, _ = possible_actions[i]
                temp_state[action] = 1
                winner = self._is_terminal(temp_state)
                if winner is not None:
                    best_action = self.tree[(0,)]['child'][i]
                    print('best action: ', action)
                    best_q = 0
                    depth_searched = 0
                    return best_action, best_q, depth_searched

            # print("SELF TREE IN SOLVE: \n", self.tree)


            # SELECT BEST ACTION
            current_state_node_id = (0,)
            action_candidates = self.tree[current_state_node_id]['child']
            
            """ minimax """
            tree = deepcopy(self.tree)
            tree_len = len(tree)
            tree[(0,)]['minimax_turn'] = 'max'

            def minimax(tree, current_node):
                player = tree[current_node]['player']
                if tree[current_node]['minimax_turn'] == 'max':
                    best = -100
                else: # player is 'o'
                    best = 100

                if tree[current_node]['child'] == [] and 'minimax_score' in tree[current_node].keys():
                    return tree[current_node]['minimax_score']

                if tree[current_node]['child'] == [] and 'minimax_score' not in tree[current_node].keys():
                    return 0

                for child in tree[current_node]['child']:
                    child_node = current_node + (child,)

                    score = minimax(tree, child_node)

                    if tree[current_node]['minimax_turn'] == 'max':
                        if score > best:
                            best = score
                    else: # minimax turn is 'min'
                        if score < best:
                            best = score
                    
                tree[current_node]['minimax_score'] = best

                return best

            #score = minimax(tree, [(0,)])
            tree[(0,)]['minimax_score'] = minimax(tree, (0,))

            """ """

            #for child in tree[(0,)]['child']:
            #    print("Child in solve: {}: \n".format(child), self.tree[(0,child)])
            tree[(0,)]['minimax_turn'] = 'max'

            """
            print("TREE IN MINIMAX: ", tree)
            print("TREE IN MINIMAX: ", len(tree))

            print("\n \n == I AM ROOT (0,): == \n \n", tree[(0,)])
            first_child = tree[(0,)]['child'][0]

            print("\n \n == CHILDREN OF ROOT == \n \n")
            for child in tree[(0,)]['child']:
                print("\n CHILD OF ROOT TREE[0,{}]: \n ".format(child), "\n")
                print("   ", tree[0,child], "\n \n")
                print("\n   GRANDCHILDREN OF ROOT: \n")
                for child2 in tree[(0,child)]['child']:
                    print("      TREE[0,{},{}]:".format(child, child2), tree[0,child,child2], "\n")
                    print("\n        GREAT-GRANDCHILDREN OF ROOT: \n")
                    for child3 in tree[(0,child, child2)]['child']:
                        print("          TREE[0,{},{},{}]:".format(child, child2, child3), tree[0,child,child2, child3], "\n")
                        print("\n            GREAT-GREAT-GRANDCHILDREN OF ROOT: \n")
                        for child4 in tree[(0,child, child2, child3)]['child']:
                            print("              TREE[0,{},{},{},{}]:".format(child, child2, child3, child4), tree[0,child,child2, child3, child4], "\n")
                            print("\n                GREAT-GREAT-GREAT-GRANDCHILDREN OF ROOT: \n")
                            for child5 in tree[(0,child, child2, child3, child4)]['child']:
                                print("                  TREE[0,{},{},{},{},{}]:".format(child, child2, child3, child4, child5), tree[0,child,child2, child3, child4, child5], "\n")
                                print("\n                    GREAT-GREAT-GREAT-GREAT-GRANDCHILDREN OF ROOT: \n")
                                for child6 in tree[(0,child, child2, child3, child4, child5)]['child']:
                                    print("                      TREE[0,{},{},{},{},{},{}]:".format(child, child2, child3, child4, child5, child6), tree[0,child,child2, child3, child4, child5, child6], "\n")
                                
                    print('          === \n')

            """
                        
            
            """
            print("NODE (0,N)): \n", tree[(0,first_child)])
            first_child2 = tree[(0,first_child)]['child'][0]
            print("NODE (0,N,N)): \n", tree[(0,first_child,first_child2)])
            first_child3 = tree[(0,first_child,first_child2)]['child'][0]
            print("NODE (0,N,N,N)): \n", tree[(0,first_child,first_child2,first_child3)])
            
            for child in tree[(0,first_child,first_child2,first_child3)]['child']:
                print("NODE (0, N, N, N, N): \n", tree[(0,first_child,first_child2,first_child3,child)])
            """


            """ q-score
            best_q = -100
            print("Action Candidates: ", action_candidates)
            for a in action_candidates:
                print("A: ", a)
                q = self.tree[(0,)+(a,)]['q']
                if q > best_q:
                    best_q = q
                    best_action = a
                    print('best action: ', best_action)
            """

            """ minimax scoring """
            best_score = -100
            for a in action_candidates:
                score = tree[(0,a)]['minimax_score']
                if score > best_score:
                    best_score = score
                    best_action = a
                    best_q = best_score

            print("GAME STATE: ", self.tree[(0,)]['state'])
            print(self.tree[(0,)]['state'][0])
            print(self.tree[(0,)]['state'][1])
            print(self.tree[(0,)]['state'][2])
            

            """
            # FOR DEBUGGING
            print('\n----------------------')
            print(' [-] game board: ')
            for row in self.tree[(0,)]['state']:
                print (row)
            print(' [-] person to play: ', self.tree[(0,)]['player'])
            print('\n [-] best_action: %d' % best_action)
            print(' best_q = %.2f' % (best_q))
            print(' [-] searching depth = %d' % (depth_searched))
            # FOR DEBUGGING
            fig = plt.figure(figsize=(5,5))
            for a in action_candidates:
                # print('a= ', a)
                _node = self.tree[(0,)+(a,)]
                _state = deepcopy(_node['state'])
                _q = _node['q']
                _action_onehot = np.zeros(len(_state)**2)
                # _state[_action_onehot] = -1
                # print('action = %d, q = %.3f' % (a, _q))
                # print('state after action: ')
                # for _row in _state:
                #     print(_row)
                plt.subplot(len(_state),len(_state),a+1)
                plt.pcolormesh(_state, alpha=0.7, cmap="RdBu")
                plt.axis('equal')
                plt.gca().invert_yaxis()
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('[%d] q=%.2f' % (a,_q))
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)
            """


        return best_action, best_q, depth_searched


'''
for test
'''
# if __name__ == '__main__':
#     mcts = VanillaMCTS(n_iterations=100, depth=10, exploration_constant=1.4, tree = None, n_rows=3, win_mark=3)
#     # leaf_node_id, depth = mcts.selection()
#     # child_node_id = mcts.expansion(leaf_node_id)
#     #
#     # print('child node id = ', child_node_id)
#     # print(' [*] simulation ...')
#     # winner = mcts.simulation(child_node_id)
#     # print(' winner', winner)
#     # mcts.backprop(child_node_id, winner)
#     best_action, max_q = mcts.solve()
#     print('best action= ', best_action, ' max_q= ', max_q)