# monte-carlo-tree-search
This repo uses a basic Monte Carlo Tree Search (MCTS) algorithm to create an AI capable of playing tic-tac-toe. While the underlying monte-carlo-tree-search algorithm remains the same throughout each notebook, the tree building policies, default policies, and selection scoring functions differ.

#### Tree Building Policies Include:
* UCT

#### Default Policy
* Terminality-Seeking Behavior
* Random Rollout

#### Selection Scoring Functions Include:
* Q-scores (w/n)
* Minimax



So far, a model combining a Minimax selection function with a UCT tree policy and terminality-seeking-behavior stands head and shoulders above the other models.
