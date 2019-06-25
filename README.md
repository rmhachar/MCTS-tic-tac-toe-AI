# monte-carlo-tree-search
This repo uses a basic Monte Carlo Tree Search (MCTS) algorithm to create an AI capable of playing tic-tac-toe. While the underlying monte-carlo-tree-search algorithm remains the same throughout each notebook, the scoring functions and tree building policies differ.

Scoring Functions Include:
* UCT (Standard MCTS Q-Scores)
* Minimax

Tree Building Policies Include:
* Random Rollout
* Terminality Seeking Behavior

So far, a model combining Minimax with a terminality-seeking tree building policy stands head and shoulders above the other models.
