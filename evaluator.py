from engine import Board
from typing import Callable

# Define the win incentive
WIN = 10000

# Prefer center columns first
CENTER_FIRST = {3: 0, 2: 1, 4: 2, 1: 3, 5: 4, 0: 5, 6: 6}

def alphabeta(board: Board, depth: int, evaluator: Callable[[Board], int],
              alpha: int = -WIN, beta: int = WIN) -> int:
    """
    Score a position from the side to move, pruning branches the opponent would block.
    Alpha represents the floor score for a player, beta represents the floor score for the opponent
    """

    # Check if the game is still ongoing (base case)
    result = board.winner()
    if result is not None:

        # Tie is neutral points
        # Lengthy losses are better than quick losses
        return 0 if result == 0 else -(WIN - len(board.history))

    # Evaluate the board from a non-terminal point
    if depth == 0:
        return evaluator(board)

    # Prioritize center columns
    moves = sorted(board.legal_moves(), key=lambda c: CENTER_FIRST[c])

    # Find the best of all possible moves (recursive step)
    best = -WIN
    for col in moves:

        # Attempt a move
        board.play(col)

        # Score from the perspective of the opponent
        score = -alphabeta(board, depth - 1, evaluator, -beta, -alpha)
        board.undo()

        # Update alpha and beta
        best = max(best, score)
        alpha = max(alpha, best)

        # Prune when alpha exceeds beta, as the opponent will try to block it
        if alpha >= beta:
            break

    return best

def search_best_move(board: Board, depth: int, evaluator: Callable[[Board], int], search: Callable = alphabeta) -> int:
    """Search for the next best move"""

    # Memory
    scored = []
    alpha = -WIN

    # Iterate across possible moves (prioritize central columns)
    for col in sorted(board.legal_moves(), key=lambda c: CENTER_FIRST[c]):

        # Attempt a move
        board.play(col)

        # Score from the perspective of the opponent
        score = -search(board, depth - 1, evaluator, -WIN, -alpha)
        board.undo()

        # Track score and alpha
        scored.append((score, col))
        alpha = max(alpha, score)

    # Return the best score (tie-breaker is the most center column)
    return max(scored, key=lambda s: (s[0], -CENTER_FIRST[s[1]]))[1]

def heuristic_eval(board: Board) -> int:
    pass