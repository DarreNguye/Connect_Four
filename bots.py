import random
from engine import Board
from evaluator import alphabeta, search_best_move
from cnn_model import load_evaluator


def random_bot(board: Board) -> int:
    """
    The bot chooses random legal moves
    """
    return random.choice(board.legal_moves())


def _wins_immediately(board: Board, col: int, player: int) -> bool:
    """
    Check if the player would win immediately
    """
    if board.side != player:
        raise ValueError("only meaningful for the side to move")
    board.play(col)
    won = board.winner() == player
    board.undo()
    return won


def survival_bot(board: Board) -> int:
    """
    Find immediate wins, block immediate losses, otherwise prefer the center
    """

    # Define bot color and legal moves
    me = board.side
    moves = board.legal_moves()

    # Check if any move would lead to a win
    for col in moves:
        if _wins_immediately(board, col, me):
            return col

    # Block any immediate losses
    for col in moves:
        board.side = -me
        threat = _wins_immediately(board, col, -me)
        board.side = me
        if threat:
            return col

    # A preset preference of moves in the center
    order = [3, 2, 4, 1, 5, 0, 6]
    return next(c for c in order if c in moves)

def make_cnn_bot(depth: int, path: str = "models/value_net.pt"):
    """Build a bot that searches to `epth using the trained CNN at the leaves."""
    net_eval = load_evaluator(path)         
    return lambda board: search_best_move(board, depth, net_eval, alphabeta)

BOTS = {
    "Random": random_bot,
    "Survival": survival_bot,
    "CNN-D4": make_cnn_bot(4),
    "CNN-D6": make_cnn_bot(6),
}