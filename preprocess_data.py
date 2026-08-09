
from typing import NamedTuple, Iterator, List, Optional, Sequence
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Board dimensions
ROWS, COLS = 6, 7

# Every possible 4 in a row
LINES = np.array([
    [(r + i * dr) * COLS + (c + i * dc) for i in range(4)]
    for r in range(ROWS)
    for c in range(COLS)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS
])

# Arithmetic trick to make a height marker
BOTTOM_MASK = sum(1 << (c * 7) for c in range(COLS))

# =============================================================================
# DATA CLEANING
# =============================================================================

class Cleaned(NamedTuple):
    boards: np.ndarray   
    winners: np.ndarray
    audit: pd.DataFrame 
    stats: dict


def clean_connect_four(data, *, relabel=True, chunk=50_000, verbose=True) -> Cleaned:
    """
    Audit, relabel, and filter finished and possible Connect-4 positions.

    Parameters
    ----------
    data : array-like, shape (N, 43) or (N, 42)
        Raw rows, where every row represents a game and every column is a position. 
    relabel : bool
        If False, rows whose label disagrees with a legal board are dropped
        rather than corrected. Useful for measuring what relabeling buys you.
    chunk : int
        Rows per block when scanning lines, to bound peak memory.
    verbose : bool
        Print the summary table.

    Returns
    -------
    Cleaned(boards, winners, audit, stats)
    """

    # Checks data dimensions
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] not in (42, 43):
        raise ValueError(f"expected (N, 42) or (N, 43), got {data.shape}")

    # Checks for valid piece labels
    cells = data[:, :42].astype(np.int8)
    N = len(cells)
    if not np.isin(cells, (-1, 0, 1)).all():
        raise ValueError("board cells contain values outside {-1, 0, 1}")

    # Pull winner information if exists
    has_labels = data.shape[1] == 43
    winner = data[:, 42].astype(np.int8) if has_labels else np.zeros(N, np.int8)

    # Accounts for pieces placed
    b = cells.reshape(N, 6, 7)
    filled = b != 0
    n1 = (b == 1).sum((1, 2))
    n2 = (b == -1).sum((1, 2))
    diff = (n1 - n2).astype(np.int8)
    total = n1 + n2

    # Possible 4 in a row combinations
    lines = [
        [(r + i * dr) * 7 + (c + i * dc) for i in range(4)]
        for r in range(6)
        for c in range(7)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
        if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7
    ]
    lines = np.array(lines)  # (69, 4)

    # Gravity rule: A filled cell must rest on a filled cell
    gravity_ok = np.all(filled[:, :-1, :] <= filled[:, 1:, :], axis=(1, 2))

    # The pieces on top that were possibly placed last
    below_empty = np.concatenate([np.ones((N, 1, 7), bool), ~filled[:, :-1, :]], 1)
    is_top = (filled & below_empty).reshape(N, 42)

    # Iterate across chunks of games
    p1_line = np.empty((N, 69), bool)
    p2_line = np.empty((N, 69), bool)
    line_top = np.empty((N, 69), bool)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)

        # Pull all possible connect 4 in a board
        w = cells[lo:hi][:, lines]

        # Mark where one player owns all pieces in a connect 4
        p1_line[lo:hi] = (w == 1).all(2)
        p2_line[lo:hi] = (w == -1).all(2)

        # Mark connect 4 lines where at least 1 piece is at the top of the column
        line_top[lo:hi] = is_top[lo:hi][:, lines].any(2)

    # Determine if the player has a connect 4
    has1, has2 = p1_line.any(1), p2_line.any(1)
    detected = np.where(has1 & ~has2, 1, np.where(has2 & ~has1, -1, 0)).astype(np.int8)

    # A win is only valid if the winning disc was just placed
    just_won = np.where(
        detected == 1,
        (p1_line & line_top).any(1),
        np.where(detected == -1, (p2_line & line_top).any(1), False),
    )

    # Identify any invalid boards
    reasons = {
        "floating_discs": ~gravity_ok,
        "piece_gap": np.abs(diff) > 1,
        "two_winners": has1 & has2,
        "loser_moved_last": (winner != 0) & (diff != 0) & (np.sign(diff) != winner),
        "win_completed_early": (detected != 0) & ~just_won,
    }
    impossible = np.logical_or.reduce(list(reasons.values()))

    # Identify any valid boards mislabeled
    mislabeled = (detected != winner) & ~impossible

    # Boards without a valid connect four and ended early
    unrecoverable = (winner == 0) & (total != 42) & (detected == 0) & ~impossible
    reasons["outcome_unknown"] = unrecoverable

    # Drop invalid boards
    drop = impossible | unrecoverable
    if relabel:
        fixed = np.where(mislabeled, detected, winner).astype(np.int8)
        n_relabeled = int(mislabeled.sum())
    else:
        drop = drop | mislabeled
        fixed = winner.copy()
        n_relabeled = 0
    keep = ~drop

    # Organize audit information
    audit = pd.DataFrame(
        {
            "row": np.arange(N),
            "diff": diff,
            "total": total,
            "winner_raw": winner,
            "winner_detected": detected,
            "winner_fixed": fixed,
            "just_won": just_won,
            "impossible": impossible,
            "mislabeled": mislabeled,
            "dropped": drop,
            **{f"why_{k}": v for k, v in reasons.items()},
        }
    )

    # General statistics
    stats = {
        "rows_in": N,
        "rows_kept": int(keep.sum()),
        "dropped": int(drop.sum()),
        "relabeled": n_relabeled,
        **{k: int(v.sum()) for k, v in reasons.items()},
        "labels": {int(k): int(v) for k, v in
                   zip(*np.unique(fixed[keep], return_counts=True))},
    }

    # Print a summary table
    if verbose:
        pad = max(len(k) for k in reasons) + 2
        print(f"{'rows in':<{pad}} {N:>9,}")
        for k, v in reasons.items():
            print(f"  {k:<{pad-2}} {int(v.sum()):>9,}")
        print("-" * (pad + 10))
        print(f"{'dropped':<{pad}} {stats['dropped']:>9,}"
              f"  ({100 * stats['dropped'] / N:.2f}%)")
        print(f"{'relabeled':<{pad}} {stats['relabeled']:>9,}")
        print(f"{'kept':<{pad}} {stats['rows_kept']:>9,}")
        print(f"{'labels':<{pad}} {stats['labels']}")

    return Cleaned(b[keep].copy(), fixed[keep], audit, stats)

# =============================================================================
# POSITION TO MOVE ORDER
# =============================================================================

def infer_starter(n_mover: int, n_other: int, winner: int) -> Optional[int]:
    """
    Who played first, from the piece counts and the result.
    """

    # Difference in amount of pieces placed
    diff = n_mover - n_other

    # The player with an extra disc started and ended
    if diff != 0:
        return 1 if diff > 0 else -1

    # The starter is whoever did not move last
    if winner != 0:
        return -winner

    # Draw
    return None


def _has_line(stacks: List[List[int]]) -> bool:
    """Does any completed four exist in the partially-peeled board?"""

    # Flatten
    flat = np.zeros(ROWS * COLS, np.int8)
    for c, stack in enumerate(stacks):
        for i, v in enumerate(stack):          
            flat[(ROWS - 1 - i) * COLS + c] = v

    # Check for connect fours
    w = flat[LINES]
    return bool(((w == 1).all(1) | (w == -1).all(1)).any())


def _peel(stacks, t, starter, n, winner, order, budget) -> bool:
    """Remove piece (t, t-1, t-2,...) one at a time in DFS in a legal way."""

    # All piece removed (base case)
    if t == 0:
        return True

    # Add a budget to stop cases that take too long
    if budget[0] <= 0:
        return False
    budget[0] -= 1

    # Determine the player
    player = starter if t % 2 == 1 else -starter

    # Possible columns that disc t could be from 
    candidates = [c for c in range(COLS) if stacks[c] and stacks[c][-1] == player]

    # For each possible column
    for col in candidates:

        # Try to remove that piece
        stacks[col].pop()

        # Check if any connect fours still exist
        legal = not (t == n and winner != 0 and _has_line(stacks))

        # Keep removing pieces (recursive call)
        if legal and _peel(stacks, t - 1, starter, n, winner, order, budget):

            # Record the order
            order[t - 1] = col
            stacks[col].append(player)
            return True

        # Add piece back in the case of failure, backtracking
        stacks[col].append(player)

    # No candidate worked
    return False

def reconstruct_order(grid, winner, budget = 10e6) -> Optional[List[int]]:
    """Find a legal move order that produces this final board.

    Parameters
    ----------
    grid : (6, 7) array, row 0 = top, values in {-1, 0, 1}
    winner : +1 / -1 / 0
    budget: Alloted budget that filters out orders that take too long

    Returns
    -------
    List of columns in play order, or None if no ordering exists
    """

    # Convert to array
    grid = np.asarray(grid, dtype=np.int8).reshape(ROWS, COLS)

    # Construct stacks for each column
    stacks = [[int(grid[r, c]) for r in range(ROWS - 1, -1, -1) if grid[r, c] != 0]
              for c in range(COLS)]

    # Check if board has pieces
    n = sum(len(s) for s in stacks)
    if n == 0:
        return []

    # Count each player pieces
    n1 = int((grid == 1).sum())
    n2 = int((grid == -1).sum())

    # Find who moved first 
    starter = infer_starter(n1, n2, winner)
    candidates = [starter] if starter is not None else [1, -1]

    # Reconstruct from the inferred starter
    for start in candidates:
        order = [0] * n

        # Start the recursive call of reconstructing order
        if _peel(stacks, n, start, n, winner, order, [budget]):
            return order

    # Unable to reconstruct
    return None


# =============================================================================
# TRAIN DATA TRANSFORM
# =============================================================================

def position_key(mover_bits: int, full_bits: int) -> int:
    """Unique 49-bit key for a board, from the side-to-move's view"""
    return mover_bits + full_bits + BOTTOM_MASK


def expand_game(order: Sequence[int], winner: int, starter: int) -> Iterator[tuple]:
    """
    For every order of moves in a complete game, generate position subsets for training data
    """

    heights = [0] * COLS
    bits = {1: 0, -1: 0}

    for t, col in enumerate(order):
        mover = starter if t % 2 == 0 else -starter

        full = bits[1] | bits[-1]
        z = 0 if winner == 0 else (1 if winner == mover else -1)
        yield position_key(bits[mover], full), z

        bits[mover] |= 1 << (col * 7 + heights[col])
        heights[col] += 1


def decode_key(key: int) -> np.ndarray:
    """Transforms the key to the board"""
    planes = np.zeros((2, ROWS, COLS), np.float32)

    for c in range(COLS):
        field = (key >> (c * 7)) & 0x7F
        height = field.bit_length() - 1
        mover_col = field & ((1 << height) - 1)
        for r in range(height):
            row = ROWS - 1 - r
            plane = 0 if (mover_col >> r) & 1 else 1
            planes[plane, row, c] = 1.0
    return planes


# =============================================================================
# DATASET BUILD
# =============================================================================

def build_dataset(boards, winners, verbose: bool = True):
    """
    Build a dataset of inter-game positions to train on from the final position data

    Parameters
    ----------
    boards : (M, 6, 7) int8
    winners : (M,) int8

    Returns
    -------
    keys   : (U,) uint64  -- unique positions
    values : (U,) float32 -- mean outcome from the mover's perspective
    counts : (U,) int32   -- how many games visited each position
    """

    # Format boards and winners
    boards = np.asarray(boards, np.int8).reshape(-1, ROWS, COLS)
    winners = np.asarray(winners, np.int8)

    # Record keeping
    all_keys, all_z, failed = [], [], []

    # Iterate across games
    for i, (grid, winner) in enumerate(tqdm(zip(boards, winners), total=len(boards),
                                        desc="reconstructing", disable=not verbose)):

        # Reconstruct the move order (failures are skipped)
        order = reconstruct_order(grid, int(winner))
        if order is None:
            failed.append(i)
            continue

        # Reinfer the player who started
        n1, n2 = int((grid == 1).sum()), int((grid == -1).sum())
        starter = infer_starter(n1, n2, int(winner))
        if starter is None:
            starter = 1 if len(order) % 2 == 0 else -1

        # Expand the game, generating position subsets for each game
        for key, z in expand_game(order, int(winner), starter):
            all_keys.append(key)
            all_z.append(z)

    # Format
    keys = np.array(all_keys, dtype=np.uint64)
    z = np.array(all_z, dtype=np.float32)

    # Calculate an average win rate per position
    uniq, inverse = np.unique(keys, return_inverse=True)
    totals = np.bincount(inverse, weights=z)
    counts = np.bincount(inverse).astype(np.int32)
    values = (totals / counts).astype(np.float32)

    # Summary
    if verbose:
        print(f"\n  raw positions   {len(keys):>10,}")
        print(f"  unique          {len(uniq):>10,}  ({100*len(uniq)/len(keys):.1f}%)")
        print(f"  seen once       {int((counts == 1).sum()):>10,}")
        print(f"  most visited    {int(counts.max()):>10,} times")
        if failed:
            print(f"  UNRECONSTRUCTED {len(failed):>10,} games")

    return uniq, values, counts, failed