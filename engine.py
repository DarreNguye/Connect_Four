
# Board dimensions
ROWS, COLS = 6, 7

# grid[0] is the top row, grid[5] is the bottom
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """
    The Connect Four Board.
    """

    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS     
        self.history = []             
        self.side = 1                 

    def legal_moves(self):
        """All possible lega moves"""
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def play(self, col):
        """Place a piece in a column"""

        # Check if legal
        if col not in self.legal_moves():
            raise ValueError(f"column {col} is not playable")

        # Place a piece
        row = ROWS - 1 - self.heights[col]
        self.grid[row][col] = self.side
        self.heights[col] += 1

        # Add history
        self.history.append(col)

        # Switch side
        self.side = -self.side

    def undo(self):
        """Undo a move"""

        # Check if there is history
        if not self.history:
            raise ValueError("no move to undo")

        # Undo
        col = self.history.pop()
        self.heights[col] -= 1
        row = ROWS - 1 - self.heights[col]
        self.grid[row][col] = 0

        # Switch side
        self.side = -self.side

    def winning_line(self):
        """The four cells of a four in a row as [(r, c), ...], or None."""

        # Iterate across grid
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]

                # No piece
                if p == 0:
                    continue

                # Check the directions for a four in a row
                for dr, dc in DIRECTIONS:

                    # Check out of bounds
                    if not (0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS):
                        continue

                    # Check if a four in a row
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(self.grid[y][x] == p for y, x in cells):
                        return cells

        # No four in a row
        return None

    def winner(self):
        """+1 / -1 if won, 0 if drawn, None if the game is still running."""

        # Check for a winning line
        line = self.winning_line()
        if line:

            # Designate winner
            r, c = line[0]
            return self.grid[r][c]

        # Either continue play or a tie
        return None if self.legal_moves() else 0

    def is_over(self):
        """Is the game done"""
        return self.winner() is not None

    def copy(self):
        """Copy the board"""
        b = Board()
        b.grid = [row[:] for row in self.grid]
        b.heights = self.heights[:]
        b.history = self.history[:]
        b.side = self.side
        return b

    def __str__(self):
        """Board to string"""
        m = {1: "X", -1: "O", 0: "."}
        rows = "\n".join(" ".join(m[v] for v in row) for row in self.grid)
        return rows + "\n" + " ".join(str(c) for c in range(COLS))


def replay(columns):
    """Build a board from a list of played columns."""
    b = Board()
    for col in columns:
        b.play(col)
    return b