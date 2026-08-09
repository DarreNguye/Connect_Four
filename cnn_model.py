from typing import Callable
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

from engine import Board

# Board Size
ROWS, COLS = 6, 7

# A nice number that depends on WIN points, informative enough
SCALE = 1000

# bit_length lookup for a 7-bit column field (used by decoder)
HEIGHT_LUT = np.array([max(0, int(v).bit_length() - 1) for v in range(128)], np.uint8)


# =============================================================================
# NETWORK
# =============================================================================

class ValueNet(nn.Module):
    """
    Basic CNN model
    Input Dimensions: 2 x 6 x 7 (the board size with a separate dimension for each player)
    Kernel Size: 3
    Padding: 1 (required to keep the board the same dimensions)
    Filters: 64 (to be optimized)
    Blocks: 3 (to be optimized)
    Head channels (8) and hidden units (64): Nice numbers
    Tanh: Matches the output of [-1, 0, +1]
    """

    def __init__(self, input_dim=2, kernel_size=3, padding=1, filters=64,
                 blocks=3, head_channels=8, hidden_units=64):
        
        super().__init__()

        # Configs
        self.config = dict(input_dim=input_dim, kernel_size=kernel_size,
                           padding=padding, filters=filters, blocks=blocks,
                           head_channels=head_channels, hidden_units=hidden_units)

        # Entry
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, filters, kernel_size, padding=padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Depth of CNN (2 convolutions per blocks)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size, padding=padding),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(filters, filters, kernel_size, padding=padding),
                nn.BatchNorm2d(filters),
            )
            for _ in range(blocks)
        )

        # Flattening and summarizing head
        self.head = nn.Sequential(
            nn.Conv2d(filters, head_channels, 1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(head_channels * ROWS * COLS, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        # Entry stem
        x = self.stem(x)

        # Depth of residual CNN (adds the block as a correction unit for optimality)
        for block in self.blocks:
            x = torch.relu(x + block(x))

        # Head squeeze
        return self.head(x).squeeze(-1)


# =============================================================================
# ENCODING
# =============================================================================

def encode(board: Board) -> np.ndarray:
    """
    Encodes the Board to 2x6x7
    Plane 0 holds the mover's discs, and plane 1 holds the opponent's, as
    piece labels are categorical, and a singular plane would convolute unecessarily
    """

    # Set as an array
    grid = np.asarray(board.grid, dtype=np.int8)

    # Specify te mover
    mover = board.side

    # Split into the mover and the opponent
    return np.stack([(grid == mover), (grid == -mover)]).astype(np.float32)


def cache_key(board: Board) -> bytes:
    """Builds a key to caches a board, saving computational time"""
    return np.asarray(board.grid, dtype=np.int8).tobytes() + bytes([board.side & 0xFF])


# =============================================================================
# EVALUATOR ADAPTER
# =============================================================================

class NetEvaluator:
    """
    Wraps a trained ValueNet as an evaluator.
    """

    def __init__(self, model: ValueNet, device: str = "cpu",
                 mirror: bool = False, cache: bool = True):

        # Sets configs and cache information
        self.model = model.to(device).eval()
        self.device = device
        self.mirror = mirror
        self.cache = {} if cache else None
        self.calls = 0      
        self.hits = 0       

    @torch.no_grad()
    def __call__(self, board: Board) -> int:
        """Allows the evaluator to be called"""

        # Checks if in cached
        if self.cache is not None:
            key = cache_key(board)
            cached = self.cache.get(key)
            if cached is not None:
                self.hits += 1
                return cached

        # Encode the board
        x = encode(board)

        # The board is symmetric, so check the mirror (optionally)
        batch = np.stack([x, x[:, :, ::-1]]) if self.mirror else x[None]

        # Convert to tensor
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)

        # Average in the mirror case, nothing meaningful in the non mirror
        value = float(self.model(tensor).mean())

        # Calculate a score for the evaluator
        self.calls += 1
        score = int(round(value * SCALE))

        # Cache
        if self.cache is not None:
            self.cache[key] = score
        return score

    def reset_stats(self):
        """Reset calls and hits"""
        self.calls = self.hits = 0

    def clear_cache(self):
        """Clears the cache"""
        if self.cache is not None:
            self.cache.clear()


def load_evaluator(path: str, device: str = "cpu", **kwargs) -> Callable[[Board], int]:
    """Load a checkpoint saved by the training script and return an evaluator"""
    ckpt = torch.load(path, map_location=device)
    model = ValueNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return NetEvaluator(model, device=device, **kwargs)


def save_model(model: ValueNet, path: str, **meta):
    """Saves the model to a file"""
    torch.save({"state_dict": model.state_dict(),
                "config": model.config, **meta}, path)

# =============================================================================
# KEY TO BOARD
# =============================================================================

def decode_keys(keys) -> np.ndarray:
    """Transforms a key to a board"""
    keys = np.asarray(keys, np.uint64)
    out = np.zeros((len(keys), 2, ROWS, COLS), np.float32)

    for c in range(COLS):
        field = ((keys >> np.uint64(7 * c)) & np.uint64(0x7F)).astype(np.uint8)
        height = HEIGHT_LUT[field]
        for r in range(ROWS):
            occupied = r < height
            is_mover = ((field >> np.uint8(r)) & np.uint8(1)).astype(bool)
            out[occupied & is_mover, 0, ROWS - 1 - r, c] = 1.0
            out[occupied & ~is_mover, 1, ROWS - 1 - r, c] = 1.0
    return out


def mirror_keys(keys) -> np.ndarray:
    """Left-right flip of the board"""
    keys = np.asarray(keys, np.uint64)
    out = np.zeros_like(keys)
    for c in range(COLS):
        field = (keys >> np.uint64(7 * c)) & np.uint64(0x7F)
        out |= field << np.uint64(7 * (COLS - 1 - c))
    return out


# =============================================================================
# SPLIT
# =============================================================================

def split_indices(keys, validation_frac=0.05, seed=22):
    """
    Train/validation split that keeps mirrors of a board on the same side.
    """

    # Choose the minimum of the key of a board and its mirror
    canon = np.minimum(np.asarray(keys, np.uint64), mirror_keys(keys))
    groups, inverse = np.unique(canon, return_inverse=True)

    # Shuffle
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(groups))
    n_val = int(len(groups) * validation_frac)
    is_val_group = np.zeros(len(groups), bool)
    is_val_group[shuffled[:n_val]] = True

    # Splits into train and validation
    is_val = is_val_group[inverse]

    # Format into index arrays
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def make_weights(counts, clip_percentile=99.9):
    """Weight each position by how many games visited it"""
    w = np.asarray(counts, np.float32)
    w = np.minimum(w, np.percentile(w, clip_percentile))
    return (w / w.mean()).astype(np.float32)


# =============================================================================
# TRAIN
# =============================================================================

def run_epoch(model, optimizer, keys, values, weights, idx, batch_size,
              device, train, rng, augment=True):
    """One pass over the positions in idx, returning the weighted mean loss.

    Parameters
    ----------
    model : ValueNet
    optimizer : torch optimiser 
    keys : (N,) uint64, the full position array
    values : (N,) float32, target win rates from the mover's perspective
    weights : (N,) float32, per-position loss weights
    idx : (M,) int, which positions this pass covers
    batch_size : positions per forward pass
    device : "cpu" or "cuda"
    train : if True, shuffle, augment, and step the optimiser
    rng : np.random.Generator
    augment : mirror roughly half of each batch left-right

    Returns
    -------
    float: loss averaged over positions, weighted by weights.
    """

    # Set this on train mode if enabled
    model.train(train)

    # Record keeping
    total_loss, total_w = 0.0, 0.0

    # Shuffle training order
    if train:
        idx = idx[rng.permutation(len(idx))]

    # Batching
    for lo in range(0, len(idx), batch_size):
        batch = idx[lo:lo + batch_size]

        # Transforms a key into a board
        x = decode_keys(keys[batch])

        # Mirrors board
        if train and augment:
            flip = rng.random(len(batch)) < 0.5
            x[flip] = x[flip][:, :, :, ::-1]

        # Tensor conversions
        xb = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        yb = torch.from_numpy(values[batch]).to(device)
        wb = torch.from_numpy(weights[batch]).to(device)

        # Weighted MSE
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = (wb * (pred - yb) ** 2).sum() / wb.sum()

        # Gradient descent
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute loss and record
        batch_w = wb.sum().item()
        total_loss += loss.detach().item() * batch_w
        total_w += batch_w

    return total_loss / total_w

def train_value_net(data, out, epochs, filters, blocks, lr, batch_size=1024, 
                    augment=True, patience=3, device=None, seed=22, verbose=True):
    
    """
    Train a ValueNet using

    Optimizer: Adam (commonly used)

    Parameters
    ----------
    data : path to an .npz holding keys, values, counts
    out : path for the checkpoint; overwritten each time val improves
    epochs : maximum passes over the data -- early stopping usually ends sooner
    filters, blocks : ValueNet architecture; 64 and 3 are sensible defaults
    lr : initial Adam learning rate. Halved after a plateau.
    batch_size : positions per forward pass
    augment : mirror roughly half of each batch left-right
    patience : consecutive epochs without val improvement before stopping
    device : "cpu" or "cuda"; auto-detected when None
    seed : seeds both the split and the shuffling, so runs are reproducible
    verbose : print per-epoch lines and the progress bar

    Returns
    -------
    (model, history) the best model, and a DataFrame with one row per epoch
    holding train and val loss.
    """

    # Set the device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    d = np.load(data)
    keys, values, counts = d["keys"], d["values"].astype(np.float32), d["counts"]

    # Compute the weights and perform a split
    weights = make_weights(counts)
    train_idx, val_idx = split_indices(keys, seed=seed)

    # Details on splits
    if verbose:
        print(f"positions {len(keys):,}   train {len(train_idx):,}   val {len(val_idx):,}")
        print(f"targets: mean {values.mean():+.3f}  "
              f"|v|=1 {np.mean(np.abs(values) == 1):.1%}   device {device}\n")

    # Initialize the model and other tools
    model = ValueNet(filters=filters, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    rng = np.random.default_rng(seed)

    # Record keeping
    history, best, stale = [], float("inf"), 0

    # Run multiple epochs
    epoch_bar = tqdm(range(epochs), desc="epochs", disable=not verbose, position=0)
    for epoch in epoch_bar:

        # Track time
        t0 = time.perf_counter()

        # Run a train epoch
        tr = run_epoch(model, optimizer, keys, values, weights, train_idx, batch_size,
                       device, train=True, rng=rng, augment=augment)

        # Run a validation epoch
        va = run_epoch(model, optimizer, keys, values, weights, val_idx, batch_size,
                       device, train=False, rng=rng)

        # Feeds validation loss to the scheduler
        sched.step(va)

        # Record
        history.append({"epoch": epoch, "train": tr, "val": va, "secs": time.perf_counter() - t0})

        # Keep the best performance, incrementing the stale counter on plateaus
        flag = ""
        if va < best - 1e-5:
            best, stale = va, 0
            save_model(model, out, epoch=epoch, val_loss=va, positions=len(keys))
            flag = "  <- saved"
        else:
            stale += 1

        # Summary information
        if verbose:
            epoch_bar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}",
                                  best=f"{best:.4f}", saved=bool(flag))

        # Stops after a certain plataeu of performance
        if stale >= patience:
            if verbose:
                tqdm.write(f"\nno improvement for {patience} epochs, stopping")
            break

    # Overwrite weights
    model.load_state_dict(torch.load(out, map_location=device)["state_dict"])

    # Summary information
    if verbose:
        print(f"\nbest val {best:.4f} -> {out}")

    return model, pd.DataFrame(history)
    