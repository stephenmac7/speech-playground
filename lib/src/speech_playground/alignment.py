import numpy as np
import matplotlib.pyplot as plt

def compute_similarity_matrix(x, y, *, dist_method, match_score, mismatch_score, alpha=None):
    """
    Computes a similarity matrix (N x M) normalized to range [-1, 1].
    Uses an RBF kernel for Euclidean distances to bound the scores.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros((x.shape[0], y.shape[0]))

    # Case 1: Discrete Matching (e.g., integers, strings)
    if dist_method is None:
        matches = x[:, None] == y[None, :]
        return np.where(matches, match_score, mismatch_score)

    # Case 2: Continuous Cosine Similarity
    elif dist_method == "cosine":
        # Normalize vectors to unit length
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
        
        # Dot product: Range [-1.0, 1.0]
        sim = np.dot(x_norm, y_norm.T)
        
        # Optional: Sharpen cosine similarities with alpha if provided
        if alpha is not None:
            sign = np.sign(sim)
            sim = sign * (np.abs(sim) ** alpha)
        
        return sim

    # Case 3: Continuous Euclidean Distance (Bounded with RBF)
    elif dist_method == "euclidean":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
        x_sq = np.sum(x**2, axis=1, keepdims=True)
        y_sq = np.sum(y**2, axis=1, keepdims=True)
        dists_sq = x_sq + y_sq.T - 2 * np.dot(x, y.T)
        dists_sq = np.maximum(dists_sq, 0) # Clip small numerical errors

        # Auto-tune alpha if not provided
        # We want the median distance to map to a score of ~0.0 (neutral)
        if alpha is None:
            sample = dists_sq.flatten()
            if len(sample) > 1000:
                sample = np.random.choice(sample, 1000, replace=False)
            median_sq = np.median(sample)
            # score = 2 * exp(-alpha * d^2) - 1
            # 0 = 2 * exp(-alpha * med) - 1  =>  0.5 = exp(...)  =>  ln(0.5) = -alpha * med
            # alpha = ln(2) / median
            if median_sq > 1e-6:
                alpha = np.log(2) / median_sq
            else:
                alpha = 1.0
            print("using alpha =", alpha)

        # RBF Kernel Shifted: Maps [0, inf) -> [1, -1]
        # Distance 0 -> Score 1.0 (Match)
        # Large Distance -> Score -1.0 (Mismatch)
        similarities = 2 * np.exp(-alpha * dists_sq) - 1
        
        return similarities * match_score

    else:
        raise ValueError(f"Unknown dist_method: {dist_method}")


def compute_alignment_grid(sim_matrix, gap_penalty, mode="global"):
    """
    Fills the DP grid using Global Alignment (Needleman-Wunsch) logic.
    """
    len_x, len_y = sim_matrix.shape
    grid = np.zeros((len_x + 1, len_y + 1))

    # Initialize first row and column with cumulative gap penalties
    grid[:, 0] = np.arange(len_x + 1) * gap_penalty
    
    if mode == "global":
        grid[0, :] = np.arange(len_y + 1) * gap_penalty
    elif mode == "semiglobal":
        grid[0, :] = 0.0
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")

    # Fill DP Grid
    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            score = sim_matrix[i - 1, j - 1]
            
            # Standard Needleman-Wunsch recurrence
            grid[i, j] = max(
                grid[i - 1, j] + gap_penalty,   # Vertical (Gap in Y / Insertion in X)
                grid[i, j - 1] + gap_penalty,   # Horizontal (Gap in X / Deletion from Y)
                grid[i - 1, j - 1] + score,     # Diagonal (Match/Mismatch)
            )

    return grid


def score_alignment(
    learner,
    reference,
    *,
    mode="global",
    gap_penalty=-0.5, # Adjusted default: since scores are [-1, 1], -0.5 is a "soft" penalty
    dist_method=None,
    match_score=1.0,
    mismatch_score=-1.0,
    alpha=None,
    tmpdir=None,
    only_return_score=False,
):
    x, y = learner, reference

    # 1. Compute Similarity Matrix (Bounded [-1, 1])
    sim_matrix = compute_similarity_matrix(
        x, y, dist_method=dist_method, match_score=match_score, mismatch_score=mismatch_score, alpha=alpha
    )
    if tmpdir:
        plt.plot()
        plt.imshow(sim_matrix.T, origin='lower', aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        plt.title('Similarity Matrix')
        plt.xlabel('Learner Frames')
        plt.ylabel('Reference Frames')
        plt.savefig(f"{tmpdir}/similarity_matrix.png")
        plt.close()

    # 2. Compute DP Grid
    grid = compute_alignment_grid(sim_matrix, gap_penalty, mode=mode)

    if only_return_score:
        if mode == "global":
            raw_score = grid[-1, -1]
            denom = max(len(x), len(y))
        elif mode == "semiglobal":
            raw_score = np.max(grid[-1, :])
            denom = len(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if denom == 0 or match_score <= 0:
            return 0.0

        # Normalize to approx [-1, 1]
        avg_score = raw_score / (denom * match_score)
        
        # Clamp to -1.0 (to handle cases where extensive gaps might push it lower)
        avg_score = np.maximum(avg_score, -1.0)
        
        # Map [-1, 1] -> [0, 1] for final output
        return (avg_score + 1.0) / 2.0

    # 3. Backtrack with explicit tie-breaking
    x_penalties = np.zeros(len(x))
    alignment_path = []

    if mode == "global":
        i, j = len(x), len(y)
    elif mode == "semiglobal":
        i = len(x)
        j = np.argmax(grid[-1, :])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    while i > 0 or (mode == "global" and j > 0):
        current_score = grid[i, j]
        
        # Retrieve predecessor scores (safe against boundary checks)
        score_diag = grid[i - 1, j - 1] + sim_matrix[i - 1, j - 1] if (i > 0 and j > 0) else -float('inf')
        score_up   = grid[i - 1, j] + gap_penalty if (i > 0) else -float('inf')
        
        # Prioritize: Diagonal (Match) > Vertical (Insertion) > Horizontal (Deletion)
        # Using np.isclose handles floating point equality issues
        if i > 0 and j > 0 and np.isclose(current_score, score_diag):
            # Diagonal: Match or Substitution
            x_penalties[i - 1] = sim_matrix[i - 1, j - 1]
            alignment_path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and np.isclose(current_score, score_up):
            # Vertical: Gap in Reference (Y)
            # This means Learner (X) has extra audio -> INSERTION
            x_penalties[i - 1] = gap_penalty
            alignment_path.append((i - 1, None))
            i -= 1
        else:
            # Horizontal: Gap in Learner (X)
            # This means Reference (Y) has audio that Learner missed -> DELETION
            # We cannot mark x_penalties because 'i' does not decrement.
            alignment_path.append((None, j - 1))
            j -= 1

    alignment_path.reverse()

    # 4. Normalization (Linear Map)
    # Our internal logic now guarantees scores are in roughly [-1, 1]
    # We map this to [0, 1] for the frontend visualization.
    #   1.0  (Perfect Match) -> 1.0
    #   0.0  (Neutral)       -> 0.5
    #   -1.0 (Gap/Mismatch)  -> 0.0
    
    # Clamp to ensure strict lower bound if gap_penalty < -1.0
    scores_clamped = np.maximum(x_penalties, -1.0)
    normalized_scores = (scores_clamped + 1.0) / 2.0

    return normalized_scores, alignment_path

def plot_waveform(waveform, sample_rate, *, agreement_scores, frame_duration=0.02):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    fig, ax = plt.subplots(num_channels, 1, figsize=(16, 3))
    if num_channels == 1:
        ax = [ax]
    for c in range(num_channels):
        ax[c].plot(time_axis, waveform[c], linewidth=1)
        ax[c].grid(True)
        if num_channels > 1:
            ax[c].set_ylabel(f"Channel {c+1}")

    assert num_channels == 1
    agreement_scores = np.convolve(agreement_scores, np.ones(3) / 3, mode="same")
    min_score = 0.4
    shade = (-1 * agreement_scores - min_score).clip(0, 1) * (1 / (1 - min_score))

    shading_intensity = (shade * 0.8).reshape(1, -1)

    time_mesh_coordinates = np.arange(0, len(agreement_scores) + 1) * frame_duration
    ymin, ymax = ax[0].get_ylim()
    mesh = ax[0].pcolormesh(
        time_mesh_coordinates,
        np.array([ymin, ymax]),
        shading_intensity,
        cmap="Reds",  # Or 'viridis', 'plasma', 'coolwarm', etc.
        vmin=0,
        vmax=1,
        alpha=0.5,
        shading="auto",
    )

    # Bring the waveform line to the front
    ax[0].get_lines()[0].set_zorder(10)

    plt.tight_layout()


def build_alignments(path, learner_len, reference_len, *, fill_backwards=False):
    """
    Generates a complete map from learner frames to reference frames.

    This function converts an alignment path (which may contain gaps) into
    a NumPy array where `map[learner_idx] = reference_idx`.

    It handles insertions (where `reference_idx` is `None`) by mapping
    them to the **next** valid reference frame encountered during a
    backward traversal of the path. Insertions at the very end
    of the learner sequence are mapped to `reference_len`.

    Args:
        path (list[tuple(int | None, int | None)]):
            The alignment path from an alignment algorithm.
            Expected as a list of `(learner_idx, reference_idx)` tuples.
        learner_len (int):
            The total number of frames in the learner sequence.
        reference_len (int):
            The total number of frames in the reference sequence. This is
            used as the fill value for insertions at the end of the path.
        fill_backwards (bool):
            If `True`, insertions in the learner sequence are filled
            with the next valid reference frame found during backward
            traversal. If `False`, they are filled with `-1`.

    Returns:
        np.ndarray:
            A 1D NumPy array of shape `(learner_len,)`, where the value
            at index `i` is the reference frame index that learner
            frame `i` is mapped to.
    """
    filled_map = np.zeros(learner_len, dtype=int)
    last_ref_idx = reference_len
    for i in range(len(path) - 1, -1, -1):
        learner_idx, reference_idx = path[i]
        if learner_idx is None:
            continue
        if reference_idx is None:
            if fill_backwards:
                filled_map[learner_idx] = last_ref_idx
            else:
                filled_map[learner_idx] = -1
        else:
            filled_map[learner_idx] = reference_idx
            last_ref_idx = reference_idx
    return filled_map
