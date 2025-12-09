import numpy as np
import matplotlib.pyplot as plt

def compute_match_grid(x, y, *, gap_penalty, match_score, mismatch_score):
    grid = np.zeros((len(x)+1, len(y)+1))
    # no penalty for starting in the middle of y
    grid[1:, 0] = np.arange(1, len(x)+1) * gap_penalty
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            if x[i-1] == y[j-1]:
                score = match_score
            else:
                score = mismatch_score
            grid[i, j] = max(
                grid[i-1, j] + gap_penalty,
                grid[i, j-1] + gap_penalty,
                grid[i-1, j-1] + score
            )
    return grid

def score_frames(learner, reference, *, gap_penalty=-1, match_score=1, mismatch_score=-1, normalize=False):
    x = learner
    y = reference

    # Step 1: Compute the alignment score grid
    grid = compute_match_grid(
        x, y,
        gap_penalty=gap_penalty,
        match_score=match_score,
        mismatch_score=mismatch_score
    )

    # Step 2: Backtrack, calculate primary penalties, and record omissions
    x_penalties = np.zeros(len(x))
    alignment_path = []

    i = len(x)
    j = np.argmax(grid[i, :])  # Start at the max in the last row

    while i > 0 and j > 0:
        current_match_score = match_score if x[i - 1] == y[j - 1] else mismatch_score

        if np.isclose(grid[i, j], grid[i - 1, j - 1] + current_match_score):
            x_penalties[i - 1] = current_match_score
            alignment_path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif np.isclose(grid[i, j], grid[i - 1, j] + gap_penalty): # Gap in reference
            x_penalties[i - 1] = gap_penalty
            alignment_path.append((i - 1, None))
            i -= 1
        else: # Gap in learner
            alignment_path.append((None, j - 1))
            j -= 1

    while i > 0:
        x_penalties[i - 1] = gap_penalty
        alignment_path.append((i - 1, None))
        i -= 1

    alignment_path.reverse()

    if normalize:
        max_val = match_score
        min_val = min(mismatch_score, gap_penalty)
        
        score_range = max_val - min_val
        
        assert score_range > 0, "Invalid score range"
        normalized_scores = (x_penalties - min_val) / score_range
        
        if not (np.all(normalized_scores >= -1e-9) and np.all(normalized_scores <= 1.0 + 1e-9)):
            raise RuntimeError(
                "Normalization failed. Scores outside [0, 1] range detected. "
                f"Min score: {np.min(normalized_scores)}, "
                f"Max score: {np.max(normalized_scores)}"
            )

        return normalized_scores, alignment_path

    return x_penalties, alignment_path

def compute_continuous_grid(x, y, *, gap_penalty):
    """
    x, y: numpy arrays of shape (N, D) and (M, D) representing sequences of vectors.
    """
    len_x, len_y = x.shape[0], y.shape[0]
    
    # 1. Pre-compute Cosine Similarity Matrix (N x M)
    # Normalize vectors to unit length
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
    
    # Dot product of normalized vectors = Cosine Similarity
    # Range: [-1.0, 1.0]
    sim_matrix = np.dot(x_norm, y_norm.T)

    # 2. Fill DP Grid
    grid = np.zeros((len_x + 1, len_y + 1))
    
    # Initialize first column with gap penalties (forcing x to be accounted for)
    grid[1:, 0] = np.arange(1, len_x + 1) * gap_penalty
    
    # No penalty for starting in the middle of y (grid[0, :] remains 0)

    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            score = sim_matrix[i-1, j-1]
            
            grid[i, j] = max(
                grid[i-1, j] + gap_penalty,   # Gap in y
                grid[i, j-1] + gap_penalty,   # Gap in x
                grid[i-1, j-1] + score        # Match/Substitution
            )
            
    return grid, sim_matrix

def score_continuous_frames(learner, reference, *, gap_penalty=-0.5, normalize=False):
    x = learner
    y = reference

    # Step 1: Compute grid and retrieve similarity matrix
    grid, sim_matrix = compute_continuous_grid(x, y, gap_penalty=gap_penalty)

    # Step 2: Backtrack
    x_penalties = np.zeros(len(x))
    alignment_path = []

    i = len(x)
    j = np.argmax(grid[i, :])  # Start at the max in the last row (semi-global)

    while i > 0 and j > 0:
        # The score for aligning these two specific vectors
        current_match_score = sim_matrix[i-1, j-1]

        # Check diagonal (Match/Substitution)
        # We use a small epsilon for float comparison
        if np.isclose(grid[i, j], grid[i-1, j-1] + current_match_score):
            x_penalties[i-1] = current_match_score
            alignment_path.append((i-1, j-1))
            i -= 1
            j -= 1
        # Check vertical (Gap in reference)
        elif np.isclose(grid[i, j], grid[i-1, j] + gap_penalty):
            x_penalties[i-1] = gap_penalty
            alignment_path.append((i-1, None))
            i -= 1
        # Check horizontal (Gap in learner)
        else:
            alignment_path.append((None, j-1))
            j -= 1

    # Handle remaining start gaps
    while i > 0:
        x_penalties[i-1] = gap_penalty
        alignment_path.append((i-1, None))
        i -= 1

    alignment_path.reverse()

    if normalize:
        # Cosine similarity is [-1, 1], Gap penalty is usually negative.
        # We define the theoretical max as 1.0 (perfect alignment)
        # and min as the lesser of -1.0 or the gap penalty.
        max_val = 1.0
        min_val = min(-1.0, gap_penalty)
        
        score_range = max_val - min_val
        normalized_scores = (x_penalties - min_val) / score_range
        return normalized_scores, alignment_path

    return x_penalties, alignment_path

def plot_waveform(waveform, sample_rate, *, agreement_scores):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    fig, ax = plt.subplots(num_channels, 1, figsize=(16,3))
    if num_channels == 1:
        ax = [ax]
    for c in range(num_channels):
        ax[c].plot(time_axis, waveform[c], linewidth=1)
        ax[c].grid(True)
        if num_channels > 1:
            ax[c].set_ylabel(f"Channel {c+1}")

    assert num_channels == 1
    agreement_scores = np.convolve(agreement_scores, np.ones(3)/3, mode='same')
    min_score = 0.4
    shade = (-1*agreement_scores - min_score).clip(0, 1) * (1 / (1-min_score))

    shading_intensity = (shade*0.8).reshape(1, -1)

    time_mesh_coordinates = np.arange(0, len(agreement_scores)+1) * 0.02
    ymin, ymax = ax[0].get_ylim()
    mesh = ax[0].pcolormesh(
        time_mesh_coordinates,
        np.array([ymin, ymax]),
        shading_intensity,
        cmap='Reds',  # Or 'viridis', 'plasma', 'coolwarm', etc.
        vmin=0,
        vmax=1,
        alpha=0.5,
        shading='auto',
    )

    # Bring the waveform line to the front
    ax[0].get_lines()[0].set_zorder(10)

    plt.tight_layout()


def build_alignments(path, learner_len, reference_len, *, fill_backwards=True):
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
    for i in range(len(path)-1, -1, -1):
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