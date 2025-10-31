import numpy as np
import matplotlib.pyplot as plt

def compute_match_grid(x, y, *, gap_penalty, match_score, mismatch_score):
    grid = np.zeros((len(x)+1, len(y)+1), dtype=x.dtype)
    grid[0, 1:] = np.arange(1, len(y)+1) * gap_penalty
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

def find_mismatches(x, y, *, gap_penalty=-1, match_score=1, mismatch_score=-1):
    """
    Performs a global alignment and distributes penalties for insertions in x
    to the surrounding tokens.
    """
    # Step 1: Compute the alignment score grid
    grid = compute_match_grid(
        x, y,
        gap_penalty=gap_penalty,
        match_score=match_score,
        mismatch_score=mismatch_score
    )

    # Step 2: Backtrack, calculate primary penalties, and record insertion locations
    x_penalties = np.zeros(len(x))
    insertion_locations = []
    i, j = len(x), len(y)

    while i > 0 and j > 0:
        current_match_score = match_score if x[i - 1] == y[j - 1] else mismatch_score

        if grid[i, j] == grid[i - 1, j - 1] + current_match_score:
            x_penalties[i - 1] = current_match_score
            i -= 1
            j -= 1
        elif grid[i, j] == grid[i - 1, j] + gap_penalty:
            x_penalties[i - 1] = gap_penalty
            i -= 1
        else: # This is a horizontal move (insertion in x)
            # Record the grid index 'i' where the insertion occurs.
            # This is the position *after* the token x[i-1].
            insertion_locations.append(i)
            j -= 1

    while i > 0:
        x_penalties[i - 1] = gap_penalty
        i -= 1
    
    # Step 3: Distribute penalties for the recorded insertions
    # We split the penalty because it affects the timing on both sides of the gap.
    penalty_share = gap_penalty / 2.0
    for loc in insertion_locations:
        # Penalize the token *before* the gap
        if loc > 0:
            x_penalties[loc - 1] += penalty_share
        
        # Penalize the token *after* the gap
        if loc < len(x):
            x_penalties[loc] += penalty_share
            
    return x_penalties

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


# def transfer_intervals(labels, alignment_map, sr=16000):
#     """
#     Transfers intervals from x to y using the alignment map.
#     'labels' is a list of tgt.Annotation objects.
#     """
#     x_samples, y_samples = alignment_map
#     
#     new_intervals = []
#     for label in labels:
#         start_time, end_time, text = label.start_time, label.end_time, label.text
#         
#         # 1. Convert x's times to sample indices
#         x_start_sample = start_time * sr
#         x_end_sample = end_time * sr
#         
#         # 2. Interpolate to find y's corresponding sample indices
#         y_start_sample = np.interp(x_start_sample, x_samples, y_samples)
#         y_end_sample = np.interp(x_end_sample, x_samples, y_samples)
#         
#         # 3. Convert y's sample indices back to times
#         y_start_time = y_start_sample / sr
#         y_end_time = y_end_sample / sr
#         
#         new_intervals.append(tgt.Annotation(
#             text=text,
#             start_time=y_start_time,
#             end_time=y_end_time
#         ))
#         
#     return new_intervals
#
# def to_sample(position):
#     return position * 320
# 
# def create_alignment_map(xcodes, ycodes, xboundaries, yboundaries, xwav, ywav, *, gap_penalty=-1, match_score=1, mismatch_score=-1):
#     grid = compute_match_grid(xcodes, ycodes, gap_penalty=gap_penalty, match_score=match_score, mismatch_score=mismatch_score)
#     
#     i, j = len(xcodes), len(ycodes)
#     
#     # Start at the end of the audio files
#     (x_current_sample,) = xwav.shape
#     (y_current_sample,) = ywav.shape
#     
#     # Store mapping points (x_sample, y_sample)
#     # We build it backwards, from the end, then reverse it.
#     mapping_points = [(x_current_sample, y_current_sample)]
# 
#     while i > 0 and j > 0:
#         is_gap_x = (grid[i, j] == grid[i, j-1] + gap_penalty)
#         is_gap_y = (grid[i, j] == grid[i-1, j] + gap_penalty)
#         
#         x_seg_len = 0
#         y_seg_len = 0
# 
#         if is_gap_y: 
#             # Gap in y, segment from x
#             frame_start, frame_end = xboundaries[i-1], xboundaries[i]
#             x_seg_len = len(xwav[to_sample(frame_start):to_sample(frame_end)])
#             i -= 1
#         elif is_gap_x: 
#             # Gap in x, segment from y
#             frame_start, frame_end = yboundaries[j-1], yboundaries[j]
#             y_seg_len = len(ywav[to_sample(frame_start):to_sample(frame_end)])
#             j -= 1
#         else: 
#             # Match or substitution
#             x_frame_start, x_frame_end = xboundaries[i-1], xboundaries[i]
#             x_seg_len = len(xwav[to_sample(x_frame_start):to_sample(x_frame_end)])
#             
#             y_frame_start, y_frame_end = yboundaries[j-1], yboundaries[j]
#             y_seg_len = len(ywav[to_sample(y_frame_start):to_sample(y_frame_end)])
#             
#             i -= 1
#             j -= 1
#             
#         # Update our position
#         x_current_sample -= x_seg_len
#         y_current_sample -= y_seg_len
#         mapping_points.append((x_current_sample, y_current_sample))
# 
#     # Handle remaining segments at the beginning
#     while i > 0:
#         frame_start, frame_end = xboundaries[i-1], xboundaries[i]
#         x_seg_len = len(xwav[to_sample(frame_start):to_sample(frame_end)])
#         x_current_sample -= x_seg_len
#         mapping_points.append((x_current_sample, y_current_sample)) # y is already 0
#         i -= 1
#     while j > 0:
#         frame_start, frame_end = yboundaries[j-1], yboundaries[j]
#         y_seg_len = len(ywav[to_sample(frame_start):to_sample(frame_end)])
#         y_current_sample -= y_seg_len
#         mapping_points.append((x_current_sample, y_current_sample)) # x is already 0
#         j -= 1
# 
#     mapping_points.reverse() # Reverse to get (0, 0) at the start
#     
#     # Separate into x and y arrays for np.interp
#     # Ensure values are monotonically increasing for interpolation
#     x_samples, y_samples = [], []
#     last_x = -1
#     for x, y in mapping_points:
#         if x > last_x:
#             x_samples.append(x)
#             y_samples.append(y)
#             last_x = x
#         elif x == last_x: # If x is the same, update y to the latest value
#             if y_samples:
#                 y_samples[-1] = y
#             
#     return np.array(x_samples), np.array(y_samples)