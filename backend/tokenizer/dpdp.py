# DPDP code is copied from bshall/dusted (MIT License)
from typing import Tuple

import numba
import numpy as np
import scipy.spatial.distance as distance
import torch
import numpy as np


def segment(
    sequence: np.ndarray, codebook: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Group speech representations into phone-like segments.

    Args:
        sequence (NDArray): speech representations of shape (T, D) where T is the number of frames and D is the feature dimension.
        codebook (NDArray): cluster centriods of the discrete units of shape (K, D) where K is the number of codes.
        gamma float: Duration regularizer weight. Larger values result in a coarser segmentation.

    Returns:
        NDArray[int]: list of discrete units representing each segment sound types of shape (N,).
        NDArray[int]: list of segment boundaries of shape (N+1,).
    """
    dists = distance.cdist(sequence, codebook).astype(np.float32)
    alpha, P = _segment(dists, gamma)
    return _backtrack(alpha, P)


@numba.njit()
def _segment(dists, gamma):
    T, K = dists.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = dists[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + dists[s, :] - gamma

    for t in range(T):
        alpha[t + 1] = np.inf
        for s in range(t + 1):
            k = np.argmin(D[s, t, :])
            alpha_min = alpha[s] + D[s, t, k]
            if alpha_min < alpha[t + 1]:
                P[t + 1, :] = s, k
                alpha[t + 1] = alpha_min
    return alpha, P


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = []
    boundaries = [rhs]
    while rhs != 0:
        lhs, code = P[rhs, :]
        segments.append(code)
        boundaries.append(lhs)
        rhs = lhs
    segments.reverse()
    boundaries.reverse()
    return np.array(segments), np.array(boundaries)


class DPDPTokenizer:
    def __init__(self, *, cluster_centers=None):
        if cluster_centers is None:
            default_kmeans, _ = torch.hub.load(
                "bshall/dusted:main", "kmeans", language="english", trust_repo=True, verbose=False
            )
            self.cluster_centers = default_kmeans.cluster_centers_
        else:
            self.cluster_centers = cluster_centers

    def tokenize_one(self, features: np.ndarray, *, gamma=0.2) -> np.ndarray:
        # features: (T, D)
        return segment(features, self.cluster_centers, gamma=gamma)
