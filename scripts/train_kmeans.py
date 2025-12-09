import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List
from tqdm.auto import tqdm

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_parser():
    """Configures command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Learn K-Means from features stored in .npy files across multiple directories."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        nargs="+",
        help="One or more sources with .npy feature files. Can be directories or JSON files containing a priority list. Each source will be sampled equally.",
    )
    parser.add_argument(
        "--exclude_dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Directories to exclude from feature loading.",
    )
    parser.add_argument(
        "--km_path",
        type=Path,
        required=True,
        help="Path to save the trained k-means model. -n_clusters will be appended to the filename automatically.",
    )
    parser.add_argument(
        "--n_clusters", type=int, required=True, nargs="+", help="Number of clusters for K-Means."
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=100.0,
        help="Stop loading features after this many hours have been accumulated. "
        "Set to -1 to use all available data.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    return parser


def list_matching_files(directory: Path, pattern: str, exclude_dirs: List[Path]) -> List[Path]:
    """
    Lists all files in a directory matching a given pattern,
    excluding any files in the specified exclude directories.
    """
    matching_files = []
    for path in directory.rglob(pattern):
        if not any(exclude in path.parents for exclude in exclude_dirs):
            matching_files.append(path)
    return matching_files


def load_features_by_duration(
    sources: List[Path], exclude_dirs: List[Path], max_hours: float
) -> np.ndarray:
    """
    Loads features from .npy files from multiple directories, sampling
    randomly until a total duration is reached.

    Args:
        sources: A list of directories to search for .npy files. Can also be a JSON file containing a priority list.
        max_hours: The maximum duration of features to load, in hours.

    Returns:
        A numpy array concatenating the loaded features.
    """
    frame_shift_s = 0.02  # Hardcoded frame shift for HuBERT/WavLM. TODO: Take as argument?
    feature_sources = []
    for source in sources:
        if source.is_dir():
            paths_in_dir = list_matching_files(source, "*.npy", exclude_dirs=exclude_dirs)
            random.shuffle(paths_in_dir)
            feature_sources.append(paths_in_dir)
            logger.info(f"Found {len(paths_in_dir)} feature files in {source}.")
        else:
            if source.suffix != ".json":
                raise ValueError(f"Unsupported source file type: {source}")
            # Load priority list from JSON file
            source_files = []
            with open(source, "r") as f:
                file_lists = [Path(p) for p in json.load(f)]
            for file_list in file_lists:
                if not file_list.is_absolute():
                    file_list = source.parent / file_list
                with open(file_list, "r") as fl:
                    paths = [Path(line.strip()) for line in fl]
                filtered_paths = [
                    p for p in paths if not any(excl in p.parents for excl in exclude_dirs)
                ]
                random.shuffle(filtered_paths)
                source_files.extend(filtered_paths)
                logger.info(f"Found {len(filtered_paths)} feature files in {file_list}.")
            logger.info(f"Found total of {len(source_files)} feature files in {source}.")
            feature_sources.append(source_files)

    total_feature_files = sum(len(paths) for paths in feature_sources)
    if total_feature_files == 0:
        raise FileNotFoundError(
            f"No .npy files found in any of the provided sources."
        )

    logger.info(f"Found a total of {total_feature_files} feature files across all directories.")

    if max_hours <= 0:
        load_all = True
        logger.info("`max_hours` is <= 0, loading all available data.")
        target_total_frames = float("inf")
        target_frames_per_dir = float("inf")
    else:
        load_all = False
        target_total_frames = max_hours * 3600 / frame_shift_s
        max_hours_per_dir = max_hours / len(feature_sources)
        target_frames_per_dir = max_hours_per_dir * 3600 / frame_shift_s
        logger.info(
            f"Attempting to load features for {max_hours_per_dir:.2f} hours per directory "
            f"({int(target_frames_per_dir)} frames)..."
        )
        logger.info(f"Global target is {max_hours:.2f} hours ({int(target_total_frames)} frames).")

    loaded_features = []
    total_files_loaded = 0
    total_frames = 0

    progress_total = total_feature_files if load_all else int(target_total_frames)

    with tqdm(total=progress_total) as progress:
        for source, paths in zip(sources, feature_sources):
            frames_in_dir = 0
            files_in_dir = 0

            for path in paths:
                features = np.load(path)
                n_frames = features.shape[0]

                loaded_features.append(features)
                frames_in_dir += n_frames
                total_frames += n_frames
                files_in_dir += 1
                total_files_loaded += 1

                if load_all:
                    progress.update(1)
                else:
                    progress.update(n_frames)

                if frames_in_dir > target_frames_per_dir:
                    break

            loaded_hours = frames_in_dir * frame_shift_s / 3600
            if not load_all and frames_in_dir < target_frames_per_dir:
                logger.warning(
                    f"{source}: "
                    f"Target duration of {max_hours_per_dir:.2f} hours could not be reached. "
                    f"Loaded all available data ({files_in_dir} files, {loaded_hours:.2f} hours)."
                )
            else:
                logger.info(
                    f"{source}: "
                    f"Loaded {files_in_dir} files with a total of {frames_in_dir} frames "
                    f"({loaded_hours:.2f} hours)."
                )

    total_loaded_hours = total_frames * frame_shift_s / 3600
    logger.info(
        f"Finished loading data. Loaded {total_files_loaded} files in total, "
        f"with {total_frames} frames ({total_loaded_hours:.2f} hours)."
    )

    return np.concatenate(loaded_features, axis=0)


def learn_kmeans(
    sources: List[Path],
    exclude_dirs: List[Path],
    km_path: Path,
    n_clusters: List[int],
    seed: int,
    max_hours: float,
    init: str,
    max_iter: int,
    batch_size: int,
    tol: float,
    n_init: int,
    reassignment_ratio: float,
    max_no_improvement: int,
):
    """
    Main function to load features and train the K-Means model.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load features from the specified directories by duration
    features = load_features_by_duration(sources, exclude_dirs, max_hours)

    km_path.parent.mkdir(parents=True, exist_ok=True)
    for n in n_clusters:
        logger.info(f"Training K-Means with {n} clusters...")
        output_path = km_path.with_name(f"{km_path.stem}-{n}{km_path.suffix}")
        if output_path.exists():
            logger.warning(
                f"K-Means model for {n} clusters already exists at {output_path}. Skipping."
            )
            continue

        # Initialize and train the MiniBatchKMeans model
        km_model = MiniBatchKMeans(
            n_clusters=n,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=1,
            compute_labels=False,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=None,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
            random_state=seed,
        )

        km_model.fit(features)

        # Save the trained model
        joblib.dump(km_model, output_path)

        # Calculate and log the inertia
        inertia = -km_model.score(features) / len(features)
        logger.info(f"Total inertia: {inertia:.5f}")
        logger.info(f"K-Means model saved to {km_path}")

    logger.info("Finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(f"Starting K-Means training with args: {args}")
    learn_kmeans(**vars(args))
