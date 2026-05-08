"""Create a manageable sample from the large LendingClub CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "accepted_2007_to_2018Q4.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "lending_club_sample.csv"
DEFAULT_N_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample creation."""

    parser = argparse.ArgumentParser(description="Create a smaller LendingClub sample CSV.")
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument(
        "--mode",
        choices=["head", "random"],
        default="random",
        help="Use first N rows or a chunked random reservoir sample.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Chunk size for memory-safe random sampling.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible random sampling.",
    )
    return parser.parse_args()


def reservoir_sample_csv(
    input_path: Path,
    sample_rows: int,
    chunk_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Randomly sample rows from a CSV without loading the full file into memory."""

    rng = np.random.default_rng(random_state)
    reservoir: pd.DataFrame | None = None
    key_column = "__sample_key__"

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        chunk = chunk.copy()
        chunk[key_column] = rng.random(len(chunk))
        if reservoir is None:
            reservoir = chunk
        else:
            reservoir = pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > sample_rows:
            reservoir = reservoir.nlargest(sample_rows, key_column)

    if reservoir is None:
        raise ValueError(f"No rows found in input CSV: {input_path}")
    return reservoir.drop(columns=[key_column]).reset_index(drop=True)


def main() -> None:
    """Create a manageable LendingClub sample CSV."""

    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if args.mode == "head":
        sample = pd.read_csv(input_path, nrows=args.rows, low_memory=False)
    else:
        sample = reservoir_sample_csv(
            input_path=input_path,
            sample_rows=args.rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows to {output_path} using mode={args.mode}")


if __name__ == "__main__":
    main()
