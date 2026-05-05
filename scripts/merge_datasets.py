"""Merge multiple HDF5 datasets into one (vertical concat).

Schemas must match across inputs (same columns, same per-row shapes &
dtypes). Episode boundaries are preserved via ``ep_len`` / ``ep_offset``;
offsets are rebased to the cumulative row count of the merged file.

Example:
    python scripts/merge_datasets.py \\
        --inputs $STABLEWM_HOME/atari_pong_random.h5 \\
                 $STABLEWM_HOME/atari_pong_v4policy.h5 \\
        --out $STABLEWM_HOME/atari_pong_v5.h5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atari_io import merge_hdf5_datasets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    merge_hdf5_datasets([Path(p) for p in args.inputs], Path(args.out))
    print(f"wrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
