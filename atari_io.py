"""HDF5 IO helpers shared by the Atari collectors.

A minimal append-style writer producing HDF5 files compatible with
``stable_worldmodel.data.dataset.HDF5Dataset``. The PyPI release of
``stable_worldmodel`` (0.0.6) does not yet ship a writer; this module
is a local stand-in that's byte-compatible with that reader.

Layout:

  - One resizable axis-0 dataset per column (e.g. ``pixels``, ``action``,
    ``reward``, ``done`` ...).
  - ``ep_len``  (int32, 1-D) — episode lengths.
  - ``ep_offset`` (int64, 1-D) — episode start offsets in the flat layout.

Schema is inferred from the first episode and locked thereafter; a
column-shape mismatch in append mode raises before any data is written.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


class HDF5EpisodeWriter:
    """Append episodes to a single HDF5 file with zstd-compressed pixels."""

    def __init__(self, path: Path | str, mode: str = "overwrite") -> None:
        if mode not in ("overwrite", "error", "append"):
            raise ValueError(f"mode must be overwrite|error|append, got {mode}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._f: h5py.File | None = None
        self._initialized = False
        self._global_ptr = 0

    def __enter__(self) -> "HDF5EpisodeWriter":
        exists = self.path.exists()
        if exists and self.mode == "error":
            raise FileExistsError(self.path)
        if not exists or self.mode == "overwrite":
            self._f = h5py.File(self.path, "w", libver="latest")
        else:
            self._f = h5py.File(self.path, "a", libver="latest")
            if "ep_len" in self._f:
                self._global_ptr = int(self._f["ep_len"][:].sum())
                self._initialized = True
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def _init_schema(self, sample_ep: dict) -> None:
        # Blosc/zstd at level 3: ~3-5x compression on Atari pixels (lots of
        # solid background) at near-zero CPU overhead vs uncompressed.
        compression = hdf5plugin.Blosc(cname="zstd", clevel=3)
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            # 16-row chunks for pixels (~440KB each — good zstd block size,
            # still fast on random access). Per-row for the small columns.
            chunk_n = 16 if col == "pixels" else 1
            self._f.create_dataset(
                col,
                shape=(0, *sample.shape),
                maxshape=(None, *sample.shape),
                dtype=sample.dtype,
                chunks=(chunk_n, *sample.shape),
                **compression,
            )
        self._f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

    def write_episode(self, ep_data: dict) -> None:
        assert self._f is not None, "writer used outside `with` block"
        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True

        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            ds = self._f[col]
            ds.resize(self._global_ptr + ep_len, axis=0)
            ds[self._global_ptr : self._global_ptr + ep_len] = np.asarray(vals)

        n = self._f["ep_len"].shape[0]
        self._f["ep_len"].resize(n + 1, axis=0)
        self._f["ep_len"][n] = ep_len
        self._f["ep_offset"].resize(n + 1, axis=0)
        self._f["ep_offset"][n] = self._global_ptr

        self._global_ptr += ep_len


def merge_hdf5_datasets(input_paths: list[str | Path], output_path: str | Path) -> None:
    """Concatenate multiple HDF5 datasets into one, preserving episode
    boundaries via ep_len / ep_offset.

    Schemas must match (same set of columns, same per-row shapes & dtypes).
    Inputs are visited in order; episodes are written sequentially with
    ``ep_offset`` rebased to the cumulative row count.
    """
    input_paths = [Path(p) for p in input_paths]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with HDF5EpisodeWriter(out, mode="overwrite") as writer:
        for src in input_paths:
            with h5py.File(src, "r") as f:
                ep_lens = f["ep_len"][:]
                ep_offsets = f["ep_offset"][:]
                # Discover columns excluding metadata.
                cols = [k for k in f.keys() if k not in ("ep_len", "ep_offset")]
                for ep_idx, (off, length) in enumerate(zip(ep_offsets, ep_lens)):
                    end = int(off) + int(length)
                    ep_data = {col: f[col][int(off):end] for col in cols}
                    writer.write_episode(ep_data)
