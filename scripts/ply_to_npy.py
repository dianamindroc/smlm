#!/usr/bin/env python3
"""Utility for converting .ply point clouds to NumPy .npy arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def try_import_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:  # pragma: no cover - helpful message for CLI use
        raise SystemExit(
            "open3d is required for reading .ply files. "
            "Install it with `pip install open3d`."
        ) from exc
    return o3d


def discover_ply_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise SystemExit(f"{path} is not a .ply file.")
        return [path]

    if not path.is_dir():
        raise SystemExit(f"{path} is neither a file nor a directory.")

    pattern = "**/*.ply" if recursive else "*.ply"
    return sorted(path.glob(pattern))


def ply_to_numpy_array(ply_path: Path) -> np.ndarray:
    o3d = try_import_open3d()
    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"{ply_path} does not contain any points.")
    return points


def target_path(ply_path: Path, output_dir: Path | None) -> Path:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / (ply_path.stem + ".npy")
    return ply_path.with_suffix(".npy")


def convert_files(files: list[Path], output_dir: Path | None, overwrite: bool) -> list[tuple[Path, Path]]:
    results: list[tuple[Path, Path]] = []
    for ply_path in files:
        dest = target_path(ply_path, output_dir)
        if dest.exists() and not overwrite:
            print(f"Skipping {ply_path} -> {dest} (exists, use --overwrite).")
            continue
        points = ply_to_numpy_array(ply_path)
        np.save(dest, points)
        results.append((ply_path, dest))
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .ply point clouds to NumPy .npy arrays."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a .ply file or a directory that contains .ply files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output directory for the .npy files (defaults to the source directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, also convert .ply files in sub-directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    files = discover_ply_files(args.input_path, args.recursive)
    if not files:
        print("No .ply files found.")
        return 1

    converted = convert_files(files, args.output, args.overwrite)
    if not converted:
        print("No files converted.")
        return 1

    for src, dst in converted:
        print(f"Saved {dst} (from {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
