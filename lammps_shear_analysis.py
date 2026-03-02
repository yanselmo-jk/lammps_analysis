#!/usr/bin/env python3
"""Analyze LAMMPS dump box deformation and compute shear-related metrics.

This script parses ``ITEM: BOX BOUNDS`` blocks from a LAMMPS dump and computes
angle/shear/strain metrics per timestep. It also provides optional integer-basis
continuity unwrapping to handle equivalent box redefinitions during runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = List[List[float]]
IntMat3 = List[List[int]]


@dataclass(frozen=True)
class FrameBox:
    timestep: int
    lx: float
    ly: float
    lz: float
    xy: float
    xz: float
    yz: float
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def h_matrix(self) -> Mat3:
        return [
            [self.lx, self.xy, self.xz],
            [0.0, self.ly, self.yz],
            [0.0, 0.0, self.lz],
        ]


def _print_progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        return
    pct = (100.0 * current) / total
    print(f"[{stage}] {pct:6.2f}% ({current}/{total})", file=sys.stderr)


def parse_dump_boxes(
    path: Path,
    progress: bool = False,
    progress_bytes_step: int = 50 * 1024 * 1024,
) -> List[FrameBox]:
    """Parse timestep + box metadata from a LAMMPS dump file."""
    frames: List[FrameBox] = []
    total_size = path.stat().st_size
    next_progress = progress_bytes_step

    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if progress and total_size > 0 and f.tell() >= next_progress:
                _print_progress("read", min(f.tell(), total_size), total_size)
                next_progress += progress_bytes_step

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break
            timestep = int(timestep_line.strip())

            box_header = None
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("ITEM: BOX BOUNDS"):
                    box_header = line.strip().split()
                    break
            if box_header is None:
                break

            row1 = _parse_float_row(f.readline())
            row2 = _parse_float_row(f.readline())
            row3 = _parse_float_row(f.readline())

            triclinic = {"xy", "xz", "yz"}.issubset(set(box_header[3:]))
            if triclinic:
                xlo_b, xhi_b, xy = row1[:3]
                ylo_b, yhi_b, xz = row2[:3]
                zlo_b, zhi_b, yz = row3[:3]

                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo = zlo_b
                zhi = zhi_b
            else:
                xlo, xhi = row1[:2]
                ylo, yhi = row2[:2]
                zlo, zhi = row3[:2]
                xy = xz = yz = 0.0

            lx = xhi - xlo
            ly = yhi - ylo
            lz = zhi - zlo
            if lx <= 0 or ly <= 0 or lz <= 0:
                raise ValueError(f"Invalid box length at timestep {timestep}: {(lx, ly, lz)}")

            frames.append(
                FrameBox(timestep, lx, ly, lz, xy, xz, yz, xlo, xhi, ylo, yhi, zlo, zhi)
            )

    if progress:
        _print_progress("read", total_size, total_size)
    return frames


def _parse_float_row(line: str) -> List[float]:
    return [float(tok) for tok in line.strip().split()]


def dot(u: Vec3, v: Vec3) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def norm(u: Vec3) -> float:
    return math.sqrt(dot(u, u))


def col(H: Mat3, j: int) -> Vec3:
    return (H[0][j], H[1][j], H[2][j])


def angle_deg(u: Vec3, v: Vec3) -> float:
    c = dot(u, v) / (norm(u) * norm(v))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def compute_angles(H: Mat3) -> Tuple[float, float, float]:
    a, b, c = col(H, 0), col(H, 1), col(H, 2)
    return angle_deg(b, c), angle_deg(a, c), angle_deg(a, b)


def matmul(A: Mat3, B: Mat3) -> Mat3:
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(A: Mat3) -> Mat3:
    return [[A[j][i] for j in range(3)] for i in range(3)]


def det(A: Mat3) -> float:
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def inv(A: Mat3) -> Mat3:
    d = det(A)
    if abs(d) < 1e-14:
        raise ValueError("Singular 3x3 matrix")
    adj = [
        [A[1][1] * A[2][2] - A[1][2] * A[2][1], A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][1] * A[1][2] - A[0][2] * A[1][1]],
        [A[1][2] * A[2][0] - A[1][0] * A[2][2], A[0][0] * A[2][2] - A[0][2] * A[2][0], A[0][2] * A[1][0] - A[0][0] * A[1][2]],
        [A[1][0] * A[2][1] - A[1][1] * A[2][0], A[0][1] * A[2][0] - A[0][0] * A[2][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]],
    ]
    return [[adj[i][j] / d for j in range(3)] for i in range(3)]


def frobenius(A: Mat3) -> float:
    return math.sqrt(sum(A[i][j] * A[i][j] for i in range(3) for j in range(3)))


def matsub(A: Mat3, B: Mat3) -> Mat3:
    return [[A[i][j] - B[i][j] for j in range(3)] for i in range(3)]


def finite_strain(H: Mat3, H_ref: Mat3) -> Mat3:
    F = matmul(H, inv(H_ref))
    C = matmul(transpose(F), F)
    I = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    return [[0.5 * (C[i][j] - I[i][j]) for j in range(3)] for i in range(3)]


def build_integer_matrices(max_abs: int = 1) -> List[IntMat3]:
    """Enumerate small integer 3x3 matrices with determinant +/- 1."""
    mats: List[IntMat3] = []
    vals = range(-max_abs, max_abs + 1)
    for entries in _product(vals, repeat=9):
        M = [[entries[3 * i + j] for j in range(3)] for i in range(3)]
        if abs(round(det([[float(x) for x in r] for r in M]))) != 1:
            continue
        mats.append(M)
    return mats


def _product(values: Iterable[int], repeat: int) -> Iterator[Tuple[int, ...]]:
    pools = [tuple(values)] * repeat
    res = [()]
    for pool in pools:
        res = [x + (y,) for x in res for y in pool]
    return iter(res)


def matmul_int(A: Mat3, M: IntMat3) -> Mat3:
    return [[sum(A[i][k] * float(M[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]


def unwrap_basis_sequence(
    H_list: Sequence[Mat3],
    max_abs: int = 1,
    progress: bool = False,
    progress_frames_step: int = 1000,
) -> Tuple[List[Mat3], List[IntMat3]]:
    if not H_list:
        return [], []
    candidates = build_integer_matrices(max_abs=max_abs)
    cont = [H_list[0]]
    picked = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]

    total = len(H_list)
    for t in range(1, total):
        H = H_list[t]
        prev = cont[-1]
        best_M: Optional[IntMat3] = None
        best_H: Optional[Mat3] = None
        best_score = float("inf")
        for M in candidates:
            HM = matmul_int(H, M)
            score = frobenius(matsub(HM, prev))
            if score < best_score:
                best_score = score
                best_M = M
                best_H = HM
        assert best_M is not None and best_H is not None
        picked.append(best_M)
        cont.append(best_H)

        if progress and (t % progress_frames_step == 0 or t == total - 1):
            _print_progress("unwrap", t + 1, total)

    return cont, picked


def _build_row(frame: FrameBox, H: Mat3, M: IntMat3, H_ref: Mat3) -> dict:
    alpha, beta, gamma = compute_angles(H)
    E = finite_strain(H, H_ref)
    a, b, c = col(H, 0), col(H, 1), col(H, 2)
    return {
        "timestep": frame.timestep,
        "lx": H[0][0],
        "ly": H[1][1],
        "lz": H[2][2],
        "a_norm": norm(a),
        "b_norm": norm(b),
        "c_norm": norm(c),
        "xy_tilt": H[0][1],
        "xz_tilt": H[0][2],
        "yz_tilt": H[1][2],
        "alpha_deg": alpha,
        "beta_deg": beta,
        "gamma_deg": gamma,
        "eng_shear_xy": H[0][1] / H[1][1],
        "eng_shear_xz": H[0][2] / H[2][2],
        "eng_shear_yz": H[1][2] / H[2][2],
        "E_xx": E[0][0],
        "E_yy": E[1][1],
        "E_zz": E[2][2],
        "E_xy": E[0][1],
        "E_xz": E[0][2],
        "E_yz": E[1][2],
        "M00": M[0][0],
        "M01": M[0][1],
        "M02": M[0][2],
        "M10": M[1][0],
        "M11": M[1][1],
        "M12": M[1][2],
        "M20": M[2][0],
        "M21": M[2][1],
        "M22": M[2][2],
    }


def _build_row_task(args: Tuple[FrameBox, Mat3, IntMat3, Mat3]) -> dict:
    frame, H, M, H_ref = args
    return _build_row(frame, H, M, H_ref)


def analyze(
    frames: Sequence[FrameBox],
    unwrap_basis: bool = True,
    max_abs_matrix: int = 1,
    progress: bool = False,
    progress_frames_step: int = 1000,
    processes: int = 1,
    mp_chunksize: int = 200,
) -> List[dict]:
    H_raw = [f.h_matrix() for f in frames]
    if unwrap_basis:
        H_used, transforms = unwrap_basis_sequence(
            H_raw,
            max_abs=max_abs_matrix,
            progress=progress,
            progress_frames_step=progress_frames_step,
        )
    else:
        H_used = H_raw
        transforms = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]] for _ in H_raw]
    if not H_used:
        return []

    H_ref = H_used[0]
    total = len(H_used)
    tasks = list(zip(frames, H_used, transforms, [H_ref] * total))

    workers = cpu_count() if processes <= 0 else processes
    chunksize = max(1, mp_chunksize)

    if workers <= 1 or total < 2:
        rows = []
        for i, task in enumerate(tasks):
            rows.append(_build_row_task(task))
            if progress and ((i + 1) % progress_frames_step == 0 or i == total - 1):
                _print_progress("analyze", i + 1, total)
        return rows

    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap(_build_row_task, tasks, chunksize=chunksize), start=1):
            rows.append(row)
            if progress and (i % progress_frames_step == 0 or i == total):
                _print_progress("analyze", i, total)
    return rows


def write_csv(rows: Sequence[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("No analysis rows to write")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("shear_analysis.csv"))
    p.add_argument("--no-unwrap", action="store_true")
    p.add_argument("--progress", action="store_true", help="Show file I/O and analysis progress")
    p.add_argument(
        "--progress-bytes-step-mb",
        type=int,
        default=50,
        help="Progress print interval for file reading (MB)",
    )
    p.add_argument(
        "--progress-frames-step",
        type=int,
        default=1000,
        help="Progress print interval for unwrapping/analyze (frames)",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes for analysis only (1=off, 0=cpu_count)",
    )
    p.add_argument(
        "--mp-chunksize",
        type=int,
        default=200,
        help="Task chunksize used by multiprocessing worker map",
    )
    p.add_argument(
        "--matrix-max-abs",
        type=int,
        default=1,
        help="GL(3,Z) search range for unwrapping; 1 is faster, 2 is broader",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = parse_dump_boxes(
        args.dump,
        progress=args.progress,
        progress_bytes_step=max(1, args.progress_bytes_step_mb) * 1024 * 1024,
    )
    if not frames:
        raise SystemExit("No frames parsed from dump")
    rows = analyze(
        frames,
        unwrap_basis=(not args.no_unwrap),
        max_abs_matrix=args.matrix_max_abs,
        progress=args.progress,
        progress_frames_step=max(1, args.progress_frames_step),
        processes=args.processes,
        mp_chunksize=max(1, args.mp_chunksize),
    )
    write_csv(rows, args.output)
    print(f"Parsed {len(frames)} frames; wrote {args.output}")


if __name__ == "__main__":
    main()
