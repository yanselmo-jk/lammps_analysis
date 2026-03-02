#!/usr/bin/env python3
"""Analyze LAMMPS dump box deformation and compute shear-related metrics."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

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


def parse_dump_boxes(path: Path) -> List[FrameBox]:
    frames: List[FrameBox] = []
    with path.open("r", encoding="utf-8") as f:
        lines = iter(f)
        for line in lines:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep_line = next(lines, None)
            if timestep_line is None:
                break
            timestep = int(timestep_line.strip())

            for line in lines:
                if line.startswith("ITEM: BOX BOUNDS"):
                    box_header = line.strip().split()
                    break
            else:
                break

            triclinic = {"xy", "xz", "yz"}.issubset(set(box_header[3:]))
            row1 = _parse_float_row(next(lines))
            row2 = _parse_float_row(next(lines))
            row3 = _parse_float_row(next(lines))

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
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


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


def unwrap_basis_sequence(H_list: Sequence[Mat3], max_abs: int = 1) -> Tuple[List[Mat3], List[IntMat3]]:
    if not H_list:
        return [], []
    candidates = build_integer_matrices(max_abs=max_abs)
    cont = [H_list[0]]
    picked = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]

    for t in range(1, len(H_list)):
        H = H_list[t]
        prev = cont[-1]
        best_M = None
        best_H = None
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
    return cont, picked


def analyze(frames: Sequence[FrameBox], unwrap_basis: bool = True, max_abs_matrix: int = 1) -> List[dict]:
    H_raw = [f.h_matrix() for f in frames]
    if unwrap_basis:
        H_used, transforms = unwrap_basis_sequence(H_raw, max_abs=max_abs_matrix)
    else:
        H_used = H_raw
        transforms = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]] for _ in H_raw]
    if not H_used:
        return []

    H_ref = H_used[0]
    rows = []
    for frame, H, M in zip(frames, H_used, transforms):
        alpha, beta, gamma = compute_angles(H)
        E = finite_strain(H, H_ref)
        a, b, c = col(H, 0), col(H, 1), col(H, 2)
        rows.append(
            {
                "timestep": frame.timestep,
                "lx": norm(a),
                "ly": norm(b),
                "lz": norm(c),
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
        )
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
    p.add_argument("--matrix-max-abs", type=int, default=1)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = parse_dump_boxes(args.dump)
    if not frames:
        raise SystemExit("No frames parsed from dump")
    rows = analyze(frames, unwrap_basis=(not args.no_unwrap), max_abs_matrix=args.matrix_max_abs)
    write_csv(rows, args.output)
    print(f"Parsed {len(frames)} frames; wrote {args.output}")


if __name__ == "__main__":
    main()
