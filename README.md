# LAMMPS Box Shear Analysis

`lammps_shear_analysis.py` parses a LAMMPS dump file and computes timestep-wise box deformation metrics useful for shear analysis:

- triclinic cell vectors (`H = [a b c]`)
- box angles (`alpha`, `beta`, `gamma`)
- engineering shear proxies (`xy/Ly`, `xz/Lz`, `yz/Lz`)
- Green-Lagrange strain components (`E_xy`, `E_xz`, `E_yz`, etc.)
- applied integer basis transform `M` per timestep when unwrapping is enabled

It supports **basis jump unwrapping** to handle equivalent box redefinitions done during simulation.

## Usage

```bash
python lammps_shear_analysis.py dump.atom -o shear_analysis.csv --progress
```

Options:

- `--no-unwrap`: disable basis continuity correction
- `--matrix-max-abs N`: integer matrix search bound for unwrapping (`default=1`, increase to `2` if needed)
- `--progress`: show read/unwrap/analyze progress in stderr for large files
- `--progress-bytes-step-mb N`: read progress print interval in MB (`default=50`)
- `--progress-frames-step N`: unwrap/analyze progress print interval in frames (`default=1000`)
- `--processes N`: multiprocessing worker count for analysis only (`1` disables, `0` uses CPU core count)
- `--mp-chunksize N`: chunksize for multiprocessing task dispatch (`default=200`)

## Output columns

- `lx`, `ly`, `lz`: LAMMPS restricted triclinic box lengths
- `a_norm`, `b_norm`, `c_norm`: Cartesian lengths of the basis vectors
- `xy_tilt`, `xz_tilt`, `yz_tilt`: tilt factors
- `alpha_deg`, `beta_deg`, `gamma_deg`
- `eng_shear_xy`, `eng_shear_xz`, `eng_shear_yz`
- `E_xx`, `E_yy`, `E_zz`, `E_xy`, `E_xz`, `E_yz`
- `M00..M22`: selected integer basis transform for each frame

## Notes

- Triclinic parsing follows LAMMPS `BOX BOUNDS xy xz yz` semantics.
- For robust strain tracking when box basis is redefined, keep unwrapping enabled.


## Multiprocessing note

- Multiprocessing is applied only to per-frame analysis row generation (not file I/O parsing).
- Basis unwrapping remains sequential because each frame depends on the previous unwrapped frame.
