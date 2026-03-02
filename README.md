# LAMMPS Box Shear Analysis

`lammps_shear_analysis.py` parses a LAMMPS dump file and computes timestep-wise box deformation metrics useful for shear analysis:

- triclinic cell vectors (`H = [a b c]`)
- box angles (`alpha`, `beta`, `gamma`)
- engineering shear proxies (`xy/Ly`, `xz/Lz`, `yz/Lz`)
- Green-Lagrange strain components (`E_xy`, `E_xz`, `E_yz`, etc.)

It also supports **basis jump unwrapping** to handle equivalent box redefinitions done during simulation.

## Usage

```bash
python lammps_shear_analysis.py dump.atom -o shear_analysis.csv
```

Options:

- `--no-unwrap`: disable basis continuity correction
- `--matrix-max-abs N`: integer matrix search bound for unwrapping (`default=1`, increase to `2` if needed)

## Notes

- Triclinic parsing follows LAMMPS `BOX BOUNDS xy xz yz` semantics.
- For robust strain tracking when box basis is redefined, keep unwrapping enabled.
