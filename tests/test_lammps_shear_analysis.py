import csv
import tempfile
import unittest
from pathlib import Path

from lammps_shear_analysis import analyze, parse_dump_boxes, write_csv


class LammpsShearAnalysisTests(unittest.TestCase):
    def test_parse_and_analyze_triclinic(self):
        dump_text = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 0.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 0 0 0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 2.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 0 0 0
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.dump"
            path.write_text(dump_text, encoding="utf-8")
            frames = parse_dump_boxes(path)
            self.assertEqual(len(frames), 2)

            rows = analyze(frames, unwrap_basis=False)
            self.assertEqual(rows[0]["gamma_deg"], 90.0)
            self.assertAlmostEqual(rows[1]["eng_shear_xy"], 0.2)
            self.assertEqual(rows[1]["ly"], 10.0)
            self.assertGreater(rows[1]["b_norm"], rows[1]["ly"])

    def test_analyze_multiprocessing_matches_serial(self):
        dump_text = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 0.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 0 0 0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 1.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 0 0 0
ITEM: TIMESTEP
2
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 10.0 2.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z
1 1 0 0 0
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.dump"
            path.write_text(dump_text, encoding="utf-8")
            frames = parse_dump_boxes(path)

            serial_rows = analyze(frames, unwrap_basis=False, processes=1)
            mp_rows = analyze(frames, unwrap_basis=False, processes=2, mp_chunksize=1)
            self.assertEqual(serial_rows, mp_rows)

    def test_write_csv(self):
        rows = [{"timestep": 0, "value": 1.23}]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.csv"
            write_csv(rows, path)
            with path.open("r", encoding="utf-8") as f:
                parsed = list(csv.DictReader(f))
            self.assertEqual(parsed[0]["timestep"], "0")
            self.assertEqual(parsed[0]["value"], "1.23")


if __name__ == "__main__":
    unittest.main()
