"""Unit tests for surfnturf.reconnect (geometry-guided Hungarian multi-altloc conformer reconnection)."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from surfnturf.reconnect import dist, main as reconnect_main, reconnect_pdb_to_conformers, write_member_pdbs


SYNTH_PDB = """\
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 21 21 21    4
ATOM      1  N  AVAL A   1       5.000   5.000   5.000  0.60  8.00           N
ATOM      2  CA AVAL A   1       6.450   5.000   5.000  0.60  8.00           C
ATOM      3  C  AVAL A   1       7.000   6.420   5.000  0.60  8.00           C
ATOM      4  O  AVAL A   1       6.300   7.420   5.000  0.60  8.00           O
ATOM      5  N  BVAL A   1       5.100   8.500   5.000  0.40 10.00           N
ATOM      6  CA BVAL A   1       6.550   8.500   5.000  0.40 10.00           C
ATOM      7  C  BVAL A   1       7.100   9.920   5.000  0.40 10.00           C
ATOM      8  O  BVAL A   1       6.400  10.920   5.000  0.40 10.00           O
ATOM      9  N  AALA A   2       8.320   9.980   5.000  0.40 10.00           N
ATOM     10  CA AALA A   2       9.050  11.240   5.000  0.40 10.00           C
ATOM     11  C  AALA A   2      10.550  11.050   5.000  0.40 10.00           C
ATOM     12  O  AALA A   2      11.050   9.930   5.000  0.40 10.00           O
ATOM     13  H  AALA A   2       8.800   9.120   5.000  0.00 10.00           H
ATOM     14  N  BALA A   2       8.220   6.480   5.000  0.60  8.00           N
ATOM     15  CA BALA A   2       8.950   7.740   5.000  0.60  8.00           C
ATOM     16  C  BALA A   2      10.450   7.550   5.000  0.60  8.00           C
ATOM     17  O  BALA A   2      10.950   6.430   5.000  0.60  8.00           O
ATOM     18  H  BALA A   2       8.700   5.620   5.000  0.60  8.00           H
ATOM     19  N  CALA A   2       8.220   6.480   5.000  0.00  8.00           N
ATOM     20  H  CALA A   2       8.700   5.620   5.000  0.25  8.00           H
HETATM   21 CL    CL A 301      14.500   7.000   5.000  1.00 12.00          CL
END
"""

HEAVY_ONLY_ICODE_NO_CRYST1_PDB = """\
ATOM      1  N  AVALAA   1       5.000   5.000   5.000  0.60  8.00           N
ATOM      2  CA AVALAA   1       6.450   5.000   5.000  0.60  8.00           C
ATOM      3  C  AVALAA   1       7.000   6.420   5.000  0.60  8.00           C
ATOM      4  O  AVALAA   1       6.300   7.420   5.000  0.60  8.00           O
ATOM      5  N  BVALAA   1       5.100   8.500   5.000  0.40 10.00           N
ATOM      6  CA BVALAA   1       6.550   8.500   5.000  0.40 10.00           C
ATOM      7  C  BVALAA   1       7.100   9.920   5.000  0.40 10.00           C
ATOM      8  O  BVALAA   1       6.400  10.920   5.000  0.40 10.00           O
ATOM      9  N  AALAAA   1A      8.320   9.980   5.000  0.40 10.00           N
ATOM     10  CA AALAAA   1A      9.050  11.240   5.000  0.40 10.00           C
ATOM     11  C  AALAAA   1A     10.550  11.050   5.000  0.40 10.00           C
ATOM     12  O  AALAAA   1A     11.050   9.930   5.000  0.40 10.00           O
ATOM     13  N  BALAAA   1A      8.220   6.480   5.000  0.60  8.00           N
ATOM     14  CA BALAAA   1A      8.950   7.740   5.000  0.60  8.00           C
ATOM     15  C  BALAAA   1A     10.450   7.550   5.000  0.60  8.00           C
ATOM     16  O  BALAAA   1A     10.950   6.430   5.000  0.60  8.00           O
END
"""

DNA_AND_ZN_PDB = """\
CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1
ATOM      1  C3'A DA B   1       2.000   2.000   2.000  0.50 10.00           C
ATOM      2  O3'A DA B   1       3.420   2.000   2.000  0.50 10.00           O
ATOM      3  C3'B DA B   1       2.000   6.000   2.000  0.50 10.00           C
ATOM      4  O3'B DA B   1       3.420   6.000   2.000  0.50 10.00           O
ATOM      5  P  A DT B   2       5.020   6.000   2.000  0.50 10.00           P
ATOM      6  OP1A DT B   2       5.600   7.350   2.000  0.50 10.00           O
ATOM      7  O5'A DT B   2       5.850   4.800   2.000  0.50 10.00           O
ATOM      8  P  B DT B   2       5.020   2.000   2.000  0.50 10.00           P
ATOM      9  OP1B DT B   2       5.600   3.350   2.000  0.50 10.00           O
ATOM     10  O5'B DT B   2       5.850   0.800   2.000  0.50 10.00           O
ATOM     11  CE1AHIS A  10      15.000  15.000  15.000  0.50 12.00           C
ATOM     12  NE2AHIS A  10      16.320  15.000  15.000  0.50 12.00           N
ATOM     13  CE1BHIS A  10      15.000  19.000  15.000  0.50 12.00           C
ATOM     14  NE2BHIS A  10      16.320  19.000  15.000  0.50 12.00           N
HETATM   15 ZN  A ZN A 201      18.370  19.000  15.000  0.50 12.00
HETATM   16 ZN  B ZN A 201      18.370  15.000  15.000  0.50 12.00
END
"""

DEUT_AND_SER_HG_PDB = """\
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
ATOM      1  N  AVAL A   1       5.000   5.000   5.000  0.50  8.00           N
ATOM      2  CA AVAL A   1       6.450   5.000   5.000  0.50  8.00           C
ATOM      3  C  AVAL A   1       7.000   6.420   5.000  0.50  8.00           C
ATOM      4  O  AVAL A   1       6.300   7.420   5.000  0.50  8.00           O
ATOM      5  N  BVAL A   1       5.100   8.500   5.000  0.50 10.00           N
ATOM      6  CA BVAL A   1       6.550   8.500   5.000  0.50 10.00           C
ATOM      7  C  BVAL A   1       7.100   9.920   5.000  0.50 10.00           C
ATOM      8  O  BVAL A   1       6.400  10.920   5.000  0.50 10.00           O
ATOM      9  N  AALA A   2       8.320   9.980   5.000  0.50 10.00           N
ATOM     10  CA AALA A   2       9.050  11.240   5.000  0.50 10.00           C
ATOM     11  C  AALA A   2      10.550  11.050   5.000  0.50 10.00           C
ATOM     12  O  AALA A   2      11.050   9.930   5.000  0.50 10.00           O
ATOM     13  D  AALA A   2       8.820   9.090   5.000  0.50 10.00           D
ATOM     14  N  BALA A   2       8.220   6.480   5.000  0.50  8.00           N
ATOM     15  CA BALA A   2       8.950   7.740   5.000  0.50  8.00           C
ATOM     16  C  BALA A   2      10.450   7.550   5.000  0.50  8.00           C
ATOM     17  O  BALA A   2      10.950   6.430   5.000  0.50  8.00           O
ATOM     18  D  BALA A   2       8.720   5.590   5.000  0.50  8.00           D
ATOM     19  N   SER A   3      11.100   8.650   5.000  1.00  9.00           N
ATOM     20  CA  SER A   3      12.520   8.700   5.000  1.00  9.00           C
ATOM     21  CB  SER A   3      13.100   9.950   5.600  1.00  9.00           C
ATOM     22  OG  SER A   3      14.500   9.950   5.600  1.00 1200.00         O
ATOM     23  HG ASER A   3      14.850  10.800   5.600  0.60  9.00           H
ATOM     24  HG BSER A   3      14.850   9.100   5.600  0.40  9.00           H
END
"""

SMALL_P1_SELF_CLASH_PDB = """\
CRYST1    4.200   15.000   15.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA AALA A   1       1.000   5.000   5.000  0.50 10.00           C
ATOM      2  CB AALA A   1       3.300   5.000   5.000  0.50 10.00           C
ATOM      3  CA BALA A   1       1.000   5.000   5.000  0.50 10.00           C
ATOM      4  CB BALA A   1       1.000   6.520   5.000  0.50 10.00           C
END
"""

RHOMBO_R3_PDB = """\
CRYST1   15.000   15.000   15.000  80.00  80.00  80.00 R 3           3
ATOM      1  CA AALA A   1       4.000   0.300   0.200  0.50 10.00           C
ATOM      2  CA BALA A   1       4.000   5.000   5.000  0.50 10.00           C
ATOM      3  CA AGLY B   2       0.200   4.000   0.300  0.50 10.00           C
ATOM      4  CA BGLY B   2       7.000   7.000   2.000  0.50 10.00           C
END
"""


class TestReconnect(unittest.TestCase):
    def test_swapped_labels_and_ghost_h_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "swapped.pdb")
            out_d = os.path.join(td, "members")
            pseudo_p = os.path.join(td, "pseudo.pdb")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(SYNTH_PDB)

            cryst1, confs, stats = reconnect_pdb_to_conformers(pdb_p, nconf=32, occ_flex=0)
            self.assertEqual(len(confs), 32)
            self.assertEqual(len(stats["dropped_ghosts"]), 1)
            self.assertGreaterEqual(stats["restored_zero_h"], 1)
            self.assertGreater(stats["alpha_clean_geom"][0], 0.0)
            self.assertEqual(stats["reconnected_geom"][0], 0.0)

            write_member_pdbs(cryst1, confs, out_d, pseudo_path=pseudo_p)
            with open(os.path.join(out_d, "m01.pdb"), "r", encoding="utf-8") as mf:
                m01_txt = mf.read()
            self.assertIn("HETATM", m01_txt)
            self.assertIn("CRYST1", m01_txt)

            with open(pseudo_p, "r", encoding="utf-8") as pf:
                cl_occs = [
                    float(line[54:60])
                    for line in pf
                    if line.startswith("HETATM") and line[17:20].strip() == "CL"
                ]
            self.assertEqual(len(cl_occs), 32)
            self.assertAlmostEqual(sum(cl_occs), 1.0, places=6)

    def test_heavy_only_2char_chain_icode_and_legacy_cli(self):
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "heavy_icode.pdb")
            out_d = os.path.join(td, "mem_cli")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(HEAVY_ONLY_ICODE_NO_CRYST1_PDB)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                reconnect_main([pdb_p, out_d, "nconf=10", "quotas=hamilton"])
            self.assertIn("no CRYST1", err_buf.getvalue())

            with open(os.path.join(out_d, "m01.pdb"), "r", encoding="utf-8") as mf:
                m01_lines = mf.read().splitlines()
            self.assertFalse(any(l.startswith("CRYST1") for l in m01_lines))
            atom_lines = [l for l in m01_lines if l.startswith("ATOM  ")]
            self.assertEqual(len(atom_lines), 8)
            self.assertTrue(all(l[76:78].strip() != "H" for l in atom_lines))
            self.assertTrue(all(l[20:22] == "AA" for l in atom_lines))

    def test_dna_phosphodiester_and_metal_coordination_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "dna_zn.pdb")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(DNA_AND_ZN_PDB)

            _, confs, stats = reconnect_pdb_to_conformers(pdb_p, nconf=8, occ_flex=0)
            # Zero steric clashes reported on 1.60 A O3'-P or 2.05 A His-Zn bonds
            self.assertEqual(stats["reconnected_geom"][2], 0.0)
            for c in confs:
                by_key = {(a["chain"], a["seqnum"], a["name"]): a for a in c}
                self.assertEqual(by_key[("A", 201, "ZN")]["elem"], "ZN")
                d_op = dist(by_key[("B", 1, "O3'")]["pos"], by_key[("B", 2, "P")]["pos"])
                d_zn = dist(by_key[("A", 10, "NE2")]["pos"], by_key[("A", 201, "ZN")]["pos"])
                self.assertAlmostEqual(d_op, 1.60, places=2)
                self.assertAlmostEqual(d_zn, 2.05, places=2)

    def test_deuterated_amide_and_h_only_rotamer_and_biso_clamp(self):
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "deut_ser.pdb")
            out_d = os.path.join(td, "out")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(DEUT_AND_SER_HG_PDB)

            cryst1, confs, _ = reconnect_pdb_to_conformers(pdb_p, nconf=10, occ_flex=0)
            for c in confs:
                ala2_hd = [a for a in c if a["seqnum"] == 2 and a["name"] in ("H", "D")]
                self.assertEqual(len(ala2_hd), 1)
                self.assertEqual(ala2_hd[0]["name"], "D")
                ser3_hg = [a for a in c if a["seqnum"] == 3 and a["name"] == "HG"]
                self.assertEqual(len(ser3_hg), 1)
            # Verify 6:4 split of Ser3 HG across 10 conformers
            n_top = sum(1 for c in confs for a in c if a["seqnum"] == 3 and a["name"] == "HG" and a["pos"][1] > 10.0)
            self.assertEqual(n_top, 6)

            write_member_pdbs(cryst1, confs, out_d)
            with open(os.path.join(out_d, "m01.pdb"), "r", encoding="utf-8") as mf:
                for line in mf:
                    if line.startswith(("ATOM  ", "HETATM")):
                        self.assertEqual(len(line.rstrip("\r\n")), 78)
                        self.assertLessEqual(float(line[60:66]), 999.99)

    def test_small_p1_lattice_self_clash_and_rhombohedral_r3(self):
        with tempfile.TemporaryDirectory() as td:
            p1_p = os.path.join(td, "p1_small.pdb")
            with open(p1_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(SMALL_P1_SELF_CLASH_PDB)
            _, _, st_p1 = reconnect_pdb_to_conformers(p1_p, nconf=4, occ_flex=0)
            # Alt A has CB at x=3.30 which is 1.90 A from translated CA at x=5.20 (50% of conformers)
            self.assertAlmostEqual(st_p1["reconnected_geom"][2], 0.5, places=5)

            r3_p = os.path.join(td, "r3_rhomb.pdb")
            with open(r3_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(RHOMBO_R3_PDB)
            _, _, st_r3 = reconnect_pdb_to_conformers(r3_p, nconf=4, occ_flex=0)
            # In rhombohedral R 3, (z,x,y) maps Res 2 A near Res 1 A (< 0.5 A clash), so optimizer pairs (A,B) and (B,A)!
            self.assertGreater(st_r3["alpha_clean_geom"][2], 0.0)
            self.assertEqual(st_r3["reconnected_geom"][2], 0.0)


if __name__ == "__main__":
    unittest.main()
