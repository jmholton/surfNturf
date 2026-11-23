"""Regression tests for the v0.2 review fixes:
  V1 HIS completion default matches field.py typing; V2/R1 intra-chain gaps vs multi-chain continuing numbers & icodes;
  V3 donors are not acceptors; V4/V7/R4a hashed work_val split (stable, cell-free, P21 negative-h symmetry-canonical);
  V8/R2 R-free flag conventions (0..19, 0/1, SHELX -1/1, empty/out-of-range warnings);
  R3/N4/N5/N9/V10 shared split across missing-SG CCP4 maps, cell/SG mismatch warnings, m*.pdb priority, CLI guardrails;
  V5 bmob gate scores stages with the fit's k_mask model; V6 divot inner scales are history-free; N1 dead start.
"""
import io
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock
import numpy as np
import gemmi

from surfnturf.cli import main as cli_main
from surfnturf.hydrogens import complete_conformer_hydrogens
from surfnturf.field import load_ensemble
from surfnturf.io import (
    load_single_alt, read_ccp4_map_checked, check_crystal_compatibility,
    _sample_structure_for_target, write_ccp4,
)
from surfnturf import bmob as bmob_mod
from surfnturf.bmob import (
    ReciprocalTarget,
    load_ensemble_with_bfac,
    load_reciprocal_target_from_fmodel_mtz,
    make_work_train_val_split,
    free_mask_from_flags,
    canonical_hkl_keys,
    fit_bmob_blind_gated,
)
from surfnturf.divot import CrystalGridOps, SolventAmplitudeProblem, extract_problem_from_fmodel


# Residues 1-4 of a heavy-atom-only peptide (VAL GLY SER LYS), as in test_v02_upgrades
PEPTIDE = [
    (1, "VAL", "N", "N", [5.000, 5.000, 5.000]), (1, "VAL", "CA", "C", [6.460, 5.000, 5.000]),
    (1, "VAL", "C", "C", [7.010, 6.420, 5.000]), (1, "VAL", "O", "O", [6.280, 7.410, 5.000]),
    (1, "VAL", "CB", "C", [6.980, 4.240, 6.230]), (1, "VAL", "CG1", "C", [8.500, 4.240, 6.280]),
    (1, "VAL", "CG2", "C", [6.420, 2.820, 6.240]),
    (2, "GLY", "N", "N", [8.330, 6.510, 5.000]), (2, "GLY", "CA", "C", [8.980, 7.810, 5.000]),
    (2, "GLY", "C", "C", [10.490, 7.690, 5.000]), (2, "GLY", "O", "O", [11.040, 6.590, 5.000]),
    (3, "SER", "N", "N", [11.150, 8.840, 5.000]), (3, "SER", "CA", "C", [12.600, 8.910, 5.000]),
    (3, "SER", "C", "C", [13.150, 10.330, 5.000]), (3, "SER", "O", "O", [12.420, 11.320, 5.000]),
    (3, "SER", "CB", "C", [13.140, 8.160, 6.220]), (3, "SER", "OG", "O", [14.550, 8.180, 6.230]),
]
LYS = [
    (4, "LYS", "N", "N", [14.470, 10.420, 5.000]), (4, "LYS", "CA", "C", [15.120, 11.720, 5.000]),
    (4, "LYS", "C", "C", [16.630, 11.600, 5.000]), (4, "LYS", "O", "O", [17.180, 10.500, 5.000]),
    (4, "LYS", "CB", "C", [14.680, 12.510, 3.770]),
]

HIS_PDB = """\
CRYST1   32.000   32.000   32.000  90.00  90.00  90.00 P 1           1
ATOM      1  CB  HIS A   8       9.250  18.700  10.000  1.00 12.00           C
ATOM      2  CG  HIS A   8      10.000  20.000  10.000  1.00 12.00           C
ATOM      3  ND1 HIS A   8      10.000  21.370  10.000  1.00 12.00           N
ATOM      4  CE1 HIS A   8      11.240  21.820  10.000  1.00 12.00           C
ATOM      5  NE2 HIS A   8      12.060  20.780  10.000  1.00 12.00           N
ATOM      6  CD2 HIS A   8      11.310  19.630  10.000  1.00 12.00           C
END
"""


def _shifted(res, seq, dz):
    return [(seq, rn, an, el, [p[0], p[1], p[2] + dz]) for _, rn, an, el, p in res]


def _h_names(atoms, seq):
    return {a[2] for a in atoms if a[0] == seq and a[3] in ("H", "D")}


def _ser_hg_dir_point(dist):
    """Point `dist` A from SER OG along one of the 12 candidate O-H directions (azimuth 60 deg)."""
    ca, cb, og = (np.array(PEPTIDE[i][4]) for i in (12, 15, 16))
    a = (cb - og) / np.linalg.norm(cb - og)
    v_ref = ca - cb
    p1 = -(v_ref - np.dot(v_ref, a) * a)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(a, p1)
    th, ph = np.radians(109.5), np.radians(60.0)
    u = np.cos(th) * a + np.sin(th) * (np.cos(ph) * p1 + np.sin(ph) * p2)
    return og + dist * u


class TestV02Fixes(unittest.TestCase):
    def test_his_default_matches_field_typing(self):
        ring = [(8, "HIS", a, a[0], p) for a, p in [
            ("CB", [9.25, 18.70, 10.0]), ("CG", [10.0, 20.0, 10.0]), ("ND1", [10.0, 21.37, 10.0]),
            ("CE1", [11.24, 21.82, 10.0]), ("NE2", [12.06, 20.78, 10.0]), ("CD2", [11.31, 19.63, 10.0])]]
        hs = _h_names(complete_conformer_hydrogens(ring), 8)
        self.assertIn("HD1", hs)
        self.assertNotIn("HE2", hs)

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "his.pdb")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(HIS_PDB)
            (m_on,), _ = load_ensemble(p, return_weights=True, complete_h=True)
            (m_off,), _ = load_ensemble(p, return_weights=True, complete_h=False)
        def type_at(m, pos):
            i = int(np.argmin(np.linalg.norm(np.asarray(m[0]) - np.array(pos), axis=1)))
            return int(m[1][i])

        for pos in ([10.0, 21.37, 10.0], [12.06, 20.78, 10.0]):   # ND1, NE2: typing unchanged by completion
            self.assertEqual(type_at(m_on, pos), type_at(m_off, pos))

    def test_chain_break_is_not_n_terminus(self):
        # 1. Intra-chain gap (same chain A, LYS renumbered 6 and moved 8 A away) -> amide N without H, NOT NH3+
        pep_a = [(sq, rn, an, el, p, "A", "") for sq, rn, an, el, p in PEPTIDE]
        lys_a_gap = [(6, rn, an, el, [p[0], p[1], p[2] + 8.0], "A", "") for _, rn, an, el, p in LYS]
        gap = complete_conformer_hydrogens(pep_a + lys_a_gap)
        self.assertEqual(_h_names(gap, 1) & {"H1", "H2", "H3"}, {"H1", "H2", "H3"})
        self.assertFalse(_h_names(gap, 6) & {"H", "H1", "H2", "H3"})
        self.assertIn("H", _h_names(gap, 3))

        # 2. Insertion-code gap inside same chain A (52 '' -> 52 'B' moved 8 A away) -> NOT misread as 52 <= 52 restart!
        ser52 = [(52, rn, an, el, p, "A", "") for sq, rn, an, el, p in PEPTIDE if sq == 3]
        lys52b_gap = [(52, rn, an, el, [p[0], p[1], p[2] + 8.0], "A", "B") for _, rn, an, el, p in LYS]
        ic_gap = complete_conformer_hydrogens(ser52 + lys52b_gap)
        lys52b_h = {a[2] for a in ic_gap if a[1] == "LYS" and a[3] == "H"}
        self.assertFalse(lys52b_h & {"H", "H1", "H2", "H3"})

        # Bonded insertion code (52 '' -> 52 'A' at normal 1.33 A peptide bond) -> normal single amide H!
        lys52a_bonded = [(52, rn, an, el, p, "A", "A") for _, rn, an, el, p in LYS]
        ic_bonded = complete_conformer_hydrogens(ser52 + lys52a_bonded)
        lys52a_h = {a[2] for a in ic_bonded if a[1] == "LYS" and a[3] == "H"}
        self.assertIn("H", lys52a_h)
        self.assertFalse(lys52a_h & {"H1", "H2", "H3"})

        # 3. R1: Multi-chain with CONTINUING residue numbers (Chain A 1..3 -> Chain B 5 -> Chain C 101) MUST get NH3+!
        lys_b5 = [(5, rn, an, el, [p[0], p[1], p[2] + 8.0], "B", "") for _, rn, an, el, p in LYS]
        multichain = complete_conformer_hydrogens(pep_a + lys_b5)
        lys_b5_h = {a[2] for a in multichain if a[5] == "B" and a[3] == "H"}
        self.assertEqual(lys_b5_h & {"H1", "H2", "H3"}, {"H1", "H2", "H3"})

        # End-to-end PDB check through load_ensemble, load_ensemble_with_bfac, and load_single_alt:
        # Chain A (VAL 1 .. SER 3) + Chain B continuing at LYS 101 -> BOTH chains get 3x H_pos (5) at N-terminus!
        mc_pdb = "\n".join([
            "CRYST1   32.000   32.000   32.000  90.00  90.00  90.00 P 1           1",
            "ATOM      1  N   VAL A   1       5.000   5.000   5.000  1.00 10.00           N",
            "ATOM      2  CA  VAL A   1       6.460   5.000   5.000  1.00 10.00           C",
            "ATOM      3  C   VAL A   1       7.010   6.420   5.000  1.00 10.00           C",
            "ATOM      4  O   VAL A   1       6.280   7.410   5.000  1.00 11.00           O",
            "ATOM      5  N   GLY A   2       8.330   6.510   5.000  1.00 10.00           N",
            "ATOM      6  CA  GLY A   2       8.980   7.810   5.000  1.00 10.00           C",
            "ATOM      7  C   GLY A   2      10.490   7.690   5.000  1.00 10.00           C",
            "ATOM      8  O   GLY A   2      11.040   6.590   5.000  1.00 11.00           O",
            "TER",
            "ATOM      9  N   VAL B 101      15.000  15.000  15.000  1.00 12.00           N",
            "ATOM     10  CA  VAL B 101      16.460  15.000  15.000  1.00 12.00           C",
            "ATOM     11  C   VAL B 101      17.010  16.420  15.000  1.00 12.00           C",
            "ATOM     12  O   VAL B 101      16.280  17.410  15.000  1.00 13.00           O",
            "END",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "homodimer.pdb")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(mc_pdb)
            (m_ens,), _ = load_ensemble(p, return_weights=True, complete_h=True)
            (m_bf,), _, _ = load_ensemble_with_bfac(p, complete_h=True)
            (m_alt,), _ = load_single_alt(p, "A", complete_h=True)
            # Chain A VAL 1 (3 H_pos) + Chain B VAL 101 (3 H_pos) == 6 cationic H_pos (type 5)!
            self.assertEqual(int(np.sum(m_ens[1] == 5)), 6)
            self.assertEqual(int(np.sum(m_bf[1] == 5)), 6)
            self.assertEqual(int(np.sum(m_alt[1] == 5)), 6)

        # 4. Legacy 5-tuple numbering restart -> new chain -> NH3+
        new_chain = complete_conformer_hydrogens(PEPTIDE + _shifted(LYS, 1, 8.0))
        lys_h = {a[2] for a in new_chain if a[1] == "LYS" and a[3] == "H"}
        self.assertEqual(lys_h & {"H1", "H2", "H3"}, {"H1", "H2", "H3"})

    def test_hydroxyl_h_ignores_donors_and_finds_acceptors(self):
        site = _ser_hg_dir_point(2.85).tolist()

        def hg_dist(extra):
            out = complete_conformer_hydrogens(PEPTIDE + [extra])
            hg = [np.array(a[4]) for a in out if a[0] == 3 and a[2] == "HG"]
            self.assertEqual(len(hg), 1)
            return float(np.linalg.norm(hg[0] - np.array(site)))

        self.assertLess(hg_dist((10, "ASP", "OD1", "O", site)), 2.1)   # acceptor: H points at it
        self.assertGreater(hg_dist((10, "LYS", "NZ", "N", site)), 2.3)  # donor NH3+: H does not

    def test_hashed_split_is_stable_symmetric_and_cell_free(self):
        sg = gemmi.SpaceGroup("P 21 21 21")
        H = np.array([[h, k, l] for h in range(0, 12) for k in range(0, 12) for l in range(0, 12)
                      if (h, k, l) != (0, 0, 0)], dtype=np.int64)
        free = (np.arange(len(H)) % 20 == 0)
        d1 = 1.0 / np.sqrt(np.asarray(gemmi.UnitCell(34, 36, 38, 90, 90, 90).calculate_1_d2_array(H)))
        d2 = 1.0 / np.sqrt(np.asarray(gemmi.UnitCell(34.05, 36, 38, 90, 90, 90).calculate_1_d2_array(H)))
        tr1, vl1 = make_work_train_val_split(H, d1, free, spacegroup=sg)
        tr2, vl2 = make_work_train_val_split(H, d2, free, spacegroup=sg)
        self.assertTrue(np.array_equal(vl1, vl2) and np.array_equal(tr1, tr2))
        self.assertFalse(np.any(vl1 & free) or np.any(tr1 & vl1))
        self.assertTrue(np.array_equal(tr1 | vl1, ~free))
        self.assertTrue(0.035 < vl1.sum() / (~free).sum() < 0.07)

        # Removing one work reflection leaves every other assignment unchanged
        drop = int(np.nonzero(~free)[0][100])
        keep = np.ones(len(H), dtype=bool)
        keep[drop] = False
        tr3, vl3 = make_work_train_val_split(H[keep], d1[keep], free[keep], spacegroup=sg)
        self.assertTrue(np.array_equal(vl3, vl1[keep]) and np.array_equal(tr3, tr1[keep]))

        # Symmetry and Friedel mates share one key; in monoclinic P 1 21 1 with negative h, dropping spacegroup
        # breaks symmetry equivalence between (h, k, l) and (-h, k, -l) (R4a)
        k = canonical_hkl_keys(np.array([[1, 2, 3], [-1, 2, -3], [1, -2, 3], [-1, -2, -3]]), sg)
        self.assertEqual(len(set(k.tolist())), 1)

        sg_p21 = gemmi.SpaceGroup("P 1 21 1")
        H_p21 = np.array(
            [[h, k_i, l_i] for h in range(-5, 6) for k_i in range(0, 6) for l_i in range(1, 6)],
            dtype=np.int64,
        )
        free_p21 = np.zeros(len(H_p21), dtype=bool)
        _, vl_p21 = make_work_train_val_split(H_p21, None, free_p21, spacegroup=sg_p21)
        _, vl_p1 = make_work_train_val_split(H_p21, None, free_p21, spacegroup=None)
        self.assertFalse(np.array_equal(vl_p21, vl_p1))  # dropping spacegroup changes negative-h P21 assignments!
        mate_map = {tuple(row.tolist()): bool(v) for row, v in zip(H_p21, vl_p21)}
        for h_v, k_v, l_v in H_p21:
            if (-int(h_v), int(k_v), -int(l_v)) in mate_map:
                self.assertEqual(mate_map[(int(h_v), int(k_v), int(l_v))], mate_map[(-int(h_v), int(k_v), -int(l_v))])

    def test_free_flag_convention(self):
        valid = np.ones(100, dtype=bool)
        f30 = (np.arange(100) < 30).astype(float)                  # 0/1, test = 1 at 30 %
        self.assertEqual(int(free_mask_from_flags(f30, valid).sum()), 30)
        f_inv = 1.0 - (np.arange(100) < 5).astype(float)           # 0/1, test = 0 at 5 %
        self.assertEqual(int(free_mask_from_flags(f_inv, valid).sum()), 5)
        f20 = (np.arange(100) % 20).astype(float)                  # CCP4 0..19, test = 0
        m = free_mask_from_flags(f20, valid)
        self.assertEqual(int(m.sum()), 5)
        self.assertTrue(np.all(f20[m] == 0))
        m1 = free_mask_from_flags(f20, valid, free_flag_value=1)   # explicit override
        self.assertTrue(np.all(f20[m1] == 1) and int(m1.sum()) == 5)

        # R2: SHELX -1 / 1 flags select -1 as test set
        f_shelx = np.where(np.arange(100) < 7, -1.0, 1.0)
        err_sh = io.StringIO()
        with redirect_stderr(err_sh):
            m_sh = free_mask_from_flags(f_shelx, valid)
        self.assertEqual(int(m_sh.sum()), 7)
        self.assertTrue(np.all(f_shelx[m_sh] == -1.0))

        # R2: 1..20 flags with no 0 warn clearly on stderr when --free-flag-value is omitted
        f_1to20 = (np.arange(100) % 20 + 1).astype(float)
        err_no0 = io.StringIO()
        with redirect_stderr(err_no0):
            m_no0 = free_mask_from_flags(f_1to20, valid)
        self.assertEqual(int(m_no0.sum()), 0)
        self.assertIn("warning", err_no0.getvalue().lower())
        self.assertIn("--free-flag-value", err_no0.getvalue())

    def test_shared_split_without_map_sg_and_mismatch_and_cli_guards(self):
        # R3 + N9 + N4 + N5 + V10
        cell = gemmi.UnitCell(31.6, 54.56, 47.98, 90.0, 99.25, 90.0)
        sg = gemmi.SpaceGroup("P 1 21 1")
        shape = (24, 32, 28)
        H = np.array(
            [[h, k, l] for h in range(-4, 5) for k in range(0, 6) for l in range(1, 5)],
            dtype=np.int32,
        )
        n = len(H)
        rng = np.random.default_rng(17)
        fc_amp = rng.uniform(5.0, 40.0, size=n).astype(np.float32)
        fc_phi = rng.uniform(-180.0, 180.0, size=n).astype(np.float32)
        fk_amp = rng.uniform(0.5, 5.0, size=n).astype(np.float32)
        fk_phi = rng.uniform(-180.0, 180.0, size=n).astype(np.float32)
        km = np.full(n, 0.35, dtype=np.float32)
        k_iso = np.ones(n, dtype=np.float32)
        k_aniso = np.ones(n, dtype=np.float32)
        fmod = fc_amp * np.exp(1j * np.deg2rad(fc_phi)) + km * fk_amp * np.exp(1j * np.deg2rad(fk_phi))
        fo = (np.abs(fmod) + 0.5).astype(np.float32)
        flags = (np.arange(n) % 20 == 0).astype(np.float32)

        with tempfile.TemporaryDirectory() as td:
            fm_path = os.path.join(td, "fmodel.mtz")
            mtz = gemmi.Mtz(with_base=True)
            mtz.cell = cell
            mtz.spacegroup = sg
            mtz.add_dataset("fmodel")
            for lbl, ctype in [
                ("FOBS", "F"), ("R_FREE_FLAGS", "I"),
                ("FCALC", "F"), ("PHIFCALC", "P"),
                ("FMASK", "F"), ("PHIFMASK", "P"),
                ("FMODEL", "F"), ("PHIFMODEL", "P"),
                ("K_MASK", "W"), ("K_ISOTROPIC", "W"), ("K_ANISOTROPIC", "W"),
            ]:
                mtz.add_column(lbl, ctype)
            data = np.column_stack([
                H, fo, flags, fc_amp, fc_phi, fk_amp, fk_phi,
                np.abs(fmod).astype(np.float32), np.degrees(np.angle(fmod)).astype(np.float32),
                km, k_iso, k_aniso,
            ]).astype(np.float32)
            mtz.set_data(data)
            mtz.write_to_file(fm_path)

            # Write a CCP4 map and zero its header ISPG word (byte offset 22*4 = 88) so it lacks a space group
            map_path = os.path.join(td, "nosg.ccp4")
            write_ccp4(map_path, np.full(shape, 0.33, dtype=np.float64), cell, sg)
            with open(map_path, "r+b") as bf:
                bf.seek(22 * 4)
                bf.write(struct.pack("<i", 0))

            err_map = io.StringIO()
            with redirect_stderr(err_map):
                arr_m, c_m, sg_m, has_sg = read_ccp4_map_checked(map_path)
            self.assertFalse(has_sg)
            self.assertEqual(sg_m.number, 1)
            self.assertIn("no space group", err_map.getvalue())

            # R3: Even if cgrid was built with P 1, extract_problem_from_fmodel takes P 1 21 1 from fmodel.mtz,
            # giving 100% identical val_mask / train_mask to load_reciprocal_target_from_fmodel_mtz!
            targ_bmob, _, _ = load_reciprocal_target_from_fmodel_mtz(fm_path)
            cg_p1 = CrystalGridOps(c_m, sg_m, arr_m.shape, conv="plus", fft_workers=1)
            err_div = io.StringIO()
            with redirect_stderr(err_div):
                prob_div = extract_problem_from_fmodel(cg_p1, fm_path)
            self.assertIn("space group mismatch", err_div.getvalue())
            self.assertTrue(np.array_equal(prob_div.val, targ_bmob.val_mask))
            self.assertTrue(np.array_equal(prob_div.train, targ_bmob.train_mask))

            # N4: Cell mismatch check warns on > 1% difference
            err_cell = io.StringIO()
            with redirect_stderr(err_cell):
                ok_c = check_crystal_compatibility(
                    gemmi.UnitCell(35.0, 54.56, 47.98, 90.0, 99.25, 90.0), sg, "pdb", cell, sg, "mtz"
                )
            self.assertFalse(ok_c)
            self.assertIn("unit-cell mismatch", err_cell.getvalue())

            # N5: Ensemble directory prefers m*.pdb over a_pseudo.pdb alphabetically earlier
            ens_dir = os.path.join(td, "ens")
            os.makedirs(ens_dir)
            with open(os.path.join(ens_dir, "a_pseudo.pdb"), "w", encoding="utf-8", newline="\n") as f:
                f.write("CRYST1   99.000   99.000   99.000  90.00  90.00  90.00 P 1           1\n")
                f.write("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n")
            with open(os.path.join(ens_dir, "m01.pdb"), "w", encoding="utf-8", newline="\n") as f:
                f.write("CRYST1   31.600   54.560   47.980  90.00  99.25  90.00 P 1 21 1      2\n")
                f.write("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n")
            st_samp = _sample_structure_for_target(ens_dir)
            self.assertAlmostEqual(st_samp.cell.a, 31.6, places=2)

            # V10: --no-holdout-val with --mode auto exits cleanly with an informative error
            with self.assertRaises(SystemExit) as cm:
                cli_main(["fit-bmob", "--pdb", os.path.join(ens_dir, "m01.pdb"),
                          "--fmodel-mtz", fm_path, "--out", os.path.join(td, "o"), "--no-holdout-val"])
            self.assertIn("--no-holdout-val", str(cm.exception))

    def test_gate_scores_with_fit_kmask_mode(self):
        n = 400
        rng = np.random.default_rng(5)
        H = np.array([[h, k, l] for h in range(1, 9) for k in range(0, 7) for l in range(0, 8)], dtype=np.int64)[:n]
        free = (np.arange(len(H)) % 20 == 0)
        fc = rng.normal(0, 10, len(H)) + 1j * rng.normal(0, 10, len(H))
        targ = ReciprocalTarget(H, np.abs(fc) + 1.0, ~free, free, fc, np.ones(len(H)),
                                np.zeros(len(H), dtype=np.int32), 1, np.full(len(H), 2.0))
        seen = []
        orig = ReciprocalTarget.eval_blind_rval

        def spy(self_t, f_part, kmask_mode='per_bin', k_bounds=(0.0, 3.0)):
            seen.append(kmask_mode)
            return orig(self_t, f_part, kmask_mode=kmask_mode, k_bounds=k_bounds)

        stage = {'f_part': np.zeros(len(H), dtype=np.complex128), 'z_phys': None}
        with mock.patch.object(ReciprocalTarget, 'eval_blind_rval', spy), \
                mock.patch.object(bmob_mod, 'multiB_fpart_loss_and_grad', return_value=dict(stage)), \
                mock.patch.object(bmob_mod, 'fit_uniform_B_on_work', return_value=dict(stage)), \
                mock.patch.object(bmob_mod, 'fit_physical_bmob_on_work', return_value=dict(stage)), \
                mock.patch.object(bmob_mod, 'fit_spatial_bmob_on_work', return_value=dict(stage)):
            ds = mock.Mock()
            ds.grid.n_asu = 8
            fit_bmob_blind_gated(None, targ, ds, None, None, kmask_mode='fixed_one')
        self.assertEqual(seen, ['fixed_one'] * 4)

    def _divot_problem(self, anti):
        rng = np.random.default_rng(42)
        cell = gemmi.UnitCell(30.0, 32.0, 28.0, 90.0, 90.0, 90.0)
        cgrid = CrystalGridOps(cell, gemmi.SpaceGroup("P 21 21 21"), (24, 24, 24), conv="plus", fft_workers=1)
        H = np.array([[h, k, l] for h in range(0, 6) for k in range(0, 6) for l in range(0, 6)
                      if (h, k, l) != (0, 0, 0)], dtype=int)
        idx = cgrid.hkl_idx(H)
        rho = cgrid.sym_s(cgrid.gaussian_blur(rng.uniform(0.05, 1.2, size=cgrid.shape), B_blur=12.0))
        Q = cgrid.sf(cgrid.gaussian_blur(rng.uniform(0.0, 2.0, size=cgrid.shape), 6.0), idx)
        free = np.zeros(len(H), dtype=bool)
        free[::20] = True
        if anti:   # data = protein only, model calc carries extra solvent -> best k_solv is the floor
            return SolventAmplitudeProblem(cgrid, H, np.abs(Q), free, np.ones(len(H)), Q + 0.5 * cgrid.sf(rho, idx)), rho
        return SolventAmplitudeProblem(cgrid, H, np.abs(Q + 0.8 * cgrid.sf(rho, idx)), free, np.ones(len(H)), Q), rho

    def test_divot_dead_start_keeps_gradient(self):
        prob, rho = self._divot_problem(anti=True)
        ev = prob.eval_rho(rho, active_mask=prob.train, compute_grad=True)
        self.assertAlmostEqual(float(ev['p'][1]), SolventAmplitudeProblem.K_SOLV_MIN, places=12)
        self.assertGreater(float(np.max(np.abs(ev['grho']))), 0.0)

    def test_divot_inner_scales_are_history_free(self):
        for anti in (False, True):
            prob, rho = self._divot_problem(anti=anti)
            ref = prob.eval_rho(rho, compute_grad=False)
            # whatever an earlier call left in p_cache, the same rho gives the same scales and loss
            for pc in ([1.0, 1.0], [1.0, 0.35], [1.5, 5.0], [0.6, 4.0], [0.5, 0.0]):
                prob.p_cache = np.array(pc, dtype=np.float64)
                ev = prob.eval_rho(rho, compute_grad=False)
                self.assertTrue(np.array_equal(ev['p'], ref['p']), (anti, pc, ev['p'], ref['p']))
                self.assertEqual(ev['loss'], ref['loss'])

        # Flat valley (P = -Q, Fo = 0.5|Q|: every p0 - p1 = 0.5 fits exactly): the warm start used to pick the answer
        prob, _ = self._divot_problem(anti=False)
        valley = SolventAmplitudeProblem(prob.cgrid, prob.H, 0.5 * np.abs(prob.Q), prob.free, prob.K, prob.Q)
        sols = []
        for pc in ([1.5, 1.0], [1.2, 0.7], [0.9, 0.4]):
            valley.p_cache = np.array(pc, dtype=np.float64)
            sols.append(valley.solve_inner_scales(-valley.Q, valley.train))
        for s in sols[1:]:
            self.assertTrue(np.array_equal(s, sols[0]), sols)


if __name__ == "__main__":
    unittest.main()
