"""Verification of v0.2.0 low-resolution & PDB robustness upgrades:
  1. Standalone riding-hydrogen completion (`surfnturf.hydrogens.complete_conformer_hydrogens`).
  2. Unified leak-free 90% work_train / 5% work_val / 5% FreeR==0 split across `bmob` and `divot`.
  3. Resolution-scaled spatial B(x) (`fwhm = max(3.2, 3.5 * d_min)`) and blind `work_val` stage gating.
"""
import os
import tempfile
import unittest
import numpy as np
import gemmi

from surfnturf.hydrogens import complete_conformer_hydrogens
from surfnturf.field import ASUGrid, WaterFieldSpec, classify_conformer_atoms, load_ensemble
from surfnturf.bmob import (
    SlimDatasetBundle,
    MultiBRenderer,
    ReciprocalTarget,
    make_work_train_val_split,
    resolution_scaled_bmob_hyperparams,
    load_ensemble_with_bfac,
    fast_build_asu_and_mobility,
    reduce_mobility_features_to_asu,
    compute_pmf_density_with_pullback,
    fit_bmob_blind_gated,
)
from surfnturf.divot import (
    CrystalGridOps,
    ContinuousPeakDict,
    SolventAmplitudeProblem,
    fit_adaptive_divot_peak_blind,
)


HEAVY_PEPTIDE_PDB = """\
CRYST1   32.000   32.000   32.000  90.00  90.00  90.00 P 21 21 21    4
ATOM      1  N   VAL A   1       5.000   5.000   5.000  1.00 10.00           N
ATOM      2  CA  VAL A   1       6.460   5.000   5.000  1.00 10.00           C
ATOM      3  C   VAL A   1       7.010   6.420   5.000  1.00 10.00           C
ATOM      4  O   VAL A   1       6.280   7.410   5.000  1.00 11.00           O
ATOM      5  CB  VAL A   1       6.980   4.240   6.230  1.00 12.00           C
ATOM      6  CG1 VAL A   1       8.500   4.240   6.280  1.00 13.00           C
ATOM      7  CG2 VAL A   1       6.420   2.820   6.240  1.00 13.00           C
ATOM      8  N   GLY A   2       8.330   6.510   5.000  1.00 10.00           N
ATOM      9  CA  GLY A   2       8.980   7.810   5.000  1.00 10.00           C
ATOM     10  C   GLY A   2      10.490   7.690   5.000  1.00 10.00           C
ATOM     11  O   GLY A   2      11.040   6.590   5.000  1.00 11.00           O
ATOM     12  N   SER A   3      11.150   8.840   5.000  1.00 11.00           N
ATOM     13  CA  SER A   3      12.600   8.910   5.000  1.00 11.00           C
ATOM     14  C   SER A   3      13.150  10.330   5.000  1.00 11.00           C
ATOM     15  O   SER A   3      12.420  11.320   5.000  1.00 12.00           O
ATOM     16  CB  SER A   3      13.140   8.160   6.220  1.00 13.00           C
ATOM     17  OG  SER A   3      14.550   8.180   6.230  1.00 14.00           O
ATOM     18  N   LYS A   4      14.470  10.420   5.000  1.00 11.00           N
ATOM     19  CA  LYS A   4      15.120  11.720   5.000  1.00 11.00           C
ATOM     20  C   LYS A   4      16.630  11.600   5.000  1.00 11.00           C
ATOM     21  O   LYS A   4      17.180  10.500   5.000  1.00 12.00           O
ATOM     22  CB  LYS A   4      14.680  12.510   3.770  1.00 12.00           C
ATOM     23  CG  LYS A   4      15.200  13.940   3.750  1.00 13.00           C
ATOM     24  CD  LYS A   4      14.760  14.720   2.520  1.00 14.00           C
ATOM     25  CE  LYS A   4      15.280  16.150   2.500  1.00 15.00           C
ATOM     26  NZ  LYS A   4      14.850  16.920   1.310  1.00 16.00           N
ATOM     27  N   CYS A   5      17.290  12.750   5.000  1.00 11.00           N
ATOM     28  CA  CYS A   5      18.740  12.820   5.000  1.00 11.00           C
ATOM     29  C   CYS A   5      19.290  14.240   5.000  1.00 11.00           C
ATOM     30  O   CYS A   5      18.560  15.230   5.000  1.00 12.00           O
ATOM     31  CB  CYS A   5      19.280  12.070   6.220  1.00 12.00           C
ATOM     32  SG  CYS A   5      21.080  12.070   6.250  1.00 13.00           S
ATOM     33  SG  CYS A   6      21.100  12.070   8.290  1.00 13.00           S
ATOM     34  CB  CYS A   6      22.880  12.070   8.600  1.00 12.00           C
ATOM     35  SG  CYS A   7      26.500   6.000   6.000  1.00 14.00           S
ATOM     36  CB  CYS A   7      25.000   6.000   7.020  1.00 13.00           C
ATOM     37  CA  CYS A   7      23.800   6.000   6.080  1.00 12.00           C
ATOM     38  CG  HIS A   8      10.000  20.000  10.000  1.00 12.00           C
ATOM     39  ND1 HIS A   8      10.000  21.370  10.000  1.00 12.00           N
ATOM     40  CE1 HIS A   8      11.240  21.820  10.000  1.00 12.00           C
ATOM     41  NE2 HIS A   8      12.060  20.780  10.000  1.00 12.00           N
ATOM     42  CD2 HIS A   8      11.310  19.630  10.000  1.00 12.00           C
HETATM   43 ZN    ZN A 201      14.100  20.780  10.000  1.00 12.00          ZN
END
"""


class TestV02Upgrades(unittest.TestCase):
    def test_riding_hydrogen_completion_and_idempotency(self):
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "pep_noH.pdb")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(HEAVY_PEPTIDE_PDB)

            m_on, w_on = load_ensemble(pdb_p, return_weights=True, complete_h=True)
            m_off, _ = load_ensemble(pdb_p, return_weights=True, complete_h=False)
            m_bf, b_bf, w_bf = load_ensemble_with_bfac(pdb_p, complete_h=True)

            self.assertEqual(len(m_on), 1)
            self.assertEqual(len(m_bf), 1)
            self.assertEqual(len(m_on[0][1]), len(m_bf[0][1]))
            self.assertLess(float(np.max(np.abs(w_on - w_bf))), 1e-12)

            # Without completion, 0 hydrogens (types 4, 5, 8) exist; with completion, all 3 H classes are populated
            t_off = set(m_off[0][1].tolist())
            t_on = m_on[0][1]
            self.assertTrue(t_off.isdisjoint({4, 5, 8}))
            self.assertGreaterEqual(int(np.sum(t_on == 4)), 4)   # GLY H, SER H, SER HG, LYS H, CYS H, HIS HD1, CYS7 HG
            self.assertGreaterEqual(int(np.sum(t_on == 5)), 6)   # VAL N-term H1..H3 + LYS HZ1..HZ3
            self.assertGreaterEqual(int(np.sum(t_on == 8)), 20)  # aliphatic & aromatic C-H

            # Verify disulfide CYS 5 & 6 (SG-SG = 2.04 A) do NOT get HG, free CYS 7 DOES get HG, and Zn-bound HIS NE2 skips HE2
            st = gemmi.read_structure(pdb_p)
            raw = [
                (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist())
                for ch in st[0] for r in ch for a in r
            ]
            comp1 = complete_conformer_hydrogens(raw)
            comp2 = complete_conformer_hydrogens(comp1)
            self.assertEqual(len(comp1), len(comp2))  # strictly idempotent!

            h_by_res = {}
            for rseq, _, an, el, _ in comp1:
                if el == "H":
                    h_by_res.setdefault(rseq, set()).add(an)
            self.assertEqual(h_by_res[1] & {"H1", "H2", "H3"}, {"H1", "H2", "H3"})
            self.assertIn("HG", h_by_res[3])
            self.assertEqual(h_by_res[4] & {"HZ1", "HZ2", "HZ3"}, {"HZ1", "HZ2", "HZ3"})
            self.assertNotIn("HG", h_by_res.get(5, set()))  # disulfide SG
            self.assertNotIn("HG", h_by_res.get(6, set()))  # disulfide SG
            self.assertIn("HG", h_by_res[7])                # free thiol SG
            self.assertIn("HD1", h_by_res[8])               # unligated ND1 protonated
            self.assertNotIn("HE2", h_by_res[8])            # Zn-ligated NE2 (2.04 A) kept unprotonated!

    def test_90_5_5_split_parity_and_resolution_scaling(self):
        # 1. Exact sub-angstrom preservation & smooth low-res scaling
        f075, l075 = resolution_scaled_bmob_hyperparams(0.75)
        f096, l096 = resolution_scaled_bmob_hyperparams(0.965)
        f170, l170 = resolution_scaled_bmob_hyperparams(1.70)
        f270, l270 = resolution_scaled_bmob_hyperparams(2.70)
        self.assertEqual(f075, 3.2)
        self.assertEqual(l075, 2e-4)
        self.assertEqual(f096, 3.2)
        self.assertAlmostEqual(l096, 2e-4, places=12)
        self.assertAlmostEqual(f170, 5.95, places=2)
        self.assertGreater(l170, 8e-4)
        self.assertAlmostEqual(f270, 9.45, places=2)
        self.assertGreater(l270, 4e-3)

        # 2. Exact reflection-set parity between ReciprocalTarget and SolventAmplitudeProblem under row permutation
        rng = np.random.default_rng(99)
        cell = gemmi.UnitCell(34.0, 36.0, 38.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        cgrid = CrystalGridOps(cell, sg, (24, 24, 24), conv="plus", fft_workers=1)

        H = np.array(
            [[h, k, l] for h in range(0, 7) for k in range(0, 7) for l in range(0, 7) if (h, k, l) != (0, 0, 0)],
            dtype=np.int64,
        )
        n = len(H)
        d_hkl = 1.0 / np.sqrt(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64))
        free_m = (np.arange(n) % 20 == 0)
        fo = rng.uniform(5.0, 80.0, size=n)
        fc = rng.normal(0.0, 20.0, size=n) + 1j * rng.normal(0.0, 20.0, size=n)

        targ = ReciprocalTarget(H, fo, ~free_m, free_m, fc, np.ones(n), np.zeros(n, dtype=np.int32), 1, d_hkl)
        perm = rng.permutation(n)
        prob = SolventAmplitudeProblem(cgrid, H[perm], fo[perm], free_m[perm], np.ones(n), fc[perm])

        set_val_bmob = {tuple(h.tolist()) for h in targ.hkl[targ.val_mask]}
        set_val_divot = {tuple(h.tolist()) for h in prob.H[prob.val]}
        set_tr_bmob = {tuple(h.tolist()) for h in targ.hkl[targ.train_mask]}
        set_tr_divot = {tuple(h.tolist()) for h in prob.H[prob.train]}
        self.assertEqual(set_val_bmob, set_val_divot)
        self.assertEqual(set_tr_bmob, set_tr_divot)
        self.assertFalse(np.any(targ.train_mask & targ.free_mask))
        self.assertFalse(np.any(targ.val_mask & targ.free_mask))

        targ_sub = targ.for_training_subset()
        self.assertTrue(np.array_equal(targ_sub.work_mask, targ.train_mask))
        self.assertFalse(np.any(targ_sub.work_mask & targ.val_mask))

    def test_blind_val_gates_reject_noise_and_accept_true_signal(self):
        rng = np.random.default_rng(2026)
        cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        shape = (24, 24, 24)
        grid = ASUGrid(cell, sg, shape, step=1)
        cgrid = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)

        # Low-resolution (d_min ~ 2.7 A) reflection list where 3.2 A blobs & spurious peaks would overfit
        H_list = [
            [h, k, l] for h in range(0, 11) for k in range(0, 11) for l in range(0, 11)
            if (h, k, l) != (0, 0, 0) and cell.calculate_d([h, k, l]) >= 2.70
        ]
        H = np.array(H_list, dtype=np.int64)
        n_ref = len(H)
        d_hkl = 1.0 / np.sqrt(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64))
        free_m = (np.arange(n_ref) % 20 == 0)

        members = [(np.array([[6.0, 6.0, 6.0], [9.5, 6.2, 6.0]], dtype=np.float64), np.array([0, 3], dtype=np.int32))]
        bisos = [np.array([25.0, 30.0], dtype=np.float64)]
        wts = np.array([1.0], dtype=np.float64)
        spec = WaterFieldSpec(n_types=10, use_langmuir=True, use_oz=True, use_hpol=True, use_lp=False, use_tet_oz=False)
        theta = spec.init_params()

        Phi_O, Phi_H, st_H, mob_f = fast_build_asu_and_mobility(grid, members, bisos, n_types=10, include_hpol=True, max_workers=1)
        ds = SlimDatasetBundle("lowres", grid, B_map=5.0, Phi_O=Phi_O, Phi_H=Phi_H, steric_H=st_H, member_weights=wts)
        mob_asu = reduce_mobility_features_to_asu(mob_f, wts)
        rend = MultiBRenderer(grid, fft_workers=1)
        gO_m, gH_m, sk, _ = compute_pmf_density_with_pullback(spec, ds, theta, need_grad=False)
        cached = (gO_m, gH_m, sk)

        # Ground truth is uniform B = 55 A^2 + pure Gaussian experimental noise (zero extra peaks, zero spatial B variation)
        g_true_asu, _, _ = rend.render_and_pullback(gO_m, gH_m, wts, np.full((1, grid.n_asu), np.log(55.0)), scale_k=sk, compute_grad=False)
        rho_true = grid.asu_to_full_3d(g_true_asu) * 0.3344
        idx = cgrid.hkl_idx(H)
        Q_prot = rng.normal(0.0, 40.0, size=n_ref) + 1j * rng.normal(0.0, 40.0, size=n_ref)
        P_true = cgrid.sf(rho_true, idx)
        Fo_noisy = np.maximum(np.abs(Q_prot + P_true) + rng.normal(0.0, 6.5, size=n_ref), 0.5)

        targ = ReciprocalTarget(H, Fo_noisy, ~free_m, free_m, Q_prot, np.ones(n_ref), np.zeros(n_ref, dtype=np.int32), 1, d_hkl, spacegroup=sg)
        gated_bmob = fit_bmob_blind_gated(rend, targ, ds, mob_asu, cached, kmask_mode="fixed_one", maxiter=15)
        self.assertIn(gated_bmob["selected_stage"], ("0shot", "uniform", "phys"))
        self.assertTrue(np.isfinite(gated_bmob["r_val"]))

        # Now feed the exact true base map into `fit_adaptive_divot_peak_blind` on noisy 2.7 A data: must select `'base'`!
        prob_noisy = SolventAmplitudeProblem(cgrid, H, Fo_noisy, free_m, np.ones(n_ref), Q_prot)
        _, inf_div_noisy = fit_adaptive_divot_peak_blind(
            prob_noisy, cgrid, rho_true, np.ones(shape), fitreso_s=10.0,
            use_divots=False, use_peaks=True, n_passes=1, hp_grid=((3.3, 2e-5),), gate_on_val=True
        )
        self.assertEqual(inf_div_noisy["selected_stage"], "base")
        self.assertEqual(inf_div_noisy["n_peaks_act"], 0)

        # Conversely, when a genuine strong water peak IS present at high resolution, gate selects `'divot_peak'`!
        H_hi = np.array(
            [[h, k, l] for h in range(0, 11) for k in range(0, 11) for l in range(0, 11) if (h, k, l) != (0, 0, 0)],
            dtype=np.int64,
        )
        free_hi = (np.arange(len(H_hi)) % 20 == 0)
        Q_hi = rng.normal(0.0, 35.0, size=len(H_hi)) + 1j * rng.normal(0.0, 35.0, size=len(H_hi))
        pk_dict = ContinuousPeakDict(cgrid, [{"xyz0": np.array([4.2, 12.5, 8.1]), "z_score": 8.0, "val_eA3": 3.0, "B_init": 9.0}], half_win=6)
        bump_3d, _ = pk_dict.forward_and_pullback(np.array([1.4, 0.0, 0.0, 0.0, np.log(9.0)]), compute_grad=False)
        Fo_sig = np.abs(Q_hi + cgrid.sf(rho_true + bump_3d, cgrid.hkl_idx(H_hi)))
        prob_sig = SolventAmplitudeProblem(cgrid, H_hi, Fo_sig, free_hi, np.ones(len(H_hi)), Q_hi)
        _, inf_sig = fit_adaptive_divot_peak_blind(
            prob_sig, cgrid, rho_true, np.ones(shape), fitreso_s=10.0,
            use_divots=False, use_peaks=True, n_passes=1, hp_grid=((3.5, 2e-5),), gate_on_val=True
        )
        self.assertEqual(inf_sig["selected_stage"], "divot_peak")
        self.assertGreaterEqual(inf_sig["n_peaks_act"], 1)
        self.assertLess(inf_sig["r_val"], inf_sig["r_val_base"] - 0.01)


if __name__ == "__main__":
    unittest.main()
