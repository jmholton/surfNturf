"""Regression tests for the 2026-11-23 independent v0.2 audit & Claude D1-D11 review fixes:
  A1. CrystalGridOps.sf / sf_adjoint real rfftn/irfftn parity & adjointness on even and odd grids (conv='plus'/'minus');
  A2/D6. Protein-core zero vacuum leak (build_solvent_support_mask with 5e-4 spline-tail floor -> exact 0.0 e/A^3);
  A3/D1. SolventAmplitudeProblem k_solv_mode='fixed_one' & strict blind rval_base gating (ignores unblind f_model_ref);
  A4/D2. build_open_solvent_envelope multi-neighbor product blockage (partial SO4 next to full-occ protein CA);
  A5/D10. ASUGrid.sym_expand(dedup_self=True) special-position solute deduplication & O(n_ops*N) F 4 3 2 speed;
  A6. load_ensemble / load_ensemble_with_bfac mode='auto' routing on heterogeneous or incomplete A/B altlocs;
  A7. Nucleic-acid nucleobase + ribose/deoxyribose C-H completion & multi-model ENDMDL handling in reconnect;
  D4/D5. CLI fit-bmob train_mask-only k_bulk baking (zero val/free leak) and symmetry-aware --fallback-fmodel;
  D7. check_crystal_compatibility sg.xhm() setting checks (P 1 21 1 vs P 1 1 21, C 1 2 1 vs I 1 2 1) & 1% cell tol;
  D8. Topology-based N-terminal cationic amine charges (H/H2/H3, 2-H H2/H3, PRO) & nucleic H1/H2/H3 neutrality;
  D9. build_asu_features(n_types=None) 2-type / 10-type / 13-type auto-inference;
  D11. global_scalar mother-liquor recovery, eps=1e-4 FD check, and fit_bmob_blind_gated in global_scalar mode.
"""
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import gemmi
from scipy import fft as sp_fft

from surfnturf.divot import (
    CrystalGridOps,
    ContinuousPeakDict,
    SolventAmplitudeProblem,
    build_open_solvent_envelope,
    build_solvent_support_mask,
    fit_scale_divot_peak_given_hp,
    fit_adaptive_divot_peak_blind,
)
from surfnturf.field import (
    N_KNOTS,
    ASUGrid,
    WaterFieldSpec,
    build_asu_features,
    classify_conformer_atoms,
    compute_pmf_density_with_pullback,
    load_ensemble,
)
from surfnturf.bmob import (
    MultiBRenderer,
    ReciprocalTarget,
    SlimDatasetBundle,
    fast_build_asu_and_mobility,
    fit_bmob_blind_gated,
    load_ensemble_with_bfac,
    reduce_mobility_features_to_asu,
)
from surfnturf.io import check_crystal_compatibility, read_ccp4_map_checked
from surfnturf.hydrogens import complete_conformer_hydrogens
from surfnturf.reconnect import parse_multi_altloc_pdb
from surfnturf.cli import main as cli_main


class TestV02AuditFixes(unittest.TestCase):
    def test_rfftn_sf_and_adjoint_even_and_odd_grids(self):
        rng = np.random.default_rng(20261123)
        cell = gemmi.UnitCell(31.4, 29.8, 27.6, 90.0, 97.5, 90.0)
        sg = gemmi.SpaceGroup("P 1")
        for shape in ((24, 20, 18), (23, 21, 19)):
            nu, nv, nw = shape
            H = np.array(
                [[h, k, l]
                 for h in range(-(nu // 2), (nu - 1) // 2 + 1, 3)
                 for k in range(-(nv // 2), (nv - 1) // 2 + 1, 3)
                 for l in range(-(nw // 2), (nw - 1) // 2 + 1, 2)],
                dtype=int,
            )
            rho = rng.standard_normal(shape)
            G = rng.standard_normal(len(H)) + 1j * rng.standard_normal(len(H))
            Af_ref = sp_fft.fftn(rho)
            ix = (np.mod(H[:, 0], nu), np.mod(H[:, 1], nv), np.mod(H[:, 2], nw))
            for conv in ("plus", "minus"):
                cg = CrystalGridOps(cell, sg, shape, conv=conv, fft_workers=1)
                idx = cg.hkl_idx(H)
                sf_got = cg.sf(rho, idx)
                sf_ref = cg.fsc * (np.conj(Af_ref)[ix] if conv == "plus" else Af_ref[ix])
                self.assertLess(float(np.max(np.abs(sf_got - sf_ref))), 1e-12)

                adj_got = cg.sf_adjoint(G, idx)
                lhs = float(np.real(np.vdot(G, sf_got)))
                rhs = float(np.sum(rho * adj_got))
                self.assertLess(abs(lhs - rhs) / max(abs(lhs), 1e-12), 1e-13)

    def test_protein_core_zero_vacuum_leak_fixed_one_and_d1_blind_gate(self):
        rng = np.random.default_rng(99)
        cell = gemmi.UnitCell(28.0, 28.0, 28.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 1")
        shape = (20, 20, 20)
        cg = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)

        # D6: even when predicted base_3d has 3e-4 e/A^3 B-spline tails inside the excluded core, supp_mask is 0.0
        d_center = np.linalg.norm(cg.xyz_full - np.array([14.0, 14.0, 14.0]), axis=1).reshape(shape)
        core = d_center < 6.5
        open_mask = np.where(core, 0.02, 0.90)
        base_3d = np.where(d_center < 7.5, 3.0e-4, 0.3344)

        supp = build_solvent_support_mask(base_3d, open_mask)
        self.assertEqual(float(np.max(supp[core])), 0.0)
        self.assertEqual(float(np.min(supp[d_center > 8.5])), 1.0)

        H = np.array([[h, k, l] for h in range(-6, 7) for k in range(-6, 7) for l in range(0, 7)
                      if (h, k, l) != (0, 0, 0)], dtype=int)
        idx = cg.hkl_idx(H)
        Q = cg.sf(np.where(core, 1.4, 0.0), idx)
        pk_true = ContinuousPeakDict(cg, [{"xyz0": np.array([4.2, 4.2, 4.2]), "z_score": 8.0, "val_eA3": 3.0, "B_init": 9.0}])
        bump, _ = pk_true.forward_and_pullback(np.array([1.3, 0.0, 0.0, 0.0, np.log(9.0)]), compute_grad=False)
        base_clean = np.where(d_center < 7.5, 0.0, 0.3344)
        Fo = np.abs(1.12 * (Q + cg.sf(base_clean + bump, idx)))
        free = (np.arange(len(H)) % 20 == 0)

        prob = SolventAmplitudeProblem(cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")
        ev = prob.eval_rho(base_clean, active_mask=prob.train, compute_grad=True)
        # FD check on grho in fixed_one mode (where k0*(rho) varies with rho -> non-trivial envelope theorem check)
        d_rho = rng.standard_normal(shape)
        d_rho /= np.linalg.norm(d_rho)
        eps = 1e-5
        lp = prob.eval_rho(base_clean + eps * d_rho, active_mask=prob.train, compute_grad=False)["loss"]
        lm = prob.eval_rho(base_clean - eps * d_rho, active_mask=prob.train, compute_grad=False)["loss"]
        fd = (lp - lm) / (2.0 * eps)
        an = float(np.sum(ev["grho"] * d_rho))
        self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

        rho_fit, _ = fit_scale_divot_peak_given_hp(
            prob, cg, base_clean, open_mask, active_mask=prob.train,
            z_thresh=3.2, lam_l1=2e-5, n_passes=1, maxiter_s0=15, maxiter_joint=25,
        )
        # Excluded protein core stays identically 0.0000 e/A^3 (zero softplus vacuum leak!)
        self.assertEqual(float(np.max(rho_fit[core])), 0.0)

        # D1: Passing an unblind f_model_ref=Fo (R_val_fmodel_in == 0.0) MUST NOT block acceptance of a real peak
        # that genuinely improves blind r_val over r_val_base!
        prob_with_unblind_ref = SolventAmplitudeProblem(
            cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one", f_model_ref=Fo.copy()
        )
        _, info_d1 = fit_adaptive_divot_peak_blind(
            prob_with_unblind_ref, cg, base_clean, open_mask,
            use_divots=False, use_peaks=True, n_passes=1, hp_grid=((3.5, 2e-5),)
        )
        self.assertEqual(info_d1["selected_stage"], "divot_peak")
        self.assertLess(info_d1["r_val"], info_d1["r_val_base"] - 0.005)
        self.assertAlmostEqual(info_d1["r_val_fmodel_in"], 0.0, places=12)

    def test_open_solvent_envelope_multi_neighbor_partial_occupancy_d2(self):
        cell = gemmi.UnitCell(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        cg = CrystalGridOps(cell, gemmi.SpaceGroup("P 1"), (20, 20, 20), conv="plus", fft_workers=1)
        pdb_45 = (
            "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
            "HETATM    1  S   SO4 A 101      10.000  10.000  10.000  0.45 20.00           S\n"
            "END\n"
        )
        # D2 scenario: full-occ CA at x=10.0 next to partial-occ SO4 S (occ=0.45) at x=11.5.
        # Voxel (11, 10, 10) is 0.5 A from S (nearest!) and 1.0 A from CA -> MUST stay blocked (<0.01) by CA!
        pdb_adj = (
            "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
            "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 12.00           C\n"
            "HETATM    2  S   SO4 A 101      11.500  10.000  10.000  0.45 20.00           S\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p45 = os.path.join(td, "s45.pdb")
            padj = os.path.join(td, "adj.pdb")
            with open(p45, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_45)
            with open(padj, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_adj)
            env45 = build_open_solvent_envelope(cg, p45, smooth_B=0.0)
            env_adj = build_open_solvent_envelope(cg, padj, smooth_B=0.0)
        self.assertAlmostEqual(float(env45[10, 10, 10]), 0.55, places=3)
        self.assertLess(float(env_adj[11, 10, 10]), 0.01)

    def test_special_position_solute_dedup_and_f432_speed_d9_d10(self):
        cell = gemmi.UnitCell(24.0, 24.0, 24.0, 90.0, 90.0, 90.0)
        g_p2 = ASUGrid(cell, gemmi.SpaceGroup("P 1 2 1"), (24, 24, 24), step=1)
        g_p1 = ASUGrid(cell, gemmi.SpaceGroup("P 1"), (24, 24, 24), step=1)
        xyz = np.array([[0.0, 6.0, 0.0]], dtype=np.float64)
        typ = np.array([2], dtype=np.int32)
        phi_p2, _, _ = build_asu_features(g_p2, [(xyz, typ)], n_types=10, include_hpol=False)
        phi_p1, _, _ = build_asu_features(g_p1, [(xyz, typ)], n_types=10, include_hpol=False)
        f3_p2 = g_p2.asu_to_full_3d(np.sum(phi_p2[0], axis=1))
        f3_p1 = g_p1.asu_to_full_3d(np.sum(phi_p1[0], axis=1))
        self.assertLess(float(np.max(np.abs(f3_p2 - f3_p1))), 1e-5)

        # D9: n_types=None auto-infers 2, 10, and 13 cleanly
        phi_t2, _, _ = build_asu_features(g_p1, [(np.array([[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]), np.array([0, 1], dtype=np.int32))],
                                          n_types=None, include_hpol=False)
        phi_t10, _, _ = build_asu_features(g_p1, [(xyz, np.array([7], dtype=np.int32))], n_types=None, include_hpol=False)
        phi_t13, _, _ = build_asu_features(g_p1, [(xyz, np.array([11], dtype=np.int32))], n_types=None, include_hpol=False)
        self.assertEqual(phi_t2.shape[2], 2 * N_KNOTS)
        self.assertEqual(phi_t10.shape[2], 10 * N_KNOTS)
        self.assertEqual(phi_t13.shape[2], 13 * N_KNOTS)

        # D10: O(n_ops * N) stabilizer pre-filter keeps high-symmetry F 4 3 2 (96 ops) fast & exact
        rng = np.random.default_rng(432)
        cell_f = gemmi.UnitCell(96.0, 96.0, 96.0, 90.0, 90.0, 90.0)
        g_f = ASUGrid(cell_f, gemmi.SpaceGroup("F 4 3 2"), (24, 24, 24), step=1)
        gen_xyz = rng.uniform(3.1, 21.7, size=(800, 3))
        spec_pt = np.array([[24.0, 24.0, 24.0]], dtype=np.float64)  # 24-fold special Wyckoff site (1/4, 1/4, 1/4)
        all_xyz = np.vstack([gen_xyz, spec_pt])
        t0 = time.perf_counter()
        sx_ded, si_ded = g_f.sym_expand(all_xyz, cutoff_A=5.6, dedup_self=True)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 1.0)
        n_spec_self = int(np.sum(np.linalg.norm(sx_ded[si_ded == 800] - spec_pt[0], axis=1) < 0.05))
        self.assertEqual(n_spec_self, 1)

    def test_d7_crystal_compatibility_xhm_and_cell_tolerance(self):
        c_eq = gemmi.UnitCell(40.0, 50.0, 60.0, 90.0, 95.0, 90.0)
        c_ok = gemmi.UnitCell(40.2, 50.0, 60.0, 90.0, 95.4, 90.0)   # 0.5% length, 0.4 deg -> within tol
        c_bad = gemmi.UnitCell(40.6, 50.0, 60.0, 90.0, 95.0, 90.0)  # 1.5% length -> exceeds 1.0% tol
        buf = io.StringIO()
        with redirect_stderr(buf):
            # Different settings with identical space-group number (#4 and #5) must fail compatibility!
            self.assertFalse(check_crystal_compatibility(c_eq, gemmi.SpaceGroup("P 1 21 1"), "a",
                                                         c_eq, gemmi.SpaceGroup("P 1 1 21"), "b"))
            self.assertFalse(check_crystal_compatibility(c_eq, gemmi.SpaceGroup("C 1 2 1"), "a",
                                                         c_eq, gemmi.SpaceGroup("I 1 2 1"), "b"))
            self.assertTrue(check_crystal_compatibility(c_eq, gemmi.SpaceGroup("P 1 21 1"), "a",
                                                        c_ok, gemmi.SpaceGroup("P 1 21 1"), "b"))
            self.assertFalse(check_crystal_compatibility(c_eq, gemmi.SpaceGroup("P 1 21 1"), "a",
                                                         c_bad, gemmi.SpaceGroup("P 1 21 1"), "b"))

    def test_d8_topological_nterm_and_nucleic_epol_charges(self):
        # 1. 1aho Challenge Val1 with H / H2 / H3 naming -> exact +1.0000
        val_h123 = [
            (1, "VAL", "N",   "N", [0.00, 0.00, 0.00], "A", ""),
            (1, "VAL", "H",   "H", [-0.85, 0.50, 0.00], "A", ""),
            (1, "VAL", "H2",  "H", [-0.20, -0.95, 0.25], "A", ""),
            (1, "VAL", "H3",  "H", [0.10, 0.25, -0.95], "A", ""),
            (1, "VAL", "CA",  "C", [1.46, 0.00, 0.00], "A", ""),
            (1, "VAL", "HA",  "H", [1.80, -0.90, -0.45], "A", ""),
            (1, "VAL", "CB",  "C", [2.00, 1.25, -0.70], "A", ""),
            (1, "VAL", "HB",  "H", [1.65, 2.10, -0.15], "A", ""),
            (1, "VAL", "C",   "C", [2.00, 0.00, 1.42], "A", ""),
            (1, "VAL", "O",   "O", [1.30, 0.00, 2.43], "A", ""),
        ]
        _, _, q_h123 = classify_conformer_atoms(val_h123, use_lp=False, return_charges=True)
        self.assertAlmostEqual(float(np.sum(q_h123)), 1.0000, places=6)

        # 2. qfit_best / cacbtb Val1 with only 2 N-terminal protons (H2 / H3) -> still sums to exact +1.0000
        val_2h = [r for r in val_h123 if r[2] != "H"]
        _, _, q_2h = classify_conformer_atoms(val_2h, use_lp=False, return_charges=True)
        self.assertAlmostEqual(float(np.sum(q_2h)), 1.0000, places=6)

        # 3. N-terminal PRO with H2 / H3 -> exact +1.0000
        pro_nt = [
            (1, "PRO", "N",   "N", [0.00, 0.00, 0.00], "A", ""),
            (1, "PRO", "H2",  "H", [-0.60, 0.75, 0.25], "A", ""),
            (1, "PRO", "H3",  "H", [-0.60, -0.75, -0.25], "A", ""),
            (1, "PRO", "CD",  "C", [0.45, 0.00, -1.40], "A", ""),
            (1, "PRO", "HD2", "H", [0.10, 0.85, -1.90], "A", ""),
            (1, "PRO", "HD3", "H", [0.10, -0.85, -1.90], "A", ""),
            (1, "PRO", "CA",  "C", [1.25, 0.00, 0.80], "A", ""),
            (1, "PRO", "HA",  "H", [1.15, 0.85, 1.45], "A", ""),
            (1, "PRO", "C",   "C", [2.50, 0.00, -0.05], "A", ""),
            (1, "PRO", "O",   "O", [2.50, 0.00, -1.28], "A", ""),
        ]
        _, _, q_pro = classify_conformer_atoms(pro_nt, use_lp=False, return_charges=True)
        self.assertAlmostEqual(float(np.sum(q_pro)), 1.0000, places=6)

        # 4. Nucleic adenine C2-H2 does NOT get N-terminal +0.3854 charge (stays neutral C=-0.05, H2=+0.05)
        ade_c2h2 = [
            (1, "A", "N1", "N", [0.00, 0.00, 0.00], "A", ""),
            (1, "A", "C2", "C", [1.34, 0.00, 0.00], "A", ""),
            (1, "A", "H2", "H", [1.88, 0.94, 0.00], "A", ""),
            (1, "A", "N3", "N", [2.05, -1.12, 0.00], "A", ""),
        ]
        _, _, q_ade = classify_conformer_atoms(ade_c2h2, use_lp=False, return_charges=True)
        self.assertAlmostEqual(float(q_ade[-1]), 0.05, places=6)
        self.assertAlmostEqual(float(q_ade[1] + q_ade[-1]), 0.0, places=6)

    def test_auto_mode_routes_heterogeneous_ab_to_reconnect(self):
        het_pdb = "\n".join([
            "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1",
            "ATOM      1  N  AGLY A   1       5.000   5.000   5.000  0.85 10.00           N",
            "ATOM      2  CA AGLY A   1       6.450   5.000   5.000  0.85 10.00           C",
            "ATOM      3  C  AGLY A   1       7.000   6.400   5.000  0.85 10.00           C",
            "ATOM      4  O  AGLY A   1       6.300   7.400   5.000  0.85 10.00           O",
            "ATOM      5  N  BGLY A   1       5.050   5.100   5.000  0.15 10.00           N",
            "ATOM      6  CA BGLY A   1       6.500   5.100   5.000  0.15 10.00           C",
            "ATOM      7  C  BGLY A   1       7.050   6.500   5.000  0.15 10.00           C",
            "ATOM      8  O  BGLY A   1       6.350   7.500   5.000  0.15 10.00           O",
            "ATOM      9  N  AGLY A   2       8.320   6.500   5.000  0.50 10.00           N",
            "ATOM     10  CA AGLY A   2       8.950   7.800   5.000  0.50 10.00           C",
            "ATOM     11  C  AGLY A   2      10.450   7.700   5.000  0.50 10.00           C",
            "ATOM     12  O  AGLY A   2      11.000   6.600   5.000  0.50 10.00           O",
            "ATOM     13  N  BGLY A   2       8.350   6.600   5.100  0.50 10.00           N",
            "ATOM     14  CA BGLY A   2       8.980   7.900   5.100  0.50 10.00           C",
            "ATOM     15  C  BGLY A   2      10.480   7.800   5.100  0.50 10.00           C",
            "ATOM     16  O  BGLY A   2      11.030   6.700   5.100  0.50 10.00           O",
            "END",
        ]) + "\n"
        unif_lines = []
        for ln in het_pdb.splitlines():
            if len(ln) > 20 and ln[16] == "A":
                unif_lines.append(ln[:54] + "  0.60" + ln[60:])
            elif len(ln) > 20 and ln[16] == "B":
                unif_lines.append(ln[:54] + "  0.40" + ln[60:])
            else:
                unif_lines.append(ln)
        unif_pdb = "\n".join(unif_lines) + "\n"

        with tempfile.TemporaryDirectory() as td:
            ph = os.path.join(td, "het.pdb")
            pu = os.path.join(td, "unif.pdb")
            with open(ph, "w", encoding="utf-8", newline="\n") as f:
                f.write(het_pdb)
            with open(pu, "w", encoding="utf-8", newline="\n") as f:
                f.write(unif_pdb)

            mh_f, _ = load_ensemble(ph, nconf=8, mode="auto", return_weights=True)
            mh_b, _, _ = load_ensemble_with_bfac(ph, nconf=8, mode="auto")
            self.assertEqual(len(mh_f), 8)
            self.assertEqual(len(mh_b), 8)

            mu_f, wu_f = load_ensemble(pu, nconf=8, mode="auto", return_weights=True)
            mu_b, _, wu_b = load_ensemble_with_bfac(pu, nconf=8, mode="auto")
            self.assertEqual(len(mu_f), 2)
            self.assertEqual(len(mu_b), 2)
            self.assertAlmostEqual(float(wu_f[0]), 0.60, places=3)
            self.assertAlmostEqual(float(wu_b[0]), 0.60, places=3)

    def test_nucleic_acid_ch_completion_and_endmdl_stop(self):
        dt = [
            (1, "DT", "O5'", "O", [0.00, 0.00, 0.00]),
            (1, "DT", "C5'", "C", [1.42, 0.00, 0.00]),
            (1, "DT", "C4'", "C", [1.95, 1.42, 0.00]),
            (1, "DT", "O4'", "O", [1.40, 2.15, 1.10]),
            (1, "DT", "C3'", "C", [3.47, 1.50, 0.05]),
            (1, "DT", "O3'", "O", [3.95, 1.00, -1.20]),
            (1, "DT", "C2'", "C", [3.80, 2.98, 0.25]),
            (1, "DT", "C1'", "C", [2.45, 3.15, 0.95]),
            (1, "DT", "N1",  "N", [2.55, 3.85, 2.25]),
            (1, "DT", "C2",  "C", [3.78, 4.10, 2.80]),
            (1, "DT", "O2",  "O", [4.82, 3.72, 2.30]),
            (1, "DT", "N3",  "N", [3.75, 4.80, 3.98]),
            (1, "DT", "C4",  "C", [2.62, 5.28, 4.62]),
            (1, "DT", "O4",  "O", [2.72, 5.90, 5.67]),
            (1, "DT", "C5",  "C", [1.36, 4.98, 4.00]),
            (1, "DT", "C7",  "C", [0.08, 5.45, 4.62]),
            (1, "DT", "C6",  "C", [1.40, 4.30, 2.84]),
        ]
        done_dt = complete_conformer_hydrogens(dt)
        h_dt = {a[2] for a in done_dt if a[3] == "H"}
        for expected in ("H3", "H6", "H71", "H72", "H73", "H1'", "H2'", "H2''", "H3'", "H4'", "H5'", "H5''"):
            self.assertIn(expected, h_dt)

        mm_pdb = (
            "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
            "MODEL        1\n"
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "ATOM      1  CA  ALA A   1       9.000   9.000   9.000  1.00 10.00           C\n"
            "ATOM      2  CA  ALA A   2      12.000  12.000  12.000  1.00 10.00           C\n"
            "ENDMDL\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pm = os.path.join(td, "mm.pdb")
            with open(pm, "w", encoding="utf-8", newline="\n") as f:
                f.write(mm_pdb)
            _, res_order, residues = parse_multi_altloc_pdb(pm)
            self.assertEqual(len(res_order), 1)
            self.assertEqual(len(residues[res_order[0]]["blank_raw"]), 1)

    def test_global_scalar_mother_liquor_fd_and_d4_d5_cli_blindness(self):
        rng = np.random.default_rng(31415)
        H = np.array([[h, k, l] for h in range(1, 8) for k in range(0, 7) for l in range(0, 6)], dtype=np.int64)
        n = len(H)
        free = (np.arange(n) % 20 == 0)
        fc = rng.normal(0.0, 25.0, n) + 1j * rng.normal(0.0, 25.0, n)
        fp = rng.normal(0.0, 6.0, n) + 1j * rng.normal(0.0, 6.0, n)
        k_tot = rng.uniform(0.85, 1.15, n)
        d_hkl = np.linspace(12.0, 1.4, n)
        bins = np.zeros(n, dtype=np.int32)

        for k0_true, kb_true in ((1.04, 1.25), (0.97, 0.80)):
            fo = np.abs(k0_true * k_tot * (fc + kb_true * fp))
            targ = ReciprocalTarget(H, fo, ~free, free, fc, k_tot, bins, 1, d_hkl)
            k0_hat, kb_hat = targ._solve_global_scalar_k0_kb(fp, targ.work_mask)
            self.assertAlmostEqual(k0_hat, k0_true, places=5)
            self.assertAlmostEqual(kb_hat, kb_true, places=5)
            r_tr, r_vl, r_wk, r_fr = targ.eval_blind_rval(fp, kmask_mode="global_scalar")
            self.assertLess(max(r_tr, r_vl, r_wk, r_fr), 1e-6)

        # D11: eps=1e-4 avoids float64 subtraction round-off on 1e-4 loss
        fo_noisy = np.abs(1.03 * k_tot * (fc + 1.22 * fp)) + rng.uniform(0.2, 2.0, n)
        targ_n = ReciprocalTarget(H, fo_noisy, ~free, free, fc, k_tot, bins, 1, d_hkl)
        _, dL_dF, _, _, kb_vec = targ_n.eval_loss_grad_and_r(fp, kmask_mode="global_scalar")
        self.assertTrue(0.65 <= float(kb_vec[0]) <= 1.50)

        dfp = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        dfp /= np.linalg.norm(dfp)
        eps = 1e-4
        lp = targ_n.eval_loss_grad_and_r(fp + eps * dfp, kmask_mode="global_scalar")[0]
        lm = targ_n.eval_loss_grad_and_r(fp - eps * dfp, kmask_mode="global_scalar")[0]
        fd = (lp - lm) / (2.0 * eps)
        an = float(np.real(np.vdot(dL_dF, dfp)))
        self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

        # D11 + D4 + D5: End-to-end fit_bmob_blind_gated (global_scalar) + CLI zero R-free / val leak + --fallback-fmodel
        cell = gemmi.UnitCell(22.0, 24.0, 26.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 1 21 1")
        pdb_txt = (
            "CRYST1   22.000   24.000   26.000  90.00  90.00  90.00 P 1 21 1      2\n"
            "ATOM      1  N   ALA A   1       5.000   6.000   7.000  1.00 10.00           N\n"
            "ATOM      2  CA  ALA A   1       6.450   6.000   7.000  1.00 10.00           C\n"
            "ATOM      3  C   ALA A   1       7.000   7.400   7.000  1.00 10.00           C\n"
            "ATOM      4  O   ALA A   1       6.300   8.400   7.000  1.00 10.00           O\n"
            "ATOM      5  CB  ALA A   1       7.000   5.200   5.820  1.00 11.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pdb_p = os.path.join(td, "m.pdb")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_txt)
            grid = ASUGrid(cell, sg, (24, 24, 28), step=1)
            spec = WaterFieldSpec()
            th0 = spec.init_params()
            mem, bis, wts = load_ensemble_with_bfac(pdb_p, nconf=1, use_lp=spec.use_lp)
            Phi_O, Phi_H, st_H, mob_f, ext = fast_build_asu_and_mobility(
                grid, mem, bis, n_types=spec.n_types, include_hpol=spec.use_hpol,
                include_tet_oz=spec.use_tet_oz, member_weights=wts, return_extra=True,
            )
            ds = SlimDatasetBundle("t", grid, 5.0, Phi_O, Phi_H, st_H, member_weights=wts, F_hbsol=ext["F_hbsol"])
            rend = MultiBRenderer(grid, fft_workers=1)
            gO, gH, sk, _ = compute_pmf_density_with_pullback(spec, ds, th0, need_grad=False)
            cached = (gO, gH, sk)
            mob_asu = reduce_mobility_features_to_asu(mob_f, wts)

            st_gem = gemmi.read_structure(pdb_p)
            dc = gemmi.DensityCalculatorX()
            dc.d_min = 2.0
            dc.set_grid_cell_and_spacegroup(st_gem)
            dc.put_model_density_on_grid(st_gem[0])
            fc_grid = gemmi.transform_map_to_f_phi(dc.grid, half_l=True)
            asu_d = fc_grid.prepare_asu_data(dmin=2.0)
            H_m = np.asarray(asu_d.miller_array, dtype=np.int64)
            Fc_m = np.asarray(asu_d.value_array, dtype=np.complex128)

            g_u, _, _ = rend.render_and_pullback(gO, gH, wts, np.full((1, grid.n_asu), np.log(22.0)), compute_grad=False)
            solv_true = np.maximum(grid.asu_to_full_3d(1.18 * g_u) * 0.3344, 0.0)
            from surfnturf.field import solv_to_fpart
            Fp_true = solv_to_fpart(solv_true, cell.volume, H_m)
            Fm_true = Fc_m + Fp_true
            Fo_m = np.maximum(np.abs(Fm_true), 0.05)
            fl_m = (np.arange(len(H_m)) % 20 == 0).astype(np.float32)

            d_m = 1.0 / np.sqrt(np.maximum(np.asarray(cell.calculate_1_d2_array(H_m), dtype=np.float64), 1e-12))
            targ_m = ReciprocalTarget(H_m, Fo_m, fl_m == 0, fl_m == 1, Fc_m, np.ones(len(H_m)),
                                      np.zeros(len(H_m), dtype=np.int32), 1, d_m, spacegroup=sg)
            res_gs = fit_bmob_blind_gated(rend, targ_m, ds, mob_asu, cached, kmask_mode="global_scalar", maxiter=15)
            self.assertIn(res_gs["selected_stage"], ("uniform", "phys", "spatial"))
            self.assertLess(res_gs["r_val"], 0.02)

            def _write_fm(path, fo_vec, fmask_vec):
                mtz = gemmi.Mtz(with_base=True)
                mtz.cell = cell
                mtz.spacegroup = sg
                mtz.add_dataset("xtal")
                for col, ctype in (("F-obs-filtered", "F"), ("R-free-flags", "I"),
                                   ("F-model", "F"), ("PHIF-model", "P"),
                                   ("FCALC", "F"), ("PHIFCALC", "P"),
                                   ("FMASK", "F"), ("PHIFMASK", "P"),
                                   ("K_ISOTROPIC", "W"), ("K_ANISOTROPIC", "W"), ("K_MASK", "W")):
                    mtz.add_column(col, ctype)
                fm_v = Fc_m + fmask_vec
                data = np.column_stack([
                    H_m, fo_vec, fl_m,
                    np.abs(fm_v), np.degrees(np.angle(fm_v)),
                    np.abs(Fc_m), np.degrees(np.angle(Fc_m)),
                    np.abs(fmask_vec), np.degrees(np.angle(fmask_vec)),
                    np.ones(len(H_m)), np.ones(len(H_m)), np.ones(len(H_m)),
                ]).astype(np.float32)
                mtz.set_data(data)
                mtz.write_to_file(path)

            fm1 = os.path.join(td, "fm1.mtz")
            fm2 = os.path.join(td, "fm2.mtz")
            fo_corrupt_free = Fo_m.copy()
            fo_corrupt_free[fl_m == 1] *= 50.0  # Massive R-free corruption MUST have zero effect on map or FPART!
            _write_fm(fm1, Fo_m, Fp_true)
            _write_fm(fm2, fo_corrupt_free, Fp_true)

            out1 = os.path.join(td, "o1")
            out2 = os.path.join(td, "o2")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                cli_main(["fit-bmob", "--pdb", pdb_p, "--fmodel-mtz", fm1, "--out", out1,
                          "--mode", "uniform", "--maxiter", "12"])
                cli_main(["fit-bmob", "--pdb", pdb_p, "--fmodel-mtz", fm2, "--out", out2,
                          "--mode", "uniform", "--maxiter", "12"])
            m1, _, _, _ = read_ccp4_map_checked(out1 + "_solvent.ccp4")
            m2, _, _, _ = read_ccp4_map_checked(out2 + "_solvent.ccp4")
            self.assertTrue(np.array_equal(m1, m2))

            # D5: --fallback-fmodel with symmetry-rotated --hkl (-h, k, -l in P 1 21 1 has 180 deg phase shift on k odd!)
            alt_hkl_p = os.path.join(td, "alt_hkl.mtz")
            H_rot = np.column_stack([-H_m[:, 0], H_m[:, 1], -H_m[:, 2]])
            mtz_h = gemmi.Mtz(with_base=True)
            mtz_h.cell = cell
            mtz_h.spacegroup = sg
            mtz_h.add_dataset("xtal")
            mtz_h.add_column("FP", "F")
            mtz_h.set_data(np.column_stack([H_rot, Fo_m]).astype(np.float32))
            mtz_h.write_to_file(alt_hkl_p)

            # Give FMASK an extra strong signal not in surfNturf so blind fallback triggers cleanly
            extra_fmask = Fp_true + 18.0 * np.exp(1j * np.linspace(0.1, 2.5, len(H_m)))
            fm_fb = os.path.join(td, "fm_fb.mtz")
            _write_fm(fm_fb, np.abs(Fc_m + extra_fmask), extra_fmask)
            out_fb = os.path.join(td, "ofb")
            s_out = io.StringIO()
            with redirect_stdout(s_out), redirect_stderr(io.StringIO()):
                cli_main(["fit-bmob", "--pdb", pdb_p, "--fmodel-mtz", fm_fb, "--hkl", alt_hkl_p,
                          "--out", out_fb, "--mode", "uniform", "--maxiter", "8", "--fallback-fmodel"])
            self.assertIn("fallback_fmodel", s_out.getvalue())
            fb_mtz = gemmi.read_mtz_file(out_fb + "_fpart.mtz")
            fb_amp = np.asarray(fb_mtz.column_with_label("FPART").array, dtype=np.float64)
            self.assertGreater(float(np.mean(fb_amp)), 5.0)


if __name__ == "__main__":
    unittest.main()
