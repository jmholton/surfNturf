"""Regression tests for the 2026-11-23 independent v0.2 audit fixes:
  A1. CrystalGridOps.sf / sf_adjoint real rfftn/irfftn parity & adjointness on even and odd grids (conv='plus'/'minus');
  A2. Protein-core zero vacuum leak (build_solvent_support_mask -> exact 0.0 e/A^3 inside excluded core);
  A3. SolventAmplitudeProblem k_solv_mode='fixed_one' (exact FD gradient, k_solv==k_prot, no L1 scale cheat) & f_model_ref gating;
  A4. build_open_solvent_envelope partial-occupancy blank-altloc atoms (e.g. SO4 occ=0.46);
  A5. ASUGrid.sym_expand(dedup_self=True) special-position solute deduplication vs wat_occ hole summing;
  A6. load_ensemble / load_ensemble_with_bfac mode='auto' routing on heterogeneous or incomplete A/B altlocs;
  A7. Nucleic-acid nucleobase + ribose/deoxyribose C-H completion & multi-model ENDMDL handling in reconnect.
"""
import os
import tempfile
import unittest
import numpy as np
import gemmi
from scipy import fft as sp_fft

from surfnturf.divot import (
    CrystalGridOps,
    ContinuousPeakDict,
    SolventAmplitudeProblem,
    build_open_solvent_envelope,
    build_solvent_support_mask,
    extract_problem_from_fmodel,
    fit_scale_divot_peak_given_hp,
    fit_adaptive_divot_peak_blind,
)
from surfnturf.field import ASUGrid, build_asu_features, load_ensemble
from surfnturf.bmob import ReciprocalTarget, load_ensemble_with_bfac
from surfnturf.hydrogens import complete_conformer_hydrogens
from surfnturf.reconnect import parse_multi_altloc_pdb


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

    def test_protein_core_zero_vacuum_leak_and_fixed_one_scale_and_gate(self):
        rng = np.random.default_rng(99)
        cell = gemmi.UnitCell(28.0, 28.0, 28.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 1")
        shape = (20, 20, 20)
        cg = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)

        # Sphere of radius 7 A around (14, 14, 14) is excluded protein core (base == 0, open_mask == 0.05)
        d_center = np.linalg.norm(cg.xyz_full - np.array([14.0, 14.0, 14.0]), axis=1).reshape(shape)
        core = d_center < 6.5
        open_mask = np.where(core, 0.05, 0.90)
        base_3d = np.where(d_center < 7.5, 0.0, 0.3344)

        supp = build_solvent_support_mask(base_3d, open_mask)
        self.assertEqual(float(np.max(supp[core])), 0.0)
        self.assertEqual(float(np.min(supp[d_center > 8.0])), 1.0)

        H = np.array([[h, k, l] for h in range(-5, 6) for k in range(-5, 6) for l in range(0, 6)
                      if (h, k, l) != (0, 0, 0)], dtype=int)
        idx = cg.hkl_idx(H)
        Q = cg.sf(np.where(core, 1.4, 0.0), idx)
        pk_true = ContinuousPeakDict(cg, [{"xyz0": np.array([4.0, 4.0, 4.0]), "z_score": 6.0, "val_eA3": 2.0, "B_init": 12.0}])
        bump, _ = pk_true.forward_and_pullback(np.array([0.8, 0.0, 0.0, 0.0, np.log(10.0)]), compute_grad=False)
        Fo = np.abs(1.12 * (Q + cg.sf(base_3d + bump, idx)))
        free = (np.arange(len(H)) % 20 == 0)

        prob = SolventAmplitudeProblem(cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")
        ev = prob.eval_rho(base_3d, active_mask=prob.train, compute_grad=True)
        self.assertAlmostEqual(float(ev["p"][0]), float(ev["p"][1]), places=14)
        # FD check on grho in fixed_one mode
        d_rho = rng.standard_normal(shape)
        d_rho /= np.linalg.norm(d_rho)
        eps = 1e-6
        lp = prob.eval_rho(base_3d + eps * d_rho, active_mask=prob.train, compute_grad=False)["loss"]
        lm = prob.eval_rho(base_3d - eps * d_rho, active_mask=prob.train, compute_grad=False)["loss"]
        fd = (lp - lm) / (2.0 * eps)
        an = float(np.sum(ev["grho"] * d_rho))
        self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

        rho_fit, _ = fit_scale_divot_peak_given_hp(
            prob, cg, base_3d, open_mask, active_mask=prob.train,
            z_thresh=3.2, lam_l1=3e-5, n_passes=1, maxiter_s0=15, maxiter_joint=25,
        )
        # Excluded protein core stays identically 0.0000 e/A^3 (zero softplus vacuum leak!)
        self.assertEqual(float(np.max(rho_fit[core])), 0.0)

        # Input f_model_ref gating: if input FMODEL already equals Fo on val, blind selector gates cleanly to 'base'
        prob_perfect_ref = SolventAmplitudeProblem(
            cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one", f_model_ref=Fo.copy()
        )
        rho_gated, info_gated = fit_adaptive_divot_peak_blind(
            prob_perfect_ref, cg, base_3d, open_mask, n_passes=1, hp_grid=((3.5, 5e-5),)
        )
        self.assertEqual(info_gated["selected_stage"], "base")
        self.assertTrue(np.array_equal(rho_gated, base_3d))

    def test_open_solvent_envelope_respects_partial_blank_occupancy(self):
        cell = gemmi.UnitCell(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        cg = CrystalGridOps(cell, gemmi.SpaceGroup("P 1"), (20, 20, 20), conv="plus", fft_workers=1)
        pdb_45 = (
            "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
            "HETATM    1  S   SO4 A 101      10.000  10.000  10.000  0.45 20.00           S\n"
            "END\n"
        )
        pdb_100 = pdb_45.replace("0.45", "1.00")
        with tempfile.TemporaryDirectory() as td:
            p45 = os.path.join(td, "s45.pdb")
            p100 = os.path.join(td, "s100.pdb")
            with open(p45, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_45)
            with open(p100, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_100)
            env45 = build_open_solvent_envelope(cg, p45, smooth_B=0.0)
            env100 = build_open_solvent_envelope(cg, p100, smooth_B=0.0)
        # At grid point (10, 10, 10) = index (10, 10, 10), d=0 -> expit(2.15/0.18) ~ 1.0
        self.assertAlmostEqual(float(env45[10, 10, 10]), 0.55, places=3)
        self.assertAlmostEqual(float(env100[10, 10, 10]), 0.00, places=3)

    def test_special_position_solute_dedup_matches_p1(self):
        # Atom sitting directly on 2-fold crystallographic axis (0, y, 0) in P 1 2 1:
        # without dedup_self, Phi_O spline features double relative to P 1!
        cell = gemmi.UnitCell(24.0, 24.0, 24.0, 90.0, 90.0, 90.0)
        g_p2 = ASUGrid(cell, gemmi.SpaceGroup("P 1 2 1"), (24, 24, 24), step=1)
        g_p1 = ASUGrid(cell, gemmi.SpaceGroup("P 1"), (24, 24, 24), step=1)
        xyz = np.array([[0.0, 6.0, 0.0]], dtype=np.float64)
        typ = np.array([2], dtype=np.int32)
        phi_p2, _, _ = build_asu_features(g_p2, [(xyz, typ)], n_types=10, include_hpol=False)
        phi_p1, _, _ = build_asu_features(g_p1, [(xyz, typ)], n_types=10, include_hpol=False)
        # Compare full 3D expanded spline sum across the unit cell
        f3_p2 = g_p2.asu_to_full_3d(np.sum(phi_p2[0], axis=1))
        f3_p1 = g_p1.asu_to_full_3d(np.sum(phi_p1[0], axis=1))
        self.assertLess(float(np.max(np.abs(f3_p2 - f3_p1))), 1e-5)

        # Default dedup_self=False still preserves both self-images for special-position ordered waters (occ=0.5)
        sw_raw, _ = g_p2.sym_expand(xyz, cutoff_A=3.0, dedup_self=False)
        sw_ded, _ = g_p2.sym_expand(xyz, cutoff_A=3.0, dedup_self=True)
        n_origin_raw = int(np.sum(np.linalg.norm(sw_raw - xyz[0], axis=1) < 0.01))
        n_origin_ded = int(np.sum(np.linalg.norm(sw_ded - xyz[0], axis=1) < 0.01))
        self.assertEqual(n_origin_raw, 2)
        self.assertEqual(n_origin_ded, 1)

    def test_auto_mode_routes_heterogeneous_ab_to_reconnect(self):
        # Residue 1 is 0.85 / 0.15 A/B, Residue 2 is 0.50 / 0.50 A/B -> mode='auto' must route to reconnect (8 confs),
        # whereas uniform 0.60 / 0.40 A/B stays on the fast 2-member letters path.
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
        # Standard B-form-like deoxythymidine (DT) heavy atoms
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

        # Multi-model PDB stops at first ENDMDL
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

    def test_global_scalar_mother_liquor_density_and_fd_grad(self):
        rng = np.random.default_rng(31415)
        H = np.array([[h, k, l] for h in range(1, 8) for k in range(0, 7) for l in range(0, 6)], dtype=np.int64)
        n = len(H)
        free = (np.arange(n) % 20 == 0)
        fc = rng.normal(0.0, 25.0, n) + 1j * rng.normal(0.0, 25.0, n)
        fp = rng.normal(0.0, 6.0, n) + 1j * rng.normal(0.0, 6.0, n)
        k_tot = rng.uniform(0.85, 1.15, n)
        d_hkl = np.linspace(12.0, 1.4, n)
        bins = np.zeros(n, dtype=np.int32)

        # Verify exact recovery of both high-salt (1.25x = 0.418 e/A^3) and detergent (0.80x = 0.268 e/A^3) mother liquors
        for k0_true, kb_true in ((1.04, 1.25), (0.97, 0.80)):
            fo = np.abs(k0_true * k_tot * (fc + kb_true * fp))
            targ = ReciprocalTarget(H, fo, ~free, free, fc, k_tot, bins, 1, d_hkl)
            k0_hat, kb_hat = targ._solve_global_scalar_k0_kb(fp, targ.work_mask)
            self.assertAlmostEqual(k0_hat, k0_true, places=5)
            self.assertAlmostEqual(kb_hat, kb_true, places=5)
            r_tr, r_vl, r_wk, r_fr = targ.eval_blind_rval(fp, kmask_mode="global_scalar")
            self.assertLess(max(r_tr, r_vl, r_wk, r_fr), 1e-6)

        # Verify analytical dL_dF matches central finite differences under noisy amplitudes
        fo_noisy = np.abs(1.03 * k_tot * (fc + 1.22 * fp)) + rng.uniform(0.2, 2.0, n)
        targ_n = ReciprocalTarget(H, fo_noisy, ~free, free, fc, k_tot, bins, 1, d_hkl)
        _, dL_dF, _, _, kb_vec = targ_n.eval_loss_grad_and_r(fp, kmask_mode="global_scalar")
        self.assertTrue(0.65 <= float(kb_vec[0]) <= 1.50)

        dfp = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        dfp /= np.linalg.norm(dfp)
        eps = 1e-5
        lp = targ_n.eval_loss_grad_and_r(fp + eps * dfp, kmask_mode="global_scalar")[0]
        lm = targ_n.eval_loss_grad_and_r(fp - eps * dfp, kmask_mode="global_scalar")[0]
        fd = (lp - lm) / (2.0 * eps)
        an = float(np.real(np.vdot(dL_dF, dfp)))
        self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)


if __name__ == "__main__":
    unittest.main()
