"""Verification suite for the 4 vacuum-offset & `F000`-conservation upgrades:
  1. `train.py`: `fit_box_scale=False` default + exact analytical gradient of `_make_voxel_plus_f000_loss` (`l2` & `kl`);
  2. `bmob.py`: `fit_joint_pmf_and_bmob_on_work` `kmask_mode='fixed_one'` default + `lam_f000` FD gradient (`< 1e-6`);
  3. `divot.py`: `VacuumConservingHazeField` exact `< 1e-13` `F000` conservation, exact `0.0000` core, non-negativity,
     even/odd monoclinic `P 1 21 1` FD gradients (`< 1e-8`), `fit_haze_arm_blind` (`haze` & `haze_peak`), and zero `R-free` leak;
  4. `bmob.py` & `cli.py`: `ReciprocalTarget.eval_unscaled_k1` & `surfnturf score-k1` unconfounded `k_mask === 1.000` ruler.
"""
import inspect
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import gemmi

from surfnturf.field import (
    RHO_BULK,
    ASUGrid,
    DatasetBundle,
    WaterFieldSpec,
    build_asu_features,
    load_ensemble,
    solv_to_fpart,
)
from surfnturf.io import load_saved_spec_and_theta, write_ccp4
from surfnturf.bmob import (
    MultiBRenderer,
    ReciprocalTarget,
    SlimDatasetBundle,
    fast_build_asu_and_mobility,
    fit_joint_pmf_and_bmob_on_work,
    load_ensemble_with_bfac,
    multiB_fpart_loss_and_grad,
)
from surfnturf.divot import (
    ContinuousPeakDict,
    CrystalGridOps,
    SolventAmplitudeProblem,
    VacuumConservingHazeField,
    build_open_solvent_envelope,
    fit_haze_arm_blind,
)
from surfnturf.train import _make_voxel_plus_f000_loss, fit_surfnturf_model
from surfnturf.cli import main as cli_main


def _mini_pdb_p21() -> str:
    return "\n".join([
        "CRYST1   20.000   22.000   21.000  90.00  99.25  90.00 P 1 21 1      2",
        "ATOM      1  N   ALA A   1       6.200   7.500   8.100  1.00 10.00           N",
        "ATOM      2  CA  ALA A   1       7.520   8.050   8.350  1.00 10.50           C",
        "ATOM      3  C   ALA A   1       7.850   9.320   7.580  1.00 10.50           C",
        "ATOM      4  O   ALA A   1       7.020  10.220   7.480  1.00 11.00           O",
        "ATOM      5  CB  ALA A   1       8.560   7.020   7.950  1.00 11.50           C",
        "ATOM      6  N   SER A   2       9.090   9.410   7.090  1.00 10.50           N",
        "ATOM      7  CA  SER A   2       9.580  10.580   6.380  1.00 11.00           C",
        "ATOM      8  C   SER A   2      11.050  10.420   6.050  1.00 11.50           C",
        "ATOM      9  O   SER A   2      11.550   9.300   5.950  1.00 12.00           O",
        "ATOM     10  OXT SER A   2      11.710  11.470   5.920  1.00 12.50           O",
        "ATOM     11  CB  SER A   2       9.360  11.840   7.210  1.00 12.00           C",
        "ATOM     12  OG  SER A   2       8.010  12.220   7.280  1.00 13.00           O",
        "END",
    ]) + "\n"


class TestVacuumHazeAndK1Ruler(unittest.TestCase):
    def test_train_defaults_and_f000_loss_fd_gradients(self):
        sig = inspect.signature(fit_surfnturf_model)
        self.assertFalse(sig.parameters["fit_box_scale"].default)
        self.assertGreater(sig.parameters["lam_f000"].default, 0.0)

        rng = np.random.default_rng(101)
        cell = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        grid = ASUGrid(cell, sg, (16, 16, 16), step=1)
        rho_t = rng.uniform(0.0, 0.65, size=(16, 16, 16))
        rho_t = grid.asu_to_full_3d(grid.full_3d_to_asu_mean(rho_t))
        PhiO = np.zeros((1, grid.n_asu, 28), dtype=np.float32)
        ds = DatasetBundle("t", grid, rho_t, 5.0, PhiO)

        g_hat = rng.uniform(0.02, 2.2, size=grid.n_asu)
        dg = rng.standard_normal(grid.n_asu)
        dg /= np.linalg.norm(dg)
        eps = 1e-5

        for mode in ("l2", "kl"):
            fn = _make_voxel_plus_f000_loss(ds, mode, lam_f000=0.40, target_mean_eA3=0.145)
            l0, grad0 = fn(g_hat)
            lp, _ = fn(g_hat + eps * dg)
            lm, _ = fn(g_hat - eps * dg)
            fd = (lp - lm) / (2.0 * eps)
            an = float(np.dot(grad0, dg))
            self.assertGreater(l0, 0.0)
            self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-7)

    def test_bmob_joint_default_and_f000_fd_gradients(self):
        sig = inspect.signature(fit_joint_pmf_and_bmob_on_work)
        self.assertEqual(sig.parameters["kmask_mode"].default, "fixed_one")
        self.assertGreater(sig.parameters["lam_f000"].default, 0.0)

        rng = np.random.default_rng(202)
        spec, theta0 = load_saved_spec_and_theta("l2")
        with tempfile.TemporaryDirectory() as td:
            p_pdb = os.path.join(td, "m.pdb")
            with open(p_pdb, "w", encoding="utf-8", newline="\n") as f:
                f.write(_mini_pdb_p21())
            cell = gemmi.UnitCell(20.0, 22.0, 21.0, 90.0, 99.25, 90.0)
            sg = gemmi.SpaceGroup("P 1 21 1")
            grid = ASUGrid(cell, sg, (20, 22, 20), step=1)
            mem, bis, wts = load_ensemble_with_bfac(p_pdb, nconf=1, use_lp=spec.use_lp)
            Phi_O, Phi_H, st_H, _, ext = fast_build_asu_and_mobility(
                grid, mem, bis, n_types=spec.n_types, include_hpol=spec.use_hpol,
                include_tet_oz=spec.use_tet_oz, member_weights=wts, return_extra=True,
            )
            ds = SlimDatasetBundle("j", grid, 5.0, Phi_O, Phi_H, st_H, member_weights=wts, F_hbsol=ext["F_hbsol"])
            rend = MultiBRenderer(grid, fft_workers=1)
            H = np.array([[h, k, l] for h in range(-4, 5) for k in range(0, 5) for l in range(0, 5)
                          if (h, k, l) != (0, 0, 0)], dtype=np.int64)
            d_hkl = 1.0 / np.sqrt(np.maximum(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64), 1e-12))
            Fc = (rng.standard_normal(len(H)) + 1j * rng.standard_normal(len(H))) * 15.0
            Fo = np.abs(Fc) + rng.uniform(1.0, 8.0, size=len(H))
            free = (np.arange(len(H)) % 20 == 0)
            targ = ReciprocalTarget(H, Fo, ~free, free, Fc, np.ones(len(H)),
                                    np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg)

            z_m = np.log(12.0) + 0.25 * rng.standard_normal((1, grid.n_asu))
            out0 = multiB_fpart_loss_and_grad(
                rend, targ, ds, z_m, spec=spec, theta=theta0, use_f32_blas=False,
                kmask_mode="fixed_one", lam_f000=0.35, target_mean_eA3=0.120, compute_grad=True,
            )
            dth = rng.standard_normal(len(theta0))
            dth /= np.linalg.norm(dth)
            dz = rng.standard_normal(z_m.shape)
            dz /= np.linalg.norm(dz)
            eps = 1e-5

            lp = multiB_fpart_loss_and_grad(
                rend, targ, ds, z_m + eps * dz, spec=spec, theta=theta0 + eps * dth, use_f32_blas=False,
                kmask_mode="fixed_one", lam_f000=0.35, target_mean_eA3=0.120, compute_grad=False,
            )["loss"]
            lm = multiB_fpart_loss_and_grad(
                rend, targ, ds, z_m - eps * dz, spec=spec, theta=theta0 - eps * dth, use_f32_blas=False,
                kmask_mode="fixed_one", lam_f000=0.35, target_mean_eA3=0.120, compute_grad=False,
            )["loss"]
            fd = (lp - lm) / (2.0 * eps)
            an = float(np.dot(out0["grad_theta"], dth) + np.sum(out0["grad_z_raw_m"] * dz))
            self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

    def test_haze_f000_exact_zero_core_fd_and_zero_rfree_leak(self):
        rng = np.random.default_rng(303)
        with tempfile.TemporaryDirectory() as td:
            p_pdb = os.path.join(td, "m.pdb")
            with open(p_pdb, "w", encoding="utf-8", newline="\n") as f:
                f.write(_mini_pdb_p21())
            cell = gemmi.UnitCell(20.0, 22.0, 21.0, 90.0, 99.25, 90.0)
            sg = gemmi.SpaceGroup("P 1 21 1")

            # Check BOTH even (24, 24, 24) AND odd-a/c (21, 22, 25) monoclinic P 1 21 1 grids!
            for shape in ((24, 24, 24), (21, 22, 25)):
                cg = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)
                open_m = build_open_solvent_envelope(cg, p_pdb)
                # Build base density with a sharp over-predicted hydration blob in open solvent
                sharp_blob = np.zeros(shape, dtype=np.float64)
                sharp_blob[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 3.2
                sharp_blob = cg.sym_s(sharp_blob) * open_m
                base_3d = RHO_BULK * open_m + sharp_blob

                haze = VacuumConservingHazeField(cg, base_3d, open_m, B_haze=75.0, fwhm_A=3.0, d_min=1.2)
                x_test = np.concatenate([np.array([-0.35]), 0.8 * rng.standard_normal(haze.n_asu)])
                rho_h, diag_h, pb_h = haze.forward_and_pullback(x_test, local=True, compute_grad=True)

                # 1. Exact F000 conservation (< 1e-13 relative)
                f000_out = float(np.sum(rho_h) * cg.vox_vol)
                self.assertLess(abs(f000_out - haze.f000_init) / max(haze.f000_init, 1e-12), 1e-13)
                # 2. Identical 0.000000 inside obligate protein core & strictly >= 0 everywhere
                core_vox = haze.M_supp == 0.0
                self.assertTrue(np.any(core_vox))
                self.assertEqual(float(np.max(np.abs(rho_h[core_vox]))), 0.0)
                self.assertGreaterEqual(float(np.min(rho_h)), 0.0)

                # 3. Exact reverse-mode adjoint vs central finite differences (< 1e-8)
                w_rand = rng.standard_normal(shape)
                g_an = pb_h(w_rand)
                dx = rng.standard_normal(len(x_test))
                dx /= np.linalg.norm(dx)
                eps = 1e-5
                rp, _, _ = haze.forward_and_pullback(x_test + eps * dx, local=True, compute_grad=False)
                rm, _, _ = haze.forward_and_pullback(x_test - eps * dx, local=True, compute_grad=False)
                fd = float(np.sum(w_rand * (rp - rm))) / (2.0 * eps)
                an = float(np.dot(g_an, dx))
                self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-7)

            # 4. End-to-end blind `fit_haze_arm_blind` (`haze` & `haze_peak`) + bit-for-bit zero R-free leak!
            cg = CrystalGridOps(cell, sg, (24, 24, 24), conv="plus", fft_workers=1)
            open_m = build_open_solvent_envelope(cg, p_pdb)
            sharp_blob = np.zeros((24, 24, 24), dtype=np.float64)
            # Find an open-solvent voxel for an over-predicted hole and another for a missing positive peak
            open_idx = np.argwhere(open_m > 0.92)
            i_hole = open_idx[len(open_idx) // 4]
            i_peak = open_idx[(3 * len(open_idx)) // 4]
            sharp_blob[tuple(i_hole)] = 18.0
            sharp_blob = cg.nops * cg.sym_s(sharp_blob) * open_m
            base_over = RHO_BULK * open_m + sharp_blob

            haze_gt = VacuumConservingHazeField(cg, base_over, open_m, B_haze=75.0, fwhm_A=3.0, d_min=1.8)
            rho_smear_true, _, _ = haze_gt.forward_and_pullback(np.array([1.2]), local=False, compute_grad=False)
            pk_seed = [{"xyz0": cg.xyz_full[np.ravel_multi_index(tuple(i_peak), (24, 24, 24))], "z_score": 6.0, "B_init": 10.0}]
            pd_gt = ContinuousPeakDict(cg, pk_seed, b_min=6.0, nonneg_dq=True)
            p_pk = pd_gt.init_params()
            p_pk[0] = 2.0
            rho_pk_true, _ = pd_gt.forward_and_pullback(p_pk, compute_grad=False)
            rho_true = rho_smear_true + haze_gt.M_supp * np.maximum(rho_pk_true, 0.0)

            H = np.array([[h, k, l] for h in range(-9, 10) for k in range(0, 10) for l in range(-9, 10)
                          if (h, k, l) != (0, 0, 0) and (h * h + k * k + l * l) <= 100], dtype=np.int64)
            idx = cg.hkl_idx(H)
            Q = cg.sf(1.4 * (1.0 - open_m), idx)
            Fo = np.abs(Q + cg.sf(rho_true, idx))
            free = (np.arange(len(H)) % 20 == 0)

            prob1 = SolventAmplitudeProblem(cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")
            rho_h1, inf_h1 = fit_haze_arm_blind(prob1, cg, base_over, open_m, use_peaks=False, maxiter=25)
            self.assertIn(inf_h1["selected_stage"], ("haze_uniform", "haze_local"))
            self.assertAlmostEqual(inf_h1["f000_init_e"], inf_h1["f000_final_e"], places=2)
            self.assertLess(inf_h1["r_val"], inf_h1["r_val_base"] - 0.002)

            rho_hp1, inf_hp1 = fit_haze_arm_blind(prob1, cg, base_over, open_m, use_peaks=True, maxiter=25)
            self.assertLess(inf_hp1["r_val"], inf_h1["r_val"])

            # Corrupt R-free amplitudes 100x -> 0 bits changed!
            Fo_bad = Fo.copy()
            Fo_bad[free] *= 100.0
            prob2 = SolventAmplitudeProblem(cg, H, Fo_bad, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")
            rho_hp2, inf_hp2 = fit_haze_arm_blind(prob2, cg, base_over, open_m, use_peaks=True, maxiter=25)
            self.assertEqual(inf_hp1["selected_stage"], inf_hp2["selected_stage"])
            self.assertTrue(np.array_equal(rho_hp1, rho_hp2))

    def test_unscaled_k1_ruler_and_cli_score_k1(self):
        rng = np.random.default_rng(404)
        cell = gemmi.UnitCell(24.0, 26.0, 28.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        H = np.array([[h, k, l] for h in range(0, 7) for k in range(0, 7) for l in range(0, 7)
                      if (h, k, l) != (0, 0, 0)], dtype=np.int64)
        d_hkl = 1.0 / np.sqrt(np.maximum(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64), 1e-12))
        Fc = (rng.standard_normal(len(H)) + 1j * rng.standard_normal(len(H))) * 25.0
        solv_3d = rng.uniform(0.0, 0.4, size=(24, 24, 24))
        sg_grid = ASUGrid(cell, sg, (24, 24, 24), step=1)
        solv_3d = sg_grid.asu_to_full_3d(sg_grid.full_3d_to_asu_mean(solv_3d))
        Fp = solv_to_fpart(solv_3d, cell.volume, H)

        k_aniso = np.exp(-0.08 * (1.0 / d_hkl ** 2))
        k_iso_scalar = 1.18
        Fo = k_iso_scalar * k_aniso * np.abs(Fc + 1.0 * Fp)
        free = (np.arange(len(H)) % 20 == 0)

        targ = ReciprocalTarget(
            H, Fo, ~free, free, Fc, k_iso_scalar * k_aniso,
            np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg, k_aniso=k_aniso,
        )
        res = targ.eval_unscaled_k1(Fp, n_iso_bins=6)
        self.assertLess(res["r_work_k1_kiso"], 1e-12)
        self.assertLess(res["r_free_k1_kiso"], 1e-12)
        self.assertAlmostEqual(res["k_iso_bin1"], k_iso_scalar, places=4)
        self.assertAlmostEqual(res["k0_1scalar"], k_iso_scalar, places=4)

        # Corrupting free_mask does NOT change fitted k_iso_bins or r_work_k1_kiso!
        Fo_c = Fo.copy()
        Fo_c[free] *= 50.0
        targ_c = ReciprocalTarget(
            H, Fo_c, ~free, free, Fc, k_iso_scalar * k_aniso,
            np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg, k_aniso=k_aniso,
        )
        res_c = targ_c.eval_unscaled_k1(Fp, n_iso_bins=6)
        self.assertEqual(res["k_iso_bins"], res_c["k_iso_bins"])
        self.assertEqual(res["r_work_k1_kiso"], res_c["r_work_k1_kiso"])

        # CLI smoke check for `surfnturf score-k1`
        with tempfile.TemporaryDirectory() as td:
            fm_p = os.path.join(td, "fm.mtz")
            map_p = os.path.join(td, "solv.ccp4")
            js_p = os.path.join(td, "k1.json")
            write_ccp4(map_p, solv_3d, cell, sg)

            mtz = gemmi.Mtz(with_base=True)
            mtz.spacegroup = sg
            mtz.set_cell_for_all(cell)
            mtz.add_dataset("xtal")
            for cn, ct in [
                ("FOBS", "F"), ("R_FREE_FLAGS", "I"),
                ("FCALC", "F"), ("PHIFCALC", "P"),
                ("FMASK", "F"), ("PHIFMASK", "P"),
                ("FMODEL", "F"), ("PHIFMODEL", "P"),
                ("K_ISOTROPIC", "W"), ("K_ANISOTROPIC", "W"), ("K_MASK", "W"),
            ]:
                mtz.add_column(cn, ct)
            fmod = (k_iso_scalar * k_aniso) * (Fc + Fp)
            mat = np.column_stack([
                H, Fo, free.astype(np.float32),
                np.abs(Fc), np.degrees(np.angle(Fc)),
                np.abs(Fp), np.degrees(np.angle(Fp)),
                np.abs(fmod), np.degrees(np.angle(fmod)),
                np.full(len(H), k_iso_scalar), k_aniso, np.ones(len(H)),
            ]).astype(np.float32)
            mtz.set_data(mat)
            mtz.write_to_file(fm_p)

            s_out = io.StringIO()
            with redirect_stdout(s_out), redirect_stderr(io.StringIO()):
                cli_main(["score-k1", "--fmodel-mtz", fm_p, "--solvent-ccp4", map_p, "--out-json", js_p])
            with open(js_p, encoding="utf-8") as jf:
                js = json.load(jf)
            self.assertLess(js["r_free_k1_kiso"], 1e-4)
            self.assertAlmostEqual(js["k_iso_bin1"], k_iso_scalar, places=3)


if __name__ == "__main__":
    unittest.main()
