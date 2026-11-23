"""Automated finite-difference and edge-case verification of surfnturf.field, surfnturf.io, and surfnturf.cli."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
import numpy as np
import gemmi

from surfnturf.cli import _split_dataset_spec
from surfnturf.data import get_pretrained_weights_path
from surfnturf.field import (
    RHO_BULK, R_CUT, ASUGrid, WaterFieldSpec, DatasetBundle, classify_conformer_atoms,
    build_asu_features, forward_and_grad, forward_direct, smoothness_penalty_and_grad,
    solv_to_fpart, fpart_grad_to_asu, fpart_forward_and_grad, fit_kmask_per_bin
)
from surfnturf.io import (
    load_saved_spec_and_theta, predict_maps_for_target, write_fpart_mtz_on_hkl, write_fpart_mtz_to_dmin
)


class TestSurfNTurfField(unittest.TestCase):
    def test_pretrained_weights_bundle(self):
        for variant in ("l2", "kl", "v1_l2", "v1_kl"):
            p = get_pretrained_weights_path(variant)
            self.assertTrue(os.path.isfile(p))
            spec, theta = load_saved_spec_and_theta(variant)
            self.assertEqual(len(theta), spec.n_params)
            self.assertTrue(np.all(np.isfinite(theta)))
            if variant.startswith("v1_"):
                self.assertEqual(spec.n_params, 173)
                self.assertEqual(spec.n_types, 10)
                self.assertFalse(spec.use_lp)
                self.assertFalse(spec.use_tet_oz)
                self.assertEqual(spec.n_oz_steps, 1)
                self.assertFalse(spec.use_pauli_cap)
                self.assertFalse(spec.zero_bulk_offset)
                self.assertFalse(spec.use_epol)
            else:
                self.assertEqual(spec.n_params, 219)
                self.assertEqual(spec.n_types, 13)
                self.assertTrue(spec.use_lp)
                self.assertTrue(spec.use_tet_oz)
                self.assertEqual(spec.n_oz_steps, 4)
                self.assertTrue(spec.use_pauli_cap)
                self.assertTrue(spec.zero_bulk_offset)
                self.assertFalse(spec.use_epol)

    def test_ion_typing_and_multichain_hip(self):
        # Chain 1 HIS 10 is doubly protonated (HIP, ND1-NE2 = 2.15 A); Chain 2 HIS 10 is neutral HIE 15 A away
        atoms = [
            (10, "HIS", "ND1", "N", [0.0, 0.0, 0.0]),
            (10, "HIS", "HD1", "H", [-0.9, 0.0, 0.0]),
            (10, "HIS", "NE2", "N", [2.14, 0.0, 0.0]),
            (10, "HIS", "HE2", "H", [3.04, 0.0, 0.0]),
            (10, "HIS", "ND1", "N", [15.0, 0.0, 0.0]),
            (10, "HIS", "NE2", "N", [17.14, 0.0, 0.0]),
            (10, "HIS", "HE2", "H", [18.04, 0.0, 0.0]),
            (20, "CL",  "CL",  "CL", [8.0, 0.0, 0.0]),
            (21, "ZN",  "ZN",  "ZN", [8.0, 4.0, 0.0]),
            (22, "MSE", "SE",  "SE", [8.0, 8.0, 0.0]),
            (23, "PO4", "P",   "P",  [8.0, 12.0, 0.0]),
        ]
        _, typ = classify_conformer_atoms(atoms)
        # Heavy atoms appear first in order: [ND1_A, NE2_A, ND1_B, NE2_B, CL, ZN, SE, P], then [HD1_A, HE2_A, HE2_B]
        self.assertEqual(list(typ[:8]), [3, 3, 2, 3, 3, 1, 9, 6])
        # Chain A HD1/HE2 are cationic H_pos (5); Chain B HE2 is neutral H_pol (4)!
        self.assertEqual(list(typ[8:]), [5, 5, 4])

    def test_grid_symmetry_validation_and_small_cell_sym_expand(self):
        cell_small = gemmi.UnitCell(14.0, 12.0, 10.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        with self.assertRaises(ValueError):
            ASUGrid(cell_small, sg, (90, 80, 60), step=2)  # 45 incompatible with 1/2 screw

        grid = ASUGrid(cell_small, sg, (16, 14, 12), step=1)
        atom_xyz = np.array([[1.2, 1.5, 9.6]], dtype=np.float64)
        sx, _ = grid.sym_expand(atom_xyz, cutoff_A=R_CUT)
        # Verify against brute-force 5x5x5 supercell expansion that zero neighbors within R_CUT of ASU are missed
        brute = []
        f = atom_xyz @ grid.F.T
        for op in grid.ops:
            R = np.array(op.rot, dtype=np.float64) / op.DEN
            t = np.array(op.tran, dtype=np.float64) / op.DEN
            g = (f @ R.T + t) % 1.0
            for kx in range(-2, 3):
                for ky in range(-2, 3):
                    for kz in range(-2, 3):
                        brute.append((g + [kx, ky, kz]) @ grid.O.T)
        brute_arr = np.vstack(brute)
        for p in grid.asu_xyz[::17]:
            d_b = np.sort(np.linalg.norm(brute_arr - p, axis=1))
            d_s = np.sort(np.linalg.norm(sx - p, axis=1))
            in_cut_b = d_b[d_b < R_CUT]
            in_cut_s = d_s[d_s < R_CUT]
            self.assertEqual(len(in_cut_b), len(in_cut_s))
            if len(in_cut_b):
                self.assertLess(float(np.max(np.abs(in_cut_b - in_cut_s))), 1e-12)

    def test_dataset_spec_splitting_and_missing_cryst1(self):
        self.assertEqual(
            _split_dataset_spec(r"ds1:C:\dir\a.ccp4:D:\dir\b.pdb:5.0"),
            ["ds1", r"C:\dir\a.ccp4", r"D:\dir\b.pdb", "5.0"],
        )
        self.assertEqual(
            _split_dataset_spec("ds1:/tmp/a.ccp4:/tmp/b.pdb:5.0"),
            ["ds1", "/tmp/a.ccp4", "/tmp/b.pdb", "5.0"],
        )
        with tempfile.TemporaryDirectory() as td:
            no_cryst_pdb = os.path.join(td, "nocell.pdb")
            with open(no_cryst_pdb, "w", encoding="utf-8", newline="\n") as f:
                f.write("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n")
            with self.assertRaises(ValueError):
                predict_maps_for_target(no_cryst_pdb)

    def test_real_and_reciprocal_gradients(self):
        rng = np.random.default_rng(42)
        cell = gemmi.UnitCell(16.0, 14.0, 12.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        grid = ASUGrid(cell, sg, (24, 20, 16), step=1)
        self.assertEqual(grid.n_asu, (24 * 20 * 16) // 4)

        u = rng.normal(size=grid.n_asu)
        v = rng.normal(size=grid.shape)
        lhs = float(np.sum(grid.asu_to_full_3d(u) * v))
        rhs = float(np.sum(u * grid.full_3d_to_asu_grad(v)))
        self.assertLess(abs(lhs - rhs), 1e-11)

        sample_atoms = [
            (1, "VAL", "N",   "N", [2.1, 3.2, 4.1]),
            (1, "VAL", "H1",  "H", [2.1, 3.2, 5.1]),
            (1, "VAL", "H2",  "H", [3.0, 3.2, 3.8]),
            (1, "VAL", "CA",  "C", [1.2, 4.1, 3.6]),
            (1, "VAL", "HA",  "H", [0.4, 3.6, 3.1]),
            (1, "VAL", "C",   "C", [1.8, 5.4, 3.2]),
            (1, "VAL", "O",   "O", [2.9, 5.6, 2.8]),
            (2, "ASP", "N",   "N", [1.0, 6.4, 3.3]),
            (2, "ASP", "H",   "H", [0.1, 6.2, 3.6]),
            (2, "ASP", "OD1", "O", [1.5, 8.2, 2.5]),
            (3, "SER", "OG",  "O", [5.5, 4.0, 6.0]),
            (3, "SER", "HG",  "H", [5.5, 4.9, 6.2]),
            (4, "CYS", "SG",  "S", [7.2, 2.5, 5.0]),
        ]
        m1 = classify_conformer_atoms(sample_atoms)
        m2_atoms = [
            (r, rn, an, el, list(p) if idx < 7 else [p[0] + 0.2, p[1] - 0.15, p[2] + 0.1])
            for idx, (r, rn, an, el, p) in enumerate(sample_atoms)
        ]
        m2 = classify_conformer_atoms(m2_atoms)
        Phi_O, Phi_H, steric_H = build_asu_features(grid, [m1, m2], n_types=10, include_hpol=True)

        raw_asu = np.abs(rng.normal(loc=0.30, scale=0.25, size=grid.n_asu))
        rho_full = grid.asu_to_full_3d(raw_asu)
        mask_asu = rng.random(grid.n_asu) > 0.15
        ds = DatasetBundle("synth", grid, rho_full, B_map=7.5, Phi_O=Phi_O, Phi_H=Phi_H,
                           steric_H=steric_H, mask_asu=mask_asu, member_weights=[0.38, 0.62])

        for loss_mode in ("l2", "kl"):
            for use_langmuir, use_oz, use_hpol, use_blur in [
                (True, True, True, True),
                (False, False, False, False),
                (True, False, True, True),
            ]:
                spec = WaterFieldSpec(
                    n_types=10, use_langmuir=use_langmuir, use_oz=use_oz,
                    use_hpol=use_hpol, use_blur=use_blur, fit_box_scale=True, n_datasets=1
                )
                theta = spec.init_params() + 0.25 * rng.normal(size=spec.n_params)
                if use_langmuir:
                    theta[spec.n_pmf] = -2.7
                B0 = 7.5

                def eval_fg(th, B_val=B0):
                    ds.B_map = B_val
                    ds.Kw, ds.Kh = grid.water_kernel(B_val)
                    l1, g1, gB, _, _ = forward_and_grad(
                        spec, ds, th, ds_index=0, loss_mode=loss_mode, compute_grad=True, return_grad_B=True
                    )
                    l2, g2 = smoothness_penalty_and_grad(spec, th)
                    return l1 + l2, g1 + g2, gB

                _, g_ana, gB_ana = eval_fg(theta, B0)
                eps = 1e-6
                g_fd = np.zeros_like(theta)
                for i in range(len(theta)):
                    tp = theta.copy(); tp[i] += eps
                    tm = theta.copy(); tm[i] -= eps
                    g_fd[i] = (eval_fg(tp, B0)[0] - eval_fg(tm, B0)[0]) / (2.0 * eps)
                gB_fd = (eval_fg(theta, B0 + eps)[0] - eval_fg(theta, B0 - eps)[0]) / (2.0 * eps)

                max_abs = float(np.max(np.abs(g_ana - g_fd)))
                rel = max_abs / max(1.0, float(np.max(np.abs(g_fd))))
                err_B = abs(gB_ana - gB_fd)
                self.assertLess(rel, 1e-6)
                self.assertLess(max_abs, 1e-5)
                self.assertLess(err_B, 1e-7)

        spec = WaterFieldSpec(n_types=10, use_langmuir=True, use_oz=True, use_hpol=True, use_blur=True, fit_box_scale=False)
        theta = spec.init_params() + 0.25 * rng.normal(size=spec.n_params)
        theta[spec.n_pmf] = -2.6
        ds.B_map = 6.5
        ds.Kw, ds.Kh = grid.water_kernel(6.5)
        g_hat, gO_ref = forward_and_grad(spec, ds, theta, compute_grad=False)
        g_hat_dir, gO_dir = forward_direct(spec, grid, [m1, m2], [0.38, 0.62], theta, B_map=6.5)
        self.assertLess(float(np.max(np.abs(g_hat - g_hat_dir))), 1e-5)
        self.assertLess(float(np.max(np.abs(gO_ref - gO_dir))), 1e-5)
        solv_3d = np.maximum(grid.asu_to_full_3d(g_hat) * RHO_BULK, 0.0)

        with tempfile.TemporaryDirectory() as td:
            dmin_mtz = os.path.join(td, "test_dmin.mtz")
            write_fpart_mtz_to_dmin(solv_3d, cell, sg, 2.5, dmin_mtz)
            m_check = gemmi.read_mtz_file(dmin_mtz)
            self.assertGreater(m_check.nreflections, 10)

            # Verify write_fpart_mtz_on_hkl handles out-of-Nyquist reflections cleanly without crashing (A2*)
            hi_mtz_path = os.path.join(td, "hi_ref.mtz")
            out_hi_mtz = os.path.join(td, "out_hi.mtz")
            hi_mtz = gemmi.Mtz(with_base=True)
            hi_mtz.cell = cell
            hi_mtz.spacegroup = sg
            hi_mtz.add_dataset("obs")
            hi_mtz.add_column("FP", "F")
            hi_mtz.set_data(np.array([[1, 2, 3, 10.0], [20, 0, 0, 5.0]], dtype=np.float32))
            hi_mtz.write_to_file(hi_mtz_path)
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                write_fpart_mtz_on_hkl(solv_3d, cell, hi_mtz_path, out_hi_mtz)
            self.assertIn("Nyquist", err_buf.getvalue())
            out_m = gemmi.read_mtz_file(out_hi_mtz)
            fpart_col = np.array(out_m.column_with_label("FPART").array)
            self.assertGreater(float(fpart_col[0]), 0.0)
            self.assertEqual(float(fpart_col[1]), 0.0)

        gg = gemmi.FloatGrid(*solv_3d.shape)
        gg.set_unit_cell(cell)
        gg.spacegroup = gemmi.SpaceGroup("P 1")
        np.array(gg, copy=False)[:] = solv_3d.astype(np.float32)
        sf_gemmi = gemmi.transform_map_to_f_phi(gg, half_l=False)

        hkl_list = [
            (h, k, l)
            for h in range(-4, 6)
            for k in range(0, 6)
            for l in range(-3, 5)
            if (h, k, l) != (0, 0, 0) and (h * h + k * k + l * l <= 30)
        ]
        hkl = np.array(hkl_list, dtype=np.int32)
        fp_gemmi = np.array([sf_gemmi.get_value(int(h), int(k), int(l)) for h, k, l in hkl])
        fp_numpy = solv_to_fpart(solv_3d.astype(np.float32), cell.volume, hkl)
        rel_sf = float(np.max(np.abs(fp_numpy - fp_gemmi)) / np.max(np.abs(fp_gemmi)))
        self.assertLess(rel_sf, 5e-7)

        n_hkl = len(hkl)
        work_mask = rng.random(n_hkl) > 0.10
        q_cart = hkl @ grid.F
        s_hkl = 0.5 * np.sqrt(np.sum(q_cart ** 2, axis=-1))
        n_bins = 4
        edges = np.quantile(s_hkl, np.linspace(0.0, 1.0, n_bins + 1))
        bin_idx = np.clip(np.digitize(s_hkl, edges[1:-1]), 0, n_bins - 1)

        k_tot = 0.9 + 0.2 * rng.random(n_hkl)
        f_calc = rng.normal(scale=12.0, size=n_hkl) + 1j * rng.normal(scale=12.0, size=n_hkl)
        true_fp = solv_to_fpart(solv_3d, cell.volume, hkl)
        f_obs = np.maximum(np.abs(k_tot * (f_calc + 0.85 * true_fp)) + rng.normal(scale=1.5, size=n_hkl), 0.1)
        norm_fo2 = float(np.sum(f_obs[work_mask] ** 2))

        theta_test = theta + 0.18 * rng.normal(size=spec.n_params)
        B_test = 8.2

        for loss_type in ("l2", "huber"):
            huber_delta = 2.0

            def fpart_loss_grad(fp):
                _, km = fit_kmask_per_bin(
                    f_calc, fp, k_tot, f_obs, work_mask, bin_idx, n_bins,
                    loss_type=loss_type, huber_delta=huber_delta, k_bounds=(0.0, 3.0)
                )
                fm = k_tot * (f_calc + km * fp)
                afm = np.maximum(np.abs(fm), 1e-12)
                r = afm - f_obs
                dL_dF = np.zeros_like(fp, dtype=np.complex128)
                rw = r[work_mask]
                if loss_type == "huber":
                    sw = np.sqrt(1.0 + (rw / huber_delta) ** 2)
                    loss_val = float(np.sum((huber_delta ** 2) * (sw - 1.0)) / norm_fo2)
                    psi = (rw / sw) / norm_fo2
                else:
                    loss_val = float(0.5 * np.sum(rw ** 2) / norm_fo2)
                    psi = rw / norm_fo2
                dL_dF[work_mask] = psi * k_tot[work_mask] * km[work_mask] * (fm[work_mask] / afm[work_mask])
                return loss_val, dL_dF

            _, g_th, g_B, fp0 = fpart_forward_and_grad(spec, ds, theta_test, B_test, hkl, fpart_loss_grad, clamp_nonneg=True)
            _, dL_dF0 = fpart_loss_grad(fp0)
            g_hat0, _ = forward_and_grad(spec, ds, theta_test, compute_grad=False)
            asu_grad_arr = fpart_grad_to_asu(grid, g_hat0, hkl, dL_dF0, clamp_nonneg=True)
            _, g_th_arr, g_B_arr, _, _ = forward_and_grad(
                spec, ds, theta_test, compute_grad=True, ext_dL_dghat=asu_grad_arr, return_grad_B=True
            )
            self.assertLess(float(np.max(np.abs(g_th - g_th_arr))), 1e-13)
            self.assertLess(abs(g_B - g_B_arr), 1e-13)

            eps = 1e-6
            g_th_fd = np.zeros_like(theta_test)
            for i in range(len(theta_test)):
                tp = theta_test.copy(); tp[i] += eps
                tm = theta_test.copy(); tm[i] -= eps
                lp = fpart_forward_and_grad(spec, ds, tp, B_test, hkl, fpart_loss_grad, clamp_nonneg=True)[0]
                lm = fpart_forward_and_grad(spec, ds, tm, B_test, hkl, fpart_loss_grad, clamp_nonneg=True)[0]
                g_th_fd[i] = (lp - lm) / (2.0 * eps)
            lpB = fpart_forward_and_grad(spec, ds, theta_test, B_test + eps, hkl, fpart_loss_grad, clamp_nonneg=True)[0]
            lmB = fpart_forward_and_grad(spec, ds, theta_test, B_test - eps, hkl, fpart_loss_grad, clamp_nonneg=True)[0]
            g_B_fd = (lpB - lmB) / (2.0 * eps)

            max_abs = float(np.max(np.abs(g_th - g_th_fd)))
            rel = max_abs / max(1e-8, float(np.max(np.abs(g_th_fd))))
            rel_B = abs(g_B - g_B_fd) / max(1e-8, abs(g_B_fd))
            self.assertLess(rel, 1e-5)
            self.assertLess(rel_B, 1e-5)

    def test_v2_lp_tetoz4_paulicap_zero_bulk_and_optional_epol(self):
        from surfnturf.cli import build_parser

        # Verify WaterFieldSpec() defaults match shipped v0.2 weights (219 params, +Epol off) & CLI oz_steps auto-resolves
        default_spec = WaterFieldSpec()
        self.assertEqual(default_spec.n_params, 219)
        self.assertTrue(default_spec.use_lp)
        self.assertTrue(default_spec.use_tet_oz)
        self.assertEqual(default_spec.n_oz_steps, 4)
        self.assertTrue(default_spec.use_pauli_cap)
        self.assertTrue(default_spec.zero_bulk_offset)
        self.assertFalse(default_spec.use_epol)
        cli_args = build_parser().parse_args(["train", "--dataset", "m.ccp4:p.pdb:5.0", "--out", "w.npz"])
        self.assertFalse(cli_args.epol)
        self.assertFalse(cli_args.no_lp)
        self.assertFalse(cli_args.no_tet_oz)
        self.assertIsNone(cli_args.oz_steps)

        rng = np.random.default_rng(7)
        cell = gemmi.UnitCell(16.0, 14.0, 12.0, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        grid = ASUGrid(cell, sg, (24, 20, 16), step=1)

        sample_atoms = [
            (1, "VAL", "N",   "N", [2.1, 3.2, 4.1]),
            (1, "VAL", "H1",  "H", [2.1, 3.2, 5.1]),
            (1, "VAL", "CA",  "C", [1.2, 4.1, 3.6]),
            (1, "VAL", "C",   "C", [1.8, 5.4, 3.2]),
            (1, "VAL", "O",   "O", [2.9, 5.6, 2.8]),
            (2, "ASP", "CB",  "C", [1.0, 7.0, 3.2]),
            (2, "ASP", "CG",  "C", [1.5, 8.0, 2.9]),
            (2, "ASP", "OD1", "O", [1.2, 9.1, 2.5]),
            (2, "ASP", "OD2", "O", [2.6, 7.9, 3.3]),
            (3, "SER", "CA",  "C", [5.0, 2.8, 5.2]),
            (3, "SER", "CB",  "C", [5.3, 3.9, 5.8]),
            (3, "SER", "OG",  "O", [5.8, 4.9, 5.0]),  # H-less SER OG exercises staggered gauche LP_sp3 fallback
            (4, "LYS", "NZ",  "N", [8.2, 6.2, 4.2]),
            (4, "LYS", "HZ1", "H", [7.6, 6.9, 4.0]),
        ]
        x1, t1, q1 = classify_conformer_atoms(sample_atoms, use_lp=True, return_charges=True)
        self.assertIn(10, set(t1.tolist()))  # LP_sp2 on VAL O
        self.assertIn(11, set(t1.tolist()))  # LP_neg on ASP OD1/OD2
        self.assertIn(12, set(t1.tolist()))  # LP_sp3 on H-less SER OG

        m2_atoms = [
            (r, rn, an, el, list(p) if idx < 6 else [p[0] + 0.18, p[1] - 0.12, p[2] + 0.09])
            for idx, (r, rn, an, el, p) in enumerate(sample_atoms)
        ]
        x2, t2, q2 = classify_conformer_atoms(m2_atoms, use_lp=True, return_charges=True)
        members = [(x1, t1), (x2, t2)]
        charges = [q1, q2]
        wts = [0.45, 0.55]

        # 1. Exact empty-bulk asymptote: g_hat == 1.000000000000 identically in an empty cell!
        spec_emp = WaterFieldSpec(
            n_types=13, use_lp=True, use_tet_oz=True, n_oz_steps=4,
            use_epol=True, use_pauli_cap=True, zero_bulk_offset=True
        )
        th_emp = spec_emp.init_params() + 0.35 * rng.normal(size=spec_emp.n_params)
        empty_mem = [(np.zeros((0, 3)), np.zeros(0, dtype=np.int32))]
        gh_emp, gO_emp = forward_direct(spec_emp, grid, empty_mem, [1.0], th_emp, B_map=5.0, charges=[np.zeros(0)])
        self.assertLess(float(np.max(np.abs(gh_emp - 1.0))), 1e-12)
        self.assertLess(float(np.max(np.abs(gO_emp - 1.0))), 1e-12)

        # 2. Central FD gradients & forward_direct parity for BOTH use_epol=False (default) and use_epol=True
        raw_asu = np.abs(rng.normal(loc=0.33, scale=0.20, size=grid.n_asu))
        rho_full = grid.asu_to_full_3d(raw_asu)

        for use_epol in (False, True):
            Phi_O, Phi_H, steric_H, extra = build_asu_features(
                grid, members, n_types=13, include_hpol=True, include_tet_oz=True,
                include_epol=use_epol, charges=(charges if use_epol else None),
                member_weights=wts, return_extra=True, max_workers=1
            )
            ds2 = DatasetBundle(
                "v2", grid, rho_full, B_map=6.0, Phi_O=Phi_O, Phi_H=Phi_H, steric_H=steric_H,
                member_weights=wts, F_hbsol=extra["F_hbsol"], E_feats=extra["E_feats"]
            )
            spec2 = WaterFieldSpec(
                n_types=13, use_lp=True, use_tet_oz=True, n_oz_steps=4, use_epol=use_epol,
                use_pauli_cap=True, zero_bulk_offset=True, fit_box_scale=True, n_datasets=1
            )
            th2 = spec2.init_params() + 0.22 * rng.normal(size=spec2.n_params)
            th2[spec2.n_pmf] = -2.2  # exercises strong saturation + Rosenfeld 0D cavity cap actively

            for loss_mode in ("l2", "kl"):
                def eval_v2(th):
                    l1, g1, _, _ = forward_and_grad(spec2, ds2, th, ds_index=0, loss_mode=loss_mode, compute_grad=True)
                    l2, g2 = smoothness_penalty_and_grad(spec2, th)
                    return l1 + l2, g1 + g2

                _, g_ana = eval_v2(th2)
                eps = 1e-6
                g_fd = np.zeros_like(th2)
                for i in range(len(th2)):
                    tp = th2.copy(); tp[i] += eps
                    tm = th2.copy(); tm[i] -= eps
                    g_fd[i] = (eval_v2(tp)[0] - eval_v2(tm)[0]) / (2.0 * eps)
                rel = float(np.max(np.abs(g_ana - g_fd))) / max(1.0, float(np.max(np.abs(g_fd))))
                self.assertLess(rel, 1e-6)

            spec2_inf = WaterFieldSpec(
                n_types=13, use_lp=True, use_tet_oz=True, n_oz_steps=4, use_epol=use_epol,
                use_pauli_cap=True, zero_bulk_offset=True, fit_box_scale=False
            )
            th2_inf = th2[:spec2_inf.n_params].copy()
            gh_b, gO_b = forward_and_grad(spec2_inf, ds2, th2_inf, compute_grad=False)
            gh_d, gO_d = forward_direct(
                spec2_inf, grid, members, wts, th2_inf, B_map=6.0, charges=(charges if use_epol else None)
            )
            self.assertLess(float(np.max(np.abs(gh_b - gh_d))), 2e-5)
            self.assertLess(float(np.max(np.abs(gO_b - gO_d))), 2e-5)

        # Verify build_asu_features(grid, members) auto-infers n_types=13 when LP virtual sites are present (V13)
        Phi_auto, _, _ = build_asu_features(grid, members, max_workers=1)
        self.assertEqual(Phi_auto.shape[2], default_spec.n_pmf)

    def test_ether_sp3_lone_pairs_charge_balance_and_trainer_roundtrip(self):
        from surfnturf.hydrogens import complete_conformer_hydrogens
        from surfnturf.io import save_spec_and_theta, write_ccp4
        from surfnturf.train import fit_surfnturf_model

        # 1. V15: Dimethyl ether (C1-O-C2) gets O_hyd (2) and two tetrahedral out-of-plane LP_sp3 (12) sites
        ether_atoms = [
            (1, "LIG", "C1", "C", [-1.160, 0.800, 0.000]),
            (1, "LIG", "O1", "O", [0.000,  0.000, 0.000]),
            (1, "LIG", "C2", "C", [1.160,  0.800, 0.000]),
        ]
        x_eth, t_eth = classify_conformer_atoms(ether_atoms, use_lp=True)
        self.assertEqual(list(t_eth[:3]), [7, 2, 7])
        self.assertEqual(list(t_eth[3:]), [12, 12])
        for lp_pos in x_eth[3:]:
            self.assertGreater(abs(float(lp_pos[2])), 0.45)  # strictly out of the C1-O-C2 z=0 plane
            self.assertGreater(float(np.linalg.norm(lp_pos - x_eth[0])), 1.65)
            self.assertGreater(float(np.linalg.norm(lp_pos - x_eth[2])), 1.65)

        # 2. V14: Formal integer charge balance on zwitterionic peptide (NH3+-VAL-GLY-ASP-LYS-SER-COO-, net 0.00 e)
        zwit_hv = [
            (1, "VAL", "N", "N", [5.00, 5.00, 5.00]), (1, "VAL", "CA", "C", [6.46, 5.00, 5.00]),
            (1, "VAL", "C", "C", [7.01, 6.42, 5.00]), (1, "VAL", "O", "O", [6.28, 7.41, 5.00]),
            (1, "VAL", "CB", "C", [6.98, 4.24, 6.23]), (1, "VAL", "CG1", "C", [8.50, 4.24, 6.28]),
            (1, "VAL", "CG2", "C", [6.42, 2.82, 6.24]),
            (2, "GLY", "N", "N", [8.33, 6.51, 5.00]), (2, "GLY", "CA", "C", [8.98, 7.81, 5.00]),
            (2, "GLY", "C", "C", [10.49, 7.69, 5.00]), (2, "GLY", "O", "O", [11.04, 6.59, 5.00]),
            (3, "ASP", "N", "N", [11.15, 8.84, 5.00]), (3, "ASP", "CA", "C", [12.60, 8.91, 5.00]),
            (3, "ASP", "C", "C", [13.15, 10.33, 5.00]), (3, "ASP", "O", "O", [12.42, 11.32, 5.00]),
            (3, "ASP", "CB", "C", [13.14, 8.16, 6.22]), (3, "ASP", "CG", "C", [14.65, 8.18, 6.25]),
            (3, "ASP", "OD1", "O", [15.25, 9.26, 6.25]), (3, "ASP", "OD2", "O", [15.22, 7.08, 6.25]),
            (4, "LYS", "N", "N", [14.47, 10.42, 5.00]), (4, "LYS", "CA", "C", [15.12, 11.72, 5.00]),
            (4, "LYS", "C", "C", [16.63, 11.60, 5.00]), (4, "LYS", "O", "O", [17.18, 10.50, 5.00]),
            (4, "LYS", "OXT", "O", [17.25, 12.68, 5.00]),
            (4, "LYS", "CB", "C", [14.68, 12.51, 3.77]), (4, "LYS", "CG", "C", [15.20, 13.94, 3.75]),
            (4, "LYS", "CD", "C", [14.76, 14.72, 2.52]), (4, "LYS", "CE", "C", [15.28, 16.15, 2.50]),
            (4, "LYS", "NZ", "N", [14.85, 16.92, 1.31]),
        ]
        zwit_comp = complete_conformer_hydrogens(zwit_hv)
        _, _, q_zwit = classify_conformer_atoms(zwit_comp, use_lp=False, return_charges=True)
        # NH3+ (+1) + ASP (-1) + LYS (+1) + COO- (-1) == 0.00 e!
        self.assertLess(abs(float(np.sum(q_zwit))), 1e-3)

        # 3. save_spec_and_theta + fit_surfnturf_model (both v0.2 219-param default and v0.1 173-param --no-tet-oz)
        with tempfile.TemporaryDirectory() as td:
            cell = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
            sg = gemmi.SpaceGroup("P 21 21 21")
            pdb_p = os.path.join(td, "mini.pdb")
            map_p = os.path.join(td, "mini.ccp4")
            with open(pdb_p, "w", encoding="utf-8", newline="\n") as f:
                f.write("CRYST1   16.000   16.000   16.000  90.00  90.00  90.00 P 21 21 21    4\n")
                f.write("ATOM      1  N   ALA A   1       4.000   4.000   4.000  1.00 10.00           N\n")
                f.write("ATOM      2  CA  ALA A   1       5.460   4.000   4.000  1.00 10.00           C\n")
                f.write("ATOM      3  C   ALA A   1       6.010   5.420   4.000  1.00 10.00           C\n")
                f.write("ATOM      4  O   ALA A   1       5.280   6.410   4.000  1.00 11.00           O\n")
                f.write("ATOM      5  CB  ALA A   1       5.980   3.240   5.230  1.00 12.00           C\nEND\n")
            grid_t = ASUGrid(cell, sg, (16, 16, 16), step=1)
            rho_syn = grid_t.asu_to_full_3d(np.full(grid_t.n_asu, 0.30, dtype=np.float64))
            write_ccp4(map_p, rho_syn, cell, sg)

            npz_v2 = os.path.join(td, "fit_v2.npz")
            r_v2 = fit_surfnturf_model([("mini", map_p, pdb_p, 5.0)], step=2, maxiter=2, out_npz=npz_v2)
            self.assertEqual(r_v2["spec"].n_params, 219)
            self.assertEqual(r_v2["spec"].n_oz_steps, 4)
            sp_ld, th_ld = load_saved_spec_and_theta(npz_v2)
            self.assertEqual(sp_ld.n_params, 219)
            self.assertTrue(np.allclose(th_ld, r_v2["theta"]))

            npz_v1 = save_spec_and_theta(os.path.join(td, "fit_v1"), WaterFieldSpec(n_types=10), np.zeros(173))
            r_v1 = fit_surfnturf_model(
                [("mini", map_p, pdb_p, 5.0)], use_lp=False, use_tet_oz=False,
                use_pauli_cap=False, zero_bulk_offset=False, step=2, maxiter=2, out_npz=npz_v1
            )
            self.assertEqual(r_v1["spec"].n_params, 173)
            self.assertEqual(r_v1["spec"].n_oz_steps, 1)


if __name__ == "__main__":
    unittest.main()
