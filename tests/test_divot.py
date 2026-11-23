"""Mathematical & finite-difference verification of surfnturf.divot (W_mc, PeakDict, WatershedDivotDict)."""
import unittest
import numpy as np
import gemmi

from surfnturf.divot import (
    CrystalGridOps,
    PeriodicFourierFields,
    ContinuousPeakDict,
    WatershedDivotDict,
    SolventAmplitudeProblem,
    pick_asu_difference_peaks,
)


class TestSurfNTurfDivot(unittest.TestCase):
    def test_divot_peak_and_transport_adjoints(self):
        rng = np.random.default_rng(42)
        cell = gemmi.UnitCell(45.9, 40.7, 30.1, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        shape = (36, 32, 24)
        cgrid = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=2)

        # 1. sf / sf_adjoint inner-product identity
        H = np.array([[h, k, l] for h in range(1, 7) for k in range(0, 6) for l in range(0, 5)], dtype=int)
        idx = cgrid.hkl_idx(H)
        drho = rng.standard_normal(shape)
        G = rng.standard_normal(len(H)) + 1j * rng.standard_normal(len(H))
        lhs_sf = float(np.real(np.vdot(G, cgrid.sf(drho, idx))))
        rhs_sf = float(np.sum(drho * cgrid.sf_adjoint(G, idx)))
        self.assertLess(abs(lhs_sf - rhs_sf) / max(abs(lhs_sf), 1e-12), 1e-13)

        # 2. Spectral grad / div skew-adjointness
        f_sc = rng.standard_normal(shape)
        v_vec = rng.standard_normal((3,) + shape)
        lhs_sp = float(np.sum(cgrid.spectral_grad_scalar(f_sc) * v_vec))
        rhs_sp = -float(np.sum(f_sc * cgrid.spectral_div_vector(v_vec)))
        self.assertLess(abs(lhs_sp - rhs_sp) / max(abs(lhs_sp), 1e-12), 1e-13)

        # 3. Mass-Conserving Wall-Clamped Transport (W_mc): F000 conservation & FD gradient
        base_raw = cgrid.sym_s(cgrid.gaussian_blur(rng.uniform(0.05, 1.2, size=shape), B_blur=12.0))
        wall_mask = cgrid.sym_s(cgrid.gaussian_blur(rng.uniform(0.0, 1.0, size=shape), B_blur=15.0))
        pf_mc = PeriodicFourierFields(
            cgrid, base_raw, fitreso_s=12.0, fitreso_w=12.0,
            use_scale=True, use_warp=True, mass_conserving=True, wall_mask_3d=wall_mask
        )
        pf_only_w_mc = PeriodicFourierFields(
            cgrid, base_raw, fitreso_s=12.0, fitreso_w=12.0,
            use_scale=False, use_warp=True, mass_conserving=True, wall_mask_3d=wall_mask
        )
        cw_test = rng.normal(0.0, 0.04, size=pf_only_w_mc.nx)
        rho_wmc, _, _, _ = pf_only_w_mc.forward_and_pullback(cw_test, compute_grad=False)
        f000_0 = float(np.sum(base_raw) * cgrid.vox_vol)
        f000_mc = float(np.sum(rho_wmc) * cgrid.vox_vol)
        self.assertLess(abs(f000_mc - f000_0) / f000_0, 2e-3)

        x_mc = rng.normal(0.0, 0.03, size=pf_mc.nx)
        w_probe = rng.standard_normal(shape)
        _, _, _, pb_mc = pf_mc.forward_and_pullback(x_mc, compute_grad=True)
        g_mc = pb_mc(w_probe)
        d_mc = rng.standard_normal(pf_mc.nx)
        d_mc /= np.linalg.norm(d_mc)
        eps = 1e-6
        rp, pp, _, _ = pf_mc.forward_and_pullback(x_mc + eps * d_mc, compute_grad=False)
        rm, pm, _, _ = pf_mc.forward_and_pullback(x_mc - eps * d_mc, compute_grad=False)
        fd_mc = ((float(np.sum(w_probe * rp)) + pp) - (float(np.sum(w_probe * rm)) + pm)) / (2.0 * eps)
        an_mc = float(np.dot(g_mc, d_mc))
        self.assertLess(abs(an_mc - fd_mc) / max(abs(an_mc), abs(fd_mc), 1e-12), 1e-6)

        # 4. ContinuousPeakDict: sub-voxel F000 invariance & FD gradient
        seeds = [
            {"xyz0": np.array([8.3, 11.2, 6.4]), "z_score": 6.5, "val_eA3": 2.5, "B_init": 10.0},
            {"xyz0": np.array([19.1, 4.8, 13.7]), "z_score": -5.1, "val_eA3": -1.8, "B_init": 18.0},
            {"xyz0": np.array([3.2, 22.5, 9.9]), "z_score": 4.2, "val_eA3": 1.2, "B_init": 28.0},
        ]
        pdict = ContinuousPeakDict(cgrid, seeds, half_win=8)
        th_a = np.array([1.0, 0.0, 0.0, 0.0, np.log(8.0)] * 3, dtype=np.float64)
        th_b = np.array([1.0, 0.37, -0.29, 0.51, np.log(22.0)] * 3, dtype=np.float64)
        ra, _ = pdict.forward_and_pullback(th_a, compute_grad=False)
        rb, _ = pdict.forward_and_pullback(th_b, compute_grad=False)
        f000_a = float(np.sum(ra) * cgrid.vox_vol)
        f000_b = float(np.sum(rb) * cgrid.vox_vol)
        self.assertLess(abs(f000_a - f000_b) / abs(f000_a), 1e-14)

        th_p = rng.uniform(-0.4, 0.6, size=pdict.n_params)
        th_p[4::5] = rng.uniform(np.log(5.0), np.log(35.0), size=3)
        _, pb_pk = pdict.forward_and_pullback(th_p, compute_grad=True)
        g_pk = pb_pk(w_probe)
        d_pk = rng.standard_normal(pdict.n_params)
        d_pk /= np.linalg.norm(d_pk)
        rpp, _ = pdict.forward_and_pullback(th_p + eps * d_pk, compute_grad=False)
        rpm, _ = pdict.forward_and_pullback(th_p - eps * d_pk, compute_grad=False)
        fd_pk = float(np.sum(w_probe * (rpp - rpm))) / (2.0 * eps)
        an_pk = float(np.dot(g_pk, d_pk))
        self.assertLess(abs(an_pk - fd_pk) / max(abs(an_pk), abs(fd_pk), 1e-12), 1e-7)

        # 5. Difference-map peak sign & WatershedDivotDict
        true_bump_dict = ContinuousPeakDict(cgrid, [seeds[0]], half_win=8)
        bump_3d, _ = true_bump_dict.forward_and_pullback(np.array([0.85, 0.15, -0.10, 0.20, np.log(8.0)]), compute_grad=False)
        rho_true = np.maximum(base_raw + bump_3d, 0.0)
        Q_prot = cgrid.sf(cgrid.gaussian_blur(rng.uniform(0.0, 2.0, size=shape), 6.0), idx)
        F_tot_true = Q_prot + 0.8 * cgrid.sf(rho_true, idx)
        free_m = np.zeros(len(H), dtype=bool)
        free_m[::20] = True
        prob = SolventAmplitudeProblem(cgrid, H, np.abs(F_tot_true), free_m, np.ones(len(H)), Q_prot)
        dmap8 = prob.compute_work_diff_map(base_raw, active_mask=prob.train, B_smooth=4.0)
        picked = pick_asu_difference_peaks(cgrid, [(4.0, dmap8)], np.ones(shape), z_thresh=3.0, max_peaks=5)
        self.assertGreaterEqual(len(picked), 1)
        self.assertGreater(picked[0]["z_score"], 3.0)

        # Verify bounded k_solv >= 0 & scaled_solvent_rho consistency (A4*)
        rho_scaled, ev_w = prob.scaled_solvent_rho(base_raw, active_mask=prob.work)
        Fm_direct = prob.K * ev_w["p"][0] * (prob.Q + cgrid.sf(rho_scaled, prob.idx))
        self.assertLess(float(np.max(np.abs(Fm_direct - ev_w["Fm"]))), 1e-11)

        prob_anti = SolventAmplitudeProblem(cgrid, H, np.abs(Q_prot), free_m, np.ones(len(H)), Q_prot + 0.5 * cgrid.sf(rho_true, idx))
        ev_anti = prob_anti.eval_rho(rho_true, active_mask=prob_anti.work, compute_grad=False)
        self.assertGreaterEqual(float(ev_anti["p"][1]), 0.0)

        wdiv = WatershedDivotDict(cgrid, dmap8, np.ones(shape), z_peak=3.0, z_floor=1.5, max_divots=10)
        self.assertGreaterEqual(wdiv.K, 1)
        w_vec = rng.normal(0.0, 0.5, size=wdiv.K)
        _, pb_wd = wdiv.forward_and_pullback(w_vec, compute_grad=True)
        g_wd = pb_wd(w_probe)
        d_wd = rng.standard_normal(wdiv.K)
        d_wd /= np.linalg.norm(d_wd)
        rdp, _ = wdiv.forward_and_pullback(w_vec + eps * d_wd, compute_grad=False)
        rdm, _ = wdiv.forward_and_pullback(w_vec - eps * d_wd, compute_grad=False)
        fd_wd = float(np.sum(w_probe * (rdp - rdm))) / (2.0 * eps)
        an_wd = float(np.dot(g_wd, d_wd))
        self.assertLess(abs(an_wd - fd_wd) / max(abs(an_wd), abs(fd_wd), 1e-12), 1e-8)

    def test_diff_map_exact_eA3_and_hexagonal_warp_symmetry_and_grid_check(self):
        rng = np.random.default_rng(7)
        # B5: P 41 must reject unequal (32, 40, 48) grid
        with self.assertRaises(ValueError):
            CrystalGridOps(gemmi.UnitCell(40.0, 40.0, 50.0, 90.0, 90.0, 90.0), gemmi.SpaceGroup("P 41"), (32, 40, 48))

        # A3: Exact e/A^3 difference map across P 21 21 21, C 1 2 1, P 61, and rhombohedral R 3 (matches P 1 to < 1e-13!)
        for sg_hm, cell_par, shape in [
            ("P 21 21 21", (28.0, 26.0, 24.0, 90.0, 90.0, 90.0), (24, 24, 24)),
            ("C 1 2 1",    (30.0, 24.0, 26.0, 90.0, 102.0, 90.0), (24, 24, 24)),
            ("P 61",       (28.0, 28.0, 30.0, 90.0, 90.0, 120.0), (24, 24, 24)),
            ("R 3",        (24.0, 24.0, 24.0, 80.0, 80.0, 80.0),  (24, 24, 24)),
        ]:
            cell = gemmi.UnitCell(*cell_par)
            sg = gemmi.find_spacegroup_by_name(sg_hm, alpha=cell.alpha, gamma=cell.gamma)
            cg = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)
            cg_p1 = CrystalGridOps(cell, gemmi.SpaceGroup("P 1"), shape, conv="plus", fft_workers=1)

            raw_sym = cg.sym_s(cg.gaussian_blur(rng.standard_normal(shape), B_blur=90.0))
            F_bl = np.fft.rfftn(raw_sym)
            F_bl[cg.stol2_r > 1.0 / (4.0 * 2.65 * 2.65)] = 0.0
            F_bl[0, 0, 0] = 0.0
            delta_true = cg.sym_s(np.fft.irfftn(F_bl, s=shape, axes=(0, 1, 2)))

            def _eval_exact_diffmap(cgrid_cur, sg_cur):
                gg = gemmi.FloatGrid(*shape)
                gg.set_unit_cell(cell)
                gg.spacegroup = sg_cur
                np.array(gg, copy=False)[:] = delta_true.astype(np.float64)
                asu_sf = gemmi.transform_map_to_f_phi(gg, half_l=True).prepare_asu_data(dmin=2.6, with_000=False)
                H_cur = np.array(asu_sf.miller_array, dtype=int)
                dF_cur = cgrid_cur.sf(delta_true, cgrid_cur.hkl_idx(H_cur))
                nz = np.abs(dF_cur) > 1e-12
                u_phase = np.where(nz, dF_cur / np.maximum(np.abs(dF_cur), 1e-12), 1.0 + 0.0j)
                Q_ref = 1e5 * u_phase
                Fo_ref = np.abs(Q_ref + dF_cur)
                pr = SolventAmplitudeProblem(cgrid_cur, H_cur, Fo_ref, np.zeros(len(H_cur), dtype=bool), np.ones(len(H_cur)), Q_ref)
                pr.P_LO = np.array([1.0, 0.0])
                pr.P_HI = np.array([1.0, 0.0])
                return pr.compute_work_diff_map(np.zeros(shape), active_mask=pr.work, B_smooth=0.0)

            dmap_sg = _eval_exact_diffmap(cg, sg)
            dmap_p1 = _eval_exact_diffmap(cg_p1, gemmi.SpaceGroup("P 1"))
            slope = float(np.dot(dmap_sg.ravel(), dmap_p1.ravel()) / np.dot(dmap_p1.ravel(), dmap_p1.ravel()))
            self.assertAlmostEqual(slope, 1.0, places=10, msg=f"slope mismatch in {sg_hm}: {slope}")
            self.assertLess(float(np.max(np.abs(dmap_sg - dmap_p1))), 1e-9, msg=f"map error in {sg_hm}")
            # Also verify agreement with delta_true in e/A^3 directly
            slope_true = float(np.dot(dmap_sg.ravel(), delta_true.ravel()) / np.dot(delta_true.ravel(), delta_true.ravel()))
            self.assertAlmostEqual(slope_true, 1.0, places=10, msg=f"e/A^3 scale mismatch in {sg_hm}: {slope_true}")
            self.assertLess(float(np.max(np.abs(dmap_sg - delta_true))), 1e-9, msg=f"e/A^3 map error in {sg_hm}")

        # B6: Hexagonal P 61 warp output is strictly P 61-symmetric and matches FD
        cell_hex = gemmi.UnitCell(28.0, 28.0, 30.0, 90.0, 90.0, 120.0)
        cg_hex = CrystalGridOps(cell_hex, gemmi.SpaceGroup("P 61"), (24, 24, 24), conv="plus", fft_workers=1)
        base_hex = cg_hex.sym_s(cg_hex.gaussian_blur(rng.uniform(0.1, 1.0, size=(24, 24, 24)), B_blur=18.0))
        pf_hex = PeriodicFourierFields(cg_hex, base_hex, fitreso_s=12.0, fitreso_w=12.0, use_scale=True, use_warp=True, mass_conserving=True)
        xh = rng.normal(0.0, 0.03, size=pf_hex.nx)
        rho_h, _, _, pb_h = pf_hex.forward_and_pullback(xh, compute_grad=True)
        self.assertLess(float(np.max(np.abs(rho_h - cg_hex.sym_s(rho_h)))), 1e-14)
        wp = rng.standard_normal((24, 24, 24))
        gh = pb_h(wp)
        dh = rng.standard_normal(pf_hex.nx)
        dh /= np.linalg.norm(dh)
        eps = 1e-6
        rp, pp, _, _ = pf_hex.forward_and_pullback(xh + eps * dh, compute_grad=False)
        rm, pm, _, _ = pf_hex.forward_and_pullback(xh - eps * dh, compute_grad=False)
        fd_h = ((float(np.sum(wp * rp)) + pp) - (float(np.sum(wp * rm)) + pm)) / (2.0 * eps)
        an_h = float(np.dot(gh, dh))
        self.assertLess(abs(an_h - fd_h) / max(abs(an_h), abs(fd_h), 1e-12), 1e-6)

    def test_monoclinic_p21_non_orthogonal_adjoints(self):
        rng = np.random.default_rng(123)
        cell = gemmi.UnitCell(31.6, 54.56, 47.98, 90.0, 99.25, 90.0)
        sg = gemmi.SpaceGroup("P 1 21 1")
        shape = (24, 32, 28)
        cgrid = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=2)

        f_sc = rng.standard_normal(shape)
        v_vec = rng.standard_normal((3,) + shape)
        lhs_sp = float(np.sum(cgrid.spectral_grad_scalar(f_sc) * v_vec))
        rhs_sp = -float(np.sum(f_sc * cgrid.spectral_div_vector(v_vec)))
        self.assertLess(abs(lhs_sp - rhs_sp) / max(abs(lhs_sp), 1e-12), 1e-13)

        base_raw = cgrid.sym_s(cgrid.gaussian_blur(rng.uniform(0.05, 1.2, size=shape), B_blur=14.0))
        wall_mask = cgrid.sym_s(cgrid.gaussian_blur(rng.uniform(0.1, 1.0, size=shape), B_blur=16.0))
        pf_mc = PeriodicFourierFields(
            cgrid, base_raw, fitreso_s=14.0, fitreso_w=14.0,
            use_scale=True, use_warp=True, mass_conserving=True, wall_mask_3d=wall_mask
        )
        x_mc = rng.normal(0.0, 0.03, size=pf_mc.nx)
        w_probe = rng.standard_normal(shape)
        _, _, _, pb_mc = pf_mc.forward_and_pullback(x_mc, compute_grad=True)
        g_mc = pb_mc(w_probe)
        d_mc = rng.standard_normal(pf_mc.nx)
        d_mc /= np.linalg.norm(d_mc)
        eps = 1e-6
        rp, pp, _, _ = pf_mc.forward_and_pullback(x_mc + eps * d_mc, compute_grad=False)
        rm, pm, _, _ = pf_mc.forward_and_pullback(x_mc - eps * d_mc, compute_grad=False)
        fd_mc = ((float(np.sum(w_probe * rp)) + pp) - (float(np.sum(w_probe * rm)) + pm)) / (2.0 * eps)
        an_mc = float(np.dot(g_mc, d_mc))
        self.assertLess(abs(an_mc - fd_mc) / max(abs(an_mc), abs(fd_mc), 1e-12), 1e-6)

        seeds = [
            {"xyz0": np.array([6.2, 14.1, 9.5]), "z_score": 5.5, "val_eA3": 2.1, "B_init": 12.0},
            {"xyz0": np.array([-2.1, 27.0, 18.2]), "z_score": -4.4, "val_eA3": -1.5, "B_init": 20.0},
        ]
        pdict = ContinuousPeakDict(cgrid, seeds, half_win=8)
        th_a = np.array([1.0, 0.0, 0.0, 0.0, np.log(9.0)] * 2, dtype=np.float64)
        th_b = np.array([1.0, 0.42, -0.31, 0.28, np.log(24.0)] * 2, dtype=np.float64)
        ra, _ = pdict.forward_and_pullback(th_a, compute_grad=False)
        rb, _ = pdict.forward_and_pullback(th_b, compute_grad=False)
        self.assertLess(abs(float(np.sum(ra) - np.sum(rb))) / abs(float(np.sum(ra))), 1e-14)

        th_p = rng.uniform(-0.4, 0.6, size=pdict.n_params)
        th_p[4::5] = rng.uniform(np.log(6.0), np.log(30.0), size=2)
        _, pb_pk = pdict.forward_and_pullback(th_p, compute_grad=True)
        g_pk = pb_pk(w_probe)
        d_pk = rng.standard_normal(pdict.n_params)
        d_pk /= np.linalg.norm(d_pk)
        rpp, _ = pdict.forward_and_pullback(th_p + eps * d_pk, compute_grad=False)
        rpm, _ = pdict.forward_and_pullback(th_p - eps * d_pk, compute_grad=False)
        fd_pk = float(np.sum(w_probe * (rpp - rpm))) / (2.0 * eps)
        an_pk = float(np.dot(g_pk, d_pk))
        self.assertLess(abs(an_pk - fd_pk) / max(abs(an_pk), abs(fd_pk), 1e-12), 1e-7)


if __name__ == "__main__":
    unittest.main()
