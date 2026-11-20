"""End-to-end analytical & finite-difference verification of surfnturf.bmob (3D water mobility B(x))."""
import os
import tempfile
import unittest
import numpy as np
import gemmi

from surfnturf.field import (
    ASUGrid, WaterFieldSpec, DatasetBundle, RHO_BULK, build_asu_features, forward_and_grad, load_ensemble
)
from surfnturf.bmob import (
    SlimDatasetBundle,
    build_mobility_features,
    fast_build_asu_and_mobility,
    load_ensemble_with_bfac,
    reduce_mobility_features_to_asu,
    PhysicalBMobSpec,
    SpatialSmoother3D,
    MultiBRenderer,
    compute_pmf_density_with_pullback,
    build_water_keep_mask_asu,
    ReciprocalTarget,
    multiB_fpart_loss_and_grad,
    fit_uniform_B_on_work,
    fit_physical_bmob_on_work,
    fit_spatial_bmob_on_work,
    fit_joint_pmf_and_bmob_on_work,
)


ABC_PDB = """\
CRYST1   25.000   25.000   25.000  90.00  90.00  90.00 P 21 21 21    4
ATOM      1  CA AALA A   1       5.000   5.000   5.000  0.50  8.00           C
ATOM      2  CA BALA A   1       6.000   5.000   5.000  0.30 12.00           C
ATOM      3  CA CALA A   1       7.000   5.000   5.000  0.20 18.00           C
ATOM      4  CA AGLY A   2       5.000   8.800   5.000  0.50  9.00           C
ATOM      5  CA BGLY A   2       6.000   8.800   5.000  0.30 13.00           C
ATOM      6  CA CGLY A   2       7.000   8.800   5.000  0.20 19.00           C
END
"""


class TestSurfNTurfBMob(unittest.TestCase):
    def test_ensemble_loader_parity(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "abc.pdb")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(ABC_PDB)
            for mode in ("auto", "letters", "reconnect", "quantiles"):
                m_f, w_f = load_ensemble(p, nconf=8, mode=mode, return_weights=True)
                m_b, _, w_b = load_ensemble_with_bfac(p, nconf=8, mode=mode)
                self.assertEqual(len(m_f), len(m_b))
                self.assertLess(float(np.max(np.abs(w_f - w_b))), 1e-12)
            m_a, _, w_a = load_ensemble_with_bfac(p, nconf=8, alt="A")
            self.assertEqual(len(m_a), 1)
            self.assertEqual(float(w_a[0]), 1.0)

    def test_bmob_adjoints_and_optimizers(self):
        rng = np.random.default_rng(42)
        cell = gemmi.UnitCell(45.9, 40.7, 30.1, 90.0, 90.0, 90.0)
        sg = gemmi.SpaceGroup("P 21 21 21")
        grid = ASUGrid(cell, sg, (18, 16, 12), step=1)

        members = []
        members_biso = []
        for _ in range(2):
            xyz = rng.uniform([5.0, 5.0, 5.0], [22.0, 20.0, 15.0], size=(40, 3))
            typ = np.arange(40, dtype=np.int32) % 10
            biso = rng.uniform(3.5, 25.0, size=40)
            members.append((xyz, typ))
            members_biso.append(biso)
        mw = np.array([0.6, 0.4], dtype=np.float64)

        PhiO_ref, PhiH_ref, stH_ref = build_asu_features(grid, members, n_types=10, include_hpol=True)
        mob_ref = build_mobility_features(grid, members, members_biso, two_type_heavy_only=False)
        PhiO_fast, PhiH_fast, stH_fast, mob_fast = fast_build_asu_and_mobility(
            grid, members, members_biso, n_types=10, include_hpol=True, max_workers=2
        )
        self.assertLess(float(np.max(np.abs(PhiO_fast - PhiO_ref))), 1e-6)
        self.assertLess(float(np.max(np.abs(PhiH_fast - PhiH_ref))), 1e-6)
        self.assertLess(float(np.max(np.abs(stH_fast - stH_ref))), 1e-6)
        self.assertLess(float(np.max(np.abs(mob_fast.d_min - mob_ref.d_min))), 1e-9)
        self.assertLess(float(np.max(np.abs(mob_fast.B_prot - mob_ref.B_prot))), 1e-9)
        self.assertLess(float(np.max(np.abs(mob_fast.N_tether - mob_ref.N_tether))), 1e-9)

        smoother = SpatialSmoother3D(grid, fwhm_A=3.2)
        x_rnd = rng.standard_normal(grid.n_asu)
        y_rnd = rng.standard_normal(grid.n_asu)
        lhs = float(np.dot(smoother.forward(x_rnd), y_rnd))
        rhs = float(np.dot(x_rnd, smoother.adjoint(y_rnd)))
        self.assertLess(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12), 1e-11)

        spec = WaterFieldSpec(n_types=10, use_langmuir=True, use_oz=True, use_hpol=True, n_datasets=1, fit_box_scale=True)
        theta = 0.15 * rng.standard_normal(spec.n_params)
        dummy_rho = rng.uniform(0.1, 0.5, size=grid.shape).astype(np.float32)
        ds_full = DatasetBundle("full", grid, dummy_rho, B_map=5.0,
                                Phi_O=PhiO_fast, Phi_H=PhiH_fast, steric_H=stH_fast, member_weights=mw)
        ds_slim = SlimDatasetBundle("slim", grid, B_map=5.0,
                                    Phi_O=PhiO_fast, Phi_H=PhiH_fast, steric_H=stH_fast, member_weights=mw)

        _, grad_th_ref, g_hat_ref, _ = forward_and_grad(spec, ds_full, theta, ds_index=0, loss_mode="l2", compute_grad=True)
        renderer = MultiBRenderer(grid)
        gO_m, gH_m, sk, pb_th = compute_pmf_density_with_pullback(spec, ds_slim, theta, ds_index=0, need_grad=True, use_f32_blas=False)
        z5 = np.full((ds_slim.M, grid.n_asu), np.log(5.0), dtype=np.float64)
        g_hat_slim, _, pb_rend = renderer.render_and_pullback(gO_m, gH_m, mw, z5, scale_k=sk, compute_grad=True)
        dL_dghat = (g_hat_slim - ds_full.g_asu) * (ds_full.weights_asu / ds_full.w_sum)
        dO, dH, _, dlogk = pb_rend(dL_dghat)
        grad_th_slim = pb_th(dO, dH, dlogk)

        self.assertLess(float(np.max(np.abs(g_hat_slim - g_hat_ref))), 1e-11)
        self.assertLess(float(np.max(np.abs(grad_th_slim - grad_th_ref))), 1e-11)

        phys_spec = PhysicalBMobSpec()
        psi_test = phys_spec.pack(alpha_prot=0.85, B0=2.5, dB_unteth=22.0, gamma_teth=0.95, B_far=450.0, d_far=4.4)
        z_phys0, _, pb_phys = phys_spec.forward_with_pullback(mob_fast, psi_test)
        eta_t = 0.25 * rng.standard_normal(grid.n_asu)
        dz0 = smoother.forward(eta_t)
        z_tot0 = z_phys0 + dz0[None, :]
        g_hat_var, _, _ = renderer.render_and_pullback(gO_m, gH_m, mw, z_tot0, scale_k=sk, compute_grad=False)

        vox = grid.cell.volume / float(grid.nvox)
        f000_b5 = float(np.sum(g_hat_slim * grid.orbit_mult) * RHO_BULK * vox)
        f000_var = float(np.sum(g_hat_var * grid.orbit_mult) * RHO_BULK * vox)
        self.assertLess(abs(f000_var - f000_b5) / max(abs(f000_b5), 1.0), 1e-11)

        hkl = np.array(
            [[h, k, l] for h in range(0, 5) for k in range(0, 5) for l in range(0, 4) if (h, k, l) != (0, 0, 0)],
            dtype=np.int64,
        )
        n_ref = len(hkl)
        d_hkl = np.array([cell.calculate_d(row.tolist()) for row in hkl], dtype=np.float64)
        f_calc = rng.normal(0.0, 20.0, size=n_ref) + 1j * rng.normal(0.0, 20.0, size=n_ref)
        f_obs = np.abs(f_calc) * rng.uniform(0.85, 1.25, size=n_ref) + 2.0
        free_mask = np.arange(n_ref) % 10 == 0
        work_mask = ~free_mask
        bin_idx = np.clip((np.argsort(np.argsort(1.0 / d_hkl)) * 4) // n_ref, 0, 3).astype(np.int32)
        target = ReciprocalTarget(hkl, f_obs, work_mask, free_mask, f_calc, np.ones(n_ref), bin_idx, 4, d_hkl)

        wat_xyz = np.array([[10.0, 10.0, 8.0], [14.0, 12.0, 9.0]], dtype=np.float64)
        wat_occ = np.array([1.0, 0.5], dtype=np.float64)
        keep_post, _ = build_water_keep_mask_asu(grid, g_hat_slim, wat_xyz, wat_occ)
        keep_pre, _ = build_water_keep_mask_asu(grid, mw @ gO_m, wat_xyz, wat_occ)

        res0 = multiB_fpart_loss_and_grad(
            renderer, target, ds_slim, z_tot0, spec=spec, theta=theta,
            displace_mode="post", keep_asu=keep_post, clamp_nonneg=True, compute_grad=True, use_f32_blas=False
        )
        g_psi_ana = pb_phys(res0["grad_z_raw_m"])
        g_eta_ana = smoother.adjoint(np.sum(res0["grad_z_raw_m"], axis=0))
        g_theta_ana = res0["grad_theta"]

        eps = 1e-5
        for j in range(len(psi_test)):
            pp = psi_test.copy(); pp[j] += eps
            pm = psi_test.copy(); pm[j] -= eps
            zp, _, _ = phys_spec.forward_with_pullback(mob_fast, pp)
            zm, _, _ = phys_spec.forward_with_pullback(mob_fast, pm)
            lp = multiB_fpart_loss_and_grad(renderer, target, ds_slim, zp + dz0[None, :], cached_pmf=(gO_m, gH_m, sk),
                                            displace_mode="post", keep_asu=keep_post, compute_grad=False)["loss"]
            lm = multiB_fpart_loss_and_grad(renderer, target, ds_slim, zm + dz0[None, :], cached_pmf=(gO_m, gH_m, sk),
                                            displace_mode="post", keep_asu=keep_post, compute_grad=False)["loss"]
            fd = (lp - lm) / (2.0 * eps)
            self.assertLess(abs(g_psi_ana[j] - fd) / max(abs(g_psi_ana[j]), abs(fd), 1e-9), 2e-4)

        d_eta = rng.standard_normal(grid.n_asu)
        d_eta /= np.linalg.norm(d_eta)
        dz_dir = smoother.forward(d_eta)
        lp_e = multiB_fpart_loss_and_grad(renderer, target, ds_slim, z_tot0 + (eps * dz_dir)[None, :], cached_pmf=(gO_m, gH_m, sk),
                                          displace_mode="post", keep_asu=keep_post, compute_grad=False)["loss"]
        lm_e = multiB_fpart_loss_and_grad(renderer, target, ds_slim, z_tot0 - (eps * dz_dir)[None, :], cached_pmf=(gO_m, gH_m, sk),
                                          displace_mode="post", keep_asu=keep_post, compute_grad=False)["loss"]
        fd_dir_eta = (lp_e - lm_e) / (2.0 * eps)
        ana_dir_eta = float(np.dot(g_eta_ana, d_eta))
        self.assertLess(abs(ana_dir_eta - fd_dir_eta) / max(abs(ana_dir_eta), abs(fd_dir_eta), 1e-9), 2e-4)

        d_th = rng.standard_normal(len(theta))
        d_th /= np.linalg.norm(d_th)
        lp_t = multiB_fpart_loss_and_grad(renderer, target, ds_slim, z_tot0, spec=spec, theta=theta + eps * d_th,
                                          displace_mode="post", keep_asu=keep_post, compute_grad=False, use_f32_blas=False)["loss"]
        lm_t = multiB_fpart_loss_and_grad(renderer, target, ds_slim, z_tot0, spec=spec, theta=theta - eps * d_th,
                                          displace_mode="post", keep_asu=keep_post, compute_grad=False, use_f32_blas=False)["loss"]
        fd_dir_th = (lp_t - lm_t) / (2.0 * eps)
        ana_dir_th = float(np.dot(g_theta_ana, d_th))
        self.assertLess(abs(ana_dir_th - fd_dir_th) / max(abs(ana_dir_th), abs(fd_dir_th), 1e-9), 2e-4)

        mob_asu = reduce_mobility_features_to_asu(mob_fast, mw)
        cached = (gO_m, gH_m, sk)
        u_out = fit_uniform_B_on_work(renderer, target, ds_slim, cached, displace_mode="pre", keep_asu=keep_pre, maxiter=3)
        p_out = fit_physical_bmob_on_work(renderer, target, ds_slim, mob_asu, cached, displace_mode="pre", keep_asu=keep_pre, maxiter=2)
        s3_out = fit_spatial_bmob_on_work(renderer, target, ds_slim, p_out["z_phys"], cached, fwhm_A=3.2, displace_mode="pre", keep_asu=keep_pre, maxiter=2)
        j_out = fit_joint_pmf_and_bmob_on_work(renderer, target, ds_slim, spec, theta, mob_asu, p_out["psi"],
                                               fwhm_A=3.2, displace_mode="pre", keep_asu=keep_pre, maxiter=2)
        self.assertTrue(np.isfinite(u_out["loss"]) and np.isfinite(s3_out["loss"]) and np.isfinite(j_out["loss"]))


if __name__ == "__main__":
    unittest.main()
