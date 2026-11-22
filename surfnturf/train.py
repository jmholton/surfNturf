"""General-purpose L-BFGS-B trainer for fitting surfNturf hydration PMF+LP+TetOZ weights to MD solvent maps."""
import sys
import time
import numpy as np
import gemmi
from scipy.optimize import minimize

from .field import (
    ASUGrid, DatasetBundle, WaterFieldSpec, load_ensemble, build_asu_features,
    forward_and_grad, smoothness_penalty_and_grad, param_bounds
)
from .io import choose_sg_grid_shape, resolve_crystal_symmetry, save_spec_and_theta


def _fourier_resample_3d(arr_3d: np.ndarray, new_shape) -> np.ndarray:
    """Periodic trigonometric (FFT) resampling of real 3D array `arr_3d` onto `new_shape`."""
    old_shape = arr_3d.shape
    new_shape = tuple(int(n) for n in new_shape)
    if old_shape == new_shape:
        return arr_3d.copy()
    F_old = np.fft.fftn(np.asarray(arr_3d, dtype=np.float64))
    F_new = np.zeros(new_shape, dtype=np.complex128)
    lims = [min(o, n) for o, n in zip(old_shape, new_shape)]
    h0, h1, h2 = [(L + 1) // 2 for L in lims]
    t0, t1, t2 = [L // 2 for L in lims]

    for s0_o, s0_n in ((slice(0, h0), slice(0, h0)), (slice(old_shape[0] - t0, old_shape[0]), slice(new_shape[0] - t0, new_shape[0]))):
        for s1_o, s1_n in ((slice(0, h1), slice(0, h1)), (slice(old_shape[1] - t1, old_shape[1]), slice(new_shape[1] - t1, new_shape[1]))):
            for s2_o, s2_n in ((slice(0, h2), slice(0, h2)), (slice(old_shape[2] - t2, old_shape[2]), slice(new_shape[2] - t2, new_shape[2]))):
                F_new[s0_n, s1_n, s2_n] = F_old[s0_o, s1_o, s2_o]

    scale = float(np.prod(new_shape)) / float(np.prod(old_shape))
    return np.real(np.fft.ifftn(F_new)) * scale


def load_ccp4_map_and_grid(map_path: str, step: int = 2):
    """Read a CCP4 solvent electron-density map (in `e/A^3`) and construct its symmetry-exact `ASUGrid`."""
    m = gemmi.read_ccp4_map(map_path, setup=True)
    full_arr = np.array(m.grid, copy=True)
    cell, sg = resolve_crystal_symmetry(
        m.grid.unit_cell,
        m.grid.spacegroup.hm if m.grid.spacegroup is not None else "",
        source_desc=str(map_path),
    )
    step = int(step)
    try:
        grid = ASUGrid(cell, sg, full_arr.shape, step=step)
        sub_arr = full_arr[::step, ::step, ::step][:grid.shape[0], :grid.shape[1], :grid.shape[2]]
        return grid, sub_arr
    except ValueError:
        approx_sp = max(
            cell.a / max(full_arr.shape[0] / max(step, 1), 1.0),
            cell.b / max(full_arr.shape[1] / max(step, 1), 1.0),
            cell.c / max(full_arr.shape[2] / max(step, 1), 1.0),
        )
        safe_shape = choose_sg_grid_shape(cell, sg, grid_spacing=approx_sp)
        print(
            f"surfnturf: {map_path} shape {full_arr.shape} with step={step} is incompatible with {sg.hm}; "
            f"Fourier-resampling onto symmetry-exact grid {safe_shape}",
            file=sys.stderr,
        )
        res_arr = _fourier_resample_3d(full_arr, safe_shape)
        grid = ASUGrid(cell, sg, safe_shape, step=1)
        return grid, res_arr


def fit_surfnturf_model(datasets_spec, loss_mode: str = "l2", n_types: int = 10,
                        use_lp: bool = True, use_tet_oz: bool = True, n_oz_steps: int = 4,
                        use_epol: bool = False, use_pauli_cap: bool = True, zero_bulk_offset: bool = True,
                        use_langmuir: bool = True, use_oz: bool = True, use_hpol: bool = True,
                        use_blur: bool = True, fit_box_scale: bool = True, step: int = 2,
                        nconf: int = 32, lam_pmf: float = 2e-3, lam_hpol: float = 2e-3,
                        lam_oz: float = 1e-4, lam_extra: float = 2e-4, maxiter: int = 220, out_npz: str = None):
    """Fit a surfNturf `WaterFieldSpec` jointly across one or more `(name, map_path, model_path, B_map)` datasets.

    By default (`use_lp=True, use_tet_oz=True, n_oz_steps=4, use_pauli_cap=True, zero_bulk_offset=True, use_epol=False`),
    trains explicit sp2/sp3 lone pairs + 4-step tetrahedral (109.47 deg) H-bond network closure + Rosenfeld 0D 1-water
    cavity cap + exact zero-offset bulk asymptote, keeping crystal E-field polarization (`use_epol`) off by default.
    """
    t0 = time.time()
    two_type = (int(n_types) == 2)
    use_lp_eff = bool(use_lp) and (not two_type)
    eff_n_types = 2 if two_type else (13 if use_lp_eff else 10)
    use_tet_eff = bool(use_tet_oz) and bool(use_oz)
    use_epol_eff = bool(use_epol) and (not two_type)

    bundles = []
    for name, map_path, model_path, b_val in datasets_spec:
        grid, rho_sub = load_ccp4_map_and_grid(map_path, step=step)
        if use_epol_eff:
            members, wts, charges = load_ensemble(
                model_path, nconf=nconf, two_type_heavy_only=two_type, return_weights=True,
                use_lp=use_lp_eff, return_charges=True
            )
        else:
            members, wts = load_ensemble(
                model_path, nconf=nconf, two_type_heavy_only=two_type, return_weights=True,
                use_lp=use_lp_eff
            )
            charges = None

        feat_res = build_asu_features(
            grid, members, n_types=eff_n_types, include_hpol=bool(use_hpol),
            include_tet_oz=use_tet_eff, include_epol=use_epol_eff,
            charges=charges, member_weights=wts, return_extra=True,
        )
        Phi_O, Phi_H, steric_H, extra = feat_res
        bundles.append(DatasetBundle(
            name=name, grid=grid, rho_full=rho_sub, B_map=float(b_val),
            Phi_O=Phi_O, Phi_H=Phi_H, steric_H=steric_H, member_weights=wts,
            F_hbsol=extra['F_hbsol'], E_feats=extra['E_feats'],
        ))

    spec = WaterFieldSpec(
        n_types=eff_n_types, use_langmuir=bool(use_langmuir), use_oz=bool(use_oz),
        use_hpol=bool(use_hpol), use_blur=bool(use_blur), fit_box_scale=bool(fit_box_scale),
        n_datasets=len(bundles), use_lp=use_lp_eff, use_tet_oz=use_tet_eff,
        n_oz_steps=int(n_oz_steps), use_epol=use_epol_eff,
        use_pauli_cap=bool(use_pauli_cap), zero_bulk_offset=bool(zero_bulk_offset),
    )
    theta0 = spec.init_params()
    bounds = param_bounds(spec)
    inv_nd = 1.0 / max(len(bundles), 1)

    def obj_and_grad(th):
        tot_loss, tot_grad = smoothness_penalty_and_grad(
            spec, th, lam_pmf=lam_pmf, lam_hpol=lam_hpol, lam_oz=lam_oz, lam_extra=lam_extra
        )
        for idx, ds in enumerate(bundles):
            l_i, g_i, _, _ = forward_and_grad(
                spec, ds, th, ds_index=idx, loss_mode=loss_mode, compute_grad=True
            )
            tot_loss += inv_nd * l_i
            tot_grad += inv_nd * g_i
        return tot_loss, tot_grad

    res = minimize(
        obj_and_grad, theta0, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-8}
    )

    spec_export = WaterFieldSpec(
        n_types=eff_n_types, use_langmuir=bool(use_langmuir), use_oz=bool(use_oz),
        use_hpol=bool(use_hpol), use_blur=bool(use_blur), fit_box_scale=False, n_datasets=1,
        use_lp=use_lp_eff, use_tet_oz=use_tet_eff, n_oz_steps=int(n_oz_steps),
        use_epol=use_epol_eff, use_pauli_cap=bool(use_pauli_cap), zero_bulk_offset=bool(zero_bulk_offset),
    )
    theta_export = res.x[:spec_export.n_params].copy()
    written_npz = None
    if out_npz is not None:
        written_npz = save_spec_and_theta(out_npz, spec_export, theta_export, extra_meta={"loss_mode": loss_mode})

    return {
        "spec": spec_export,
        "theta": theta_export,
        "theta_full": res.x,
        "loss": float(res.fun),
        "nit": int(res.nit),
        "success": bool(res.success),
        "dt_s": round(time.time() - t0, 2),
        "out_npz": written_npz,
    }
