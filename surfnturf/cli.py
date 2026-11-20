"""Unified command-line interface for surfNturf (`surfnturf`, `surfNturf`, or `python -m surfnturf`)."""
import argparse
import json
import math
import os
import sys
import numpy as np
import gemmi

from . import __version__
from .field import ASUGrid
from .io import (
    choose_sg_grid_shape,
    resolve_crystal_symmetry,
    load_saved_spec_and_theta,
    extract_ordered_waters,
    displace_ordered_waters,
    write_ccp4,
    write_fpart_mtz_on_hkl,
    write_fpart_mtz_to_dmin,
    predict_maps_for_target,
)
from .reconnect import reconnect_pdb_to_conformers, write_member_pdbs

_ALLOWED_PREDICT_KEYS = {
    "weights", "B", "grid", "alt", "nconf", "mode", "dmin", "hkl", "bmob", "displace", "ignore_waters", "shape"
}


def _nyquist_safe_grid_spacing(gsp: float, d_hi: float) -> float:
    """Ensure 2 * grid_spacing < d_hi so all reflections up to d_hi sit strictly inside the FFT Nyquist box."""
    if math.isfinite(d_hi) and d_hi > 0.0 and 2.05 * gsp > d_hi:
        return max(0.15, round(d_hi / 2.12, 4))
    return gsp


def _split_dataset_spec(spec_str: str):
    """Split `[name:]map.ccp4:model.pdb:B` on delimiter colons while preserving Windows `C:\\...` drive letters."""
    parts = []
    cur = []
    n = len(spec_str)
    for i, ch in enumerate(spec_str):
        if ch == ":":
            is_win_drive = (
                i > 0
                and spec_str[i - 1].isalpha()
                and (i == 1 or spec_str[i - 2] == ":")
                and (i + 1 < n)
                and (spec_str[i + 1] == "\\" or (os.name == "nt" and spec_str[i + 1] == "/"))
            )
            if not is_win_drive:
                parts.append("".join(cur))
                cur = []
                continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _cmd_predict(args):
    kv = {}
    for tok in args.extra:
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip()
            if k not in _ALLOWED_PREDICT_KEYS:
                raise SystemExit(
                    f"Unknown predict option {k!r} in {tok!r}; allowed keys: {sorted(_ALLOWED_PREDICT_KEYS)}"
                )
            kv[k] = v.strip()
        else:
            raise SystemExit(f"Unrecognized positional argument {tok!r}; expected key=value or --flag")

    weights = kv.get("weights", args.weights)
    B_map = float(kv.get("B", args.B))
    gsp = float(kv.get("grid", args.grid))
    alt = kv.get("alt", args.alt)
    nconf = int(kv.get("nconf", args.nconf))
    mode = kv.get("mode", args.mode)
    dmin = kv.get("dmin", args.dmin)
    hkl_mtz = kv.get("hkl", args.hkl)
    bmob_preset = kv.get("bmob", args.bmob)
    if bmob_preset in ("", "none", "None"):
        bmob_preset = None
    displace = kv.get("displace", args.displace).lower()
    if displace not in ("auto", "0", "1"):
        raise SystemExit(f"Invalid displace={displace!r}; choose 'auto', '0', or '1'")
    if kv.get("ignore_waters", "0").lower() in ("1", "true", "yes") or args.ignore_waters:
        displace = "0"
    shape_str = kv.get("shape", args.shape)
    exact_shape = tuple(int(x) for x in shape_str.split(",")) if shape_str else None
    if exact_shape is None:
        if hkl_mtz and os.path.isfile(hkl_mtz):
            ref_m = gemmi.read_mtz_file(hkl_mtz)
            ref_m.update_reso()
            gsp = _nyquist_safe_grid_spacing(gsp, float(ref_m.resolution_high()))
        elif str(dmin).lower() != "none":
            gsp = _nyquist_safe_grid_spacing(gsp, float(dmin))

    cell, sg, grid, members, want_3d, cannot_3d, solv_3d = predict_maps_for_target(
        args.target,
        weights=weights,
        B_map=B_map,
        grid_spacing=gsp,
        exact_shape=exact_shape,
        alt=alt,
        nconf=nconf,
        mode=mode,
        bmob_preset=bmob_preset,
    )
    out = args.out_prefix
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_ccp4(f"{out}_want.ccp4", want_3d, cell, sg)
    write_ccp4(f"{out}_cannot.ccp4", cannot_3d, cell, sg)
    write_ccp4(f"{out}_solvent.ccp4", solv_3d, cell, sg)

    wx, wo = extract_ordered_waters(args.target)
    use_disp = (displace == "1") or (displace == "auto" and len(wx) > 0)
    if use_disp and len(wx) > 0:
        bulk_3d, rex, e_ord = displace_ordered_waters(grid, solv_3d, wx, wo)
        write_ccp4(f"{out}_bulk.ccp4", bulk_3d, cell, sg)
        sf_map = bulk_3d
        disp_msg = f", bulk(r_ex={rex:.3f}A, e_ord={e_ord:.0f}e)"
    else:
        sf_map = solv_3d
        disp_msg = ""

    mtz_msg = ""
    if hkl_mtz:
        mtz_out = f"{out}_fpart.mtz"
        write_fpart_mtz_on_hkl(sf_map, cell, hkl_mtz, mtz_out)
        mtz_msg = f", {mtz_out}"
    elif str(dmin).lower() != "none":
        mtz_out = f"{out}_fpart.mtz"
        write_fpart_mtz_to_dmin(sf_map, cell, sg, float(dmin), mtz_out)
        mtz_msg = f", {mtz_out}"

    b_desc = f"bmob={bmob_preset}" if bmob_preset else f"B={B_map:g}"
    print(
        f"{args.target} (alt={alt}, {len(members)} confs, grid={want_3d.shape}, {b_desc}): "
        f"cannot>0.95={100.0 * (cannot_3d > 0.95).mean():.1f}%, want_max={want_3d.max():.2f}, "
        f"solv_mean={solv_3d.mean():.4f} e/A^3{disp_msg} -> wrote {out}_*.ccp4{mtz_msg}"
    )


def _cmd_reconnect(args):
    cryst1_line, conformers, st = reconnect_pdb_to_conformers(
        args.pdb,
        nconf=args.nconf,
        occ_flex=args.occ_flex,
        quotas=args.quotas.lower(),
        rebuild_h=not args.no_rebuild_h,
    )
    write_member_pdbs(cryst1_line, conformers, args.out_dir, pseudo_path=args.pseudo)
    na = [len(c) for c in conformers]
    a_pep, a_ss, a_cl = st["alpha_clean_geom"]
    u_pep, u_ss, u_cl = st["reconnected_geom"]
    print(
        f"{args.pdb}: {st['n_res']} residues, {args.nconf} conformers, atoms/conformer {min(na)}-{max(na)} "
        f"(dropped {len(st['dropped_ghosts'])} H-only ghost altlocs fixing {st['buggy_alpha_holes']} holes; "
        f"restored {st['restored_zero_h']} zero-occ H, bisected {st['rebuilt_amide_h']} cross-letter amide H)"
    )
    print(
        f"  alphabetical quantiles (clean): broken peptide {a_pep:.2f}/{st['n_pep_links']}, "
        f"broken SS {a_ss:.2f}/{st['n_ss_links']}, clashes <2.2A {a_cl:.2f}"
    )
    print(
        f"  geometry-reconnected (flex={args.occ_flex}): broken peptide {u_pep:.2f}/{st['n_pep_links']}, "
        f"broken SS {u_ss:.2f}/{st['n_ss_links']}, clashes <2.2A {u_cl:.2f}"
    )


def _cmd_fit_bmob(args):
    from .bmob import (
        SlimDatasetBundle,
        MultiBRenderer,
        load_ensemble_with_bfac,
        fast_build_asu_and_mobility,
        reduce_mobility_features_to_asu,
        compute_pmf_density_with_pullback,
        build_water_keep_mask_asu,
        fit_uniform_B_on_work,
        fit_physical_bmob_on_work,
        fit_spatial_bmob_on_work,
        fit_joint_pmf_and_bmob_on_work,
        summarize_bmob_microenvironments,
        load_reciprocal_target_from_fmodel_mtz,
    )

    spec, theta = load_saved_spec_and_theta(args.weights)
    target, cell, sg = load_reciprocal_target_from_fmodel_mtz(args.fmodel_mtz, n_bins=args.bins)
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))
    else:
        d_hi = float(np.min(target.d_hkl))
        if args.hkl and os.path.isfile(args.hkl):
            rm = gemmi.read_mtz_file(args.hkl)
            rm.update_reso()
            d_hi = min(d_hi, float(rm.resolution_high()))
        gsp = _nyquist_safe_grid_spacing(float(args.grid), d_hi)
        shape = choose_sg_grid_shape(cell, sg, gsp)

    grid = ASUGrid(cell, sg, shape, step=1)
    members, bisos, wts = load_ensemble_with_bfac(args.pdb, nconf=args.nconf, two_type_heavy_only=(spec.n_types == 2))
    Phi_O, Phi_H, steric_H, mob_feat = fast_build_asu_and_mobility(
        grid, members, bisos, n_types=spec.n_types, include_hpol=spec.use_hpol
    )
    ds_slim = SlimDatasetBundle("bmob", grid, B_map=5.0, Phi_O=Phi_O, Phi_H=Phi_H, steric_H=steric_H, member_weights=wts)
    mob_asu = reduce_mobility_features_to_asu(mob_feat, wts)
    renderer = MultiBRenderer(grid)
    gO_m, gH_m, sk, _ = compute_pmf_density_with_pullback(spec, ds_slim, theta, need_grad=False)
    cached = (gO_m, gH_m, sk)

    wx, wo = extract_ordered_waters(args.pdb)
    disp_mode = args.displace
    if disp_mode == "auto":
        disp_mode = "pre" if len(wx) > 0 else "none"
    keep_asu = None
    if disp_mode == "pre" and len(wx) > 0:
        keep_asu, _ = build_water_keep_mask_asu(grid, wts @ gO_m, wx, wo)
    elif disp_mode == "post" and len(wx) > 0:
        g_hat_init, _, _ = renderer.render_and_pullback(
            gO_m, gH_m, wts, np.full((1, grid.n_asu), np.log(5.0)), scale_k=sk, compute_grad=False
        )
        keep_asu, _ = build_water_keep_mask_asu(grid, g_hat_init, wx, wo)

    if args.mode == "uniform":
        out = fit_uniform_B_on_work(
            renderer, target, ds_slim, cached, displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )
    elif args.mode == "phys":
        out = fit_physical_bmob_on_work(
            renderer, target, ds_slim, mob_asu, cached, displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )
    elif args.mode == "tb32":
        p_out = fit_physical_bmob_on_work(
            renderer, target, ds_slim, mob_asu, cached, displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )
        out = fit_spatial_bmob_on_work(
            renderer, target, ds_slim, p_out["z_phys"], cached, fwhm_A=args.fwhm, lam_z=args.lam_z,
            displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )
    else:
        p_out = fit_physical_bmob_on_work(
            renderer, target, ds_slim, mob_asu, cached, displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )
        out = fit_joint_pmf_and_bmob_on_work(
            renderer, target, ds_slim, spec, theta, mob_asu, p_out["psi"], fwhm_A=args.fwhm, lam_z=args.lam_z,
            displace_mode=disp_mode, keep_asu=keep_asu, maxiter=args.maxiter
        )

    summarize_bmob_microenvironments(args.mode, out["B_eff_asu"], mob_asu, wts)
    solv_3d = out["solv_3d"]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_ccp4(f"{args.out}_solvent.ccp4", solv_3d, cell, sg)
    ref_mtz = args.hkl or args.fmodel_mtz
    write_fpart_mtz_on_hkl(solv_3d, cell, ref_mtz, f"{args.out}_fpart.mtz")
    print(
        f"fit-bmob ({args.mode}, displace={disp_mode}): internal R_work / R_free = "
        f"{100.0 * out['r_work']:.2f}% / {100.0 * out['r_free']:.2f}% -> wrote {args.out}_solvent.ccp4, {args.out}_fpart.mtz"
    )


def _cmd_fit_divot(args):
    from .divot import (
        CrystalGridOps,
        PeriodicFourierFields,
        build_open_solvent_envelope,
        extract_problem_from_fmodel,
        fit_periodic_arm,
        fit_adaptive_divot_peak_blind,
        write_fpart_cgrid,
    )

    fm_probe = gemmi.read_mtz_file(args.fmodel_mtz)
    fm_probe.update_reso()
    d_hi = float(fm_probe.resolution_high())
    if args.data_mtz and os.path.isfile(args.data_mtz):
        dm_probe = gemmi.read_mtz_file(args.data_mtz)
        dm_probe.update_reso()
        d_hi = min(d_hi, float(dm_probe.resolution_high()))

    if args.base_ccp4:
        m = gemmi.read_ccp4_map(args.base_ccp4, setup=True)
        base_3d = np.array(m.grid, copy=True, dtype=np.float64)
        cell, sg = resolve_crystal_symmetry(
            m.grid.unit_cell,
            m.grid.spacegroup.hm if m.grid.spacegroup is not None else "",
            source_desc=str(args.base_ccp4),
        )
    else:
        gsp = _nyquist_safe_grid_spacing(float(args.grid), d_hi)
        cell, sg, grid, _, _, _, solv_3d = predict_maps_for_target(
            args.pdb, weights=args.weights, grid_spacing=gsp, nconf=args.nconf
        )
        wx, wo = extract_ordered_waters(args.pdb)
        if len(wx) > 0 and not args.ignore_waters:
            base_3d, _, _ = displace_ordered_waters(grid, solv_3d, wx, wo)
        else:
            base_3d = solv_3d

    cgrid = CrystalGridOps(cell, sg, base_3d.shape, conv="plus", fft_workers=args.workers)
    prob = extract_problem_from_fmodel(cgrid, args.fmodel_mtz, data_mtz_path=args.data_mtz)
    open_mask = build_open_solvent_envelope(cgrid, args.pdb)

    arm = args.arm.lower()
    if arm == "sw6_mc":
        pf = PeriodicFourierFields(
            cgrid, base_3d, fitreso_s=6.0, fitreso_w=6.0,
            use_scale=True, use_warp=True, mass_conserving=True, wall_mask_3d=open_mask
        )
        rho_opt, _, info = fit_periodic_arm(prob, pf)
    elif arm == "sw6_std":
        pf = PeriodicFourierFields(
            cgrid, base_3d, fitreso_s=6.0, fitreso_w=6.0,
            use_scale=True, use_warp=True, mass_conserving=False
        )
        rho_opt, _, info = fit_periodic_arm(prob, pf)
    elif arm == "peak":
        rho_opt, info = fit_adaptive_divot_peak_blind(
            prob, cgrid, base_3d, open_mask, fitreso_s=10.0, use_divots=False, use_peaks=True, n_passes=2
        )
    elif arm == "divot":
        rho_opt, info = fit_adaptive_divot_peak_blind(
            prob, cgrid, base_3d, open_mask, fitreso_s=10.0, use_divots=True, use_peaks=False, n_passes=1
        )
    else:
        rho_opt, info = fit_adaptive_divot_peak_blind(
            prob, cgrid, base_3d, open_mask, fitreso_s=8.0, use_divots=True, use_peaks=True, n_passes=2
        )

    rho_scaled, ev = prob.scaled_solvent_rho(rho_opt, active_mask=prob.work)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_ccp4(f"{args.out}_solvent.ccp4", rho_scaled, cell, sg)
    ref_mtz = args.data_mtz or args.fmodel_mtz
    write_fpart_cgrid(cgrid, rho_scaled, ref_mtz, f"{args.out}_fpart.mtz", col_names=("FPART", "PHIPART"))
    print(
        f"fit-divot (arm={arm}): internal R_work / R_free = {100.0 * ev['r_work']:.2f}% / {100.0 * ev['r_free']:.2f}% "
        f"(blind r_val={100.0 * info.get('r_val', 0.0):.2f}%, k_solv/k_prot={ev['p'][1] / max(ev['p'][0], 1e-8):.3f}) "
        f"-> wrote {args.out}_solvent.ccp4, {args.out}_fpart.mtz"
    )


def _cmd_train(args):
    from .train import fit_surfnturf_model

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            items = json.load(f)
        datasets = [(d["name"], d["map"], d["model"], float(d["B"])) for d in items]
    else:
        datasets = []
        for idx, spec_str in enumerate(args.dataset):
            parts = _split_dataset_spec(spec_str)
            if len(parts) == 3:
                mp, mdl, bv = parts
                nm = f"ds{idx + 1}"
            elif len(parts) == 4:
                nm, mp, mdl, bv = parts
            else:
                raise SystemExit(f"--dataset must be 'map.ccp4:model.pdb:B' or 'name:map.ccp4:model.pdb:B', got {spec_str!r}")
            datasets.append((nm, mp, mdl, float(bv)))

    if not datasets:
        raise SystemExit("Provide at least one --dataset or --manifest.")

    res = fit_surfnturf_model(
        datasets, loss_mode=args.loss, n_types=args.n_types, step=args.step,
        nconf=args.nconf, maxiter=args.maxiter, out_npz=args.out
    )
    written = res.get("out_npz") or args.out
    print(
        f"Trained surfNturf ({args.loss}, {len(datasets)} datasets, {res['nit']} L-BFGS-B iters in {res['dt_s']}s): "
        f"loss={res['loss']:.6f} -> wrote {written}"
    )


def _cmd_test(_args):
    import unittest

    repo_tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
    if not os.path.isdir(repo_tests):
        raise SystemExit(f"Test directory not found at {repo_tests}; run pytest from a source checkout.")
    suite = unittest.defaultTestLoader.discover(start_dir=repo_tests, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="surfnturf",
        description="surfNturf: Differentiable multi-conformer crystal hydration (Surf), conformer reconnection (Turf), 3D B(x) mobility, and X-ray solvent modelling.",
    )
    p.add_argument("--version", action="version", version=f"surfNturf {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # 1. predict
    sp = sub.add_parser("predict", help="Predict hydration maps (_want, _cannot, _solvent, _bulk) and FPART/PHIPART MTZ.")
    sp.add_argument("target", help="Input PDB/mmCIF file or directory of conformer member PDBs.")
    sp.add_argument("out_prefix", help="Output file prefix.")
    sp.add_argument("extra", nargs="*", help="Optional key=value overrides (e.g. B=5.0 hkl=refme.mtz).")
    sp.add_argument("--weights", default="l2", help="Pretrained weights ('l2', 'kl', or path to .npz; default: l2).")
    sp.add_argument("--B", type=float, default=5.0, help="Isotropic water B-factor in A^2 when --bmob is not set (default: 5.0).")
    sp.add_argument("--bmob", default=None, help="Optional 0-shot physical 3D mobility preset ('dry_a4' or 'wet_b5b').")
    sp.add_argument("--grid", type=float, default=0.45, help="Target grid spacing in Angstroms (default: 0.45).")
    sp.add_argument("--shape", default=None, help="Explicit grid dimensions 'nu,nv,nw'.")
    sp.add_argument("--alt", default="all", help="Conformer selection ('all' or single altloc letter; default: all).")
    sp.add_argument("--nconf", type=int, default=32, help="Number of reconnected conformers for multi-altloc PDBs (default: 32).")
    sp.add_argument("--mode", default="auto", help="Ensemble mode ('auto', 'letters', 'reconnect', 'quantiles').")
    sp.add_argument("--hkl", default=None, help="Reference diffraction MTZ onto whose Miller indices FPART/PHIPART are written.")
    sp.add_argument("--dmin", default="0.965", help="Resolution limit in A for standalone FPART MTZ when --hkl is omitted ('none' to skip).")
    sp.add_argument("--displace", default="auto", choices=["auto", "0", "1"], help="Displace ordered HOH atoms ('auto', '0', '1').")
    sp.add_argument("--ignore-waters", action="store_true", help="Ignore ordered HOH atoms in input PDB (DRY prediction).")
    sp.set_defaults(func=_cmd_predict)

    # 2. reconnect
    su = sub.add_parser("reconnect", help="Expand a multi-altloc PDB into clash-free geometry-reconnected complete conformers.")
    su.add_argument("pdb", help="Input multi-altloc PDB/mmCIF file.")
    su.add_argument("out_dir", help="Output directory for m01.pdb .. mNN.pdb.")
    su.add_argument("--nconf", type=int, default=32, help="Number of complete conformers (default: 32).")
    su.add_argument("--occ-flex", type=int, default=0, help="Bounded quota slack +-k/nconf (default: 0).")
    su.add_argument("--quotas", default="hamilton", choices=["hamilton", "quantile"], help="Quota rounding rule.")
    su.add_argument("--no-rebuild-h", action="store_true", help="Disable cross-letter amide H/D bisector rebuilding.")
    su.add_argument("--pseudo", default=None, help="Optional combined multi-altloc pseudo-PDB output path.")
    su.set_defaults(func=_cmd_reconnect)

    # 3. fit-bmob
    sb = sub.add_parser("fit-bmob", help="Fit spatially varying 3D water mobility B(x) against work-set |Fo|.")
    sb.add_argument("--pdb", required=True, help="Input macromolecular model PDB/mmCIF or member directory.")
    sb.add_argument("--fmodel-mtz", required=True, help="Phenix *_f_model.mtz (from export_final_f_model=True).")
    sb.add_argument("--out", required=True, help="Output prefix for _solvent.ccp4 and _fpart.mtz.")
    sb.add_argument("--hkl", default=None, help="Optional reference MTZ for output HKL list (defaults to --fmodel-mtz).")
    sb.add_argument("--weights", default="l2", help="Starting surfNturf weights ('l2', 'kl', or .npz).")
    sb.add_argument("--mode", default="joint", choices=["uniform", "phys", "tb32", "joint"], help="Mobility fit mode.")
    sb.add_argument("--fwhm", type=float, default=3.2, help="Spatial coupling FWHM in A (default: 3.2).")
    sb.add_argument("--lam-z", type=float, default=2e-4, help="L2 regularization on spatial ln B nudges (default: 2e-4).")
    sb.add_argument("--displace", default="auto", choices=["auto", "none", "pre", "post"], help="Ordered-water displacement mode.")
    sb.add_argument("--grid", type=float, default=0.45, help="Grid spacing in A (default: 0.45).")
    sb.add_argument("--shape", default=None, help="Explicit grid shape 'nu,nv,nw'.")
    sb.add_argument("--nconf", type=int, default=32, help="Conformer count for multi-altloc models (default: 32).")
    sb.add_argument("--bins", type=int, default=24, help="Resolution shells for per-bin k_mask (default: 24).")
    sb.add_argument("--maxiter", type=int, default=55, help="Max L-BFGS-B iterations per stage (default: 55).")
    sb.set_defaults(func=_cmd_fit_bmob)

    # 4. fit-divot
    sd = sub.add_parser("fit-divot", help="Fit sparse ContinuousPeakDict / WatershedDivotDict / W_mc solvent on work |Fo|.")
    sd.add_argument("--pdb", required=True, help="Input macromolecular model PDB/mmCIF (defines open-solvent envelope).")
    sd.add_argument("--fmodel-mtz", required=True, help="Phenix *_f_model.mtz (from export_final_f_model=True).")
    sd.add_argument("--out", required=True, help="Output prefix for _solvent.ccp4 and _fpart.mtz.")
    sd.add_argument("--data-mtz", default=None, help="Optional original diffraction MTZ with 20-bin FreeR_flag for work_val split.")
    sd.add_argument("--base-ccp4", default=None, help="Optional starting solvent CCP4 map (defaults to surfNturf prediction).")
    sd.add_argument("--weights", default="l2", help="surfNturf weights if generating starting base ('l2', 'kl', or .npz).")
    sd.add_argument("--arm", default="hybrid", choices=["hybrid", "peak", "divot", "sw6_mc", "sw6_std"], help="Refinement arm.")
    sd.add_argument("--grid", type=float, default=0.45, help="Grid spacing in A when generating starting base (default: 0.45).")
    sd.add_argument("--nconf", type=int, default=32, help="Conformer count when generating starting base (default: 32).")
    sd.add_argument("--ignore-waters", action="store_true", help="Do not displace ordered HOH when generating starting base.")
    sd.add_argument("--workers", type=int, default=4, help="FFT worker threads (default: 4).")
    sd.set_defaults(func=_cmd_fit_divot)

    # 5. train
    st = sub.add_parser("train", help="Fit surfNturf hydration PMF+OZ weights to MD solvent CCP4 maps.")
    st.add_argument("--dataset", action="append", default=[], help="'[name:]map.ccp4:model.pdb_or_dir:B_map' (repeatable).")
    st.add_argument("--manifest", default=None, help="JSON list of {'name','map','model','B'} objects.")
    st.add_argument("--out", required=True, help="Output .npz weights path.")
    st.add_argument("--loss", default="l2", choices=["l2", "kl"], help="Real-space training loss (default: l2).")
    st.add_argument("--n-types", type=int, default=10, choices=[2, 10], help="Chemical atom classes (default: 10).")
    st.add_argument("--step", type=int, default=2, help="Grid subsampling stride during training (default: 2).")
    st.add_argument("--nconf", type=int, default=32, help="Conformer count for multi-altloc PDBs (default: 32).")
    st.add_argument("--maxiter", type=int, default=220, help="Max L-BFGS-B iterations (default: 220).")
    st.set_defaults(func=_cmd_train)

    # 6. test
    stest = sub.add_parser("test", help="Run surfNturf's analytical & finite-difference verification suite.")
    stest.set_defaults(func=_cmd_test)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
