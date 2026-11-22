"""High-level prediction, CCP4/MTZ I/O, and ordered-water displacement for surfNturf."""
import os
import sys
import numpy as np
import gemmi
from scipy.spatial import cKDTree

from .data import get_pretrained_weights_path, load_bmob_phys_preset
from .field import (
    RHO_BULK, ASUGrid, WaterFieldSpec,
    classify_conformer_atoms, load_ensemble, forward_direct, solv_to_fpart
)
from .bmob import (
    build_mobility_features, load_ensemble_with_bfac,
    reduce_mobility_features_to_asu, PhysicalBMobSpec, MultiBRenderer
)


def resolve_crystal_symmetry(cell: gemmi.UnitCell, sg_or_hm, source_desc: str = "<structure>"):
    """Validate crystallographic unit cell and resolve space group cleanly (handling rhombohedral vs hexagonal R)."""
    if cell is None or not cell.is_crystal():
        raise ValueError(f"{source_desc}: missing or invalid crystallographic unit cell (CRYST1)")
    if isinstance(sg_or_hm, gemmi.SpaceGroup):
        return cell, sg_or_hm
    sg_hm = (sg_or_hm or "").strip()
    if not sg_hm:
        print(f"surfnturf: {source_desc} has no space group; assuming 'P 1'", file=sys.stderr)
        sg_hm = "P 1"
    sg = gemmi.find_spacegroup_by_name(sg_hm, alpha=cell.alpha, gamma=cell.gamma)
    if sg is None:
        try:
            sg = gemmi.SpaceGroup(sg_hm)
        except Exception as exc:
            raise ValueError(f"{source_desc}: unrecognized space group {sg_hm!r}") from exc
    return cell, sg


def choose_sg_grid_shape(cell: gemmi.UnitCell, sg: gemmi.SpaceGroup, grid_spacing: float = 0.45):
    """Choose FFT-friendly grid dimensions `(nu, nv, nw)` exact for all symmetry operations of `sg`."""
    gg = gemmi.FloatGrid()
    gg.set_unit_cell(cell)
    gg.spacegroup = sg
    gg.set_size_from_spacing(float(grid_spacing), gemmi.GridSizeRounding.Up)
    return (int(gg.nu), int(gg.nv), int(gg.nw))


def load_saved_spec_and_theta(npz_or_variant="l2"):
    """Load `(WaterFieldSpec, theta)` from a `.npz` path or bundled preset name ('l2', 'kl', 'v1_l2', 'v1_kl')."""
    npz_path = get_pretrained_weights_path(npz_or_variant)
    D = dict(np.load(npz_path, allow_pickle=True))
    n_types = int(D["n_types"])
    use_lp = bool(D["use_lp"]) if "use_lp" in D else (n_types == 13)
    use_tet_oz = bool(D["use_tet_oz"]) if "use_tet_oz" in D else False
    n_oz_steps = int(D["n_oz_steps"]) if "n_oz_steps" in D else (4 if use_tet_oz else 1)
    use_epol = bool(D["use_epol"]) if "use_epol" in D else False
    use_pauli_cap = bool(D["use_pauli_cap"]) if "use_pauli_cap" in D else False
    zero_bulk_offset = bool(D["zero_bulk_offset"]) if "zero_bulk_offset" in D else False

    spec = WaterFieldSpec(
        n_types=n_types,
        use_langmuir=bool(D["use_langmuir"]),
        use_oz=bool(D["use_oz"]),
        use_hpol=bool(D["use_hpol"]),
        use_blur=bool(D["use_blur"]),
        fit_box_scale=False,
        n_datasets=1,
        use_lp=use_lp,
        use_tet_oz=use_tet_oz,
        n_oz_steps=n_oz_steps,
        use_epol=use_epol,
        use_pauli_cap=use_pauli_cap,
        zero_bulk_offset=zero_bulk_offset,
    )
    theta_raw = D["theta"].astype(np.float64)
    theta = theta_raw[:spec.n_params].copy()
    return spec, theta


def save_spec_and_theta(npz_path: str, spec: WaterFieldSpec, theta: np.ndarray, extra_meta: dict = None) -> str:
    """Save `(WaterFieldSpec, theta)` to a portable surfNturf `.npz` weights file and return the written path."""
    out_path = str(npz_path) if str(npz_path).endswith(".npz") else f"{npz_path}.npz"
    payload = {
        "n_types": np.int32(spec.n_types),
        "use_langmuir": np.bool_(spec.use_langmuir),
        "use_oz": np.bool_(spec.use_oz),
        "use_hpol": np.bool_(spec.use_hpol),
        "use_blur": np.bool_(spec.use_blur),
        "use_lp": np.bool_(spec.use_lp),
        "use_tet_oz": np.bool_(spec.use_tet_oz),
        "n_oz_steps": np.int32(spec.n_oz_steps),
        "use_epol": np.bool_(spec.use_epol),
        "use_pauli_cap": np.bool_(spec.use_pauli_cap),
        "zero_bulk_offset": np.bool_(spec.zero_bulk_offset),
        "theta": np.asarray(theta[:spec.n_params], dtype=np.float64),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            payload[k] = v
    np.savez(out_path, **payload)
    return out_path


def load_single_alt(pdb_path: str, alt_letter: str, two_type: bool = False,
                    use_lp: bool = False, return_charges: bool = False, complete_h: bool = True):
    """Extract a single conformer (`blank + alt_letter`) from a PDB/mmCIF structure."""
    from .hydrogens import complete_conformer_hydrogens
    st = gemmi.read_structure(pdb_path)
    al = []
    matched_alt = 0
    seen_alts = set()
    for ch in st[0]:
        for r in ch:
            if r.name in ("HOH", "WAT", "DOD"):
                continue
            for a in r:
                if a.occ <= 0:
                    continue
                alt_s = a.altloc.strip() if a.altloc != "\0" else ""
                if alt_s:
                    seen_alts.add(alt_s)
                if not alt_s or alt_s == alt_letter:
                    if alt_s == alt_letter:
                        matched_alt += 1
                    al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist()))
    if matched_alt == 0 and seen_alts:
        print(
            f"surfnturf: altloc {alt_letter!r} not found in {pdb_path} (available: {sorted(seen_alts)}); "
            f"using blank-altloc atoms only",
            file=sys.stderr,
        )
    if not al:
        raise ValueError(f"{pdb_path}: no non-water atoms with occupancy > 0 matched altloc {alt_letter!r}")
    if complete_h and not two_type:
        al = complete_conformer_hydrogens(al)
    wts = np.array([1.0], dtype=np.float64)
    if return_charges:
        x, t, q = classify_conformer_atoms(
            al, two_type_heavy_only=two_type, use_lp=(use_lp and not two_type), return_charges=True
        )
        return [(x, t)], wts, [q]
    return [classify_conformer_atoms(al, two_type_heavy_only=two_type, use_lp=(use_lp and not two_type))], wts


def extract_ordered_waters(pdb_path: str):
    """Extract ordered water heavy-atom Cartesian positions and occupancies (`occ > 0`) from a structure."""
    if os.path.isdir(pdb_path):
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    st = gemmi.read_structure(pdb_path)
    wx, wo = [], []
    for ch in st[0]:
        for r in ch:
            if r.name not in ("HOH", "WAT", "DOD"):
                continue
            for a in r:
                if a.element.name in ("H", "D") or a.occ <= 0:
                    continue
                wx.append(a.pos.tolist())
                wo.append(float(a.occ))
    if not wx:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.array(wx, dtype=np.float64), np.array(wo, dtype=np.float64)


def displace_ordered_waters(grid: ASUGrid, total_3d: np.ndarray, wat_xyz: np.ndarray, wat_occ: np.ndarray,
                            rex_override: float = None):
    """Displace ordered crystallographic waters from total solvent density while conserving total solvent electrons."""
    if len(wat_xyz) == 0:
        return total_3d.copy(), 0.0, 0.0
    vox = grid.cell.volume / float(grid.nvox)
    e_ord = 8.0 * float(np.sum(wat_occ)) * len(grid.ops)
    total_asu = grid.full_3d_to_asu_mean(total_3d)
    mult_vox = grid.orbit_mult * vox
    tot_elec = float(np.sum(total_asu * mult_vox))
    target_elec = max(0.0, tot_elec - e_ord)

    SW, Swi = grid.sym_expand(wat_xyz, cutoff_A=3.35)
    tg = cKDTree(grid.asu_xyz)
    nb = tg.query_ball_point(SW, 3.3)
    lens = np.array([len(x) for x in nb], dtype=np.int32)
    if lens.sum() == 0:
        return total_3d.copy(), 0.0, e_ord
    gi = np.concatenate([np.array(x, dtype=np.int32) for x in nb if len(x)])
    wi = np.repeat(np.arange(len(SW), dtype=np.int32), lens)
    dw = np.linalg.norm(grid.asu_xyz[gi] - SW[wi], axis=1)
    ow = wat_occ[Swi[wi]]

    def bulk_asu_of(rex):
        occ_hole = np.bincount(gi, ow / (1.0 + np.exp(-(rex - dw) / 0.15)), minlength=grid.n_asu)
        return total_asu * np.clip(1.0 - occ_hole, 0.0, 1.0)

    if rex_override is not None:
        rex = float(rex_override)
    else:
        lo, hi = 0.50, 3.10
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if float(np.sum(bulk_asu_of(mid) * mult_vox)) > target_elec:
                lo = mid
            else:
                hi = mid
        rex = 0.5 * (lo + hi)

    bulk_asu = bulk_asu_of(rex)
    bulk_3d = grid.asu_to_full_3d(bulk_asu)
    return bulk_3d, rex, e_ord


def write_ccp4(path: str, arr_3d: np.ndarray, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup):
    """Write a 3D float array (`e/A^3` or dimensionless relative density) to a CCP4 map file."""
    gg = gemmi.FloatGrid(*arr_3d.shape)
    gg.set_unit_cell(cell)
    gg.spacegroup = sg
    np.array(gg, copy=False)[:] = arr_3d.astype(np.float32)
    mo = gemmi.Ccp4Map()
    mo.grid = gg
    mo.update_ccp4_header()
    mo.write_ccp4_map(path)


def write_fpart_mtz_on_hkl(arr_3d: np.ndarray, cell: gemmi.UnitCell, ref_mtz_path: str, out_mtz_path: str,
                           col_names=("FPART", "PHIPART")):
    """Direct FFT structure-factor evaluation onto `ref_mtz_path`'s exact Miller indices (Nyquist-safe)."""
    ref_mtz = gemmi.read_mtz_file(ref_mtz_path)
    H = np.array(ref_mtz.make_miller_array(), dtype=np.int64)
    nu, nv, nw = arr_3d.shape
    in_nyq = (
        (H[:, 0] >= -(nu // 2)) & (H[:, 0] <= (nu - 1) // 2) &
        (H[:, 1] >= -(nv // 2)) & (H[:, 1] <= (nv - 1) // 2) &
        (H[:, 2] >= -(nw // 2)) & (H[:, 2] <= (nw - 1) // 2)
    )
    n_out = int(np.sum(~in_nyq))
    if n_out > 0:
        print(
            f"surfnturf: {n_out}/{len(H)} reflections in {ref_mtz_path} exceed grid {arr_3d.shape} Nyquist limit "
            f"and are set to FPART=0",
            file=sys.stderr,
        )
    vals = solv_to_fpart(arr_3d, cell.volume, H)

    out = gemmi.Mtz(with_base=True)
    out.cell = ref_mtz.cell
    out.spacegroup = ref_mtz.spacegroup
    out.add_dataset("bulk")
    out.add_column(col_names[0], "F")
    out.add_column(col_names[1], "P")
    data = np.column_stack([H, np.abs(vals), np.degrees(np.angle(vals))]).astype(np.float32)
    out.set_data(data)
    out.write_to_file(out_mtz_path)


def write_fpart_mtz_to_dmin(arr_3d: np.ndarray, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup,
                            dmin: float, out_mtz_path: str, col_names=("FPART", "PHIPART")):
    """Write reciprocal-space asymmetric-unit FPART/PHIPART structure factors up to `dmin` (pure gemmi)."""
    F_mat = np.array(cell.frac.mat.tolist(), dtype=np.float64)
    perp_spacings = 1.0 / (np.linalg.norm(F_mat, axis=1) * np.array(arr_3d.shape, dtype=np.float64))
    nyq_dmin = float(2.0 * np.max(perp_spacings))
    if nyq_dmin > float(dmin) * 1.02:
        print(
            f"surfnturf: grid {arr_3d.shape} has Nyquist limit {nyq_dmin:.3f} A > requested dmin={float(dmin):.3f} A; "
            f"use a finer grid spacing (<= {0.48 * float(dmin):.3f} A) to cover the full resolution sphere",
            file=sys.stderr,
        )

    gg = gemmi.FloatGrid(*arr_3d.shape)
    gg.set_unit_cell(cell)
    gg.spacegroup = sg
    np.array(gg, copy=False)[:] = arr_3d.astype(np.float32)
    sf = gemmi.transform_map_to_f_phi(gg, half_l=True)
    asu_data = sf.prepare_asu_data(dmin=float(dmin), with_000=False)

    out = gemmi.Mtz(with_base=True)
    out.cell = cell
    out.spacegroup = sg
    out.add_dataset("bulk")
    out.add_column(col_names[0], "F")
    out.add_column(col_names[1], "P")
    H = asu_data.miller_array
    vals = asu_data.value_array
    data = np.column_stack([H, np.abs(vals), np.degrees(np.angle(vals))]).astype(np.float32)
    out.set_data(data)
    out.write_to_file(out_mtz_path)


def _sample_structure_for_target(target_path: str) -> gemmi.Structure:
    if os.path.isdir(target_path):
        pdbs = sorted([os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith(".pdb")])
        if not pdbs:
            raise ValueError(f"No .pdb files found in ensemble directory {target_path!r}")
        return gemmi.read_structure(pdbs[0])
    return gemmi.read_structure(target_path)


def predict_maps_for_target(target_path: str, weights="l2", B_map: float = 5.0, grid_spacing: float = 0.45,
                            exact_shape=None, alt: str = "all", nconf: int = 32, mode: str = "auto",
                            bmob_preset: str = None, members_override=None, weights_override=None,
                            cell_override=None, sg_override=None, charges_override=None, complete_h: bool = True):
    """Predict `(cell, sg, grid, members, want_3d, cannot_3d, solv_3d)` for any PDB/mmCIF or member directory."""
    spec, theta = load_saved_spec_and_theta(weights)
    two_type = (spec.n_types == 2)
    need_lp = bool(spec.use_lp)
    need_q = bool(spec.use_epol)

    if bmob_preset is not None and members_override is None:
        st = _sample_structure_for_target(target_path)
        cell, sg = resolve_crystal_symmetry(
            cell_override or st.cell,
            sg_override or st.spacegroup_hm,
            source_desc=str(target_path),
        )
        if need_q:
            members, bisos, wts, charges = load_ensemble_with_bfac(
                target_path, nconf=nconf, two_type_heavy_only=two_type, mode=mode, alt=alt,
                use_lp=need_lp, return_charges=True, complete_h=complete_h,
            )
        else:
            members, bisos, wts = load_ensemble_with_bfac(
                target_path, nconf=nconf, two_type_heavy_only=two_type, mode=mode, alt=alt,
                use_lp=need_lp, return_charges=False, complete_h=complete_h,
            )
            charges = None

        shape = (
            tuple(int(k) for k in exact_shape)
            if exact_shape is not None
            else choose_sg_grid_shape(cell, sg, grid_spacing)
        )
        grid = ASUGrid(cell, sg, shape, step=1)
        _, gO_asu, gO_m, gH_m = forward_direct(
            spec, grid, members, wts, theta, B_map=B_map, return_per_member=True, charges=charges
        )
        mob_feat = build_mobility_features(grid, members, bisos, two_type_heavy_only=two_type)
        mob_asu = reduce_mobility_features_to_asu(mob_feat, wts)
        phys_spec = PhysicalBMobSpec()
        pdict = load_bmob_phys_preset(bmob_preset)
        psi = phys_spec.pack(
            alpha_prot=pdict["alpha_prot"], B0=pdict["B0"], dB_unteth=pdict["dB_unteth"],
            gamma_teth=pdict["gamma_teth"], B_far=pdict["B_far"], d_far=pdict["d_far"]
        )
        z_phys, _, _ = phys_spec.forward_with_pullback(mob_asu, psi)
        renderer = MultiBRenderer(grid)
        g_hat_asu, _, _ = renderer.render_and_pullback(gO_m, gH_m, wts, z_phys, scale_k=1.0, compute_grad=False)
        want_3d = grid.asu_to_full_3d(gO_asu)
        cannot_3d = 1.0 - np.clip(want_3d / 0.5, 0.0, 1.0)
        solv_3d = np.maximum(grid.asu_to_full_3d(g_hat_asu) * RHO_BULK, 0.0)
        return cell, sg, grid, members, want_3d, cannot_3d, solv_3d

    charges = charges_override
    if members_override is not None:
        members = members_override
        wts = weights_override if weights_override is not None else np.full(len(members), 1.0 / len(members))
        cell, sg = resolve_crystal_symmetry(cell_override, sg_override, source_desc="members_override")
    elif os.path.isdir(target_path):
        st = _sample_structure_for_target(target_path)
        cell, sg = resolve_crystal_symmetry(
            cell_override or st.cell, sg_override or st.spacegroup_hm, source_desc=str(target_path)
        )
        if need_q:
            members, wts, charges = load_ensemble(
                target_path, two_type_heavy_only=two_type, return_weights=True,
                use_lp=need_lp, return_charges=True, complete_h=complete_h,
            )
        else:
            members, wts = load_ensemble(
                target_path, two_type_heavy_only=two_type, return_weights=True,
                use_lp=need_lp, complete_h=complete_h,
            )
    else:
        st = gemmi.read_structure(target_path)
        cell, sg = resolve_crystal_symmetry(
            cell_override or st.cell, sg_override or st.spacegroup_hm, source_desc=str(target_path)
        )
        if alt != "all":
            if need_q:
                members, wts, charges = load_single_alt(
                    target_path, alt, two_type=two_type, use_lp=need_lp, return_charges=True, complete_h=complete_h
                )
            else:
                members, wts = load_single_alt(
                    target_path, alt, two_type=two_type, use_lp=need_lp, complete_h=complete_h
                )
        else:
            if need_q:
                members, wts, charges = load_ensemble(
                    target_path, nconf=nconf, two_type_heavy_only=two_type,
                    mode=mode, return_weights=True, use_lp=need_lp, return_charges=True, complete_h=complete_h,
                )
            else:
                members, wts = load_ensemble(
                    target_path, nconf=nconf, two_type_heavy_only=two_type,
                    mode=mode, return_weights=True, use_lp=need_lp, complete_h=complete_h,
                )

    shape = tuple(int(k) for k in exact_shape) if exact_shape is not None else choose_sg_grid_shape(cell, sg, grid_spacing)
    grid = ASUGrid(cell, sg, shape, step=1)
    g_hat_asu, gO_asu = forward_direct(spec, grid, members, wts, theta, B_map=B_map, charges=charges)
    want_3d = grid.asu_to_full_3d(gO_asu)
    cannot_3d = 1.0 - np.clip(want_3d / 0.5, 0.0, 1.0)
    solv_3d = np.maximum(grid.asu_to_full_3d(g_hat_asu) * RHO_BULK, 0.0)
    return cell, sg, grid, members, want_3d, cannot_3d, solv_3d
