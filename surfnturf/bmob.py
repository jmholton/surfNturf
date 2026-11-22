"""Spatially varying water mobility B(x) for analytical multi-conformer hydration fields in surfNturf.

Implements two complementary mobility mechanisms:
  1. Physical protein-anchored + H-bond-tethered mobility law B_phys^(m)(x; psi):
     - Propagates each conformer's crystallographic protein B_iso across ~3.2 A contacts into the first hydration shell,
     - Modulates first-shell excess mobility by local polar H-bond tethering N_tether^(m)(x),
     - Transitions smoothly beyond d_far (~4.5 A) to wide disordered bulk mobility B_far (up to B = 999+ A^2).
  2. Regularized |Fo| / difference-map spatial mobility field dz_asu(x) = S_ell(eta_asu):
     - Continuous ASU log-mobility nudge field coupled across an ell = 3.2 A neighborhood, fitted under either
       strict `kmask_mode='fixed_one'` (`k_mask == 1.000` across all shells, eliminating gauge collinearity between
       local B(x) and per-bin k_mask(s)) or `'per_bin'`.

Rendering uses a 13-level C^1 Catmull-Rom partition-of-unity basis on ln B:
    B_levels = 5.0 * 2^(k - 3)  for k = 0..12  ([0.625 .. 2560.0] A^2, exact knot at 5.0 A^2).
"""
import glob
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gemmi
from scipy import fft as sp_fft
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit

from .field import (
    ASUGrid, WaterFieldSpec, DatasetBundle, RHO_BULK,
    R_MIN, DR, N_KNOTS, R_CUT, RH_MIN, DRH, N_HKNOTS, RH_CUT, ACCEPTOR_TYPES,
    _COMPACT_IONS, _LARGE_POLAR_IONS, _DEFAULT_CLASS_Q, _bspline_weights, _validate_ensemble_mode,
    _compute_epol_additive_terms, _epol_features_from_additive,
    softplus, classify_conformer_atoms, compute_pmf_density_with_pullback,
    solv_to_fpart, fpart_grad_to_asu, fit_kmask_per_bin,
    smoothness_penalty_and_grad, param_bounds
)
from .reconnect import reconnect_pdb_to_conformers

B_LEVELS = 5.0 * (2.0 ** (np.arange(13, dtype=np.float64) - 3.0))
N_BLEVELS = len(B_LEVELS)
Z0_BASIS = float(np.log(B_LEVELS[0]))
DZ_BASIS = float(np.log(2.0))
Z_CLIP_LO = float(np.log(1.30))
Z_CLIP_HI = float(np.log(1220.0))
BETA_CLIP = 20.0


def _atom_biso_from_gemmi(a):
    """Extract isotropic-equivalent B-factor (A^2) from a gemmi.Atom (falling back to ANISOU trace if needed)."""
    b = float(a.b_iso)
    if b <= 0.0 and a.aniso.nonzero():
        b = (8.0 * (np.pi ** 2) / 3.0) * float(a.aniso.u11 + a.aniso.u22 + a.aniso.u33)
    return float(np.clip(b if b > 0.0 else 15.0, 0.5, 999.0))


def classify_conformer_atoms_with_bfac(atoms_list, two_type_heavy_only=False, use_lp=False, return_charges=False):
    """Classify [(res_seq, res_name, atom_name, elem, xyz, biso), ...] -> (xyz, typ, biso[, q])."""
    if two_type_heavy_only:
        xyz, typ, biso = [], [], []
        for _, _, _, el, pos, b in atoms_list:
            el_u = el.upper()
            if el_u in ('H', 'D'):
                continue
            xyz.append(pos)
            is_pol = (el_u in ('N', 'O') or el_u in _COMPACT_IONS or el_u in _LARGE_POLAR_IONS)
            typ.append(0 if is_pol else 1)
            biso.append(float(np.clip(b, 0.5, 999.0)))
        xyz_a = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        typ_a = np.array(typ, dtype=np.int32)
        bi_a = np.array(biso, dtype=np.float64)
        if return_charges:
            return xyz_a, typ_a, bi_a, np.zeros(len(typ_a), dtype=np.float64)
        return xyz_a, typ_a, bi_a

    stripped = [rec[:5] for rec in atoms_list]
    heavy_b = [float(np.clip(rec[5], 0.5, 999.0)) for rec in atoms_list if rec[3].strip().upper() not in ('H', 'D')]
    hyd_b = [float(np.clip(rec[5], 0.5, 999.0)) for rec in atoms_list if rec[3].strip().upper() in ('H', 'D')]
    real_biso = np.array(heavy_b + hyd_b, dtype=np.float64)

    if return_charges:
        xyz_out, typ_out, q_out, lp_parents = classify_conformer_atoms(
            stripped, two_type_heavy_only=False, use_lp=use_lp, return_charges=True, return_lp_parents=True
        )
    else:
        xyz_out, typ_out, lp_parents = classify_conformer_atoms(
            stripped, two_type_heavy_only=False, use_lp=use_lp, return_charges=False, return_lp_parents=True
        )
        q_out = None

    biso_out = np.concatenate([real_biso, real_biso[lp_parents]]) if len(lp_parents) > 0 else real_biso
    assert len(biso_out) == len(xyz_out)
    if return_charges:
        return xyz_out, typ_out, biso_out, q_out
    return xyz_out, typ_out, biso_out


def load_ensemble_with_bfac(path_or_dir, nconf=32, two_type_heavy_only=False, mode='auto',
                            alt='all', use_lp=False, return_charges=False, complete_h=True):
    """Load conformers with per-atom B_iso: returns `(members, bisos, weights[, charges])`."""
    from .hydrogens import complete_conformer_hydrogens
    mode = _validate_ensemble_mode(mode)
    use_lp_eff = bool(use_lp) and (not two_type_heavy_only)

    def _finish(raw_lists, wts):
        members, bisos = [], []
        charges = [] if return_charges else None
        for al in raw_lists:
            if complete_h and not two_type_heavy_only:
                al = complete_conformer_hydrogens(al)
            if return_charges:
                x, t, bi, q = classify_conformer_atoms_with_bfac(
                    al, two_type_heavy_only=two_type_heavy_only, use_lp=use_lp_eff, return_charges=True
                )
                members.append((x, t))
                bisos.append(bi)
                charges.append(q)
            else:
                x, t, bi = classify_conformer_atoms_with_bfac(
                    al, two_type_heavy_only=two_type_heavy_only, use_lp=use_lp_eff, return_charges=False
                )
                members.append((x, t))
                bisos.append(bi)
        if return_charges:
            return members, bisos, wts, charges
        return members, bisos, wts

    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, 'm*.pdb')))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_dir, '*.pdb')))
        if not files:
            raise ValueError(f"No .pdb files found in ensemble directory {path_or_dir!r}")
        raw_lists = []
        for f in files:
            st = gemmi.read_structure(f)
            al = []
            for ch in st[0]:
                for r in ch:
                    if r.name in ('HOH', 'WAT', 'DOD'):
                        continue
                    for a in r:
                        if a.occ <= 0:
                            continue
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), _atom_biso_from_gemmi(a)))
            raw_lists.append(al)
        wts = np.full(len(raw_lists), 1.0 / max(len(raw_lists), 1), dtype=np.float64)
        return _finish(raw_lists, wts)

    st = gemmi.read_structure(path_or_dir)

    if alt != 'all':
        al = []
        matched_alt = 0
        seen_alts = set()
        for ch in st[0]:
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                for a in r:
                    if a.occ <= 0:
                        continue
                    alt_s = a.altloc.strip() if a.altloc != '\0' else ''
                    if alt_s:
                        seen_alts.add(alt_s)
                    if not alt_s or alt_s == alt:
                        if alt_s == alt:
                            matched_alt += 1
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), _atom_biso_from_gemmi(a)))
        if matched_alt == 0 and seen_alts:
            print(
                f"surfnturf: altloc {alt!r} not found in {path_or_dir} (available: {sorted(seen_alts)}); "
                f"using blank-altloc atoms only",
                file=sys.stderr,
            )
        if not al:
            raise ValueError(f"{path_or_dir}: no non-water atoms with occupancy > 0 matched altloc {alt!r}")
        return _finish([al], np.array([1.0], dtype=np.float64))

    all_alts = set()
    any_h_split_on_blank_hv = False
    n_pos_hv = 0
    for ch in st[0]:
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            r_blank_hv = False
            r_alt_hv = False
            r_alt_h_letters = set()
            for a in r:
                is_h = a.element.name in ('H', 'D')
                alt_s = a.altloc.strip() if a.altloc != '\0' else ''
                if not is_h and a.occ > 0:
                    n_pos_hv += 1
                    if alt_s:
                        all_alts.add(alt_s)
                        r_alt_hv = True
                    else:
                        r_blank_hv = True
                elif is_h and a.occ > 0 and alt_s:
                    r_alt_h_letters.add(alt_s)
                elif not is_h and alt_s:
                    r_alt_hv = True
            if r_blank_hv and not r_alt_hv and r_alt_h_letters:
                any_h_split_on_blank_hv = True
                all_alts.update(r_alt_h_letters)

    if n_pos_hv == 0:
        raise ValueError(f"{path_or_dir}: no non-water heavy atoms with occupancy > 0 found")

    if not all_alts and not any_h_split_on_blank_hv:
        al = []
        for ch in st[0]:
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                for a in r:
                    if a.occ > 0:
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), _atom_biso_from_gemmi(a)))
        return _finish([al], np.array([1.0], dtype=np.float64))

    use_letters = (mode == 'letters') or (mode == 'auto' and all_alts.issubset({'A', 'B'}))
    if use_letters:
        letters = sorted(all_alts)
        mem_lists = {L: [] for L in letters}
        occ_lists = {L: [] for L in letters}
        for ch in st[0]:
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                blank_atoms = []
                blank_hv = False
                any_alt_hv = False
                by_alt_raw = {}
                for a in r:
                    rec = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), _atom_biso_from_gemmi(a))
                    alt_s = a.altloc.strip() if a.altloc != '\0' else ''
                    is_h = a.element.name in ('H', 'D')
                    if not alt_s:
                        if a.occ > 0:
                            blank_atoms.append(rec)
                            if not is_h:
                                blank_hv = True
                    else:
                        if not is_h:
                            any_alt_hv = True
                        by_alt_raw.setdefault(alt_s, []).append((rec, a.occ, a.element.name))
                for L in letters:
                    mem_lists[L].extend(blank_atoms)
                    if L in by_alt_raw:
                        hv_occs = [occ for (_, occ, el) in by_alt_raw[L] if el not in ('H', 'D') and occ > 0]
                        if hv_occs:
                            mem_lists[L].extend([rec for (rec, occ, el) in by_alt_raw[L] if occ > 0 or el in ('H', 'D')])
                            occ_lists[L].extend(hv_occs)
                        elif blank_hv and not any_alt_hv:
                            h_occs = [occ for (_, occ, _) in by_alt_raw[L] if occ > 0]
                            if h_occs:
                                mem_lists[L].extend([rec for (rec, occ, _) in by_alt_raw[L] if occ > 0])
                                occ_lists[L].extend(h_occs)
        wts = np.array([np.mean(occ_lists[L]) if occ_lists[L] else 1.0 for L in letters], dtype=np.float64)
        wts = wts / wts.sum()
        return _finish([mem_lists[L] for L in letters], wts)

    if mode == 'auto' or mode.startswith(('reconnect', 'geometry')):
        occ_flex = 0
        for pfx in ('reconnect_flex', 'geometry_flex'):
            if mode.startswith(pfx):
                occ_flex = int(mode[len(pfx):] or '0')
        _, confs, _ = reconnect_pdb_to_conformers(
            path_or_dir, nconf=nconf, occ_flex=occ_flex, quotas='hamilton', rebuild_h=True, add_missing_h=True
        )
        raw_lists = [
            [(a['seqnum'], a['resname'], a['name'], a['elem'], list(a['pos']), float(np.clip(a['biso'], 0.5, 999.0)))
             for a in c]
            for c in confs
        ]
        wts = np.full(len(raw_lists), 1.0 / len(raw_lists), dtype=np.float64)
        return _finish(raw_lists, wts)

    quantiles = (np.arange(nconf, dtype=np.float64) + 0.5) / nconf
    mem_recs = [[] for _ in range(nconf)]
    for ch in st[0]:
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            blank_atoms = []
            blank_hv = False
            any_alt_hv = False
            by_alt_raw = {}
            for a in r:
                rec = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), _atom_biso_from_gemmi(a), a.occ)
                if a.altloc == '\0' or not a.altloc.strip():
                    if a.occ > 0:
                        blank_atoms.append(rec[:6])
                        if a.element.name not in ('H', 'D'):
                            blank_hv = True
                else:
                    if a.element.name not in ('H', 'D'):
                        any_alt_hv = True
                    by_alt_raw.setdefault(a.altloc.strip(), []).append(rec)
            by_alt = {}
            hv_weights = {}
            for L, recs in by_alt_raw.items():
                hv_occs = [x[6] for x in recs if x[3] not in ('H', 'D') and x[6] > 0]
                if hv_occs:
                    by_alt[L] = [x[:6] for x in recs if x[6] > 0 or x[3] in ('H', 'D')]
                    hv_weights[L] = float(np.mean(hv_occs))
                elif blank_hv and not any_alt_hv:
                    h_occs = [x[6] for x in recs if x[6] > 0]
                    if h_occs:
                        by_alt[L] = [x[:6] for x in recs if x[6] > 0]
                        hv_weights[L] = float(np.mean(h_occs))
            if not by_alt:
                for m in range(nconf):
                    mem_recs[m].extend(blank_atoms)
                continue
            letters = sorted(by_alt.keys())
            weights = np.array([hv_weights[L] for L in letters], dtype=np.float64)
            weights = weights / weights.sum()
            cum = np.cumsum(weights)
            for m, q in enumerate(quantiles):
                mem_recs[m].extend(blank_atoms)
                k = min(int(np.searchsorted(cum, q, side='right')), len(letters) - 1)
                mem_recs[m].extend(by_alt[letters[k]])
    wts = np.full(nconf, 1.0 / nconf, dtype=np.float64)
    return _finish(mem_recs, wts)


class MobilityFeatures:
    """Per-conformer protein anchor B_iso, nearest heavy-atom distance, and polar tether count on ASUGrid."""

    def __init__(self, d_min, B_prot, B_prot_tb32, N_tether):
        self.d_min = np.ascontiguousarray(d_min, dtype=np.float64)
        self.B_prot = np.ascontiguousarray(B_prot, dtype=np.float64)
        self.B_prot_tb32 = np.ascontiguousarray(B_prot_tb32, dtype=np.float64)
        self.N_tether = np.ascontiguousarray(N_tether, dtype=np.float64)


def reduce_mobility_features_to_asu(mob_feat: MobilityFeatures, mw):
    """Accessibility-weighted reduction of M conformers' anchor features to (1, n_asu)."""
    if mob_feat.d_min.shape[0] == 1:
        return mob_feat
    mw_col = np.asarray(mw, dtype=np.float64)[:, None]
    w_open = mw_col * (1e-3 + expit((mob_feat.d_min - 2.05) / 0.20))
    w_norm = w_open / np.sum(w_open, axis=0, keepdims=True)
    return MobilityFeatures(
        np.sum(w_norm * mob_feat.d_min, axis=0, keepdims=True),
        np.sum(w_norm * mob_feat.B_prot, axis=0, keepdims=True),
        np.sum(w_norm * mob_feat.B_prot_tb32, axis=0, keepdims=True),
        np.sum(w_norm * mob_feat.N_tether, axis=0, keepdims=True),
    )


def build_mobility_features(grid: ASUGrid, members, members_biso, two_type_heavy_only=False):
    """Precompute per-conformer mobility anchor features on ASUGrid from real heavy protein atoms."""
    M = len(members)
    n_asu = grid.n_asu
    pts = grid.asu_xyz
    tree_grid = cKDTree(pts)

    d_min = np.full((M, n_asu), 15.0, dtype=np.float64)
    B_prot = np.zeros((M, n_asu), dtype=np.float64)
    B_prot_tb32 = np.zeros((M, n_asu), dtype=np.float64)
    N_tether = np.zeros((M, n_asu), dtype=np.float64)

    for m, ((xyz, typ), biso) in enumerate(zip(members, members_biso)):
        if two_type_heavy_only:
            hv_mask = np.ones(len(typ), dtype=bool)
            pol_types = (typ == 0)
        else:
            hv_mask = (typ < 10) & (typ != 4) & (typ != 5) & (typ != 8)
            pol_types = (typ == 0) | (typ == 1) | (typ == 2) | (typ == 3)

        hv_xyz = xyz[hv_mask]
        hv_b = biso[hv_mask]
        hv_pol = pol_types[hv_mask]
        mean_b = float(np.mean(hv_b)) if len(hv_b) else 12.0

        SX, Si = grid.sym_expand(hv_xyz, cutoff_A=5.3)
        if len(SX) == 0:
            B_prot[m] = mean_b
            B_prot_tb32[m] = mean_b
            continue
        Sb = hv_b[Si]
        Spol = hv_pol[Si]
        tree = cKDTree(SX)

        d_nearest, idx_nearest = tree.query(pts)
        d_min[m] = d_nearest
        b_nearest = Sb[idx_nearest]

        nb = tree_grid.query_ball_point(SX, 5.2)
        lens = np.array([len(x) for x in nb], dtype=np.int32)
        if lens.sum() == 0:
            B_prot[m] = mean_b
            B_prot_tb32[m] = mean_b
            continue

        ai = np.repeat(np.arange(len(SX), dtype=np.int32), lens)
        pi = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb if len(x)])
        d = np.linalg.norm(pts[pi] - SX[ai], axis=1)
        bi = Sb[ai]
        poli = Spol[ai]

        w_c = np.exp(-np.maximum(d - 2.4, 0.0) / 0.65) * expit((3.8 - d) / 0.30)
        w_sum = np.bincount(pi, weights=w_c, minlength=n_asu)
        wb_sum = np.bincount(pi, weights=w_c * bi, minlength=n_asu)
        B_prot[m] = (wb_sum + 1e-3 * b_nearest) / (w_sum + 1e-3)

        w_tb = expit((3.2 - d) / 0.18)
        wtb_sum = np.bincount(pi, weights=w_tb, minlength=n_asu)
        wbtb_sum = np.bincount(pi, weights=w_tb * bi, minlength=n_asu)
        B_prot_tb32[m] = (wbtb_sum + 1e-3 * b_nearest) / (wtb_sum + 1e-3)

        if np.any(poli):
            dp = d[poli]
            pip = pi[poli]
            w_teth = expit((3.25 - dp) / 0.20) * expit((dp - 2.25) / 0.15)
            N_tether[m] = np.bincount(pip, weights=w_teth, minlength=n_asu)

    return MobilityFeatures(d_min, B_prot, B_prot_tb32, N_tether)


class SlimDatasetBundle:
    """Memory-efficient DatasetBundle keeping Phi_O / Phi_H in float32 (3x lower RAM on large ASU grids)."""

    def __init__(self, name, grid: ASUGrid, B_map, Phi_O, Phi_H=None, steric_H=None,
                 member_weights=None, F_hbsol=None, E_feats=None):
        self.name = name
        self.grid = grid
        self.B_map = float(B_map)
        self.Kw, self.Kh = grid.water_kernel(B_map)
        self.M = int(Phi_O.shape[0])
        if member_weights is None:
            self.mw = np.full(self.M, 1.0 / max(self.M, 1), dtype=np.float64)
        else:
            mw = np.asarray(member_weights, dtype=np.float64)
            self.mw = mw / mw.sum()
        self.Phi_O_f32 = np.ascontiguousarray(Phi_O, dtype=np.float32)
        self.Phi_O_2d_f32 = self.Phi_O_f32.reshape(self.M * grid.n_asu, -1)
        self.Phi_H_f32 = np.ascontiguousarray(Phi_H, dtype=np.float32) if Phi_H is not None else None
        self.Phi_H_2d_f32 = self.Phi_H_f32.reshape(self.M * grid.n_asu, -1) if self.Phi_H_f32 is not None else None
        self.steric_H_64 = np.ascontiguousarray(steric_H, dtype=np.float64) if steric_H is not None else None
        self.F_hbsol = F_hbsol
        self.E_feats_2d = (np.ascontiguousarray(E_feats, dtype=np.float64).reshape(self.M * grid.n_asu, 3)
                           if E_feats is not None else None)
        self.Phi_O_flat = None
        self.Phi_H_flat = None


def fast_build_asu_and_mobility(grid: ASUGrid, members, members_biso, n_types=10, include_hpol=True,
                                include_tet_oz=False, include_epol=False, charges=None,
                                member_weights=None, return_extra=False, max_workers=8):
    """Build `(Phi_O, Phi_H, steric_H, MobilityFeatures[, extra])` ~50x faster via `cKDTree(asu_xyz)`."""
    M = len(members)
    n_asu = grid.n_asu
    pts = grid.asu_xyz
    n_feat = int(n_types) * N_KNOTS
    nhf = len(ACCEPTOR_TYPES) * N_HKNOTS
    two_type = (int(n_types) == 2)
    vox_vol = float(grid.cell.volume) / float(grid.nvox)

    Phi_O = np.zeros((M, n_asu, n_feat), dtype=np.float32)
    Phi_H = np.zeros((M, n_asu, nhf), dtype=np.float32) if include_hpol else None
    steric_H = np.ones((M, n_asu), dtype=np.float32) if include_hpol else None
    E_feats = np.zeros((M, n_asu, 3), dtype=np.float32) if include_epol else None
    sol_hb_m = np.zeros((M, n_asu), dtype=np.float64) if include_tet_oz else None

    d_min = np.full((M, n_asu), 15.0, dtype=np.float64)
    B_prot = np.zeros((M, n_asu), dtype=np.float64)
    B_prot_tb32 = np.zeros((M, n_asu), dtype=np.float64)
    N_tether = np.zeros((M, n_asu), dtype=np.float64)

    tree_pts = cKDTree(pts)

    def _work_one(m):
        xyz, typ = members[m]
        biso = members_biso[m]
        if int(n_types) == 10 and np.any(typ >= 10):
            keep_use = typ < 10
            xyz_use, typ_use, biso_use = xyz[keep_use], typ[keep_use], biso[keep_use]
        else:
            xyz_use, typ_use, biso_use = xyz, typ, biso

        SX, Si = grid.sym_expand(xyz_use, cutoff_A=R_CUT)
        St = typ_use[Si]
        Sb = biso_use[Si]
        nb_a = tree_pts.query_ball_point(SX, R_CUT) if len(SX) else []
        lens = np.array([len(x) for x in nb_a], dtype=np.int32) if len(SX) else np.zeros(0, dtype=np.int32)
        if lens.sum() > 0:
            ai = np.repeat(np.arange(len(SX), dtype=np.int32), lens)
            pi = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb_a if len(x)])
            d = np.linalg.norm(pts[pi] - SX[ai], axis=1)
            ti = St[ai]

            buf = np.zeros(n_asu * n_feat, dtype=np.float64)
            for kk, bw in _bspline_weights(d, R_MIN, DR, N_KNOTS):
                valid = kk < N_KNOTS
                if np.any(valid):
                    flat_col = ti[valid] * N_KNOTS + kk[valid]
                    flat_idx = pi[valid].astype(np.int64) * n_feat + flat_col
                    buf += np.bincount(flat_idx, weights=bw[valid], minlength=n_asu * n_feat)
            Phi_O[m] = buf.reshape(n_asu, n_feat).astype(np.float32)

            if include_hpol:
                real_m = np.ones_like(ti, dtype=bool) if two_type else (ti < 10)
                if np.any(real_m):
                    pr, dr, tr = pi[real_m], d[real_m], ti[real_m]
                    is_h_atom = np.zeros_like(tr, dtype=bool) if two_type else ((tr == 4) | (tr == 5) | (tr == 8))
                    r_clash = np.where(is_h_atom, 1.25, 1.55)
                    close = dr < (r_clash + 0.80)
                    if np.any(close):
                        l_excl = -np.logaddexp(0.0, -(dr[close] - r_clash[close]) / 0.12)
                        steric_H[m] = np.exp(np.bincount(pr[close], weights=l_excl, minlength=n_asu)).astype(np.float32)

                    acc_mask = ((tr == 0) if two_type else ((tr == 0) | (tr == 1) | (tr == 2))) & (dr < RH_CUT)
                    if np.any(acc_mask):
                        pi_a, d_a, ti_a = pr[acc_mask], dr[acc_mask], tr[acc_mask]
                        buf_h = np.zeros(n_asu * nhf, dtype=np.float64)
                        for kk, bw in _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS):
                            valid = kk < N_HKNOTS
                            if np.any(valid):
                                flat_col = ti_a[valid] * N_HKNOTS + kk[valid]
                                flat_idx = pi_a[valid].astype(np.int64) * nhf + flat_col
                                buf_h += np.bincount(flat_idx, weights=bw[valid], minlength=n_asu * nhf)
                        Phi_H[m] = buf_h.reshape(n_asu, nhf).astype(np.float32)
        else:
            pi = np.zeros(0, dtype=np.int32)
            ai = np.zeros(0, dtype=np.int32)
            d = np.zeros(0, dtype=np.float64)
            ti = np.zeros(0, dtype=np.int32)

        real_atoms = np.ones(len(typ), dtype=bool) if two_type else (typ < 10)
        xyz_r = xyz[real_atoms]
        typ_r = typ[real_atoms]

        if include_tet_oz and len(xyz_r) > 0:
            pol_hv = (typ_r == 0) if two_type else ((typ_r == 0) | (typ_r == 1) | (typ_r == 2) | (typ_r == 3))
            if np.any(pol_hv):
                SP, _ = grid.sym_expand(xyz_r[pol_hv], cutoff_A=1.8)
                if len(SP) > 0:
                    nb_p = tree_pts.query_ball_point(SP, 1.5)
                    lp = np.array([len(x) for x in nb_p], dtype=np.int32)
                    if lp.sum() > 0:
                        ap = np.repeat(np.arange(len(SP), dtype=np.int32), lp)
                        pp = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb_p if len(x)])
                        dp = np.linalg.norm(pts[pp] - SP[ap], axis=1)
                        sig_b = 0.40
                        norm_vox = vox_vol / ((2.0 * math.pi * sig_b * sig_b) ** 1.5)
                        sol_hb_m[m] = np.bincount(pp, weights=norm_vox * np.exp(-0.5 * (dp / sig_b) ** 2), minlength=n_asu)

        if include_epol and len(xyz_r) > 0:
            if charges is not None and charges[m] is not None and len(charges[m]) >= len(typ):
                q_r = np.asarray(charges[m], dtype=np.float64)[real_atoms]
            elif charges is not None and charges[m] is not None and len(charges[m]) == len(xyz_r):
                q_r = np.asarray(charges[m], dtype=np.float64)
            else:
                q_r = _DEFAULT_CLASS_Q[np.clip(typ_r, 0, len(_DEFAULT_CLASS_Q) - 1)]
            ex_v, ey_v, ez_v, es_v, lg_v = _compute_epol_additive_terms(grid, tree_pts, xyz_r, typ_r, q_r)
            E_feats[m] = _epol_features_from_additive(ex_v, ey_v, ez_v, es_v, lg_v).astype(np.float32)

        if two_type:
            hv_atom = np.ones(len(typ_use), dtype=bool)
            hv_pair = d <= 5.2
            pol_pair = (ti == 0)
        else:
            hv_atom = (typ_use < 10) & (typ_use != 4) & (typ_use != 5) & (typ_use != 8)
            hv_pair = (ti < 10) & (ti != 4) & (ti != 5) & (ti != 8) & (d <= 5.2)
            pol_pair = (ti == 0) | (ti == 1) | (ti == 2) | (ti == 3)

        hv_xyz = xyz_use[hv_atom]
        hv_b = biso_use[hv_atom]
        SX_hv, Si_hv = grid.sym_expand(hv_xyz, cutoff_A=5.3)
        if len(SX_hv) == 0:
            mean_b = float(np.mean(hv_b)) if len(hv_b) else 12.0
            B_prot[m] = mean_b
            B_prot_tb32[m] = mean_b
            return
        Sb_hv = hv_b[Si_hv]
        d_nearest, idx_nearest = cKDTree(SX_hv).query(pts)
        d_min[m] = d_nearest
        b_nearest = Sb_hv[idx_nearest]

        if np.any(hv_pair):
            pi_h = pi[hv_pair]
            d_h = d[hv_pair]
            bi_h = Sb[ai[hv_pair]]
            pol_h = pol_pair[hv_pair]

            w_c = np.exp(-np.maximum(d_h - 2.4, 0.0) / 0.65) * expit((3.8 - d_h) / 0.30)
            w_sum = np.bincount(pi_h, weights=w_c, minlength=n_asu)
            wb_sum = np.bincount(pi_h, weights=w_c * bi_h, minlength=n_asu)
            B_prot[m] = (wb_sum + 1e-3 * b_nearest) / (w_sum + 1e-3)

            w_tb = expit((3.2 - d_h) / 0.18)
            wtb_sum = np.bincount(pi_h, weights=w_tb, minlength=n_asu)
            wbtb_sum = np.bincount(pi_h, weights=w_tb * bi_h, minlength=n_asu)
            B_prot_tb32[m] = (wbtb_sum + 1e-3 * b_nearest) / (wtb_sum + 1e-3)

            if np.any(pol_h):
                dp = d_h[pol_h]
                pip = pi_h[pol_h]
                w_teth = expit((3.25 - dp) / 0.20) * expit((dp - 2.25) / 0.15)
                N_tether[m] = np.bincount(pip, weights=w_teth, minlength=n_asu)
        else:
            B_prot[m] = b_nearest
            B_prot_tb32[m] = b_nearest

    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), M))) as ex:
        list(ex.map(_work_one, range(M)))

    mob_feat = MobilityFeatures(d_min, B_prot, B_prot_tb32, N_tether)
    if not return_extra and not include_tet_oz and not include_epol:
        return Phi_O, Phi_H, steric_H, mob_feat

    mw = np.asarray(member_weights if member_weights is not None else np.full(M, 1.0 / max(M, 1)), dtype=np.float64)
    mw = mw / mw.sum()
    F_hbsol = np.fft.rfftn(grid.asu_to_full_3d(mw @ sol_hb_m), axes=(0, 1, 2)) if include_tet_oz else None
    return Phi_O, Phi_H, steric_H, mob_feat, {'F_hbsol': F_hbsol, 'E_feats': E_feats}


class PhysicalBMobSpec:
    """6-parameter physical hydration mobility model B_phys^(m)(x; psi)."""

    def __init__(self, w_far=0.75, use_tb32_anchor=False):
        self.w_far = float(w_far)
        self.use_tb32_anchor = bool(use_tb32_anchor)
        self.n_params = 6

    @staticmethod
    def _inv_softplus(y):
        y = max(float(y), 1e-6)
        return float(np.log(np.expm1(y))) if y < 20.0 else float(y)

    def pack(self, alpha_prot=0.80, B0=2.5, dB_unteth=18.0, gamma_teth=0.90, B_far=120.0, d_far=4.5):
        return np.array([
            self._inv_softplus(alpha_prot),
            np.log(B0),
            np.log(dB_unteth),
            np.log(gamma_teth),
            np.log(B_far),
            float(d_far),
        ], dtype=np.float64)

    def unpack_physical(self, psi):
        psi = np.asarray(psi, dtype=np.float64)
        alpha_prot = float(softplus(psi[0], beta=1.0))
        B0 = float(np.exp(psi[1]))
        dB_unteth = float(np.exp(psi[2]))
        gamma_teth = float(np.exp(psi[3]))
        B_far = float(np.exp(psi[4]))
        d_far = float(psi[5])
        return alpha_prot, B0, dB_unteth, gamma_teth, B_far, d_far

    def bounds(self):
        return [
            (-5.0, 3.0),
            (np.log(0.5), np.log(80.0)),
            (np.log(0.05), np.log(300.0)),
            (np.log(0.05), np.log(5.0)),
            (np.log(2.0), np.log(1100.0)),
            (2.8, 8.5),
        ]

    def forward_with_pullback(self, mob_feat: MobilityFeatures, psi):
        alpha_prot, B0, dB_unteth, gamma_teth, B_far, d_far = self.unpack_physical(psi)
        Bp = mob_feat.B_prot_tb32 if self.use_tb32_anchor else mob_feat.B_prot
        Nt = mob_feat.N_tether
        dm = mob_feat.d_min

        p_near = expit((d_far - dm) / self.w_far)
        E_teth = np.exp(-gamma_teth * Nt)
        B_near = alpha_prot * Bp + B0 + dB_unteth * E_teth
        B_phys = p_near * B_near + (1.0 - p_near) * B_far
        z_phys = np.log(np.maximum(B_phys, 1e-6))

        sig_a = float(expit(psi[0]))
        w_far = self.w_far

        def pullback(dL_dz_phys):
            dL_dB = dL_dz_phys / B_phys
            dL_dBnear = dL_dB * p_near
            g = np.zeros(6, dtype=np.float64)
            g[0] = sig_a * float(np.sum(dL_dBnear * Bp))
            g[1] = B0 * float(np.sum(dL_dBnear))
            g[2] = dB_unteth * float(np.sum(dL_dBnear * E_teth))
            g[3] = -gamma_teth * dB_unteth * float(np.sum(dL_dBnear * Nt * E_teth))
            g[4] = B_far * float(np.sum(dL_dB * (1.0 - p_near)))
            g[5] = (1.0 / w_far) * float(np.sum(dL_dB * (B_near - B_far) * p_near * (1.0 - p_near)))
            return g

        return z_phys, B_phys, pullback


class SpatialSmoother3D:
    """Exact crystal-symmetric Gaussian spatial smoother on ASUGrid."""

    def __init__(self, grid: ASUGrid, fwhm_A=3.2, fft_workers=None):
        self.grid = grid
        self.fwhm_A = float(fwhm_A)
        self.fft_workers = int(fft_workers) if fft_workers is not None else max(1, min(8, int(os.environ.get('OMP_NUM_THREADS', '4'))))
        if self.fwhm_A > 0.0:
            sigma = self.fwhm_A / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            B_smooth = 8.0 * (np.pi ** 2) * (sigma ** 2)
            self.G_k = np.exp(-B_smooth * grid.stol2)
        else:
            self.G_k = None

    def forward(self, eta_asu):
        eta_asu = np.asarray(eta_asu, dtype=np.float64)
        if self.G_k is None:
            return eta_asu.copy()
        x_3d = self.grid.asu_to_full_3d(eta_asu)
        y_3d = sp_fft.irfftn(self.G_k * sp_fft.rfftn(x_3d, axes=(0, 1, 2), workers=self.fft_workers),
                             s=self.grid.shape, axes=(0, 1, 2), workers=self.fft_workers)
        return y_3d.ravel()[self.grid.asu_reps]

    def adjoint(self, dL_ddz_asu):
        dL_ddz_asu = np.asarray(dL_ddz_asu, dtype=np.float64)
        if self.G_k is None:
            return dL_ddz_asu.copy()
        u_3d = self.grid.asu_to_full_3d(dL_ddz_asu / self.grid.orbit_mult)
        v_3d = sp_fft.irfftn(self.G_k * sp_fft.rfftn(u_3d, axes=(0, 1, 2), workers=self.fft_workers),
                             s=self.grid.shape, axes=(0, 1, 2), workers=self.fft_workers)
        return v_3d.ravel()[self.grid.asu_reps] * self.grid.orbit_mult


class MultiBRenderer:
    """Precomputes the 13-level water scattering kernels on ASUGrid and provides fast forward + adjoint."""

    def __init__(self, grid: ASUGrid, fft_workers=None):
        self.grid = grid
        self.K = N_BLEVELS
        self.B_levels = B_LEVELS.copy()
        self.fft_workers = int(fft_workers) if fft_workers is not None else max(1, min(8, int(os.environ.get('OMP_NUM_THREADS', '4'))))
        dw = np.exp(-self.B_levels[:, None, None, None] * grid.stol2[None, :, :, :])
        self.Kw_stack = grid.f_water_0[None, :, :, :] * dw
        self.Kh_stack = grid.f_H_0[None, :, :, :] * dw

    @staticmethod
    def _softclip_z(z_raw):
        sp_lo = softplus(z_raw - Z_CLIP_LO, beta=BETA_CLIP)
        sp_hi = softplus(z_raw - Z_CLIP_HI, beta=BETA_CLIP)
        z = Z_CLIP_LO + sp_lo - sp_hi
        dz_draw = expit(BETA_CLIP * (z_raw - Z_CLIP_LO)) - expit(BETA_CLIP * (z_raw - Z_CLIP_HI))
        return z, dz_draw

    def render_and_pullback(self, gO_m, gH_m, mw, z_raw_m, scale_k=1.0, compute_grad=True):
        grid = self.grid
        M, n_asu = gO_m.shape
        K = self.K
        use_hpol = gH_m is not None
        w_fft = self.fft_workers

        collapse_m = (z_raw_m.shape[0] == 1) and (M > 1)
        if collapse_m:
            gO_eff = (mw @ gO_m)[None, :]
            gH_eff = ((mw @ gH_m)[None, :]) if use_hpol else None
            mw_eff = np.array([1.0], dtype=np.float64)
            M_eff = 1
        else:
            gO_eff = gO_m
            gH_eff = gH_m
            mw_eff = mw
            M_eff = M

        z, dz_draw = self._softclip_z(z_raw_m)
        s = (z - Z0_BASIS) / DZ_BASIS
        i = np.clip(np.floor(s).astype(np.int32), 1, K - 3)
        f = s - i
        f2 = f * f
        f3 = f2 * f

        wm1 = 0.5 * (-f3 + 2.0 * f2 - f)
        w0 = 0.5 * (3.0 * f3 - 5.0 * f2 + 2.0)
        w1 = 0.5 * (-3.0 * f3 + 4.0 * f2 + f)
        w2 = 0.5 * (f3 - f2)

        mw_col = mw_eff[:, None]
        mw_gO = mw_col * gO_eff
        mw_gH = (mw_col * gH_eff) if use_hpol else None

        k_lo = int(i.min()) - 1
        k_hi = int(i.max()) + 2
        active_ks = np.arange(k_lo, k_hi + 1, dtype=np.int32)

        if M_eff == 1:
            ar = np.arange(n_asu, dtype=np.intp)
            i0 = i[0]
            uO_asu = np.zeros((K, n_asu), dtype=np.float64)
            uO_asu[i0 - 1, ar] += mw_gO[0] * wm1[0]
            uO_asu[i0,     ar] += mw_gO[0] * w0[0]
            uO_asu[i0 + 1, ar] += mw_gO[0] * w1[0]
            uO_asu[i0 + 2, ar] += mw_gO[0] * w2[0]
            if use_hpol:
                uH_asu = np.zeros((K, n_asu), dtype=np.float64)
                uH_asu[i0 - 1, ar] += mw_gH[0] * wm1[0]
                uH_asu[i0,     ar] += mw_gH[0] * w0[0]
                uH_asu[i0 + 1, ar] += mw_gH[0] * w1[0]
                uH_asu[i0 + 2, ar] += mw_gH[0] * w2[0]
            else:
                uH_asu = None
            flat_indices = None
        else:
            weights = ((-1, wm1), (0, w0), (1, w1), (2, w2))
            a_idx = np.broadcast_to(np.arange(n_asu, dtype=np.int64)[None, :], (M_eff, n_asu))
            uO_flat = np.zeros(K * n_asu, dtype=np.float64)
            uH_flat = np.zeros(K * n_asu, dtype=np.float64) if use_hpol else None
            flat_indices = []
            for off, wr in weights:
                fidx = ((i + off).astype(np.int64) * n_asu + a_idx).ravel()
                flat_indices.append(fidx)
                uO_flat += np.bincount(fidx, weights=(mw_gO * wr).ravel(), minlength=K * n_asu)
                if use_hpol:
                    uH_flat += np.bincount(fidx, weights=(mw_gH * wr).ravel(), minlength=K * n_asu)
            uO_asu = uO_flat.reshape(K, n_asu)
            uH_asu = uH_flat.reshape(K, n_asu) if use_hpol else None

        F_elec = np.zeros(grid.stol2.shape, dtype=np.complex128)
        for k in active_ks:
            if np.any(uO_asu[k]):
                uO_3d = grid.asu_to_full_3d(grid.n_bulk * uO_asu[k])
                F_elec += self.Kw_stack[k] * sp_fft.rfftn(uO_3d, axes=(0, 1, 2), workers=w_fft)
            if use_hpol and np.any(uH_asu[k]):
                uH_3d = grid.asu_to_full_3d(grid.n_bulk * uH_asu[k])
                F_elec += self.Kh_stack[k] * sp_fft.rfftn(uH_3d, axes=(0, 1, 2), workers=w_fft)

        rho_3d = sp_fft.irfftn(F_elec, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
        g_hat_base = rho_3d.ravel()[grid.asu_reps] / RHO_BULK
        g_hat = float(scale_k) * g_hat_base

        gO_sum = np.maximum(mw_eff @ gO_eff, 1e-8)
        B_eff_asu = np.sum(mw_gO * np.exp(z), axis=0) / gO_sum

        if not compute_grad:
            return g_hat, B_eff_asu, None

        inv_dz = 0.5 / DZ_BASIS
        dwm1 = inv_dz * (-3.0 * f2 + 4.0 * f - 1.0)
        dw0 = inv_dz * (9.0 * f2 - 10.0 * f)
        dw1 = inv_dz * (-9.0 * f2 + 8.0 * f + 1.0)
        dw2 = inv_dz * (3.0 * f2 - 2.0 * f)

        def pullback(dL_dghat):
            dL_dghat = np.asarray(dL_dghat, dtype=np.float64)
            dL_dlogk = float(np.sum(dL_dghat * g_hat))
            dL_dbase = dL_dghat * float(scale_k)

            g_rho_3d = grid.asu_to_full_3d((dL_dbase / RHO_BULK) / grid.orbit_mult)
            F_grho = sp_fft.rfftn(g_rho_3d, axes=(0, 1, 2), workers=w_fft)

            dL_duO_asu = np.zeros((K, n_asu), dtype=np.float64)
            dL_duH_asu = np.zeros((K, n_asu), dtype=np.float64) if use_hpol else None
            for k in active_ks:
                grad_uO_3d = grid.n_bulk * sp_fft.irfftn(self.Kw_stack[k] * F_grho, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
                dL_duO_asu[k] = grad_uO_3d.ravel()[grid.asu_reps] * grid.orbit_mult
                if use_hpol:
                    grad_uH_3d = grid.n_bulk * sp_fft.irfftn(self.Kh_stack[k] * F_grho, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
                    dL_duH_asu[k] = grad_uH_3d.ravel()[grid.asu_reps] * grid.orbit_mult

            if M_eff == 1:
                ar = np.arange(n_asu, dtype=np.intp)
                i0 = i[0]
                v0, v1, v2, v3 = dL_duO_asu[i0 - 1, ar], dL_duO_asu[i0, ar], dL_duO_asu[i0 + 1, ar], dL_duO_asu[i0 + 2, ar]
                dO_0 = wm1[0] * v0 + w0[0] * v1 + w1[0] * v2 + w2[0] * v3
                dz_0 = mw_gO[0] * (dwm1[0] * v0 + dw0[0] * v1 + dw1[0] * v2 + dw2[0] * v3)
                if use_hpol:
                    vh0, vh1, vh2, vh3 = dL_duH_asu[i0 - 1, ar], dL_duH_asu[i0, ar], dL_duH_asu[i0 + 1, ar], dL_duH_asu[i0 + 2, ar]
                    dH_0 = wm1[0] * vh0 + w0[0] * vh1 + w1[0] * vh2 + w2[0] * vh3
                    dz_0 = dz_0 + mw_gH[0] * (dwm1[0] * vh0 + dw0[0] * vh1 + dw1[0] * vh2 + dw2[0] * vh3)
                else:
                    dH_0 = None
                dL_dz_raw = dz_0[None, :] * dz_draw
                if collapse_m:
                    dL_dgO_m = np.outer(mw, dO_0)
                    dL_dgH_m = np.outer(mw, dH_0) if use_hpol else None
                    return dL_dgO_m, dL_dgH_m, dL_dz_raw, dL_dlogk
                return (mw_col * dO_0[None, :]), ((mw_col * dH_0[None, :]) if use_hpol else None), dL_dz_raw, dL_dlogk

            dL_duO_flat = dL_duO_asu.ravel()
            dL_duH_flat = dL_duH_asu.ravel() if use_hpol else None

            dL_dgO_m = np.zeros((M_eff, n_asu), dtype=np.float64)
            dL_dgH_m = np.zeros((M_eff, n_asu), dtype=np.float64) if use_hpol else None
            dL_dz = np.zeros((M_eff, n_asu), dtype=np.float64)
            weights = ((-1, wm1), (0, w0), (1, w1), (2, w2))
            dweights = (dwm1, dw0, dw1, dw2)

            for fidx, (_, wr), dwr in zip(flat_indices, weights, dweights):
                vO = dL_duO_flat[fidx].reshape(M_eff, n_asu)
                dL_dgO_m += mw_col * wr * vO
                dL_dz += mw_gO * dwr * vO
                if use_hpol:
                    vH = dL_duH_flat[fidx].reshape(M_eff, n_asu)
                    dL_dgH_m += mw_col * wr * vH
                    dL_dz += mw_gH * dwr * vH

            dL_dz_raw = dL_dz * dz_draw
            return dL_dgO_m, dL_dgH_m, dL_dz_raw, dL_dlogk

        return g_hat, B_eff_asu, pullback


def build_water_keep_mask_asu(grid: ASUGrid, base_g_asu, wat_xyz, wat_occ, rex_override=None):
    """Solve electron-conserving ordered-water exclusion mask keep_asu in [0, 1] on ASUGrid."""
    if wat_xyz is None or len(wat_xyz) == 0:
        return np.ones(grid.n_asu, dtype=np.float64), 0.0
    vox = grid.cell.volume / float(grid.nvox)
    e_ord = 8.0 * float(np.sum(wat_occ)) * len(grid.ops)
    total_asu = np.maximum(base_g_asu, 0.0) * RHO_BULK
    mult_vox = grid.orbit_mult * vox
    tot_elec = float(np.sum(total_asu * mult_vox))
    target_elec = max(0.0, tot_elec - e_ord)

    SW, Swi = grid.sym_expand(wat_xyz, cutoff_A=3.35)
    tg = cKDTree(grid.asu_xyz)
    nb = tg.query_ball_point(SW, 3.3)
    lens = np.array([len(x) for x in nb], dtype=np.int32)
    if lens.sum() == 0:
        return np.ones(grid.n_asu, dtype=np.float64), 0.0
    gi = np.concatenate([np.array(x, dtype=np.int32) for x in nb if len(x)])
    wi = np.repeat(np.arange(len(SW), dtype=np.int32), lens)
    dw = np.linalg.norm(grid.asu_xyz[gi] - SW[wi], axis=1)
    ow = wat_occ[Swi[wi]]

    def keep_of(rex):
        occ_hole = np.bincount(gi, ow / (1.0 + np.exp(-(rex - dw) / 0.15)), minlength=grid.n_asu)
        return np.clip(1.0 - occ_hole, 0.0, 1.0)

    if rex_override is not None:
        rex = float(rex_override)
    else:
        lo, hi = 0.50, 3.10
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if float(np.sum(total_asu * keep_of(mid) * mult_vox)) > target_elec:
                lo = mid
            else:
                hi = mid
        rex = 0.5 * (lo + hi)
    return keep_of(rex), rex


def free_mask_from_flags(flags, valid, free_flag_value=None, source_desc="R-free flags"):
    """Return the test-set mask from an R-free flag column.

    `free_flag_value` given: test = flags == value. Otherwise 0/1 flags take the minority value as test (phenix
    writes test = 1, but a 0/1 file with test = 0 is also read correctly), and multi-valued 0..N-1 flags take 0
    (CCP4 convention; pass `free_flag_value` for files that mark test with another value).
    """
    fl = np.asarray(flags, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(fl)
    fl_i = np.where(np.isfinite(fl), np.round(fl), -1).astype(np.int64)
    if free_flag_value is not None:
        return ok & (fl_i == int(free_flag_value))
    vals, counts = np.unique(fl_i[ok], return_counts=True)
    if len(vals) == 0:
        return np.zeros(len(fl_i), dtype=bool)
    if set(vals.tolist()).issubset({0, 1}) and len(vals) == 2:
        test_val = int(vals[int(np.argmin(counts))])
        if counts[0] == counts[1]:
            test_val = 1
        if test_val == 0:
            print(f"surfnturf: {source_desc}: 0/1 flags with 0 in the minority; treating flag 0 as the test set",
                  file=sys.stderr)
        return ok & (fl_i == test_val)
    if len(vals) == 1:
        print(f"surfnturf: {source_desc}: all flags equal {int(vals[0])}; no test set", file=sys.stderr)
        return np.zeros(len(fl_i), dtype=bool)
    frac0 = float(np.mean(fl_i[ok] == 0))
    if frac0 > 0.5:
        print(f"surfnturf: {source_desc}: {100 * frac0:.0f}% of reflections have flag 0, which would be the test "
              f"set under the CCP4 convention; pass the test-set flag value explicitly", file=sys.stderr)
    return ok & (fl_i == 0)


def canonical_hkl_keys(hkl, spacegroup=None):
    """Integer key per reflection, identical for all symmetry (and Friedel) mates under `spacegroup`'s Laue group."""
    H = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    rots = [np.eye(3, dtype=np.int64)]
    if spacegroup is not None:
        rots = [np.array(op.rot, dtype=np.int64) // op.DEN for op in spacegroup.operations().sym_ops]
    best = None
    for R in rots:
        Hr = H @ R
        for sgn in (1, -1):
            Hs = sgn * Hr
            key = ((Hs[:, 0] + 2048) << 24) | ((Hs[:, 1] + 2048) << 12) | (Hs[:, 2] + 2048)
            best = key if best is None else np.maximum(best, key)
    return best


def _hkl_hash_mod(keys, mod):
    """Deterministic splitmix64 hash of integer reflection keys, reduced modulo `mod`."""
    with np.errstate(over='ignore'):
        z = np.asarray(keys, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(mod)).astype(np.int64)


def make_work_train_val_split(hkl, d_hkl, free_mask, data_mtz_path: str = None, free_flag_col: str = 'FreeR_flag',
                              spacegroup=None):
    """Deterministic 90% work_train / 5% work_val split strictly disjoint from free_mask.

    With a data MTZ carrying 0..19 FreeR flags (and flag 1 disjoint from the free set), work_val = flag 1.
    Otherwise each work reflection is assigned by a hash of its symmetry-canonical Miller index (1 in 19 to
    work_val), so the split depends only on the reflection itself: it is the same in every program, in every
    file of the same data, and does not change when other reflections are added or removed. `d_hkl` is unused
    and kept for call compatibility.
    """
    H = np.asarray(hkl, dtype=np.int64)
    free_v = np.asarray(free_mask, dtype=bool)
    work_v = ~free_v
    keys = canonical_hkl_keys(H, spacegroup)

    if data_mtz_path is not None:
        if not os.path.isfile(data_mtz_path):
            print(f"surfnturf: data MTZ {data_mtz_path!r} not found; using the hashed work_val split",
                  file=sys.stderr)
        else:
            dm = gemmi.read_mtz_file(data_mtz_path)
            col_fl = dm.column_with_label(free_flag_col)
            reason = None
            if col_fl is None:
                reason = f"no {free_flag_col!r} column"
            else:
                dm_keys = canonical_hkl_keys(np.asarray(dm.make_miller_array(), dtype=np.int64), spacegroup)
                dm_fl = np.asarray(col_fl.array, dtype=np.float64)
                fl_lookup = {int(k): int(round(float(v))) for k, v in zip(dm_keys, dm_fl) if np.isfinite(v)}
                orig_flags = np.array([fl_lookup.get(int(k), -1) for k in keys], dtype=int)
                if len(np.unique(orig_flags[orig_flags >= 0])) < 10:
                    reason = "fewer than 10 distinct flag values"
                elif not (0.025 < float(np.mean(orig_flags == 1)) < 0.085):
                    reason = f"flag 1 covers {100 * float(np.mean(orig_flags == 1)):.1f}% of reflections"
                elif np.any((orig_flags == 1) & free_v):
                    reason = f"flag 1 overlaps the free set on {int(np.sum((orig_flags == 1) & free_v))} reflections"
                else:
                    val_v = (orig_flags == 1) & work_v
                    train_v = work_v & (~val_v)
                    return train_v, val_v
            print(f"surfnturf: {data_mtz_path}: cannot take work_val from {free_flag_col} ({reason}); "
                  f"using the hashed work_val split", file=sys.stderr)

    if not np.any(work_v):
        return np.zeros_like(work_v), np.zeros_like(work_v)
    h19 = _hkl_hash_mod(keys, 19)
    val_v = work_v & (h19 == 9)
    if not np.any(val_v) and int(np.sum(work_v)) >= 2:
        work_idx = np.nonzero(work_v)[0]
        val_v[work_idx[int(np.argmin(_hkl_hash_mod(keys[work_idx], 2 ** 31)))]] = True
    train_v = work_v & (~val_v)
    if not np.any(train_v):
        train_v = work_v.copy()
        val_v = np.zeros(len(free_v), dtype=bool)
    return train_v, val_v


def resolution_scaled_bmob_hyperparams(d_min: float, fwhm_A='auto', lam_z='auto',
                                       base_fwhm_A: float = 3.2, base_lam_z: float = 2e-4):
    """Return resolution-scaled `(fwhm_eff, lam_z_eff)` preserving 3.2 A / 2e-4 at sub-angstrom d_min <= 1.0 A."""
    d = max(float(d_min), 0.40)
    if fwhm_A is None or (isinstance(fwhm_A, str) and fwhm_A.strip().lower() == 'auto'):
        fwhm_eff = max(float(base_fwhm_A), round(3.5 * d, 2)) if d > 1.00 else float(base_fwhm_A)
    else:
        fwhm_eff = float(fwhm_A)
    if lam_z is None or (isinstance(lam_z, str) and lam_z.strip().lower() == 'auto'):
        lam_eff = float(base_lam_z) * max(1.0, (d / 0.965) ** 3)
    else:
        lam_eff = float(lam_z)
    return fwhm_eff, lam_eff


class ReciprocalTarget:
    """Holds work/train/val/free reflections, fixed atomic F_calc, Phenix k_total, and bins for |Fo| fitting."""

    def __init__(self, hkl, f_obs, work_mask, free_mask, f_calc, k_tot, bin_idx, n_bins, d_hkl,
                 train_mask=None, val_mask=None, f_model_ref=None):
        self.hkl = np.ascontiguousarray(hkl, dtype=np.int64)
        self.f_obs = np.ascontiguousarray(f_obs, dtype=np.float64)
        self.work_mask = np.asarray(work_mask, dtype=bool)
        self.free_mask = np.asarray(free_mask, dtype=bool)
        assert not np.any(self.work_mask & self.free_mask), "FREE SET LEAK IN WORK_MASK!"
        self.f_calc = np.ascontiguousarray(f_calc, dtype=np.complex128)
        self.k_tot = np.ascontiguousarray(k_tot, dtype=np.float64)
        self.bin_idx = np.ascontiguousarray(bin_idx, dtype=np.int32)
        self.n_bins = int(n_bins)
        self.d_hkl = np.ascontiguousarray(d_hkl, dtype=np.float64)
        if train_mask is None or val_mask is None:
            tr_m, vl_m = make_work_train_val_split(self.hkl, self.d_hkl, self.free_mask)
            self.train_mask = tr_m & self.work_mask
            self.val_mask = vl_m & (~self.free_mask)
        else:
            self.train_mask = np.asarray(train_mask, dtype=bool) & (~self.free_mask)
            self.val_mask = np.asarray(val_mask, dtype=bool) & (~self.free_mask)
        self.f_model_ref = np.asarray(f_model_ref) if f_model_ref is not None else None
        self.work_fo2_sum = float(np.sum(self.f_obs[self.work_mask] ** 2))
        self.work_fo_sum = float(np.sum(self.f_obs[self.work_mask]))
        self.free_fo_sum = float(np.sum(self.f_obs[self.free_mask]))

    def for_training_subset(self) -> 'ReciprocalTarget':
        """Return a ReciprocalTarget whose active `work_mask` is restricted to `train_mask` (90%), holding `val_mask` unseen."""
        use_tr = self.train_mask if np.any(self.train_mask) else self.work_mask
        return ReciprocalTarget(
            hkl=self.hkl, f_obs=self.f_obs, work_mask=use_tr, free_mask=self.free_mask,
            f_calc=self.f_calc, k_tot=self.k_tot, bin_idx=self.bin_idx, n_bins=self.n_bins,
            d_hkl=self.d_hkl, train_mask=use_tr, val_mask=self.val_mask, f_model_ref=self.f_model_ref,
        )

    def eval_blind_rval(self, f_part, kmask_mode: str = 'per_bin', k_bounds=(0.0, 3.0)):
        """Fit solvent scales strictly on `train_mask` and return `(r_train, r_val, r_work, r_free)`."""
        tr = self.train_mask if np.any(self.train_mask) else self.work_mask
        vl = self.val_mask
        wk = self.work_mask | vl
        fr = self.free_mask

        if kmask_mode == 'fixed_one':
            u_tot = self.k_tot * (self.f_calc + f_part)
            au = np.abs(u_tot)
            denom = float(np.sum(au[tr] ** 2))
            k0 = float(np.sum(au[tr] * self.f_obs[tr]) / max(denom, 1e-12))
            afm = np.abs(k0 * u_tot)
        else:
            _, k_hkl = fit_kmask_per_bin(
                self.f_calc, f_part, self.k_tot, self.f_obs, tr,
                self.bin_idx, self.n_bins, loss_type='l2', k_bounds=k_bounds
            )
            afm = np.abs(self.k_tot * (self.f_calc + k_hkl * f_part))

        def _r(sel):
            s = float(np.sum(self.f_obs[sel])) if np.any(sel) else 0.0
            return float(np.sum(np.abs(afm[sel] - self.f_obs[sel])) / s) if s > 0.0 else float('nan')

        return _r(tr), _r(vl), _r(wk), _r(fr)

    def eval_fmodel_ref_rval(self) -> float:
        """Evaluate the input `*_f_model.mtz` amplitude agreement on unseen `val_mask`."""
        if self.f_model_ref is None or not np.any(self.val_mask):
            return float('nan')
        denom = float(np.sum(self.f_obs[self.val_mask]))
        if denom <= 0.0:
            return float('nan')
        return float(np.sum(np.abs(np.abs(self.f_model_ref[self.val_mask]) - self.f_obs[self.val_mask])) / denom)

    def eval_loss_grad_and_r(self, f_part, k_bounds=(0.0, 3.0), kmask_mode: str = 'per_bin'):
        w = self.work_mask
        fr = self.free_mask
        inv_norm = 1.0 / max(self.work_fo2_sum, 1e-12)

        if kmask_mode == 'fixed_one':
            u_tot = self.k_tot * (self.f_calc + f_part)
            au = np.abs(u_tot)
            denom = float(np.sum(au[w] ** 2))
            k0 = float(np.sum(au[w] * self.f_obs[w]) / max(denom, 1e-12))
            fm = k0 * u_tot
            afm = np.maximum(np.abs(fm), 1e-12)
            res = afm - self.f_obs
            loss = 0.5 * float(np.sum(res[w] ** 2)) * inv_norm
            dL_dF = np.zeros_like(f_part, dtype=np.complex128)
            dL_dF[w] = (res[w] * inv_norm) * (k0 * self.k_tot[w]) * (fm[w] / afm[w])
            r_work = float(np.sum(np.abs(res[w])) / max(self.work_fo_sum, 1e-12))
            r_free = float(np.sum(np.abs(res[fr])) / self.free_fo_sum) if np.any(fr) and self.free_fo_sum > 0 else float('nan')
            return loss, dL_dF, r_work, r_free, np.full(self.n_bins, 1.0, dtype=np.float64)

        k_bins, k_hkl = fit_kmask_per_bin(
            self.f_calc, f_part, self.k_tot, self.f_obs, w,
            self.bin_idx, self.n_bins, loss_type='l2', k_bounds=k_bounds
        )
        fm = self.k_tot * (self.f_calc + k_hkl * f_part)
        afm = np.maximum(np.abs(fm), 1e-12)
        res = afm - self.f_obs
        loss = 0.5 * float(np.sum(res[w] ** 2)) * inv_norm
        dL_dF = np.zeros_like(f_part, dtype=np.complex128)
        dL_dF[w] = (res[w] * inv_norm) * self.k_tot[w] * k_hkl[w] * (fm[w] / afm[w])

        r_work = float(np.sum(np.abs(res[w])) / max(self.work_fo_sum, 1e-12))
        r_free = float(np.sum(np.abs(res[fr])) / self.free_fo_sum) if np.any(fr) and self.free_fo_sum > 0 else float('nan')
        return loss, dL_dF, r_work, r_free, k_bins


def multiB_fpart_loss_and_grad(
    renderer: MultiBRenderer,
    target: ReciprocalTarget,
    ds: DatasetBundle,
    z_raw_m,
    spec: WaterFieldSpec = None,
    theta=None,
    cached_pmf=None,
    displace_mode='none',
    keep_asu=None,
    clamp_nonneg=True,
    compute_grad=True,
    use_f32_blas=True,
    kmask_mode='per_bin',
):
    """Evaluate reciprocal-space work-set loss, R_work/R_free, solv_3d, and exact analytical gradients."""
    grid = ds.grid
    need_theta_grad = compute_grad and (cached_pmf is None)

    if cached_pmf is not None:
        gO_m_raw, gH_m_raw, scale_k = cached_pmf
        pb_theta = None
    else:
        gO_m_raw, gH_m_raw, scale_k, pb_theta = compute_pmf_density_with_pullback(
            spec, ds, theta, ds_index=0, need_grad=need_theta_grad, use_f32_blas=use_f32_blas
        )

    if displace_mode == 'pre' and keep_asu is not None:
        gO_m = gO_m_raw * keep_asu[None, :]
        gH_m = (gH_m_raw * keep_asu[None, :]) if gH_m_raw is not None else None
    else:
        gO_m = gO_m_raw
        gH_m = gH_m_raw

    g_hat, B_eff_asu, pb_render = renderer.render_and_pullback(
        gO_m, gH_m, ds.mw, z_raw_m, scale_k=scale_k, compute_grad=compute_grad
    )

    if displace_mode == 'post' and keep_asu is not None:
        g_eff_asu = g_hat * keep_asu
    else:
        g_eff_asu = g_hat

    rho_3d = grid.asu_to_full_3d(g_eff_asu) * RHO_BULK
    solv_3d = np.maximum(rho_3d, 0.0) if clamp_nonneg else rho_3d
    f_part = solv_to_fpart(solv_3d, grid.cell.volume, target.hkl)
    loss, dL_dF, r_work, r_free, k_bins = target.eval_loss_grad_and_r(f_part, kmask_mode=kmask_mode)

    if not compute_grad:
        return {
            'loss': loss, 'r_work': r_work, 'r_free': r_free,
            'solv_3d': solv_3d, 'f_part': f_part, 'B_eff_asu': B_eff_asu, 'k_bins': k_bins,
        }

    dL_dg_eff = fpart_grad_to_asu(grid, g_eff_asu, target.hkl, dL_dF, clamp_nonneg=clamp_nonneg)
    dL_dghat = (dL_dg_eff * keep_asu) if (displace_mode == 'post' and keep_asu is not None) else dL_dg_eff

    dL_dgO_m, dL_dgH_m, dL_dz_raw_m, dL_dlogk = pb_render(dL_dghat)

    grad_theta = None
    if need_theta_grad and pb_theta is not None:
        if displace_mode == 'pre' and keep_asu is not None:
            dL_dgO_raw = dL_dgO_m * keep_asu[None, :]
            dL_dgH_raw = (dL_dgH_m * keep_asu[None, :]) if dL_dgH_m is not None else None
        else:
            dL_dgO_raw = dL_dgO_m
            dL_dgH_raw = dL_dgH_m
        grad_theta = pb_theta(dL_dgO_raw, dL_dgH_raw, dL_dlogk)

    return {
        'loss': loss,
        'r_work': r_work,
        'r_free': r_free,
        'grad_z_raw_m': dL_dz_raw_m,
        'grad_theta': grad_theta,
        'solv_3d': solv_3d,
        'f_part': f_part,
        'B_eff_asu': B_eff_asu,
        'k_bins': k_bins,
    }


def fit_uniform_B_on_work(renderer, target, ds, cached_pmf, displace_mode='none', keep_asu=None,
                          kmask_mode='per_bin', maxiter=35):
    """Fit 1 scalar uniform B_const on work |Fo| via L-BFGS-B."""
    def obj(x):
        z_arr = np.full((1, ds.grid.n_asu), float(x[0]), dtype=np.float64)
        out = multiB_fpart_loss_and_grad(
            renderer, target, ds, z_arr, cached_pmf=cached_pmf,
            displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=True, kmask_mode=kmask_mode
        )
        return out['loss'], np.array([float(np.sum(out['grad_z_raw_m']))], dtype=np.float64)

    best_z = np.log(5.0)
    best_l = np.inf
    for b_try in (2.5, 5.0, 10.0, 20.0, 40.0, 80.0):
        l_try, _ = obj([np.log(b_try)])
        if l_try < best_l:
            best_l = l_try
            best_z = np.log(b_try)

    res = minimize(obj, np.array([best_z]), method='L-BFGS-B', jac=True,
                   bounds=[(np.log(1.6), np.log(999.0))], options={'maxiter': int(maxiter), 'ftol': 1e-10})
    z_opt = float(res.x[0])
    out = multiB_fpart_loss_and_grad(
        renderer, target, ds, np.full((1, ds.grid.n_asu), z_opt), cached_pmf=cached_pmf,
        displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=False, kmask_mode=kmask_mode
    )
    out['B_const'] = float(np.exp(z_opt))
    return out


def fit_physical_bmob_on_work(renderer, target, ds, mob_feat, cached_pmf,
                              displace_mode='none', keep_asu=None, use_tb32_anchor=False,
                              kmask_mode='per_bin', maxiter=65):
    """Fit 6-parameter physical mobility model B_phys^(m)(x; psi) on work |Fo| (PMF frozen)."""
    phys_spec = PhysicalBMobSpec(use_tb32_anchor=use_tb32_anchor)
    inits = [
        phys_spec.pack(alpha_prot=0.60, B0=2.5, dB_unteth=15.0, gamma_teth=0.90, B_far=80.0, d_far=4.2),
        phys_spec.pack(alpha_prot=0.80, B0=2.0, dB_unteth=25.0, gamma_teth=1.00, B_far=350.0, d_far=4.5),
    ]

    def obj(psi):
        z_p, _, pb_p = phys_spec.forward_with_pullback(mob_feat, psi)
        out = multiB_fpart_loss_and_grad(
            renderer, target, ds, z_p, cached_pmf=cached_pmf,
            displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=True, kmask_mode=kmask_mode
        )
        return out['loss'], pb_p(out['grad_z_raw_m'])

    best_psi = min(inits, key=lambda p: obj(p)[0])
    res = minimize(obj, best_psi, method='L-BFGS-B', jac=True,
                   bounds=phys_spec.bounds(), options={'maxiter': int(maxiter), 'ftol': 1e-11, 'gtol': 1e-8})
    psi_opt = res.x
    z_opt, B_phys_opt, _ = phys_spec.forward_with_pullback(mob_feat, psi_opt)
    out = multiB_fpart_loss_and_grad(
        renderer, target, ds, z_opt, cached_pmf=cached_pmf,
        displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=False, kmask_mode=kmask_mode
    )
    out['psi'] = psi_opt
    out['phys_params'] = phys_spec.unpack_physical(psi_opt)
    out['z_phys'] = z_opt
    out['B_phys'] = B_phys_opt
    return out


def fit_spatial_bmob_on_work(renderer, target, ds, z_base_m, cached_pmf, fwhm_A='auto', lam_z='auto',
                             displace_mode='none', keep_asu=None, kmask_mode='per_bin', maxiter=75, eta_init=None):
    """Nudge spatial log-mobility field dz_asu = S_fwhm(eta_asu) on work |Fo| (PMF frozen)."""
    grid = ds.grid
    d_min_t = float(np.min(target.d_hkl)) if len(target.d_hkl) else 0.965
    fwhm_eff, lam_eff = resolution_scaled_bmob_hyperparams(d_min_t, fwhm_A=fwhm_A, lam_z=lam_z)
    smoother = SpatialSmoother3D(grid, fwhm_A=fwhm_eff)
    w_reg = float(lam_eff) * (grid.orbit_mult / float(grid.nvox))

    def obj(eta):
        dz_asu = smoother.forward(eta)
        out = multiB_fpart_loss_and_grad(
            renderer, target, ds, z_base_m + dz_asu[None, :], cached_pmf=cached_pmf,
            displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=True, kmask_mode=kmask_mode
        )
        pen = 0.5 * float(np.sum(w_reg * (eta ** 2)))
        g_dz = np.sum(out['grad_z_raw_m'], axis=0)
        g_eta = smoother.adjoint(g_dz) + w_reg * eta
        return out['loss'] + pen, g_eta

    eta0 = np.clip(np.array(eta_init, dtype=np.float64, copy=True), -4.5, 4.5) if eta_init is not None else np.zeros(grid.n_asu, dtype=np.float64)
    bounds = [(-4.5, 4.5)] * grid.n_asu
    res = minimize(obj, eta0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': int(maxiter), 'ftol': 1e-11, 'gtol': 1e-8})
    eta_opt = res.x
    dz_opt = smoother.forward(eta_opt)
    out = multiB_fpart_loss_and_grad(
        renderer, target, ds, z_base_m + dz_opt[None, :], cached_pmf=cached_pmf,
        displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=False, kmask_mode=kmask_mode
    )
    out['eta_asu'] = eta_opt
    out['dz_asu'] = dz_opt
    out['fwhm_A'] = fwhm_eff
    out['lam_z'] = lam_eff
    return out


def fit_joint_pmf_and_bmob_on_work(renderer, target, ds, spec, theta_init, mob_feat, psi_init,
                                   eta_init=None, fwhm_A='auto', lam_z='auto', lam_md=2e-5,
                                   displace_mode='none', keep_asu=None, kmask_mode='per_bin', maxiter=45):
    """Two-block well-conditioned optimization of (theta_PMF, psi_phys) + res-scaled spatial dz_asu on work |Fo|."""
    grid = ds.grid
    d_min_t = float(np.min(target.d_hkl)) if len(target.d_hkl) else 0.965
    fwhm_eff, lam_eff = resolution_scaled_bmob_hyperparams(d_min_t, fwhm_A=fwhm_A, lam_z=lam_z)
    phys_spec = PhysicalBMobSpec()
    smoother = SpatialSmoother3D(grid, fwhm_A=fwhm_eff)
    dz_fixed = smoother.forward(eta_init) if eta_init is not None else np.zeros(grid.n_asu, dtype=np.float64)

    n_th = spec.n_params
    tp_bounds = param_bounds(spec) + phys_spec.bounds()
    tp0 = np.concatenate([theta_init, psi_init])

    def obj_tp(tp):
        th = tp[:n_th]
        ps = tp[n_th:]
        z_p, _, pb_p = phys_spec.forward_with_pullback(mob_feat, ps)
        out = multiB_fpart_loss_and_grad(
            renderer, target, ds, z_p + dz_fixed[None, :], spec=spec, theta=th,
            displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=True, kmask_mode=kmask_mode
        )
        pen_sm, g_sm = smoothness_penalty_and_grad(spec, th, lam_pmf=8e-5, lam_hpol=8e-5, lam_oz=3e-5)
        diff_th = th - theta_init
        pen_md = 0.5 * float(lam_md) * float(np.sum(diff_th ** 2))
        g_th = out['grad_theta'] + g_sm + float(lam_md) * diff_th
        g_ps = pb_p(out['grad_z_raw_m'])
        return out['loss'] + pen_sm + pen_md, np.concatenate([g_th, g_ps])

    res_tp = minimize(obj_tp, tp0, method='L-BFGS-B', jac=True, bounds=tp_bounds,
                      options={'maxiter': int(maxiter), 'ftol': 1e-9, 'gtol': 1e-7})
    th_opt = res_tp.x[:n_th]
    ps_opt = res_tp.x[n_th:]

    gO_new, gH_new, sk_new, _ = compute_pmf_density_with_pullback(spec, ds, th_opt, need_grad=False)
    cached_new = (gO_new, gH_new, sk_new)
    z_p_opt, _, _ = phys_spec.forward_with_pullback(mob_feat, ps_opt)

    sp_maxiter = max(2, int(maxiter))
    out = fit_spatial_bmob_on_work(
        renderer, target, ds, z_p_opt, cached_new, fwhm_A=fwhm_eff, lam_z=lam_eff,
        displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode, maxiter=sp_maxiter, eta_init=eta_init
    )
    out['theta'] = th_opt
    out['psi'] = ps_opt
    out['phys_params'] = phys_spec.unpack_physical(ps_opt)
    return out


def fit_bmob_blind_gated(renderer, target: ReciprocalTarget, ds, mob_feat, cached_pmf,
                         displace_mode='none', keep_asu=None, kmask_mode='fixed_one',
                         fwhm_A='auto', lam_z='auto', maxiter=55, min_rval_gain: float = 1e-4):
    """Fit uniform, 6-param physical, and res-scaled spatial B(x) strictly on 90% `train_mask` and pick the blind `val_mask` winner.

    Keeping `target.val_mask` (5%) strictly unseen prevents spatial B(x) overfitting at low resolution
    and preserves `val_mask` uncontaminated for any downstream `fit-divot` stage, without ever touching `FreeR_flag == 0`.
    """
    targ_tr = target.for_training_subset()
    z_b5 = np.full((1, ds.grid.n_asu), np.log(5.0), dtype=np.float64)
    out_0 = multiB_fpart_loss_and_grad(
        renderer, targ_tr, ds, z_b5, cached_pmf=cached_pmf,
        displace_mode=displace_mode, keep_asu=keep_asu, compute_grad=False, kmask_mode=kmask_mode,
    )
    out_u = fit_uniform_B_on_work(
        renderer, targ_tr, ds, cached_pmf, displace_mode=displace_mode, keep_asu=keep_asu,
        kmask_mode=kmask_mode, maxiter=maxiter,
    )
    out_p = fit_physical_bmob_on_work(
        renderer, targ_tr, ds, mob_feat, cached_pmf, displace_mode=displace_mode, keep_asu=keep_asu,
        kmask_mode=kmask_mode, maxiter=maxiter,
    )
    out_s = fit_spatial_bmob_on_work(
        renderer, targ_tr, ds, out_p['z_phys'], cached_pmf, fwhm_A=fwhm_A, lam_z=lam_z,
        displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode, maxiter=maxiter,
    )

    stages = [('0shot', out_0), ('uniform', out_u), ('phys', out_p), ('spatial', out_s)]
    rval_table = {}
    best_name, best_out = stages[0]
    best_rv = float('inf')

    for s_name, s_out in stages:
        # Score each stage with the same k_mask model it was fitted with (a per-bin refit of a k_mask = 1 fit
        # re-opens the B / k_mask trade-off that fixed_one removes, and ranks a model nobody fitted)
        r_tr, r_vl, r_wk, r_fr = target.eval_blind_rval(s_out['f_part'], kmask_mode=kmask_mode)
        score = r_vl if np.isfinite(r_vl) else r_tr
        rval_table[s_name] = round(float(score), 6)
        s_out['r_train'] = r_tr
        s_out['r_val'] = r_vl
        s_out['r_work'] = r_wk
        s_out['r_free'] = r_fr
        if score < best_rv - (0.0 if s_name == '0shot' else float(min_rval_gain)):
            best_rv = score
            best_name = s_name
            best_out = s_out

    best_out['selected_stage'] = best_name
    best_out['r_val_table'] = rval_table
    best_out['r_val_fmodel_ref'] = target.eval_fmodel_ref_rval()
    return best_out


def summarize_bmob_microenvironments(label, B_eff_asu, mob_feat, mw):
    """Report median/IQR of learned B(x) and Gaussian FWHM across physical hydration environments."""
    if mob_feat.d_min.shape[0] == 1:
        dm = mob_feat.d_min[0]
        bp = mob_feat.B_prot[0]
        nt = mob_feat.N_tether[0]
    else:
        dm = mw @ mob_feat.d_min
        bp = mw @ mob_feat.B_prot
        nt = mw @ mob_feat.N_tether

    zones = [
        ('Multi-H-bond cleft (d<3.3, Nt>=1.8)', (dm < 3.3) & (nt >= 1.8)),
        ('Single-H-bond 1st  (d<3.3, 0.8..1.8)', (dm < 3.3) & (nt >= 0.8) & (nt < 1.8)),
        ('Apolar 1st shell   (2.4<d<3.6, Nt<0.4)', (dm >= 2.4) & (dm < 3.6) & (nt < 0.4)),
        ('2nd hydration shell(3.6<=d<5.0)',       (dm >= 3.6) & (dm < 5.0)),
        ('Bulk channel       (d>=5.0)',           (dm >= 5.0)),
    ]
    b_pos = B_eff_asu > 0.1
    print(f"\n  Microenvironment mobility breakdown — {label}:")
    for zname, m in zones:
        mz = m & b_pos
        if not np.any(mz):
            continue
        b_sub = B_eff_asu[mz]
        med_b = float(np.median(b_sub))
        p25, p75 = np.percentile(b_sub, [25, 75])
        fwhm = 2.3548 * np.sqrt(med_b / (8.0 * np.pi ** 2))
        print(f"    {zname:38s}: n={int(np.sum(mz)):5d} | B_med = {med_b:6.1f} A^2 "
              f"[IQR {p25:5.1f}..{p75:5.1f}] (FWHM = {fwhm:4.2f} A)", flush=True)

    m_1st = (dm >= 2.3) & (dm < 3.5) & b_pos
    if np.sum(m_1st) > 20:
        r_bp = float(np.corrcoef(B_eff_asu[m_1st], bp[m_1st])[0, 1])
        r_nt = float(np.corrcoef(np.log(B_eff_asu[m_1st]), nt[m_1st])[0, 1])
        print(f"    1st-shell correlations (n={int(np.sum(m_1st))}): "
              f"r(B_wat, B_prot) = {r_bp:+.3f},  r(ln B_wat, N_tether) = {r_nt:+.3f}", flush=True)


def _norm_col_label(s: str) -> str:
    return s.upper().replace('-', '').replace('_', '')


def load_reciprocal_target_from_fmodel_mtz(fmodel_mtz_path: str, n_bins: int = 24,
                                           data_mtz_path: str = None, free_flag_col: str = 'FreeR_flag',
                                           free_flag_value=None):
    """Build a ReciprocalTarget directly from a Phenix `*_f_model.mtz` file (`export_final_f_model=True`)."""
    mtz = gemmi.read_mtz_file(fmodel_mtz_path)
    if not mtz.cell.is_crystal():
        raise ValueError(f"{fmodel_mtz_path}: MTZ has no valid crystallographic unit cell")
    raw_labels = mtz.column_labels()
    arr = np.array(mtz, copy=True)
    col_map = {_norm_col_label(lbl): arr[:, idx] for idx, lbl in enumerate(raw_labels)}

    def pick(*names, required=True):
        for nm in names:
            k = _norm_col_label(nm)
            if k in col_map:
                return col_map[k]
        if required:
            raise ValueError(
                f"{fmodel_mtz_path} is missing required column ({'/'.join(names)}); "
                f"present columns: {raw_labels}. Export with `phenix.refine ... output.export_final_f_model=True`."
            )
        return None

    hkl = arr[:, :3].astype(np.int64)
    f_obs = pick('FOBS', 'F-obs-filtered', 'F-obs', 'FP').astype(np.float64)
    flags = pick('R_FREE_FLAGS', 'R-free-flags', 'FreeR_flag')

    valid = np.isfinite(f_obs) & (f_obs > 0.0) & np.isfinite(flags)
    if int(np.sum(valid)) < 2:
        raise ValueError(f"{fmodel_mtz_path}: fewer than 2 valid positive reflections found")
    free_raw = free_mask_from_flags(flags, valid, free_flag_value, source_desc=fmodel_mtz_path)

    work_mask = valid & (~free_raw)
    free_mask = valid & free_raw
    if int(np.sum(work_mask)) < 2:
        raise ValueError(f"{fmodel_mtz_path}: fewer than 2 work reflections found")

    fc_amp = pick('FCALC', 'FCALCATOMS', required=False)
    fc_phi = pick('PHIFCALC', 'PHIFCALCATOMS', required=False)
    fm_amp = pick('FMODEL', 'F-model')
    fm_phi = pick('PHIFMODEL', 'PHIF-model')
    f_model = fm_amp.astype(np.float64) * np.exp(1j * np.deg2rad(fm_phi.astype(np.float64)))

    k_iso = pick('K_ISOTROPIC', 'K_ISO', 'KISO', required=False)
    k_aniso = pick('K_ANISOTROPIC', 'K_ANISO', 'KANISO', required=False)
    k_total_col = pick('K_TOTAL', 'KTOTAL', required=False)
    k_mask_col = pick('K_MASK', 'KMASK', required=False)
    fmask_amp = pick('FMASK', 'FPART', required=False)
    fmask_phi = pick('PHIFMASK', 'PHIFPART', 'PHIPART', required=False)

    if k_iso is not None and k_aniso is not None:
        k_tot = (k_iso * k_aniso).astype(np.float64)
    elif k_total_col is not None:
        k_tot = k_total_col.astype(np.float64)
    else:
        k_tot = None

    if fc_amp is not None and fc_phi is not None:
        f_calc = fc_amp.astype(np.float64) * np.exp(1j * np.deg2rad(fc_phi.astype(np.float64)))
    elif k_tot is not None and k_mask_col is not None and fmask_amp is not None and fmask_phi is not None:
        f_mask_phx = fmask_amp.astype(np.float64) * np.exp(1j * np.deg2rad(fmask_phi.astype(np.float64)))
        f_calc = (f_model / np.maximum(k_tot, 1e-8)) - k_mask_col.astype(np.float64) * f_mask_phx
    else:
        raise ValueError(f"{fmodel_mtz_path} lacks FCALC/PHIFCALC columns needed to build ReciprocalTarget.")

    if k_mask_col is not None and fmask_amp is not None and fmask_phi is not None:
        f_mask_phx = fmask_amp.astype(np.float64) * np.exp(1j * np.deg2rad(fmask_phi.astype(np.float64)))
        unscaled = f_calc + k_mask_col.astype(np.float64) * f_mask_phx
        if k_tot is not None:
            pred0 = k_tot * unscaled
            nz = valid & (np.abs(pred0) > 1e-6)
            if np.any(nz):
                scale_k1 = float(np.median(np.abs(f_model[nz]) / np.abs(pred0[nz])))
                k_tot = scale_k1 * k_tot
        else:
            k_tot = np.abs(f_model) / np.maximum(np.abs(unscaled), 1e-8)

    if k_tot is None:
        k_tot = np.ones(len(hkl), dtype=np.float64)

    k_tot = np.where(np.isfinite(k_tot) & (k_tot > 0.0), k_tot, 1.0)
    d_hkl = 1.0 / np.sqrt(np.maximum(np.asarray(mtz.cell.calculate_1_d2_array(hkl), dtype=np.float64), 1e-12))

    inv_d3 = 1.0 / np.maximum(d_hkl, 0.1) ** 3
    edges = np.quantile(inv_d3[work_mask], np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[0] = -1.0
    edges[-1] = np.inf
    bin_idx = np.clip(np.searchsorted(edges, inv_d3, side='right') - 1, 0, int(n_bins) - 1)

    sg_raw = mtz.spacegroup
    sg = (
        gemmi.find_spacegroup_by_name(sg_raw.hm, alpha=mtz.cell.alpha, gamma=mtz.cell.gamma)
        if sg_raw is not None
        else gemmi.SpaceGroup("P 1")
    ) or sg_raw or gemmi.SpaceGroup("P 1")

    train_v, val_v = make_work_train_val_split(
        hkl[valid], d_hkl[valid], free_mask[valid],
        data_mtz_path=data_mtz_path, free_flag_col=free_flag_col, spacegroup=sg,
    )
    target = ReciprocalTarget(
        hkl=hkl[valid], f_obs=f_obs[valid],
        work_mask=work_mask[valid], free_mask=free_mask[valid],
        f_calc=f_calc[valid], k_tot=k_tot[valid],
        bin_idx=bin_idx[valid], n_bins=int(n_bins), d_hkl=d_hkl[valid],
        train_mask=train_v, val_mask=val_v, f_model_ref=f_model[valid],
    )
    return target, mtz.cell, sg
