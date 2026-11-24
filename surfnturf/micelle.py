"""surfnturf.micelle: Differentiable space-group-exact 3-zone amphiphilic detergent micelle / lipid-belt model.

Models the physical electron-density contrast of membrane-protein detergent micelles, bicelles, and nanodiscs:
  1. Automatic transmembrane hydrophobic belt & interior-pore detector (`detect_transmembrane_belt`): identifies the
     exterior lipid-facing hydrophobic belt and `TRP/TYR` interfacial girdles while keeping interior aqueous pores
     (such as beta-barrel lumens in `3QUE` or ion-channel vestibules) purely aqueous via 12-sector 2D azimuthal
     ray-enclosure filtering.
  2. Space-group-exact distance fields on `ASUGrid` (`d_TM`, `d_pol`, `d_pore`, and multi-neighbor `f_open`), naturally
     fusing micelles across hydrophobic crystal contacts and pinching off alkane tails at hydrophilic crystal contacts.
  3. Smooth C-infinity 3-zone partition of unity (`w_tail + w_head + w_aq == 1`) separating the low-density aliphatic
     hydrocarbon tail core (`rho_tail ~ 0.24..0.29 e/A^3`), higher-density polar/glycosidic headgroup collar
     (`rho_head ~ 0.37..0.45 e/A^3`), and structured aqueous `surfNturf` hydration (`k_wat * rho_surfnturf`), with
     strict zero protein-core leakage.
  4. Zero R-free leakage: fitted strictly on `90% work_train` `|Fo|`, gated on unseen `5% work_val`, and locked with
     hard assertions so `R-free` is never read.
"""
import math
import os
import numpy as np
import gemmi
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit

from .field import ASUGrid, RHO_BULK
from .divot import (
    CrystalGridOps,
    SolventAmplitudeProblem,
    build_open_solvent_envelope,
    build_solvent_support_mask,
)


_AA_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "MSE",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
_APOLAR_LIPID_RES = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "MSE", "PRO", "CYS", "CYX"}
_CHARGED_SIDECHAIN_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("HIP", "ND1"), ("HIP", "NE2"),
}
_POLAR_SIDECHAIN_RES = {"ASP", "GLU", "LYS", "ARG", "HIS", "HID", "HIE", "HIP", "ASN", "GLN", "SER", "THR"}
_GIRDLE_ATOMS = {
    ("TRP", "NE1"), ("TRP", "CZ2"), ("TRP", "CH2"),
    ("TYR", "OH"), ("TYR", "CZ"),
}

MICELLE_PARAM_NAMES = (
    "L_tail_A",       # 0: aliphatic hydrocarbon chain belt thickness (A)
    "w_head_A",       # 1: polar/glycosidic headgroup collar thickness (A)
    "d_pinch_A",      # 2: polar-residue exclusion distance pinching off alkane tails (A)
    "sigma_tail_A",   # 3: inner tail-to-headgroup transition width (A)
    "sigma_outer_A",  # 4: outer headgroup-to-water thermal boundary width (A)
    "rho_tail_eA3",   # 5: hydrocarbon core electron density (e/A^3; octane/dodecane ~ 0.25-0.28)
    "rho_head_eA3",   # 6: polar headgroup collar electron density (e/A^3; glucoside/maltoside ~ 0.38-0.43)
    "k_wat",          # 7: aqueous mother-liquor scale relative to surfNturf base
)


def _fibonacci_hemisphere(n_dirs: int = 81) -> np.ndarray:
    idx = np.arange(n_dirs, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = (idx + 0.5) / float(n_dirs)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = golden * idx
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _azimuthal_open_fraction(slab_xy: np.ndarray, query_indices: np.ndarray,
                             r_min: float = 3.5, r_max: float = 22.0, n_sectors: int = 12) -> np.ndarray:
    """Fraction of 2D azimuthal sectors in `[0, 2*pi)` NOT blocked by protein atoms at `r_min <= r_2d <= r_max`.

    Atoms lining the interior lumen of a beta-barrel (`3QUE`) or central ion channel are enclosed in almost all 12
    azimuthal directions (`open_frac <= 0.25`), whereas exterior lipid-facing perimeter atoms face open space over at
    least one outward hemisphere (`open_frac >= 0.33`).
    """
    if len(query_indices) == 0:
        return np.zeros(0, dtype=np.float64)
    q_xy = slab_xy[query_indices]
    diff = slab_xy[None, :, :] - q_xy[:, None, :]   # (Q, N_slab, 2)
    r2 = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
    in_ring = (r2 >= float(r_min) ** 2) & (r2 <= float(r_max) ** 2)
    ang = np.arctan2(diff[:, :, 1], diff[:, :, 0]) + math.pi  # [0, 2*pi]
    sec = np.clip(np.floor(ang * (float(n_sectors) / (2.0 * math.pi))).astype(np.int32), 0, n_sectors - 1)

    blocked = np.zeros((len(query_indices), n_sectors), dtype=bool)
    qi_nz, ji_nz = np.nonzero(in_ring)
    if len(qi_nz) > 0:
        blocked[qi_nz, sec[qi_nz, ji_nz]] = True
    return 1.0 - np.mean(blocked, axis=1)


def detect_transmembrane_belt(
    pdb_or_dir: str,
    min_residues: int = 70,
    min_belt_score: float = 0.40,
    force_micelle: bool = False,
):
    """Detect an exterior hydrophobic transmembrane belt & distinguish it from soluble caps and interior aqueous pores.

    Returns a dict with:
      - `is_membrane`: bool whether a genuine transmembrane belt was detected (or `force_micelle=True`),
      - `belt_score`: float hydrophobic slab contrast score,
      - `normal`: unit 3-vector of the inferred membrane normal,
      - `z_center`, `thickness_A`: slab center projection and thickness in Angstroms,
      - `tm_xyz`: `(N_tm, 3)` exterior lipid-facing transmembrane heavy-atom coordinates,
      - `pol_xyz`: `(N_pol, 3)` hydrophilic / extramembranous cap heavy-atom coordinates,
      - `pore_xyz`: `(N_pore, 3)` interior enclosed pore/cavity heavy-atom coordinates.
    """
    from .io import _sample_structure_for_target
    st = _sample_structure_for_target(str(pdb_or_dir))
    if len(st) == 0:
        return {
            "is_membrane": False, "belt_score": 0.0, "reason": "empty_structure",
            "normal": np.array([0.0, 0.0, 1.0]), "z_center": 0.0, "thickness_A": 24.0,
            "tm_xyz": np.zeros((0, 3)), "pol_xyz": np.zeros((0, 3)), "pore_xyz": np.zeros((0, 3)),
        }

    # Extract highest-occupancy heavy conformer per protein residue atom (ignoring HOH and modeled detergents)
    best_atoms = {}
    res_keys = set()
    for ch_idx, ch in enumerate(st[0]):
        ch_id = ch.name.strip() or f"#{ch_idx}"
        for r in ch:
            rn = r.name.strip().upper()
            if rn not in _AA_RESNAMES:
                continue
            ic_id = r.seqid.icode.strip() if r.seqid.icode != "\0" else ""
            rkey = (ch_id, int(r.seqid.num), ic_id, rn)
            res_keys.add(rkey)
            for a in r:
                el = a.element.name.strip().upper()
                if el in ("H", "D") or a.occ <= 0.0:
                    continue
                an = a.name.strip().upper()
                akey = (rkey, an)
                pos = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=np.float64)
                if akey not in best_atoms or float(a.occ) > best_atoms[akey][0]:
                    best_atoms[akey] = (float(a.occ), rn, an, el, pos)

    if len(best_atoms) < 30:
        return {
            "is_membrane": False, "belt_score": 0.0, "reason": "too_few_protein_atoms",
            "normal": np.array([0.0, 0.0, 1.0]), "z_center": 0.0, "thickness_A": 24.0,
            "tm_xyz": np.zeros((0, 3)), "pol_xyz": np.zeros((0, 3)), "pore_xyz": np.zeros((0, 3)),
        }

    recs = list(best_atoms.values())
    xyz = np.array([r[4] for r in recs], dtype=np.float64)
    rnames = [r[1] for r in recs]
    anames = [r[2] for r in recs]
    elems = [r[3] for r in recs]
    n_atoms = len(xyz)
    n_res = len(res_keys)

    tree_all = cKDTree(xyz)
    nb5 = tree_all.query_ball_point(xyz, 5.0)
    n_nb5 = np.array([len(x) - 1 for x in nb5], dtype=np.int32)
    is_surf = n_nb5 <= 24

    on_mask = np.array([el in ("O", "N") for el in elems], dtype=bool)
    on_xyz = xyz[on_mask]
    tree_on = cKDTree(on_xyz) if len(on_xyz) else None

    is_charged = np.zeros(n_atoms, dtype=bool)
    is_polar = np.zeros(n_atoms, dtype=bool)
    is_apolar = np.zeros(n_atoms, dtype=bool)
    is_girdle = np.zeros(n_atoms, dtype=bool)

    for i in range(n_atoms):
        rn, an, el = rnames[i], anames[i], elems[i]
        if (rn, an) in _GIRDLE_ATOMS:
            is_girdle[i] = True
        if (rn, an) in _CHARGED_SIDECHAIN_ATOMS or an == "OXT":
            is_charged[i] = True
            is_polar[i] = True
            continue
        if el in ("O", "N"):
            if an in ("N", "O"):
                # Transmembrane helix/sheet backbone N and O are intramolecularly H-bonded (2.45..3.40 A);
                # only unpartnered exposed loop N/O count as hydrophilic water-seeking groups.
                if tree_on is not None:
                    d_on = np.linalg.norm(on_xyz[tree_on.query_ball_point(xyz[i], 3.40)] - xyz[i], axis=1)
                    has_hb = bool(np.any((d_on >= 2.42) & (d_on <= 3.40)))
                else:
                    has_hb = False
                if not has_hb:
                    is_polar[i] = True
            elif rn in _POLAR_SIDECHAIN_RES:
                is_polar[i] = True
        elif el in ("C", "S", "SE") and an not in ("C",):
            if rn in _APOLAR_LIPID_RES or (rn == "TYR" and an in ("CB", "CG", "CD1", "CD2", "CE1", "CE2")):
                is_apolar[i] = True

    # Count nearby hydrophilic O/N within 4.6 A of each atom
    pol_pts = xyz[is_polar]
    tree_pol = cKDTree(pol_pts) if len(pol_pts) else None
    n_pol_near = (
        np.array([len(x) for x in tree_pol.query_ball_point(xyz, 4.6)], dtype=np.int32)
        if tree_pol is not None else np.zeros(n_atoms, dtype=np.int32)
    )

    com = np.mean(xyz, axis=0)
    centered = xyz - com
    # Include principal inertia eigenvectors alongside 81 Fibonacci directions
    cov = (centered.T @ centered) / max(n_atoms, 1)
    _, eigvecs = np.linalg.eigh(cov)
    dirs = np.vstack([_fibonacci_hemisphere(81), eigvecs.T])
    surf_idx = np.nonzero(is_surf)[0]

    # Coarse vectorized scan across (direction, thickness H, slab center z0)
    proj_all = centered @ dirs.T    # (N, D)
    candidates = []
    for d_i in range(len(dirs)):
        z_all = proj_all[:, d_i]
        z_s = z_all[surf_idx]
        if len(z_s) < 20:
            continue
        z_lo, z_hi = float(np.percentile(z_s, 5)), float(np.percentile(z_s, 95))
        span = z_hi - z_lo
        if span < 16.0 and not force_micelle:
            continue
        r2_perp = np.sum(centered[surf_idx] ** 2, axis=1) - z_s ** 2
        for H in (20.0, 24.0, 28.0, 31.0):
            half_h = 0.5 * H
            z0_grid = np.linspace(z_lo + 0.35 * H, z_hi - 0.35 * H, 5) if span > H else np.array([0.5 * (z_lo + z_hi)])
            for z0 in z0_grid:
                dz = np.abs(z_s - float(z0))
                in_core = dz <= (half_h - 2.0)
                if int(np.sum(in_core)) < 16:
                    continue
                # Coarse peripheral filter (outer 60% of cylindrical radius in the slab)
                r2_med = float(np.percentile(r2_perp[in_core], 40))
                ext_c = in_core & (r2_perp >= r2_med)
                if int(np.sum(ext_c)) < 12:
                    continue
                idx_ec = surf_idx[ext_c]
                f_ap = float(np.mean(is_apolar[idx_ec]))
                f_ch = float(np.mean(is_charged[idx_ec]))
                out_cap = dz > half_h
                f_pol_out = float(np.mean(is_polar[surf_idx[out_cap]])) if int(np.sum(out_cap)) >= 8 else 0.42
                in_gird = (dz >= half_h - 4.5) & (dz <= half_h + 2.0)
                n_gird = int(np.sum(is_girdle[surf_idx[in_gird]]))
                sc = (f_ap - 2.2 * f_ch) + 0.35 * (f_pol_out - (1.0 - f_ap)) + 0.08 * min(1.0, n_gird / 5.0)
                candidates.append((sc, d_i, float(z0), float(H)))

    if not candidates:
        return {
            "is_membrane": False, "belt_score": 0.0, "reason": "no_slab_candidates",
            "normal": np.array([0.0, 0.0, 1.0]), "z_center": 0.0, "thickness_A": 24.0,
            "tm_xyz": np.zeros((0, 3)), "pol_xyz": xyz[is_polar], "pore_xyz": np.zeros((0, 3)),
        }

    # Refine top distinct candidate axes with exact 12-sector 2D azimuthal pore-vs-exterior ray enclosure
    candidates.sort(key=lambda t: -t[0])
    best_pack = None
    seen_dirs = []
    for _, d_i, z0, H in candidates:
        n_vec = dirs[d_i] / np.linalg.norm(dirs[d_i])
        if any(abs(float(np.dot(n_vec, prev))) > 0.94 for prev in seen_dirs):
            continue
        seen_dirs.append(n_vec)
        if len(seen_dirs) > 6:
            break

        # Construct orthonormal 2D basis (e1, e2) in the membrane plane perpendicular to n_vec
        ref_ax = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n_vec, ref_ax)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_vec, e1)

        z_all = centered @ n_vec
        half_h = 0.5 * H
        in_slab_all = np.abs(z_all - z0) <= (half_h + 1.5)
        slab_idx = np.nonzero(in_slab_all)[0]
        if len(slab_idx) < 20:
            continue
        slab_xy = np.column_stack([centered[slab_idx] @ e1, centered[slab_idx] @ e2])
        surf_in_slab_local = np.nonzero(is_surf[slab_idx])[0]
        open_frac_surf = _azimuthal_open_fraction(slab_xy, surf_in_slab_local)

        open_by_global = np.zeros(n_atoms, dtype=np.float64)
        open_by_global[slab_idx[surf_in_slab_local]] = open_frac_surf

        dz_all = np.abs(z_all - z0)
        ext_slab_surf = is_surf & (dz_all <= (half_h - 1.5)) & (open_by_global >= 0.33)
        pore_slab = (dz_all <= half_h) & (open_by_global < 0.26) & is_surf

        if int(np.sum(ext_slab_surf)) < 14:
            continue
        f_ap = float(np.mean(is_apolar[ext_slab_surf]))
        f_ch = float(np.mean(is_charged[ext_slab_surf]))
        out_surf = is_surf & (dz_all > half_h)
        f_pol_out = float(np.mean(is_polar[out_surf])) if int(np.sum(out_surf)) >= 8 else 0.42
        n_gird = int(np.sum(is_girdle & is_surf & (dz_all >= half_h - 4.5) & (dz_all <= half_h + 2.0) & (open_by_global >= 0.30)))
        sc_exact = (f_ap - 2.2 * f_ch) + 0.35 * (f_pol_out - (1.0 - f_ap)) + 0.08 * min(1.0, n_gird / 5.0)

        tm_mask = (
            is_surf
            & (dz_all <= half_h)
            & (open_by_global >= 0.33)
            & is_apolar
            & (n_pol_near <= 2)
        )
        pol_cap_mask = is_polar | (is_surf & (dz_all > half_h + 1.5))
        pack = (sc_exact, n_vec, z0, H, xyz[tm_mask], xyz[pol_cap_mask], xyz[pore_slab], f_ap, f_ch)
        if best_pack is None or sc_exact > best_pack[0]:
            best_pack = pack

    if best_pack is None:
        return {
            "is_membrane": False, "belt_score": 0.0, "reason": "no_exterior_belt",
            "normal": np.array([0.0, 0.0, 1.0]), "z_center": 0.0, "thickness_A": 24.0,
            "tm_xyz": np.zeros((0, 3)), "pol_xyz": xyz[is_polar], "pore_xyz": np.zeros((0, 3)),
        }

    sc_best, n_best, z0_best, H_best, tm_xyz, pol_xyz, pore_xyz, f_ap_b, f_ch_b = best_pack
    size_ok = (n_res >= int(min_residues)) or force_micelle
    score_ok = (sc_best >= float(min_belt_score) and len(tm_xyz) >= 15) or (force_micelle and len(tm_xyz) >= 6)
    if force_micelle and len(tm_xyz) < 6:
        # Fallback if forced on a synthetic minimal apolar test helix/bundle
        tm_xyz = xyz[is_apolar] if np.any(is_apolar) else xyz
        score_ok = len(tm_xyz) > 0

    is_mem = bool(size_ok and score_ok)
    reason = "detected" if is_mem else ("below_min_residues" if not size_ok else "soluble_surface_polarity")
    return {
        "is_membrane": is_mem,
        "belt_score": round(float(sc_best), 4),
        "f_apolar_belt": round(float(f_ap_b), 4),
        "f_charged_belt": round(float(f_ch_b), 4),
        "n_residues": n_res,
        "n_tm_atoms": int(len(tm_xyz)),
        "n_pore_atoms": int(len(pore_xyz)),
        "reason": reason,
        "normal": n_best,
        "z_center": round(float(com @ n_best + z0_best), 3),
        "thickness_A": float(H_best),
        "tm_xyz": tm_xyz,
        "pol_xyz": pol_xyz,
        "pore_xyz": pore_xyz,
    }


class AmphiphilicMicelleModel:
    """Space-group-exact differentiable 3-zone amphiphilic torus (`tail` + `headgroup` + `aqueous surfNturf`) on `ASUGrid`."""

    def __init__(self, asu_grid: ASUGrid, base_3d: np.ndarray, open_3d: np.ndarray, belt_info: dict):
        self.grid = asu_grid
        self.belt_info = belt_info
        self.n_asu = asu_grid.n_asu
        pts = asu_grid.asu_xyz

        supp_3d = build_solvent_support_mask(base_3d, open_3d)
        self.base_asu = asu_grid.full_3d_to_asu_mean(np.asarray(base_3d, dtype=np.float64))
        self.open_asu = asu_grid.full_3d_to_asu_mean(np.asarray(open_3d, dtype=np.float64))
        self.supp_asu = asu_grid.full_3d_to_asu_mean(supp_3d)

        tm_xyz = np.asarray(belt_info.get("tm_xyz", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        pol_xyz = np.asarray(belt_info.get("pol_xyz", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        pore_xyz = np.asarray(belt_info.get("pore_xyz", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)

        # Expand across all space-group symmetry operations and unit-cell lattice translations up to 24 A
        S_tm, _ = asu_grid.sym_expand(tm_xyz, cutoff_A=24.0, dedup_self=True) if len(tm_xyz) else (np.zeros((0, 3)), None)
        if len(S_tm) > 0:
            self.d_tm = np.asarray(cKDTree(S_tm).query(pts)[0], dtype=np.float64)
        else:
            self.d_tm = np.full(self.n_asu, 50.0, dtype=np.float64)

        pol_all = np.vstack([a for a in (pol_xyz, pore_xyz) if len(a) > 0]) if (len(pol_xyz) + len(pore_xyz)) > 0 else np.zeros((0, 3))
        S_pol, _ = asu_grid.sym_expand(pol_all, cutoff_A=12.0, dedup_self=True) if len(pol_all) else (np.zeros((0, 3)), None)
        if len(S_pol) > 0:
            self.d_pol = np.asarray(cKDTree(S_pol).query(pts)[0], dtype=np.float64)
        else:
            self.d_pol = np.full(self.n_asu, 20.0, dtype=np.float64)

        g_rim = expit((self.d_pol - 2.25) / 0.35)
        S_pore, _ = asu_grid.sym_expand(pore_xyz, cutoff_A=20.0, dedup_self=True) if len(pore_xyz) else (np.zeros((0, 3)), None)
        if len(S_pore) >= 6:
            d_pore = np.asarray(cKDTree(S_pore).query(pts)[0], dtype=np.float64)
            # Prevent outer detergent belt from leaking through a thin (~5 A) beta-barrel wall into an interior lumen
            g_ext = expit((d_pore - self.d_tm + 1.0) / 0.60)
        else:
            g_ext = 1.0
        self.g_fixed = g_rim * g_ext

    @staticmethod
    def default_params():
        # [L_tail, w_head, d_pinch, sigma_tail, sigma_outer, rho_tail, rho_head, k_wat]
        return np.array([9.5, 4.2, 3.4, 1.00, 1.60, 0.270, 0.395, 1.000], dtype=np.float64)

    @staticmethod
    def bounds():
        return [
            (5.5, 18.0),    # L_tail_A (BOG ~8-10 A, DDM ~12-15 A)
            (2.0, 8.0),     # w_head_A (glucoside/maltoside/PC headgroup collar)
            (2.4, 5.2),     # d_pinch_A (polar cap pinch-off distance)
            (0.45, 2.8),    # sigma_tail_A
            (0.60, 4.0),    # sigma_outer_A (outer micelle disorder)
            (0.200, 0.334), # rho_tail_eA3 (aliphatic alkane <= water)
            (0.334, 0.500), # rho_head_eA3 (polar headgroup >= water)
            (0.75, 1.35),   # k_wat (aqueous mother-liquor scale)
        ]

    def forward_and_pullback(self, params, compute_grad: bool = True):
        """Return `(rho_3d, zone_stats, pullback_fn)` with exact C-infinity analytical derivatives."""
        p = np.asarray(params, dtype=np.float64)
        L_tail, w_head, d_pinch, sig_t, sig_o, rho_tail, rho_head, k_wat = [float(x) for x in p]

        u_out = (L_tail + w_head - self.d_tm) / sig_o
        s_out = expit(u_out)
        u_in = (L_tail - self.d_tm) / sig_t
        s_in = expit(u_in)
        u_pin = (self.d_pol - d_pinch) / 0.55
        h_pol = expit(u_pin)

        W_mic = self.g_fixed * s_out
        sh = s_in * h_pol
        w_tail = W_mic * sh
        w_head_z = W_mic - w_tail

        f_op = self.open_asu
        b_aq = self.base_asu
        supp = self.supp_asu

        diff_hw = f_op * rho_head - k_wat * b_aq
        diff_th = f_op * (rho_tail - rho_head)
        rho_asu = supp * (k_wat * b_aq + W_mic * diff_hw + w_tail * diff_th)
        rho_3d = self.grid.asu_to_full_3d(rho_asu)

        vox_vol = float(self.grid.cell.volume) / float(self.grid.nvox)
        mult_v = self.grid.orbit_mult * vox_vol
        tail_vol = float(np.sum(supp * f_op * w_tail * mult_v))
        head_vol = float(np.sum(supp * f_op * w_head_z * mult_v))
        stats = {
            "tail_vol_A3": round(tail_vol, 1),
            "head_vol_A3": round(head_vol, 1),
            "params": {nm: round(float(v), 4) for nm, v in zip(MICELLE_PARAM_NAMES, p)},
        }
        if not compute_grad:
            return rho_3d, stats, None

        def pullback(dL_drho_3d):
            g_asu = self.grid.full_3d_to_asu_grad(np.asarray(dL_drho_3d, dtype=np.float64))
            gs = g_asu * supp

            grad = np.zeros(8, dtype=np.float64)
            grad[5] = float(np.dot(gs, f_op * w_tail))
            grad[6] = float(np.dot(gs, f_op * w_head_z))
            grad[7] = float(np.dot(gs, b_aq * (1.0 - W_mic)))

            d_wtail = gs * diff_th
            d_W = gs * diff_hw + d_wtail * sh

            du_out = (d_W * self.g_fixed) * s_out * (1.0 - s_out)
            du_in = (d_wtail * W_mic * h_pol) * s_in * (1.0 - s_in)
            du_pin = (d_wtail * W_mic * s_in) * h_pol * (1.0 - h_pol)

            sum_out = float(np.sum(du_out)) / sig_o
            sum_in = float(np.sum(du_in)) / sig_t

            grad[0] = sum_out + sum_in
            grad[1] = sum_out
            grad[2] = -float(np.sum(du_pin)) / 0.55
            grad[3] = -float(np.dot(du_in, u_in)) / sig_t
            grad[4] = -float(np.dot(du_out, u_out)) / sig_o
            return grad

        return rho_3d, stats, pullback


def fit_micelle_blind(
    prob: SolventAmplitudeProblem,
    cgrid: CrystalGridOps,
    base_3d: np.ndarray,
    pdb_or_dir: str,
    open_3d: np.ndarray = None,
    force_micelle: bool = False,
    gate_on_val: bool = True,
    refit_on_full_work: bool = True,
    lam_reg: float = 2e-5,
    maxiter: int = 45,
    min_val_gain: str = "auto",
):
    """Fit `AmphiphilicMicelleModel` on `90% prob.train` `|Fo|` and gate acceptance strictly on unseen `5% prob.val`.

    Enforces `R-free` blindness (`assert not np.any(active_mask & prob.free)`) throughout.
    """
    assert not np.any(prob.train & prob.free), "R-FREE LEAK IN MICELLE TRAIN MASK!"
    assert not np.any(prob.val & prob.free), "R-FREE LEAK IN MICELLE VAL MASK!"

    if open_3d is None:
        open_3d = build_open_solvent_envelope(cgrid, pdb_or_dir)

    ev_base = prob.eval_rho(base_3d, active_mask=prob.train, compute_grad=False)
    rval_base = float(ev_base["r_val"])
    n_val = max(1, int(np.sum(prob.val)))
    if str(min_val_gain).lower() == "auto":
        gain_req = max(5e-4, 0.55 * rval_base / math.sqrt(2.0 * n_val))
    else:
        gain_req = float(min_val_gain)

    belt = detect_transmembrane_belt(pdb_or_dir, force_micelle=force_micelle)
    if not belt["is_membrane"]:
        return base_3d.copy(), {
            "selected_stage": "aqueous_soluble",
            "belt": {k: v for k, v in belt.items() if k not in ("tm_xyz", "pol_xyz", "pore_xyz", "normal")},
            "r_val_base": rval_base,
            "r_val": rval_base,
            "r_work": float(ev_base["r_work"]),
            "r_free": float(ev_base["r_free"]),
        }

    asu_grid = ASUGrid(cgrid.cell, cgrid.sg, cgrid.shape, step=1)
    model = AmphiphilicMicelleModel(asu_grid, base_3d, open_3d, belt)
    supp_3d = build_solvent_support_mask(base_3d, open_3d)

    # Guard against unblurred B=5 A^2 base maps: evaluate 1-scalar isotropic aqueous B_sol blur on prob.train first
    # so AmphiphilicMicelleModel only wins on prob.val when genuine spatial alkane/headgroup contrast beats B_sol!
    from scipy.optimize import minimize_scalar
    res_b = minimize_scalar(
        lambda b_val: prob.eval_rho(supp_3d * cgrid.gaussian_blur(base_3d, float(b_val)),
                                    active_mask=prob.train, compute_grad=False)["loss"],
        bounds=(0.0, 120.0), method="bounded", options={"xatol": 0.5, "maxiter": 12},
    )
    rho_aq_blur = supp_3d * cgrid.gaussian_blur(base_3d, float(res_b.x))
    ev_aq_blur = prob.eval_rho(rho_aq_blur, active_mask=prob.train, compute_grad=False)
    rval_aq_blur = float(ev_aq_blur["r_val"])
    if rval_aq_blur < rval_base - gain_req:
        aq_ref_rho = rho_aq_blur
        rval_gate_ref = rval_aq_blur
        ev_aq_ref = ev_aq_blur
    else:
        aq_ref_rho = base_3d.copy()
        rval_gate_ref = rval_base
        ev_aq_ref = ev_base

    p_ref = np.array([9.5, 4.2, 3.4, 1.00, 1.60, RHO_BULK, RHO_BULK, 1.00], dtype=np.float64)
    reg_scales = np.array([0.04, 0.06, 0.10, 0.15, 0.15, 1.0, 1.0, 0.5], dtype=np.float64)

    def _fit_on_mask(active_mask):
        assert not np.any(active_mask & prob.free), "R-FREE LEAK BLOCKED IN MICELLE FIT!"
        inits = [
            np.array([9.0, 4.0, 3.3, 0.95, 1.50, 0.268, 0.395, 1.00], dtype=np.float64),   # BOG-like compact micelle
            np.array([13.0, 4.8, 3.5, 1.15, 2.00, 0.265, 0.405, 1.00], dtype=np.float64),  # DDM-like wider micelle
        ]

        def _obj(x):
            rho_3d, _, pb = model.forward_and_pullback(x, compute_grad=True)
            ev = prob.eval_rho(rho_3d, active_mask=active_mask, compute_grad=True)
            diff = (x - p_ref) * reg_scales
            pen = 0.5 * float(lam_reg) * float(np.dot(diff, diff))
            grad = pb(ev["grho"]) + float(lam_reg) * (diff * reg_scales)
            return ev["loss"] + pen, grad

        x0 = min(inits, key=lambda p0: _obj(p0)[0])
        res = minimize(_obj, x0, method="L-BFGS-B", jac=True, bounds=model.bounds(),
                       options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-8})
        rho_opt, stats_opt, _ = model.forward_and_pullback(res.x, compute_grad=False)
        return res.x, rho_opt, stats_opt

    fit_mask = prob.train if gate_on_val else prob.work
    p_tr, rho_tr, stats_tr = _fit_on_mask(fit_mask)
    ev_mic = prob.eval_rho(rho_tr, active_mask=prob.train, compute_grad=False)
    rval_mic = float(ev_mic["r_val"])

    belt_summary = {
        "is_membrane": belt["is_membrane"],
        "belt_score": belt["belt_score"],
        "f_apolar_belt": belt.get("f_apolar_belt", 0.0),
        "f_charged_belt": belt.get("f_charged_belt", 0.0),
        "n_tm_atoms": belt.get("n_tm_atoms", 0),
        "n_pore_atoms": belt.get("n_pore_atoms", 0),
        "thickness_A": belt["thickness_A"],
        "z_center": belt["z_center"],
        "normal": [round(float(v), 4) for v in belt["normal"]],
    }

    if gate_on_val and not (rval_mic < rval_gate_ref - gain_req):
        return aq_ref_rho, {
            "selected_stage": "aqueous_gated",
            "belt": belt_summary,
            "candidate_micelle": stats_tr,
            "r_val_base": rval_base,
            "r_val_aq_blur": round(rval_aq_blur, 6),
            "r_val_micelle": rval_mic,
            "r_val": rval_gate_ref,
            "min_val_gain": round(gain_req, 6),
            "r_work": float(ev_aq_ref["r_work"]),
            "r_free": float(ev_aq_ref["r_free"]),
        }

    if gate_on_val and refit_on_full_work:
        p_final, rho_final, stats_final = _fit_on_mask(prob.work)
    else:
        p_final, rho_final, stats_final = p_tr, rho_tr, stats_tr

    _, ev_wk = prob.scaled_solvent_rho(rho_final, active_mask=prob.work)
    return rho_final, {
        "selected_stage": "micelle",
        "belt": belt_summary,
        "micelle": stats_final,
        "params": p_final.tolist(),
        "r_val_base": rval_base,
        "r_val_micelle": rval_mic,
        "r_val": rval_mic,
        "min_val_gain": round(gain_req, 6),
        "r_work": float(ev_wk["r_work"]),
        "r_free": float(ev_wk["r_free"]),
    }
