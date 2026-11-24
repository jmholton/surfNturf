"""surfnturf.chemfit: Differentiable crystallographic chemical feedback (zero R-free / work_val leakage).

Feeds observed X-ray diffraction amplitudes (`90% work_train` ONLY; `5% work_val` held out as a strict blind
acceptance gate; `5% R-free` locked behind hard assertions and never read) back into both:
  1. Mother-liquor & crystal chemical environment (`CrystalChemAdapter`: 6 physical knobs modulating anionic pH/salt
     screening, neutral H-bond acceptors, NH/OH donors, apolar cosolvent contrast, pocket water activity, and
     cryo/RT tetrahedral water-network cooperativity around the pretrained MD parameterization).
  2. Ambiguous macromolecular chemical states (`ChemicalSiteBasis`: sparse Boltzmann probabilities `p_{r,s}` for
     `ASN/GLN` 180-deg amide flips, `HIS` `HID/HIE/HIP+` tautomers/protonation, `ASP/GLU` `COO-/COOH` protonation,
     and rotatable `SER/THR/TYR` `-OH` + lone-pair orientations), combining stereochemical solute H-bond priors,
     direct `IT92` atomic structure-factor shifts `Delta F_calc`, and 10-electron first-shell hydration shifts `Delta F_part`.
"""
import json
import math
import os
import numpy as np
import gemmi
from scipy import fft as sp_fft
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree
from scipy.special import expit, logsumexp

from .field import (
    ACCEPTOR_TYPES,
    DR,
    DRH,
    D_LP,
    N_HKNOTS,
    N_KNOTS,
    OZ_DAMP,
    RH_CUT,
    RH_MIN,
    RHO_BULK,
    R_CUT,
    R_MIN,
    BETA_CAV,
    N_CAV_MAX,
    ASUGrid,
    WaterFieldSpec,
    _bspline_weights,
    _dg_dU_and_dz0,
    _run_oz_and_pauli_forward,
    _unit_vec,
    fpart_grad_to_asu,
    softplus,
    solv_to_fpart,
)
from .bmob import MultiBRenderer, ReciprocalTarget, build_water_keep_mask_asu
from .hydrogens import D_NH_SP2, D_OH, _sp2_bisector, _sp2_nh2, complete_conformer_hydrogens


CHEM_KNOB_NAMES = (
    "anionic_well",      # 0: O_neg + LP_neg attractive well (pH / divalent & salt screening)
    "neutral_acc_well",  # 1: O_carb + O_hyd + LP_sp2 + LP_sp3 + w_hpol (cosolvent acceptor competition)
    "donor_well",        # 2: N_don + H_pol + H_pos attractive shell (halide/anion donor modulation)
    "apolar_contrast",   # 3: C_sp2 + C_sp3 + H_apol + S_sul 1st-shell solvation contrast (PEG/MPD wetting)
    "pocket_activity",   # 4: Langmuir chemical-potential offset logit_alpha (osmotic water activity)
    "network_coop",      # 5: OZ + TetOZ H-bond cooperativity scale (100 K cryo vs RT network stiffness)
)


class CrystalChemAdapter:
    """Smooth C-infinity 6-parameter chemical-environment adapter `theta(gamma)` around `theta_base` (`theta(0) == theta_base`)."""

    def __init__(self, spec: WaterFieldSpec, theta_base: np.ndarray):
        self.spec = spec
        self.theta_base = np.asarray(theta_base, dtype=np.float64).copy()
        self.n_knobs = 6

        w_pmf0, logit_alpha0, v_oz0, w_tet0, w_epol0, w_hpol0, _ = spec.unpack_all(self.theta_base)
        W0 = w_pmf0.reshape(spec.n_types, N_KNOTS)
        W_neg = np.minimum(W0, 0.0)

        self.M_anion = np.zeros_like(W0)
        self.M_acc = np.zeros_like(W0)
        self.M_don = np.zeros_like(W0)
        self.M_apol = np.zeros_like(W0)

        if spec.n_types >= 10:
            self.M_anion[1] = W_neg[1]
            if spec.n_types >= 13:
                self.M_anion[11] = W_neg[11]

            for t in (0, 2) + ((10, 12) if spec.n_types >= 13 else ()):
                self.M_acc[t] = W_neg[t]

            for t in (3, 4, 5):
                self.M_don[t] = W_neg[t]

            knots = R_MIN + DR * np.arange(N_KNOTS, dtype=np.float64)
            apol_win = -0.80 * expit((knots - 3.10) / 0.25) * expit((4.80 - knots) / 0.35)
            for t in (6, 7, 8, 9):
                self.M_apol[t] = apol_win
        else:
            self.M_acc[0] = W_neg[0]
            knots = R_MIN + DR * np.arange(N_KNOTS, dtype=np.float64)
            self.M_apol[1] = -0.80 * expit((knots - 3.10) / 0.25) * expit((4.80 - knots) / 0.35)

        self.w_hpol0 = w_hpol0.copy() if (spec.use_hpol and w_hpol0 is not None) else None
        self.v_oz0 = v_oz0.copy() if (spec.use_oz and v_oz0 is not None) else None
        self.w_tet0 = w_tet0.copy() if (spec.use_tet_oz and w_tet0 is not None) else None
        self.has_alpha = bool(spec.use_langmuir and logit_alpha0 is not None)
        self.B5_pmf = np.column_stack([
            w_pmf0,
            self.M_anion.ravel(),
            self.M_acc.ravel(),
            self.M_don.ravel(),
            self.M_apol.ravel(),
        ]).astype(np.float64)
        self._ds_cache = {}

    def get_ds_basis(self, ds):
        """Return cached float64 `(U5, rh0_flat)` on `ds` (eliminates float32 step quantization & 95% of BLAS ops)."""
        key = id(ds)
        if key in self._ds_cache:
            return self._ds_cache[key][1], self._ds_cache[key][2]
        n_rows = ds.M * ds.grid.n_asu
        n_pmf = self.spec.n_pmf
        if getattr(ds, "Phi_O_flat", None) is not None:
            U5 = ds.Phi_O_flat[:, :n_pmf] @ self.B5_pmf
        else:
            Phi32 = ds.Phi_O_2d_f32[:, :n_pmf]
            U5 = np.empty((n_rows, 5), dtype=np.float64)
            step = 65536
            for i0 in range(0, n_rows, step):
                i1 = min(i0 + step, n_rows)
                U5[i0:i1] = Phi32[i0:i1].astype(np.float64) @ self.B5_pmf

        rh0 = None
        if self.w_hpol0 is not None:
            if getattr(ds, "Phi_H_flat", None) is not None:
                rh0 = ds.Phi_H_flat @ self.w_hpol0
            elif getattr(ds, "Phi_H_2d_f32", None) is not None:
                PhiH32 = ds.Phi_H_2d_f32
                rh0 = np.empty(n_rows, dtype=np.float64)
                step = 65536
                for i0 in range(0, n_rows, step):
                    i1 = min(i0 + step, n_rows)
                    rh0[i0:i1] = PhiH32[i0:i1].astype(np.float64) @ self.w_hpol0
        self._ds_cache[key] = (ds, U5, rh0)
        return U5, rh0

    def bounds(self):
        return [(-0.65, 0.65)] * self.n_knobs

    def theta_and_pullback(self, gamma):
        g = np.asarray(gamma, dtype=np.float64)
        spec = self.spec
        th = self.theta_base.copy()

        eg0 = math.exp(float(g[0]))
        eg1 = math.exp(float(g[1]))
        eg2 = math.exp(float(g[2]))
        g3 = float(g[3])
        g4 = float(g[4])
        eg5 = math.exp(float(g[5]))

        dW = (
            (eg0 - 1.0) * self.M_anion
            + (eg1 - 1.0) * self.M_acc
            + (eg2 - 1.0) * self.M_don
            + g3 * self.M_apol
        )
        th[:spec.n_pmf] += dW.ravel()

        idx = spec.n_pmf
        idx_alpha = idx if self.has_alpha else None
        if self.has_alpha:
            th[idx] = float(th[idx] + 1.25 * g4)
            idx += 1
        idx_oz = idx if self.v_oz0 is not None else None
        if self.v_oz0 is not None:
            th[idx:idx + spec.n_oz] = eg5 * self.v_oz0
            idx += spec.n_oz
        idx_tet = idx if self.w_tet0 is not None else None
        if self.w_tet0 is not None:
            th[idx:idx + 3] = eg5 * self.w_tet0[:3]
            idx += 4
        if spec.use_epol:
            idx += spec.n_epol
        idx_hpol = idx if self.w_hpol0 is not None else None
        if self.w_hpol0 is not None:
            th[idx:idx + spec.n_hpol] = eg1 * self.w_hpol0

        def pullback(dL_dtheta):
            gt = np.asarray(dL_dtheta, dtype=np.float64)
            gW = gt[:spec.n_pmf].reshape(spec.n_types, N_KNOTS)
            gg = np.zeros(6, dtype=np.float64)
            gg[0] = eg0 * float(np.sum(gW * self.M_anion))
            gg[1] = eg1 * float(np.sum(gW * self.M_acc))
            if idx_hpol is not None:
                gg[1] += eg1 * float(np.dot(gt[idx_hpol:idx_hpol + spec.n_hpol], self.w_hpol0))
            gg[2] = eg2 * float(np.sum(gW * self.M_don))
            gg[3] = float(np.sum(gW * self.M_apol))
            if self.has_alpha:
                gg[4] = 1.25 * float(gt[idx_alpha])
            if idx_oz is not None:
                gg[5] += eg5 * float(np.dot(gt[idx_oz:idx_oz + spec.n_oz], self.v_oz0))
            if idx_tet is not None:
                gg[5] += eg5 * float(np.dot(gt[idx_tet:idx_tet + 3], self.w_tet0[:3]))
            return gg

        return th, pullback


def _it92_form_factor(elem_symbol: str, stol2: np.ndarray) -> np.ndarray:
    it = gemmi.Element(elem_symbol).it92
    f = np.full_like(stol2, float(it.c), dtype=np.float64)
    for ai, bi in zip(it.a, it.b):
        f = f + float(ai) * np.exp(-float(bi) * stol2)
    return f


def _calc_atom_list_fcalc(cell: gemmi.UnitCell, sg: gemmi.SpaceGroup, hkl: np.ndarray,
                          stol2: np.ndarray, atom_specs, ff_cache: dict) -> np.ndarray:
    """Exact crystallographic IT92 structure factor sum for a small list of `(xyz_cart, elem, biso, occ)` atoms."""
    if not atom_specs:
        return np.zeros(len(hkl), dtype=np.complex128)
    F_mat = np.array(cell.frac.mat.tolist(), dtype=np.float64)
    ops = [(np.array(op.rot, dtype=np.float64) / op.DEN, np.array(op.tran, dtype=np.float64) / op.DEN)
           for op in sg.operations()]
    H = np.asarray(hkl, dtype=np.float64)
    out = np.zeros(len(H), dtype=np.complex128)
    twopi_i = 2j * math.pi
    for xyz_c, elem, biso, occ in atom_specs:
        el = "H" if elem.strip().upper() == "D" else elem.strip().upper()
        if el not in ff_cache:
            ff_cache[el] = _it92_form_factor(el, stol2)
        amp = float(occ) * ff_cache[el] * np.exp(-float(biso) * stol2)
        xf = F_mat @ np.asarray(xyz_c, dtype=np.float64)
        ph_sum = np.zeros(len(H), dtype=np.complex128)
        for R_op, t_op in ops:
            xr = R_op @ xf + t_op
            ph_sum += np.exp(twopi_i * (H @ xr))
        out += amp * ph_sum
    return out


def _rotate_around_axis(pt, origin, axis_unit, angle_rad):
    v = np.asarray(pt, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    k = np.asarray(axis_unit, dtype=np.float64)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    v_rot = v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)
    return np.asarray(origin, dtype=np.float64) + v_rot


def _sp2_oxygen_lps(o_pos, c_pos, plane_ref1, plane_ref2, lp_class: int):
    u = _unit_vec(o_pos - c_pos)
    if u is None:
        return []
    p_vec = plane_ref1 - plane_ref2
    v = _unit_vec(p_vec - float(np.dot(p_vec, u)) * u)
    if v is None:
        return [(o_pos + D_LP * u, lp_class)]
    c60, s60 = 0.5, 0.5 * math.sqrt(3.0)
    return [
        (o_pos + D_LP * (c60 * u + s60 * v), lp_class),
        (o_pos + D_LP * (c60 * u - s60 * v), lp_class),
    ]


def _amide_nh2_protons(n_pos, c_pos, o_pos, r_nh: float = D_NH_SP2):
    pair = _sp2_nh2(n_pos, c_pos, o_pos, d=float(r_nh))
    if pair is not None:
        return [pair[0], pair[1]]
    u_cn = _unit_vec(n_pos - c_pos)
    u_co = _unit_vec(o_pos - c_pos)
    if u_cn is None or u_co is None:
        return []
    v_in = _unit_vec(u_co - float(np.dot(u_co, u_cn)) * u_cn)
    if v_in is None:
        return []
    c60, s60 = 0.5, 0.5 * math.sqrt(3.0)
    return [
        n_pos + float(r_nh) * (c60 * u_cn + s60 * v_in),
        n_pos + float(r_nh) * (c60 * u_cn - s60 * v_in),
    ]


def _sp3_hydroxyl_lps(o_pos, c_pos, h_pos):
    a_u = _unit_vec(c_pos - o_pos)
    b_u = _unit_vec(h_pos - o_pos)
    if a_u is None or b_u is None:
        return []
    u = _unit_vec(-(a_u + b_u))
    w = _unit_vec(np.cross(a_u, b_u))
    if u is None or w is None:
        return []
    c_tet, s_tet = 1.0 / math.sqrt(3.0), math.sqrt(2.0 / 3.0)
    return [
        (o_pos + D_LP * (c_tet * u + s_tet * w), 12),
        (o_pos + D_LP * (c_tet * u - s_tet * w), 12),
    ]


def _local_asu_fields(grid: ASUGrid, tree_pts: cKDTree, local_atoms, n_types: int, include_hpol: bool):
    """Evaluate exact local `(csr_O, csr_H, lster_vec)` on `ASUGrid` in vectorized C/CSR (< 1 ms per state)."""
    n_asu = grid.n_asu
    n_pmf = int(n_types) * N_KNOTS
    n_hpol = len(ACCEPTOR_TYPES) * N_HKNOTS
    empty_o = csr_matrix((n_asu, n_pmf), dtype=np.float64)
    empty_h = csr_matrix((n_asu, max(n_hpol, 1)), dtype=np.float64)
    empty_st = np.zeros(n_asu, dtype=np.float64)
    if not local_atoms:
        return empty_o, empty_h, empty_st

    xyz = np.array([p for p, _ in local_atoms], dtype=np.float64).reshape(-1, 3)
    typ = np.array([int(t) for _, t in local_atoms], dtype=np.int32)
    if int(n_types) == 10 and np.any(typ >= 10):
        keep = typ < 10
        xyz, typ = xyz[keep], typ[keep]
    if len(xyz) == 0:
        return empty_o, empty_h, empty_st

    SX, Si = grid.sym_expand(xyz, cutoff_A=R_CUT, dedup_self=True)
    if len(SX) == 0:
        return empty_o, empty_h, empty_st
    St = typ[Si]
    nb = tree_pts.query_ball_point(SX, R_CUT)
    lens = np.array([len(x) for x in nb], dtype=np.int32)
    if lens.sum() == 0:
        return empty_o, empty_h, empty_st

    pts = grid.asu_xyz
    ai = np.repeat(np.arange(len(SX), dtype=np.int32), lens)
    pi = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb if len(x)])
    d = np.linalg.norm(pts[pi] - SX[ai], axis=1)
    ti = St[ai]

    r_o, c_o, v_o = [], [], []
    for kk, bw in _bspline_weights(d, R_MIN, DR, N_KNOTS):
        valid = (kk < N_KNOTS) & (bw > 1e-12)
        if np.any(valid):
            r_o.append(pi[valid])
            c_o.append(ti[valid] * N_KNOTS + kk[valid])
            v_o.append(bw[valid])
    csr_o = (
        coo_matrix((np.concatenate(v_o), (np.concatenate(r_o), np.concatenate(c_o))),
                   shape=(n_asu, n_pmf), dtype=np.float64).tocsr()
        if r_o else empty_o
    )

    csr_h = empty_h
    lster = empty_st
    if include_hpol:
        real_m = (ti < 10) if int(n_types) > 2 else np.ones_like(ti, dtype=bool)
        if np.any(real_m):
            pr, dr, tr = pi[real_m], d[real_m], ti[real_m]
            is_h_atom = ((tr == 4) | (tr == 5) | (tr == 8)) if int(n_types) > 2 else np.zeros_like(tr, dtype=bool)
            r_clash = np.where(is_h_atom, 1.25, 1.55)
            close = dr < (r_clash + 0.80)
            if np.any(close):
                l_excl = -np.logaddexp(0.0, -(dr[close] - r_clash[close]) / 0.12)
                lster = np.bincount(pr[close], weights=l_excl, minlength=n_asu)

            acc_mask = (((tr == 0) | (tr == 1) | (tr == 2)) if int(n_types) > 2 else (tr == 0)) & (dr < RH_CUT)
            if np.any(acc_mask):
                pi_a, d_a, ti_a = pr[acc_mask], dr[acc_mask], tr[acc_mask]
                r_h, c_h, v_h = [], [], []
                for kk, bw in _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS):
                    valid = (kk < N_HKNOTS) & (bw > 1e-12)
                    if np.any(valid):
                        r_h.append(pi_a[valid])
                        c_h.append(ti_a[valid] * N_HKNOTS + kk[valid])
                        v_h.append(bw[valid])
                if r_h:
                    csr_h = coo_matrix(
                        (np.concatenate(v_h), (np.concatenate(r_h), np.concatenate(c_h))),
                        shape=(n_asu, max(n_hpol, 1)), dtype=np.float64,
                    ).tocsr()
    return csr_o, csr_h, lster


def _diff_sparse_matrix(csr_alt: csr_matrix, csr_ref: csr_matrix, n_cols: int):
    diff = (csr_alt - csr_ref).tocsr()
    if diff.nnz == 0:
        return np.zeros(0, dtype=np.int32), csr_matrix((0, n_cols), dtype=np.float64)
    diff.data[np.abs(diff.data) <= 1e-11] = 0.0
    diff.eliminate_zeros()
    if diff.nnz == 0:
        return np.zeros(0, dtype=np.int32), csr_matrix((0, n_cols), dtype=np.float64)
    u_asu = np.nonzero(np.diff(diff.indptr) > 0)[0].astype(np.int32)
    return u_asu, diff[u_asu]


def _diff_sparse_vec(v_alt: np.ndarray, v_ref: np.ndarray):
    dv = v_alt - v_ref
    u_idx = np.nonzero(np.abs(dv) > 1e-11)[0].astype(np.int32)
    if len(u_idx) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    return u_idx, dv[u_idx].astype(np.float64)


class ChemicalSiteBasis:
    """Sparse precomputed ASU hydration & IT92 `Delta F_calc` basis over ambiguous macromolecular chemical sites."""

    def __init__(self, sites, alt_records, n_pmf: int, n_hpol: int):
        self.sites = sites                  # list of site metadata dicts
        self.alt_records = alt_records      # flat list of length n_zeta (one per non-reference state s >= 1)
        self.n_sites = len(sites)
        self.n_zeta = len(alt_records)
        self.n_pmf = int(n_pmf)
        self.n_hpol = int(n_hpol)

    def probs_and_kl(self, zeta):
        """Return `(p_alt_flat, site_probs_list, kl_sum, dkl_dzeta)` for unconstrained `zeta` (`zeta_{r,0} == 0`)."""
        z = np.asarray(zeta, dtype=np.float64)
        p_alt = np.zeros(self.n_zeta, dtype=np.float64)
        site_probs = []
        kl_tot = 0.0
        dkl_dz = np.zeros(self.n_zeta, dtype=np.float64)

        for st in self.sites:
            z0 = st["log_prior"]
            sl = st["zeta_slice"]
            u = z0.copy()
            u[1:] += z[sl]
            log_p = u - float(logsumexp(u))
            p = np.exp(log_p)
            log_p0 = st["log_p0"]
            diff_lp = log_p - log_p0
            kl_r = float(np.dot(p, diff_lp))
            kl_tot += kl_r
            # Exact gradient d D_KL(p || p0) / du_s = p_s * ((log p_s - log p0_s) - kl_r)
            dkl_dz[sl] = p[1:] * (diff_lp[1:] - kl_r)
            p_alt[sl] = p[1:]
            site_probs.append(p)
        return p_alt, site_probs, kl_tot, dkl_dz

    def pullback_probs_to_zeta(self, site_probs, dL_dp_alt):
        """Pull gradient `dL / dp_{r,s}` (`s >= 1`, `dL / dp_{r,0} == 0`) back through per-site softmax to `zeta`."""
        gp = np.asarray(dL_dp_alt, dtype=np.float64)
        gz = np.zeros(self.n_zeta, dtype=np.float64)
        for st, p in zip(self.sites, site_probs):
            sl = st["zeta_slice"]
            g_full = np.zeros(len(p), dtype=np.float64)
            g_full[1:] = gp[sl]
            mean_g = float(np.dot(p, g_full))
            gz[sl] = p[1:] * (g_full[1:] - mean_g)
        return gz


def build_chemical_site_basis(
    pdb_path: str,
    grid: ASUGrid,
    spec: WaterFieldSpec,
    hkl: np.ndarray = None,
    d_hkl: np.ndarray = None,
    site_families=("asn_gln", "his", "asp_glu", "hydroxyl"),
    max_sites: int = 160,
) -> ChemicalSiteBasis:
    """Enumerate single-conformer `ASN/GLN`, `HIS`, `ASP/GLU`, and `SER/THR/TYR` sites and precompute sparse deltas.

    Priors `log_prior` are computed strictly from macromolecular solute coordinates and crystal contacts (ZERO `f_obs`
    access), guaranteeing zero leakage into `work_val` or `R_free`.
    """
    n_pmf = spec.n_pmf
    n_hpol = spec.n_hpol
    if not os.path.isfile(str(pdb_path)) or spec.n_types < 10:
        return ChemicalSiteBasis([], [], n_pmf, n_hpol)

    st = gemmi.read_structure(str(pdb_path))
    if len(st) == 0:
        return ChemicalSiteBasis([], [], n_pmf, n_hpol)

    # Collect all solute polar O/N and run complete_conformer_hydrogens once so state-0 H positions
    # match load_ensemble_with_bfac / build_asu_features to machine precision!
    pol_xyz = []
    pdb_has_explicit_h = False
    raw_conf_recs = []
    for ch_idx, ch in enumerate(st[0]):
        ch_id = ch.name.strip() or f"#{ch_idx}"
        for r in ch:
            if r.name.strip().upper() in ("HOH", "WAT", "DOD"):
                continue
            ic_id = r.seqid.icode.strip() if r.seqid.icode != "\0" else ""
            for a in r:
                if a.occ <= 0.0:
                    continue
                el_u = a.element.name.strip().upper()
                if el_u in ("H", "D"):
                    pdb_has_explicit_h = True
                elif el_u in ("O", "N", "S"):
                    pol_xyz.append([a.pos.x, a.pos.y, a.pos.z])
                if a.altloc == "\0" or not a.altloc.strip():
                    raw_conf_recs.append((
                        int(r.seqid.num), r.name.strip().upper(), a.name.strip().upper(),
                        el_u, [a.pos.x, a.pos.y, a.pos.z], ch_id, ic_id
                    ))
    completed_recs = complete_conformer_hydrogens(raw_conf_recs)
    comp_h_by_res = {}
    for rec in completed_recs:
        if rec[3] in ("H", "D"):
            rkey = (rec[5], int(rec[0]), rec[6], rec[1])
            comp_h_by_res.setdefault(rkey, {})[rec[2].strip().upper()] = np.asarray(rec[4], dtype=np.float64)
    pol_arr = np.asarray(pol_xyz, dtype=np.float64).reshape(-1, 3)
    sym_pol, _ = grid.sym_expand(pol_arr, cutoff_A=4.0, dedup_self=True) if len(pol_arr) else (np.zeros((0, 3)), None)
    tree_pol = cKDTree(sym_pol) if len(sym_pol) else None
    tree_pts = cKDTree(grid.asu_xyz)

    stol2 = (0.25 / np.maximum(np.asarray(d_hkl, dtype=np.float64), 1e-6) ** 2) if d_hkl is not None else None
    ff_cache = {}
    use_lp = bool(spec.use_lp)
    fams = set(site_families)

    def _hbond_donor_bonus(d_pos, h_pos):
        """Purely coordinate-based donor-to-solute-acceptor H-bond bonus (excludes self within 2.35 A)."""
        if tree_pol is None:
            return 0.0
        nbrs = tree_pol.query_ball_point(d_pos, 3.25)
        u_dh = _unit_vec(h_pos - d_pos)
        if u_dh is None:
            return 0.0
        bonus = 0.0
        for j in nbrs:
            vec = sym_pol[j] - d_pos
            dist = float(np.linalg.norm(vec))
            if dist < 2.38:
                continue
            cos_ang = float(np.dot(u_dh, vec / dist))
            if cos_ang > 0.55:
                bonus += 1.4 * cos_ang * float(expit((3.10 - dist) / 0.18))
        return min(bonus, 2.8)

    def _acceptor_clash_penalty(o_pos):
        """Penalize placing a bare carbonyl O within <2.85 A of surrounding unprotonated carbonyl/anionic oxygens."""
        if tree_pol is None:
            return 0.0
        nbrs = tree_pol.query_ball_point(o_pos, 2.75)
        pen = 0.0
        for j in nbrs:
            dist = float(np.linalg.norm(sym_pol[j] - o_pos))
            if 2.10 < dist < 2.75:
                pen += 0.6 * float(expit((2.65 - dist) / 0.12))
        return min(pen, 1.8)

    raw_candidates = []
    for ch_idx, ch in enumerate(st[0]):
        ch_id = ch.name.strip() or f"#{ch_idx}"
        for r in ch:
            rn = r.name.strip().upper()
            if rn not in ("ASN", "GLN", "HIS", "HID", "HIE", "HIP", "ASP", "GLU", "SER", "THR", "TYR"):
                continue
            # Restrict to single-conformer (blank-altloc) residues so local site deltas apply cleanly across all M
            has_alt = any((a.altloc != "\0" and a.altloc.strip() != "") and a.occ > 0.0 for a in r)
            if has_alt:
                continue
            amap = {}
            for a in r:
                if a.occ > 0.0:
                    amap[a.name.strip().upper()] = (
                        np.array([a.pos.x, a.pos.y, a.pos.z], dtype=np.float64),
                        a.element.name.strip().upper(),
                        float(a.b_iso),
                        float(a.occ),
                    )
            ic_id = r.seqid.icode.strip() if r.seqid.icode != "\0" else ""
            res_tag = f"{ch_id}:{rn}{r.seqid.num}{ic_id}"
            comp_h = comp_h_by_res.get((ch_id, int(r.seqid.num), ic_id, rn), {})

            # 1. ASN / GLN 180-deg amide flip
            if rn in ("ASN", "GLN") and "asn_gln" in fams:
                c_nm, o_nm, n_nm, cb_nm, h1_nm, h2_nm = (
                    ("CG", "OD1", "ND2", "CB", "HD21", "HD22") if rn == "ASN"
                    else ("CD", "OE1", "NE2", "CG", "HE21", "HE22")
                )
                if not all(k in amap for k in (c_nm, o_nm, n_nm, cb_nm)):
                    continue
                p_c, _, _, _ = amap[c_nm]
                p_o, _, b_o, occ_o = amap[o_nm]
                p_n, _, b_n, occ_n = amap[n_nm]
                b_h = 1.2 * b_n
                occ_m = 0.5 * (occ_o + occ_n)

                has_amide_h = (h1_nm in amap and h2_nm in amap)
                if has_amide_h:
                    hs_0 = [amap[h1_nm][0], amap[h2_nm][0]]
                elif h1_nm in comp_h and h2_nm in comp_h:
                    hs_0 = [comp_h[h1_nm], comp_h[h2_nm]]
                else:
                    hs_0 = _amide_nh2_protons(p_n, p_c, p_o)
                hs_1 = _amide_nh2_protons(p_o, p_c, p_n)
                if len(hs_0) < 2 or len(hs_1) < 2:
                    continue

                loc_0 = [(p_o, 0), (p_n, 3), (hs_0[0], 4), (hs_0[1], 4)]
                loc_1 = [(p_n, 0), (p_o, 3), (hs_1[0], 4), (hs_1[1], 4)]
                if use_lp:
                    loc_0.extend(_sp2_oxygen_lps(p_o, p_c, p_n, amap[cb_nm][0], 10))
                    loc_1.extend(_sp2_oxygen_lps(p_n, p_c, p_o, amap[cb_nm][0], 10))

                # Match target.f_calc: include riding H in Delta F_calc iff the input PDB carries explicit amide H!
                fc_0 = [(p_o, "O", b_o, occ_o), (p_n, "N", b_n, occ_n)]
                fc_1 = [(p_n, "O", b_n, occ_o), (p_o, "N", b_o, occ_n)]
                if has_amide_h:
                    fc_0.extend([(hs_0[0], "H", b_h, occ_m), (hs_0[1], "H", b_h, occ_m)])
                    fc_1.extend([(hs_1[0], "H", b_h, occ_m), (hs_1[1], "H", b_h, occ_m)])
                hb0 = _hbond_donor_bonus(p_n, hs_0[0]) + _hbond_donor_bonus(p_n, hs_0[1]) - _acceptor_clash_penalty(p_o)
                hb1 = _hbond_donor_bonus(p_o, hs_1[0]) + _hbond_donor_bonus(p_o, hs_1[1]) - _acceptor_clash_penalty(p_n)
                z_prior = np.array([0.45 + hb0, hb1], dtype=np.float64)
                raw_candidates.append((
                    10, res_tag, "asn_gln", ("as_built", "flip_180"), z_prior,
                    [loc_0, loc_1], [fc_0, fc_1],
                    {"chain": ch_id, "seqnum": r.seqid.num, "icode": ic_id, "resname": rn,
                     "o_name": o_nm, "n_name": n_nm, "h_names": (h1_nm, h2_nm), "hs_flip": hs_1},
                ))

            # 2. HIS tautomer & protonation (HID / HIE / HIP+)
            elif rn in ("HIS", "HID", "HIE", "HIP") and "his" in fams:
                if not all(k in amap for k in ("CG", "ND1", "CE1", "NE2", "CD2")):
                    continue
                p_cg = amap["CG"][0]
                p_nd1, _, b_nd1, occ_nd1 = amap["ND1"]
                p_ce1 = amap["CE1"][0]
                p_ne2, _, b_ne2, occ_ne2 = amap["NE2"]
                p_cd2 = amap["CD2"][0]

                u_d1 = _unit_vec(_unit_vec(p_nd1 - p_cg) + _unit_vec(p_nd1 - p_ce1))
                u_e2 = _unit_vec(_unit_vec(p_ne2 - p_ce1) + _unit_vec(p_ne2 - p_cd2))
                if u_d1 is None or u_e2 is None:
                    continue
                hd1_bis = _sp2_bisector(p_nd1, p_cg, p_ce1, D_NH_SP2)
                he2_bis = _sp2_bisector(p_ne2, p_ce1, p_cd2, D_NH_SP2)
                hd1_pos = amap["HD1"][0] if "HD1" in amap else comp_h.get("HD1", hd1_bis if hd1_bis is not None else p_nd1 + D_NH_SP2 * u_d1)
                he2_pos = amap["HE2"][0] if "HE2" in amap else comp_h.get("HE2", he2_bis if he2_bis is not None else p_ne2 + D_NH_SP2 * u_e2)
                lp_d1 = p_nd1 + D_LP * u_d1
                lp_e2 = p_ne2 + D_LP * u_e2

                has_hd1 = ("HD1" in amap) or ("HD1" in comp_h) or (rn == "HID")
                has_he2 = ("HE2" in amap) or ("HE2" in comp_h) or (rn == "HIE")
                if rn == "HIP" or (("HD1" in amap or "HD1" in comp_h) and ("HE2" in amap or "HE2" in comp_h)):
                    init_state = "HIP"
                elif has_hd1 and not has_he2:
                    init_state = "HID"
                else:
                    init_state = "HIE"

                def _his_state(tag):
                    if tag == "HID":
                        la = [(p_nd1, 3), (hd1_pos, 4), (p_ne2, 2)] + ([(lp_e2, 10)] if use_lp else [])
                        fc = [(hd1_pos, "H", 1.2 * b_nd1, occ_nd1)] if pdb_has_explicit_h else []
                        zp = -0.25 + _hbond_donor_bonus(p_nd1, hd1_pos)
                    elif tag == "HIE":
                        la = [(p_nd1, 2), (p_ne2, 3), (he2_pos, 4)] + ([(lp_d1, 10)] if use_lp else [])
                        fc = [(he2_pos, "H", 1.2 * b_ne2, occ_ne2)] if pdb_has_explicit_h else []
                        zp = 0.00 + _hbond_donor_bonus(p_ne2, he2_pos)
                    else:
                        la = [(p_nd1, 3), (hd1_pos, 5), (p_ne2, 3), (he2_pos, 5)]
                        fc = (
                            [(hd1_pos, "H", 1.2 * b_nd1, occ_nd1), (he2_pos, "H", 1.2 * b_ne2, occ_ne2)]
                            if pdb_has_explicit_h else []
                        )
                        zp = -0.85 + 0.8 * (_hbond_donor_bonus(p_nd1, hd1_pos) + _hbond_donor_bonus(p_ne2, he2_pos))
                    return la, fc, zp

                order = [init_state] + [s for s in ("HIE", "HID", "HIP") if s != init_state]
                locs, fcs, zps = zip(*[_his_state(s) for s in order])
                z_prior = np.array(zps, dtype=np.float64)
                z_prior[0] += 0.40
                raw_candidates.append((
                    9, res_tag, "his", tuple(order), z_prior, list(locs), list(fcs),
                    {"chain": ch_id, "seqnum": r.seqid.num, "icode": ic_id, "resname": rn,
                     "hd1_pos": hd1_pos, "he2_pos": he2_pos},
                ))

            # 3. ASP / GLU carboxylate vs neutral COOH
            elif rn in ("ASP", "GLU") and "asp_glu" in fams:
                c_nm, o1_nm, o2_nm, cb_nm = (
                    ("CG", "OD1", "OD2", "CB") if rn == "ASP" else ("CD", "OE1", "OE2", "CG")
                )
                if not all(k in amap for k in (c_nm, o1_nm, o2_nm, cb_nm)):
                    continue
                p_c = amap[c_nm][0]
                p_cb = amap[cb_nm][0]
                p_o1, _, b_o1, occ_o1 = amap[o1_nm]
                p_o2, _, b_o2, occ_o2 = amap[o2_nm]

                # Syn-planar carboxyl H positions (O-H = D_OH = 0.84 A, C-O-H = 112 deg toward the other O)
                def _cooh_h(p_oh, p_other):
                    u_oc = _unit_vec(p_c - p_oh)
                    u_oo = _unit_vec(p_other - p_oh)
                    if u_oc is None or u_oo is None:
                        return None
                    v_p = _unit_vec(u_oo - float(np.dot(u_oo, u_oc)) * u_oc)
                    if v_p is None:
                        return None
                    ang = math.radians(112.0)
                    return p_oh + D_OH * (math.cos(ang) * u_oc + math.sin(ang) * v_p)

                h_o1 = _cooh_h(p_o1, p_o2)
                h_o2 = _cooh_h(p_o2, p_o1)
                if h_o1 is None or h_o2 is None:
                    continue

                loc_0 = [(p_o1, 1), (p_o2, 1)]
                loc_1 = [(p_o1, 2), (h_o1, 4), (p_o2, 0)]
                loc_2 = [(p_o1, 0), (p_o2, 2), (h_o2, 4)]
                if use_lp:
                    loc_0.extend(_sp2_oxygen_lps(p_o1, p_c, p_o2, p_cb, 11))
                    loc_0.extend(_sp2_oxygen_lps(p_o2, p_c, p_o1, p_cb, 11))
                    loc_1.extend(_sp3_hydroxyl_lps(p_o1, p_c, h_o1))
                    loc_1.extend(_sp2_oxygen_lps(p_o2, p_c, p_o1, p_cb, 10))
                    loc_2.extend(_sp2_oxygen_lps(p_o1, p_c, p_o2, p_cb, 10))
                    loc_2.extend(_sp3_hydroxyl_lps(p_o2, p_c, h_o2))

                fc_0 = []
                fc_1 = [(h_o1, "H", 1.2 * b_o1, occ_o1)] if pdb_has_explicit_h else []
                fc_2 = [(h_o2, "H", 1.2 * b_o2, occ_o2)] if pdb_has_explicit_h else []
                z_prior = np.array([
                    2.60,
                    _hbond_donor_bonus(p_o1, h_o1),
                    _hbond_donor_bonus(p_o2, h_o2),
                ], dtype=np.float64)
                raw_candidates.append((
                    6, res_tag, "asp_glu", ("COO-", f"{o1_nm}_H", f"{o2_nm}_H"), z_prior,
                    [loc_0, loc_1, loc_2], [fc_0, fc_1, fc_2],
                    {"chain": ch_id, "seqnum": r.seqid.num, "icode": ic_id, "resname": rn},
                ))

            # 4. SER / THR / TYR rotatable -OH + LP_sp3
            elif rn in ("SER", "THR", "TYR") and "hydroxyl" in fams:
                c_nm, o_nm, ref_nm, h_nm = (
                    ("CB", "OG", "CA", "HG") if rn == "SER"
                    else (("CB", "OG1", "CA", "HG1") if rn == "THR" else ("CZ", "OH", "CE1", "HH"))
                )
                if not all(k in amap for k in (c_nm, o_nm, ref_nm)):
                    continue
                p_c = amap[c_nm][0]
                p_o, _, b_o, occ_o = amap[o_nm]
                p_ref = amap[ref_nm][0]
                axis_co = _unit_vec(p_o - p_c)
                if axis_co is None:
                    continue

                if h_nm in amap:
                    h0 = amap[h_nm][0]
                elif h_nm in comp_h:
                    h0 = comp_h[h_nm]
                else:
                    u_oc = -axis_co
                    v_cr = p_ref - p_c
                    p_perp = _unit_vec(v_cr - float(np.dot(v_cr, u_oc)) * u_oc)
                    if p_perp is None:
                        continue
                    ang = math.radians(109.5)
                    h0 = p_o + D_OH * (math.cos(ang) * u_oc - math.sin(ang) * p_perp)

                angles = (0.0, math.radians(120.0), math.radians(-120.0)) if rn != "TYR" else (0.0, math.pi)
                labels = ("rot_0", "rot_+120", "rot_-120") if rn != "TYR" else ("rot_0", "rot_180")
                locs, fcs, zps = [], [], []
                for idx_a, ang in enumerate(angles):
                    h_s = h0 if idx_a == 0 else _rotate_around_axis(h0, p_o, axis_co, ang)
                    la = [(h_s, 4)] + (_sp3_hydroxyl_lps(p_o, p_c, h_s) if use_lp else [])
                    locs.append(la)
                    fcs.append([(h_s, "H", 1.2 * b_o, occ_o)] if (h_nm in amap) else [])
                    zps.append((0.35 if idx_a == 0 else 0.0) + _hbond_donor_bonus(p_o, h_s))
                raw_candidates.append((
                    4, res_tag, "hydroxyl", labels, np.asarray(zps, dtype=np.float64),
                    locs, fcs,
                    {"chain": ch_id, "seqnum": r.seqid.num, "icode": ic_id, "resname": rn, "h_name": h_nm},
                ))

    # Prioritize highest-leverage chemical sites (ASN/GLN flips & HIS tautomers first) if capping at max_sites
    raw_candidates.sort(key=lambda x: -x[0])
    raw_candidates = raw_candidates[:int(max_sites)]

    sites = []
    alt_records = []
    z_cursor = 0
    for _prio, res_tag, fam, state_labels, z_prior, locs, fcs, meta in raw_candidates:
        o_ref, h_ref, st_ref = _local_asu_fields(grid, tree_pts, locs[0], spec.n_types, spec.use_hpol)
        fc_ref = (
            _calc_atom_list_fcalc(grid.cell, grid.sg, hkl, stol2, fcs[0], ff_cache)
            if (hkl is not None and stol2 is not None) else None
        )
        n_alt = len(locs) - 1
        any_effect = False
        site_alts = []
        for s in range(1, len(locs)):
            o_s, h_s, st_s = _local_asu_fields(grid, tree_pts, locs[s], spec.n_types, spec.use_hpol)
            u_o, csr_o = _diff_sparse_matrix(o_s, o_ref, n_pmf)
            u_h, csr_h = _diff_sparse_matrix(h_s, h_ref, n_hpol) if spec.use_hpol else (
                np.zeros(0, dtype=np.int32), csr_matrix((0, max(n_hpol, 1)), dtype=np.float64)
            )
            u_st, d_st = _diff_sparse_vec(st_s, st_ref) if spec.use_hpol else (
                np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
            )
            dfc = (
                _calc_atom_list_fcalc(grid.cell, grid.sg, hkl, stol2, fcs[s], ff_cache) - fc_ref
                if fc_ref is not None else None
            )
            if len(u_o) > 0 or len(u_h) > 0 or (dfc is not None and float(np.max(np.abs(dfc))) > 1e-6):
                any_effect = True
            site_alts.append({
                "u_o": u_o, "csr_o": csr_o,
                "u_h": u_h, "csr_h": csr_h,
                "u_st": u_st, "d_st": d_st,
                "dfc": dfc,
                "state_label": state_labels[s],
                "res_tag": res_tag,
                "family": fam,
            })
        if not any_effect:
            continue
        sl = slice(z_cursor, z_cursor + n_alt)
        z_cursor += n_alt
        log_p0 = z_prior - float(logsumexp(z_prior))
        sites.append({
            "res_tag": res_tag,
            "family": fam,
            "state_labels": state_labels,
            "log_prior": z_prior,
            "log_p0": log_p0,
            "p0": np.exp(log_p0),
            "zeta_slice": sl,
            "meta": meta,
        })
        alt_records.extend(site_alts)

    return ChemicalSiteBasis(sites, alt_records, n_pmf, n_hpol)


def _pullback_oz_and_pauli_to_Usol(spec: WaterFieldSpec, grid: ASUGrid, mw: np.ndarray,
                                   U_sol: np.ndarray, v_oz, w_tet, tape, dL_dgO_m: np.ndarray, w_fft: int = 1):
    """Exact reverse-mode pullback of `_run_oz_and_pauli_forward` returning `(dL_dUsol, grad_logit_alpha, grad_voz, grad_wtet)`."""
    g0_m, alpha, step_tape, cav_cache, F_hbs = tape
    n_asu = grid.n_asu

    dL_dg_cur = np.asarray(dL_dgO_m, dtype=np.float64)
    if spec.use_pauli_cap and cav_cache is not None:
        g_raw_m, n_cav_asu, inv_D, K_cav_conj, vox_mol_c = cav_cache
        dL_dgraw_m = dL_dg_cur * inv_D[None, :]
        dL_dDcav = -np.sum(dL_dg_cur * g_raw_m, axis=0) * (inv_D * inv_D)
        dL_dncav_asu = dL_dDcav * expit(BETA_CAV * (n_cav_asu - N_CAV_MAX))
        dL_dncav_3d = grid.asu_to_full_3d(dL_dncav_asu / grid.orbit_mult)
        dL_dgraw_3d = vox_mol_c * sp_fft.irfftn(
            K_cav_conj * sp_fft.rfftn(dL_dncav_3d, axes=(0, 1, 2), workers=w_fft),
            s=grid.shape, axes=(0, 1, 2), workers=w_fft,
        )
        dL_dgraw_bar = grid.full_3d_to_asu_grad(dL_dgraw_3d)
        dL_dg_cur = dL_dgraw_m + np.outer(mw, dL_dgraw_bar)

    dL_dUsol = np.zeros_like(U_sol, dtype=np.float64)
    grad_logit_alpha = 0.0
    grad_voz = np.zeros(spec.n_oz, dtype=np.float64)
    grad_wtet = np.zeros(4, dtype=np.float64) if spec.use_tet_oz else None

    if spec.use_oz:
        K_steps = len(step_tape)
        eta = 1.0 if K_steps == 1 else OZ_DAMP
        vox_mol = (float(grid.cell.volume) / float(grid.nvox)) * float(grid.n_bulk)
        xi_bulk = math.log(2.0) / 4.0
        W_hat = np.tensordot(v_oz, grid.oz_kernels[:spec.n_oz], axes=(0, 0))

        if spec.use_tet_oz:
            w_frust, w_tcoop, w_nc, raw_bsol = [float(x) for x in w_tet]
            dsig_bsol = float(expit(raw_bsol))
            tk = grid.get_tet_kernels()
            rho_sol_3d = (
                sp_fft.irfftn(F_hbs, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
                if F_hbs is not None else None
            )
        else:
            w_frust = w_tcoop = w_nc = dsig_bsol = 0.0
            tk = None
            rho_sol_3d = None

        dL_ddU_cum_next = np.zeros(n_asu, dtype=np.float64)
        for s in range(K_steps - 1, -1, -1):
            g_in_bar, psi_xi_3d, tet_cache, U_tot_s, g_m_s = step_tape[s]
            dg_dU, dg_dz = _dg_dU_and_dz0(g_m_s, U_tot_s, spec.use_langmuir, alpha)
            dL_dUtot_s = dL_dg_cur * dg_dU
            if spec.use_langmuir:
                grad_logit_alpha += float(np.sum(dL_dg_cur * dg_dz))
            dL_dUsol += dL_dUtot_s

            dL_ddU_cum_s = dL_ddU_cum_next + np.sum(dL_dUtot_s, axis=0)
            dL_ddU_step_asu = eta * dL_ddU_cum_s
            dL_ddU_cum_prev = (1.0 - eta) * dL_ddU_cum_s

            G3 = grid.asu_to_full_3d(dL_ddU_step_asu / grid.orbit_mult)
            sum_G3 = float(np.sum(G3)) if spec.zero_bulk_offset else 0.0

            for q in range(spec.n_oz):
                grad_voz[q] += float(np.sum(G3 * psi_xi_3d[q]))
                if spec.zero_bulk_offset:
                    grad_voz[q] -= sum_G3 * xi_bulk * float(grid.oz_dc[q])

            F_G3 = sp_fft.rfftn(G3, axes=(0, 1, 2), workers=w_fft)
            dL_dxi_3d = sp_fft.irfftn(W_hat * F_G3, s=grid.shape, axes=(0, 1, 2), workers=w_fft)

            if spec.use_tet_oz:
                Nc, J1, Q2, den_f, A_frust, den_c, sat_nc, exp_af, T_coop = tet_cache
                grad_wtet[0] += float(np.sum(G3 * A_frust)) - (sum_G3 * tk.A_frust_0 if spec.zero_bulk_offset else 0.0)
                grad_wtet[1] += float(np.sum(G3 * T_coop)) - (sum_G3 * tk.T_coop_0 if spec.zero_bulk_offset else 0.0)
                grad_wtet[2] += float(np.sum(G3 * sat_nc)) - (sum_G3 * tk.sat_nc_0 if spec.zero_bulk_offset else 0.0)

                dG_dA = G3 * (w_frust - 2.0 * w_tcoop * T_coop)
                dG_dsat = G3 * (w_tcoop * exp_af + w_nc)
                inv_denf = 1.0 / den_f
                dG_dnum = dG_dA * inv_denf
                dsat_dNc = (2.0 * Nc) / (den_c * den_c)
                dG_dNc = dG_dsat * dsat_dNc - dG_dA * (2.0 * Nc * A_frust * inv_denf)

                F_adj_hb = tk.K0_conj * sp_fft.rfftn(dG_dNc, axes=(0, 1, 2), workers=w_fft)
                dG_dJ1 = (4.0 / 3.0) * dG_dnum[None, ...] * J1
                F_adj_hb = F_adj_hb + np.sum(
                    tk.K1_conj * sp_fft.rfftn(dG_dJ1, axes=(1, 2, 3), workers=w_fft), axis=0
                )
                dG_dQ2 = 2.0 * dG_dnum[None, ...] * Q2
                F_adj_hb = F_adj_hb + np.sum(
                    tk.K2_conj * sp_fft.rfftn(dG_dQ2, axes=(1, 2, 3), workers=w_fft), axis=0
                )

                dL_drho_hb_3d = sp_fft.irfftn(F_adj_hb, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
                dL_dxi_3d = dL_dxi_3d + vox_mol * dL_drho_hb_3d
                if rho_sol_3d is not None:
                    grad_wtet[3] += dsig_bsol * float(np.sum(dL_drho_hb_3d * rho_sol_3d))

            dL_dxi_asu = grid.full_3d_to_asu_grad(dL_dxi_3d)
            dL_dg_in_bar = dL_dxi_asu * expit(4.0 * (g_in_bar - 1.0))
            dL_dg_cur = np.outer(mw, dL_dg_in_bar)
            dL_ddU_cum_next = dL_ddU_cum_prev

    dg0_dU, dg0_dz = _dg_dU_and_dz0(g0_m, U_sol, spec.use_langmuir, alpha)
    dL_dUsol += dL_dg_cur * dg0_dU
    if spec.use_langmuir:
        grad_logit_alpha += float(np.sum(dL_dg_cur * dg0_dz))

    return dL_dUsol, grad_logit_alpha, grad_voz, grad_wtet


def _eval_target_with_dfc(target: ReciprocalTarget, f_part: np.ndarray, dfc: np.ndarray = None,
                          kmask_mode: str = "global_scalar", compute_grad: bool = True):
    """Evaluate reciprocal amplitude loss & exact complex gradients wrt BOTH `f_part` and `f_calc` on `target.work_mask`."""
    assert not np.any(target.work_mask & target.free_mask), "R-FREE LEAK BLOCKED IN CHEMFIT!"
    fc_eff = target.f_calc if dfc is None else (target.f_calc + dfc)
    w = target.work_mask
    fr = target.free_mask
    inv_norm = 1.0 / max(target.work_fo2_sum, 1e-12)

    if dfc is not None:
        t_eff = ReciprocalTarget(
            target.hkl, target.f_obs, target.work_mask, target.free_mask,
            fc_eff, target.k_tot, target.bin_idx, target.n_bins, target.d_hkl,
            train_mask=target.train_mask, val_mask=target.val_mask, spacegroup=target.spacegroup,
        )
    else:
        t_eff = target

    if kmask_mode == "fixed_one":
        u_tot = t_eff.k_tot * (fc_eff + f_part)
        au = np.abs(u_tot)
        k0 = float(np.sum(au[w] * t_eff.f_obs[w]) / max(float(np.sum(au[w] ** 2)), 1e-12))
        kb = 1.0
        fm = k0 * u_tot
        k_vec = np.full(t_eff.n_bins, 1.0, dtype=np.float64)
    elif kmask_mode == "global_scalar":
        k0, kb = t_eff._solve_global_scalar_k0_kb(f_part, w)
        fm = k0 * t_eff.k_tot * (fc_eff + kb * f_part)
        k_vec = np.full(t_eff.n_bins, kb, dtype=np.float64)
    else:
        from .field import fit_kmask_per_bin
        k_bins, k_hkl = fit_kmask_per_bin(
            fc_eff, f_part, t_eff.k_tot, t_eff.f_obs, w, t_eff.bin_idx, t_eff.n_bins, loss_type="l2"
        )
        k0 = 1.0
        kb = k_hkl
        fm = t_eff.k_tot * (fc_eff + k_hkl * f_part)
        k_vec = k_bins

    afm = np.maximum(np.abs(fm), 1e-12)
    res = afm - t_eff.f_obs
    loss = 0.5 * float(np.sum(res[w] ** 2)) * inv_norm
    r_work = float(np.sum(np.abs(res[w])) / max(t_eff.work_fo_sum, 1e-12))
    r_free = float(np.sum(np.abs(res[fr])) / t_eff.free_fo_sum) if np.any(fr) and t_eff.free_fo_sum > 0 else float("nan")

    if not compute_grad:
        return loss, None, None, r_work, r_free, k_vec

    phase_unit = fm[w] / afm[w]
    base_g = (res[w] * inv_norm) * (k0 * t_eff.k_tot[w]) * phase_unit
    dL_dFc = np.zeros_like(f_part, dtype=np.complex128)
    dL_dFc[w] = base_g
    dL_dFpart = np.zeros_like(f_part, dtype=np.complex128)
    if np.ndim(kb) == 0:
        dL_dFpart[w] = float(kb) * base_g
    else:
        dL_dFpart[w] = kb[w] * base_g
    return loss, dL_dFpart, dL_dFc, r_work, r_free, k_vec


def eval_blind_rval_chem(target_full: ReciprocalTarget, f_part: np.ndarray, dfc: np.ndarray = None,
                         kmask_mode: str = "global_scalar"):
    """Evaluate `(r_train, r_val, r_work, r_free)` fitting scale factors strictly on `train_mask` (`free_mask` untouched)."""
    fc_eff = target_full.f_calc if dfc is None else (target_full.f_calc + dfc)
    t_eval = ReciprocalTarget(
        target_full.hkl, target_full.f_obs, target_full.work_mask, target_full.free_mask,
        fc_eff, target_full.k_tot, target_full.bin_idx, target_full.n_bins, target_full.d_hkl,
        train_mask=target_full.train_mask, val_mask=target_full.val_mask, spacegroup=target_full.spacegroup,
    )
    return t_eval.eval_blind_rval(f_part, kmask_mode=kmask_mode)


def chem_loss_and_grad(
    spec: WaterFieldSpec,
    ds,
    renderer: MultiBRenderer,
    target: ReciprocalTarget,
    z_mob_m: np.ndarray,
    adapter: CrystalChemAdapter,
    gamma: np.ndarray,
    site_basis: ChemicalSiteBasis = None,
    p_alt_override: np.ndarray = None,
    zeta: np.ndarray = None,
    include_atomic_dfc: bool = True,
    displace_mode: str = "none",
    keep_asu: np.ndarray = None,
    kmask_mode: str = "global_scalar",
    lam_gamma: float = 2e-4,
    lam_kl: float = 1.5e-4,
    compute_grad: bool = True,
    strict_val_blind: bool = False,
):
    """Unified forward & exact analytical reverse-mode gradient for `(gamma, zeta)` on `target.work_mask`."""
    assert not np.any(target.work_mask & target.free_mask), "FREE SET LEAK IN CHEM_LOSS_AND_GRAD!"
    if strict_val_blind:
        assert not np.any(target.work_mask & target.val_mask), "VAL SET LEAK IN STRICT BLIND CHEM FIT!"

    grid = ds.grid
    M = ds.M
    n_asu = grid.n_asu
    mw = ds.mw
    w_fft = renderer.fft_workers

    theta, pb_gamma = adapter.theta_and_pullback(gamma)
    w_pmf, logit_alpha, v_oz, w_tet, w_epol, w_hpol, _ = spec.unpack_all(theta)
    g_arr = np.asarray(gamma, dtype=np.float64)
    eg0, eg1, eg2, g3 = math.exp(float(g_arr[0])), math.exp(float(g_arr[1])), math.exp(float(g_arr[2])), float(g_arr[3])
    U5_base, rh0_base = adapter.get_ds_basis(ds)

    # 1. Baseline solute potential U_sol (exact float64 linear combination of 5 cached basis channels)
    c5 = np.array([1.0, eg0 - 1.0, eg1 - 1.0, eg2 - 1.0, g3], dtype=np.float64)
    U_sol = (U5_base @ c5).reshape(M, n_asu)

    E2d = getattr(ds, "E_feats_2d", None)
    if spec.use_epol and E2d is not None and w_epol is not None:
        U_sol = U_sol + (E2d @ w_epol).reshape(M, n_asu)

    has_sites = (site_basis is not None and site_basis.n_zeta > 0 and (zeta is not None or p_alt_override is not None))
    if has_sites:
        if p_alt_override is not None:
            p_alt = np.asarray(p_alt_override, dtype=np.float64)
            site_probs, kl_val, dkl_dzeta = None, 0.0, np.zeros(site_basis.n_zeta, dtype=np.float64)
        else:
            p_alt, site_probs, kl_val, dkl_dzeta = site_basis.probs_and_kl(zeta)
    else:
        p_alt, site_probs, kl_val, dkl_dzeta = None, None, 0.0, np.zeros(0, dtype=np.float64)

    du_list = []
    dfc_tot = None
    if has_sites:
        for k, rec in enumerate(site_basis.alt_records):
            pk = float(p_alt[k])
            du_k = rec["csr_o"] @ w_pmf if len(rec["u_o"]) else None
            du_list.append(du_k)
            if du_k is not None and pk > 0.0:
                U_sol[:, rec["u_o"]] += pk * du_k[None, :]
            if include_atomic_dfc and rec["dfc"] is not None and pk > 0.0:
                dfc_tot = (pk * rec["dfc"]) if dfc_tot is None else (dfc_tot + pk * rec["dfc"])

    gO_m_raw, _, tape = _run_oz_and_pauli_forward(
        spec, grid, U_sol, mw, logit_alpha, v_oz, w_tet, getattr(ds, "F_hbsol", None),
        need_grad=compute_grad, fft_workers=w_fft,
    )

    # 2. Water-H polarization field
    has_hpol = bool(spec.use_hpol and w_hpol is not None and rh0_base is not None)
    drh_list = []
    if has_hpol:
        raw_h = (eg1 * rh0_base).reshape(M, n_asu)
        lster = np.log(np.maximum(ds.steric_H_64, 1e-37))
        if has_sites:
            for k, rec in enumerate(site_basis.alt_records):
                pk = float(p_alt[k])
                drh_k = rec["csr_h"] @ w_hpol if len(rec["u_h"]) else None
                drh_list.append(drh_k)
                if drh_k is not None and pk > 0.0:
                    raw_h[:, rec["u_h"]] += pk * drh_k[None, :]
                if len(rec["u_st"]) and pk > 0.0:
                    lster[:, rec["u_st"]] += pk * rec["d_st"][None, :]
        ster_m = np.exp(lster)
        th_h = np.tanh(raw_h)
        gH_m_raw = ster_m * th_h
    else:
        ster_m = th_h = gH_m_raw = None

    if displace_mode == "pre" and keep_asu is not None:
        gO_m = gO_m_raw * keep_asu[None, :]
        gH_m = (gH_m_raw * keep_asu[None, :]) if gH_m_raw is not None else None
    else:
        gO_m, gH_m = gO_m_raw, gH_m_raw

    g_hat, B_eff_asu, pb_render = renderer.render_and_pullback(
        gO_m, gH_m, mw, z_mob_m, scale_k=1.0, compute_grad=compute_grad
    )
    g_eff_asu = (g_hat * keep_asu) if (displace_mode == "post" and keep_asu is not None) else g_hat
    rho_3d = grid.asu_to_full_3d(g_eff_asu) * RHO_BULK
    solv_3d = np.maximum(rho_3d, 0.0)
    f_part = solv_to_fpart(solv_3d, grid.cell.volume, target.hkl)

    data_loss, dL_dFpart, dL_dFc, r_work, r_free, k_vec = _eval_target_with_dfc(
        target, f_part, dfc=dfc_tot, kmask_mode=kmask_mode, compute_grad=compute_grad
    )
    pen_gamma = 0.5 * float(lam_gamma) * float(np.sum(np.asarray(gamma, dtype=np.float64) ** 2))
    pen_kl = float(lam_kl) * float(kl_val)
    tot_loss = data_loss + pen_gamma + pen_kl

    if not compute_grad:
        return {
            "loss": tot_loss, "data_loss": data_loss, "r_work": r_work, "r_free": r_free,
            "solv_3d": solv_3d, "f_part": f_part, "dfc": dfc_tot, "B_eff_asu": B_eff_asu,
            "k_bins": k_vec, "p_alt": p_alt, "site_probs": site_probs, "theta": theta,
        }

    # Reverse pass
    dL_dg_eff = fpart_grad_to_asu(grid, g_eff_asu, target.hkl, dL_dFpart, clamp_nonneg=True)
    dL_dghat = (dL_dg_eff * keep_asu) if (displace_mode == "post" and keep_asu is not None) else dL_dg_eff
    dL_dgO_m, dL_dgH_m, dL_dz_mob, _ = pb_render(dL_dghat)

    if displace_mode == "pre" and keep_asu is not None:
        dL_dgO_raw = dL_dgO_m * keep_asu[None, :]
        dL_dgH_raw = (dL_dgH_m * keep_asu[None, :]) if dL_dgH_m is not None else None
    else:
        dL_dgO_raw, dL_dgH_raw = dL_dgO_m, dL_dgH_m

    grad_theta = np.zeros_like(theta)
    dL_dp_alt = np.zeros(site_basis.n_zeta, dtype=np.float64) if has_sites else np.zeros(0, dtype=np.float64)

    if has_sites and include_atomic_dfc and dL_dFc is not None:
        for k, rec in enumerate(site_basis.alt_records):
            if rec["dfc"] is not None:
                dL_dp_alt[k] += float(np.real(np.vdot(dL_dFc, rec["dfc"])))

    grad_gamma_base = np.zeros(6, dtype=np.float64)
    if has_hpol and dL_dgH_raw is not None:
        dL_draw = dL_dgH_raw * ster_m * (1.0 - th_h ** 2)
        dL_dlster = dL_dgH_raw * gH_m_raw
        dL_draw_sum = np.sum(dL_draw, axis=0)
        dL_dlster_sum = np.sum(dL_dlster, axis=0)
        grad_gamma_base[1] += eg1 * float(np.dot(dL_draw.ravel(), rh0_base))

        i_hp = spec.n_pmf + spec.n_alpha + spec.n_oz + spec.n_tet + spec.n_epol
        gw_h_sparse = np.zeros(spec.n_hpol, dtype=np.float64)
        if has_sites:
            for k, rec in enumerate(site_basis.alt_records):
                pk = float(p_alt[k])
                if len(rec["u_h"]):
                    sub_r = dL_draw_sum[rec["u_h"]]
                    dL_dp_alt[k] += float(np.dot(sub_r, drh_list[k]))
                    if pk > 0.0:
                        gw_h_sparse = gw_h_sparse + pk * (sub_r @ rec["csr_h"])
                if len(rec["u_st"]):
                    dL_dp_alt[k] += float(np.dot(dL_dlster_sum[rec["u_st"]], rec["d_st"]))
        grad_theta[i_hp:i_hp + spec.n_hpol] = gw_h_sparse

    dL_dUsol, g_alpha, g_voz, g_wtet = _pullback_oz_and_pauli_to_Usol(
        spec, grid, mw, U_sol, v_oz, w_tet, tape, dL_dgO_raw, w_fft=w_fft
    )
    dots_u = dL_dUsol.ravel() @ U5_base[:, 1:5]
    grad_gamma_base[0] += eg0 * float(dots_u[0])
    grad_gamma_base[1] += eg1 * float(dots_u[1])
    grad_gamma_base[2] += eg2 * float(dots_u[2])
    grad_gamma_base[3] += float(dots_u[3])

    dL_dU_sum = np.sum(dL_dUsol, axis=0)
    gw_o_sparse = np.zeros(spec.n_pmf, dtype=np.float64)
    if has_sites:
        for k, rec in enumerate(site_basis.alt_records):
            pk = float(p_alt[k])
            if len(rec["u_o"]):
                sub_u = dL_dU_sum[rec["u_o"]]
                dL_dp_alt[k] += float(np.dot(sub_u, du_list[k]))
                if pk > 0.0:
                    gw_o_sparse = gw_o_sparse + pk * (sub_u @ rec["csr_o"])
    grad_theta[:spec.n_pmf] = gw_o_sparse

    idx = spec.n_pmf
    if spec.use_langmuir:
        grad_theta[idx] = g_alpha
        idx += 1
    if spec.use_oz:
        grad_theta[idx:idx + spec.n_oz] = g_voz
        idx += spec.n_oz
    if spec.use_tet_oz:
        grad_theta[idx:idx + 4] = g_wtet
        idx += 4
    if spec.use_epol and E2d is not None:
        grad_theta[idx:idx + 3] = dL_dUsol.ravel() @ E2d

    grad_gamma = grad_gamma_base + pb_gamma(grad_theta) + float(lam_gamma) * np.asarray(gamma, dtype=np.float64)
    if has_sites and site_probs is not None:
        grad_zeta = site_basis.pullback_probs_to_zeta(site_probs, dL_dp_alt) + float(lam_kl) * dkl_dzeta
    else:
        grad_zeta = np.zeros(0, dtype=np.float64)

    return {
        "loss": tot_loss,
        "data_loss": data_loss,
        "r_work": r_work,
        "r_free": r_free,
        "grad_gamma": grad_gamma,
        "grad_zeta": grad_zeta,
        "grad_z_mob": dL_dz_mob,
        "solv_3d": solv_3d,
        "f_part": f_part,
        "dfc": dfc_tot,
        "B_eff_asu": B_eff_asu,
        "k_bins": k_vec,
        "p_alt": p_alt,
        "site_probs": site_probs,
        "theta": theta,
    }


def apply_confident_chemistry_to_pdb(pdb_in_path: str, pdb_out_path: str, site_basis: ChemicalSiteBasis,
                                     site_probs, prob_thresh: float = 0.72, min_shift: float = 0.25):
    """Write `pdb_out_path` applying high-confidence `ASN/GLN` 180-deg flips and `REMARK 3` chemistry annotations."""
    st = gemmi.read_structure(str(pdb_in_path))
    applied = []
    if site_basis is None or site_probs is None or len(st) == 0:
        st.write_pdb(str(pdb_out_path))
        return applied

    flip_lookup = {}
    for s_info, p_vec in zip(site_basis.sites, site_probs):
        best_s = int(np.argmax(p_vec))
        if best_s == 0:
            continue
        p_best = float(p_vec[best_s])
        p_init = float(s_info["p0"][best_s])
        if p_best < float(prob_thresh) or (p_best - p_init) < float(min_shift):
            continue
        meta = s_info["meta"]
        rec = {
            "res_tag": s_info["res_tag"],
            "family": s_info["family"],
            "from_state": s_info["state_labels"][0],
            "to_state": s_info["state_labels"][best_s],
            "posterior": round(p_best, 4),
            "prior": round(p_init, 4),
        }
        applied.append(rec)
        if s_info["family"] == "asn_gln" and best_s == 1:
            key = (meta["chain"], int(meta["seqnum"]), meta["icode"], meta["resname"])
            flip_lookup[key] = meta

    if flip_lookup:
        for ch_idx, ch in enumerate(st[0]):
            ch_id = ch.name.strip() or f"#{ch_idx}"
            for r in ch:
                ic_id = r.seqid.icode.strip() if r.seqid.icode != "\0" else ""
                key = (ch_id, int(r.seqid.num), ic_id, r.name.strip().upper())
                if key not in flip_lookup:
                    continue
                meta = flip_lookup[key]
                o_atoms = [a for a in r if a.name.strip().upper() == meta["o_name"]]
                n_atoms = [a for a in r if a.name.strip().upper() == meta["n_name"]]
                if len(o_atoms) == 1 and len(n_atoms) == 1:
                    oa, na = o_atoms[0], n_atoms[0]
                    old_o_pos = gemmi.Position(oa.pos.x, oa.pos.y, oa.pos.z)
                    oa.pos = gemmi.Position(na.pos.x, na.pos.y, na.pos.z)
                    na.pos = old_o_pos
                    h1_nm, h2_nm = meta["h_names"]
                    hs_flip = meta["hs_flip"]
                    for a in r:
                        an = a.name.strip().upper()
                        if an == h1_nm:
                            a.pos = gemmi.Position(*hs_flip[0])
                        elif an == h2_nm:
                            a.pos = gemmi.Position(*hs_flip[1])

    for rec in applied:
        st.raw_remarks.append(
            f"REMARK   3 SURFNTURF CHEMFIT {rec['res_tag']} {rec['from_state']} -> {rec['to_state']} "
            f"(p={rec['posterior']:.3f}, prior={rec['prior']:.3f})"
        )
    out_dir = os.path.dirname(os.path.abspath(pdb_out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    st.write_pdb(str(pdb_out_path))
    return applied


def fit_chemistry_blind_gated(
    spec: WaterFieldSpec,
    theta_init: np.ndarray,
    ds,
    renderer: MultiBRenderer,
    target_full: ReciprocalTarget,
    z_mob_m: np.ndarray,
    site_basis: ChemicalSiteBasis = None,
    mode: str = "auto",
    kmask_mode: str = "global_scalar",
    displace_mode: str = "none",
    keep_asu: np.ndarray = None,
    include_atomic_dfc: bool = True,
    lam_gamma: str = "auto",
    lam_kl: str = "auto",
    maxiter: int = 45,
    min_rval_gain: str = "auto",
):
    """Fit crystal environment `gamma` and/or per-site `zeta` on 90% `train_mask`, gating strictly on 5% `val_mask`.

    Guarantees zero R-free leakage (`free_mask` is never read during fitting or model selection) and zero `val_mask`
    leakage into parameter updates.
    """
    assert not np.any(target_full.train_mask & target_full.free_mask), "FREE SET LEAK IN TRAIN_MASK!"
    assert not np.any(target_full.val_mask & target_full.free_mask), "FREE SET LEAK IN VAL_MASK!"
    assert not np.any(target_full.train_mask & target_full.val_mask), "TRAIN/VAL OVERLAP!"

    targ_tr = target_full.for_training_subset()
    d_min = float(np.min(target_full.d_hkl)) if len(target_full.d_hkl) else 1.5
    res_scale = max(1.0, (d_min / 1.10) ** 3)
    v_asu = float(ds.grid.cell.volume) / float(max(len(ds.grid.ops), 1))
    vol_norm = 2500.0 / max(v_asu, 1500.0)
    lam_g_eff = (2.0e-4 * res_scale) if str(lam_gamma).lower() == "auto" else float(lam_gamma)
    lam_k_eff = (2.5e-5 * vol_norm * max(1.0, (d_min / 1.10) ** 2)) if str(lam_kl).lower() == "auto" else float(lam_kl)

    adapter = CrystalChemAdapter(spec, theta_init)
    gamma_zero = np.zeros(adapter.n_knobs, dtype=np.float64)
    has_sites = (site_basis is not None and site_basis.n_zeta > 0)

    # Stage 0: exact as-built PDB (`p_alt == 0`) + unmodified MD `theta_init` (`gamma == 0`)
    out_base = chem_loss_and_grad(
        spec, ds, renderer, targ_tr, z_mob_m, adapter, gamma_zero,
        site_basis=site_basis if has_sites else None,
        p_alt_override=(np.zeros(site_basis.n_zeta, dtype=np.float64) if has_sites else None),
        include_atomic_dfc=False, displace_mode=displace_mode, keep_asu=keep_asu,
        kmask_mode=kmask_mode, lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=False, strict_val_blind=True,
    )
    r_tr0, r_vl0, r_wk0, r_fr0 = eval_blind_rval_chem(target_full, out_base["f_part"], dfc=None, kmask_mode=kmask_mode)
    out_base.update({"r_train": r_tr0, "r_val": r_vl0, "r_work": r_wk0, "r_free": r_fr0, "gamma": gamma_zero})

    n_val = max(1, int(np.sum(target_full.val_mask)))
    rval_ref = r_vl0 if np.isfinite(r_vl0) else r_tr0
    if str(min_rval_gain).lower() == "auto":
        gain_req = max(1.5e-4, 0.45 * float(rval_ref) / math.sqrt(2.0 * n_val))
    else:
        gain_req = float(min_rval_gain)

    candidates = [("base", out_base)]

    # Stage 1: 6-knob mother-liquor environment adapter `gamma` (solute states held as-built)
    if mode in ("auto", "env", "joint"):
        def _obj_env(g_vec):
            ev = chem_loss_and_grad(
                spec, ds, renderer, targ_tr, z_mob_m, adapter, g_vec,
                site_basis=None, include_atomic_dfc=False,
                displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
                lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=True, strict_val_blind=True,
            )
            return ev["loss"], ev["grad_gamma"]

        res_env = minimize(_obj_env, gamma_zero, method="L-BFGS-B", jac=True, bounds=adapter.bounds(),
                           options={"maxiter": min(int(maxiter), 28), "ftol": 1e-10, "gtol": 1e-8})
        g_env = res_env.x
        out_env = chem_loss_and_grad(
            spec, ds, renderer, targ_tr, z_mob_m, adapter, g_env,
            site_basis=None, include_atomic_dfc=False,
            displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
            lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=False, strict_val_blind=True,
        )
        r_tr, r_vl, r_wk, r_fr = eval_blind_rval_chem(target_full, out_env["f_part"], dfc=None, kmask_mode=kmask_mode)
        out_env.update({"r_train": r_tr, "r_val": r_vl, "r_work": r_wk, "r_free": r_fr, "gamma": g_env})
        candidates.append(("env", out_env))
    else:
        g_env = gamma_zero

    # Stage 2: Per-site solute chemical states `zeta` (environment held at `gamma = 0`)
    if has_sites and mode in ("auto", "sites"):
        z0_init = np.zeros(site_basis.n_zeta, dtype=np.float64)
        z_bounds = [(-6.0, 6.0)] * site_basis.n_zeta

        def _obj_sites(z_vec):
            ev = chem_loss_and_grad(
                spec, ds, renderer, targ_tr, z_mob_m, adapter, gamma_zero,
                site_basis=site_basis, zeta=z_vec, include_atomic_dfc=include_atomic_dfc,
                displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
                lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=True, strict_val_blind=True,
            )
            return ev["loss"], ev["grad_zeta"]

        res_st = minimize(_obj_sites, z0_init, method="L-BFGS-B", jac=True, bounds=z_bounds,
                          options={"maxiter": min(int(maxiter), 28), "ftol": 1e-10, "gtol": 1e-8})
        z_st_warm = res_st.x
        out_st = chem_loss_and_grad(
            spec, ds, renderer, targ_tr, z_mob_m, adapter, gamma_zero,
            site_basis=site_basis, zeta=res_st.x, include_atomic_dfc=include_atomic_dfc,
            displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
            lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=False, strict_val_blind=True,
        )
        r_tr, r_vl, r_wk, r_fr = eval_blind_rval_chem(
            target_full, out_st["f_part"], dfc=(out_st["dfc"] if include_atomic_dfc else None), kmask_mode=kmask_mode
        )
        out_st.update({"r_train": r_tr, "r_val": r_vl, "r_work": r_wk, "r_free": r_fr,
                       "gamma": gamma_zero, "zeta": res_st.x})
        candidates.append(("sites", out_st))
    else:
        z_st_warm = np.zeros(site_basis.n_zeta, dtype=np.float64) if has_sites else np.zeros(0, dtype=np.float64)

    # Stage 3: Joint `(gamma, zeta)` optimization warm-started from `(g_env, z_st_warm)`
    if has_sites and mode in ("auto", "joint"):
        x0 = np.concatenate([g_env, z_st_warm])
        x_bounds = adapter.bounds() + [(-6.0, 6.0)] * site_basis.n_zeta
        ng = adapter.n_knobs
        jt_iters = min(int(maxiter), 18) if mode == "auto" else int(maxiter)

        def _obj_joint(x_vec):
            ev = chem_loss_and_grad(
                spec, ds, renderer, targ_tr, z_mob_m, adapter, x_vec[:ng],
                site_basis=site_basis, zeta=x_vec[ng:], include_atomic_dfc=include_atomic_dfc,
                displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
                lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=True, strict_val_blind=True,
            )
            return ev["loss"], np.concatenate([ev["grad_gamma"], ev["grad_zeta"]])

        res_jt = minimize(_obj_joint, x0, method="L-BFGS-B", jac=True, bounds=x_bounds,
                          options={"maxiter": jt_iters, "ftol": 1e-10, "gtol": 1e-8})
        g_jt, z_jt = res_jt.x[:ng], res_jt.x[ng:]
        out_jt = chem_loss_and_grad(
            spec, ds, renderer, targ_tr, z_mob_m, adapter, g_jt,
            site_basis=site_basis, zeta=z_jt, include_atomic_dfc=include_atomic_dfc,
            displace_mode=displace_mode, keep_asu=keep_asu, kmask_mode=kmask_mode,
            lam_gamma=lam_g_eff, lam_kl=lam_k_eff, compute_grad=False, strict_val_blind=True,
        )
        r_tr, r_vl, r_wk, r_fr = eval_blind_rval_chem(
            target_full, out_jt["f_part"], dfc=(out_jt["dfc"] if include_atomic_dfc else None), kmask_mode=kmask_mode
        )
        out_jt.update({"r_train": r_tr, "r_val": r_vl, "r_work": r_wk, "r_free": r_fr,
                       "gamma": g_jt, "zeta": z_jt})
        candidates.append(("joint", out_jt))

    rval_table = {}
    for c_name, c_out in candidates:
        sc = c_out["r_val"] if np.isfinite(c_out["r_val"]) else c_out["r_train"]
        rval_table[c_name] = round(float(sc), 6)

    if mode != "auto":
        # Explicit non-auto mode requested by caller (still fitted strictly on train_mask, zero R-free leak)
        chosen_name, chosen_out = candidates[-1]
    else:
        chosen_name, chosen_out = candidates[0]
        best_sc = rval_table["base"]
        for c_name, c_out in candidates[1:]:
            sc = rval_table[c_name]
            if sc < best_sc - gain_req:
                best_sc = sc
                chosen_name, chosen_out = c_name, c_out

    # Build human-readable chemistry summary
    gamma_sel = chosen_out.get("gamma", gamma_zero)
    knob_dict = {nm: round(float(val), 4) for nm, val in zip(CHEM_KNOB_NAMES, gamma_sel)}
    site_shifts = []
    if has_sites and chosen_out.get("site_probs") is not None:
        for s_info, p_vec in zip(site_basis.sites, chosen_out["site_probs"]):
            best_s = int(np.argmax(p_vec))
            shift = float(p_vec[best_s] - s_info["p0"][best_s])
            if best_s != 0 and (float(p_vec[best_s]) >= 0.50 or abs(shift) >= 0.15):
                site_shifts.append({
                    "res_tag": s_info["res_tag"],
                    "family": s_info["family"],
                    "as_built": s_info["state_labels"][0],
                    "favored": s_info["state_labels"][best_s],
                    "p_favored": round(float(p_vec[best_s]), 4),
                    "p_prior": round(float(s_info["p0"][best_s]), 4),
                    "delta_p": round(shift, 4),
                })
        site_shifts.sort(key=lambda d: -d["p_favored"])

    chosen_out["selected_stage"] = chosen_name
    chosen_out["r_val_table"] = rval_table
    chosen_out["min_rval_gain"] = round(float(gain_req), 6)
    chosen_out["chem_knobs"] = knob_dict
    chosen_out["n_sites_enumerated"] = site_basis.n_sites if has_sites else 0
    chosen_out["site_shifts"] = site_shifts
    return chosen_out
