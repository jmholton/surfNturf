"""Differentiable Sparse Divot/Peak Dictionary & Mass-Conserving Wall-Clamped Transport for surfNturf.

Implements:
  1. Occupancy-weighted open-solvent envelope f_open(x) across multi-altloc protein conformers.
  2. Mass-Conserving, Wall-Clamped Tricubic Transport:
       rho_mc(x) = sym_s(det(I - grad u_eff(x)) * rho_0(x - u_eff(x))),   u_eff(x) = m_solv(x) * u(x),
     with exact skew-adjoint general-triclinic spectral derivatives and exact space-group symmetry across all 230 groups.
  3. Continuous-Coordinate Space-Group-Symmetric IT92 Peak Dictionary (ContinuousPeakDict).
  4. Bader Steepest-Ascent Watershed Divot Dictionary (WatershedDivotDict) seeded from exact e/A^3 work-set mFo - DFc maps.
  5. Strict Blind Hyperparameter Selection on Internal Work Split (work_train 90% / work_val 5%; R-free test set untouched).
"""
import math
import os
import sys
import time
import numpy as np
import gemmi
from scipy import fft as sp_fft
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.optimize import minimize

RHO_BULK = 0.3344
R_OH = 0.9572


def softplus_pos(x, tau=0.015):
    """Smooth C^infty non-negativity clamp tau * log(1 + exp(x / tau)) -> max(x, 0) with derivative expit(x / tau)."""
    z = x / float(tau)
    out = np.where(z > 30.0, x, float(tau) * np.log1p(np.exp(np.clip(z, -60.0, 30.0))))
    deriv = expit(z)
    return out, deriv


def smooth_l1_penalty(w, eps=0.02):
    """Charbonnier / pseudo-Huber smooth L1 sparsity sum(sqrt(w^2 + eps^2) - eps) and exact gradient."""
    s = np.sqrt(w * w + eps * eps)
    return float(np.sum(s - eps)), w / s


class CrystalGridOps:
    """Periodic unit-cell FFT, space-group symmetry projectors, exact IT92 water kernel, and spectral derivatives."""

    def __init__(self, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup, shape, conv='plus', fsc=None, fft_workers=4):
        if cell is None or not cell.is_crystal():
            raise ValueError("CrystalGridOps requires a valid crystallographic UnitCell")
        self.cell = cell
        self.sg = sg if sg is not None else gemmi.SpaceGroup('P 1')
        self.shape = tuple(int(n) for n in shape)
        self.nvox = int(np.prod(self.shape))
        self.Ns = np.array(self.shape, dtype=np.float64)
        self.Lc = np.array([cell.a, cell.b, cell.c], dtype=np.float64)
        self.dx = self.Lc / self.Ns
        self.vox_vol = float(cell.volume) / float(self.nvox)
        self.Om = np.array(cell.orth.mat.tolist(), dtype=np.float64)
        self.Fm = np.array(cell.frac.mat.tolist(), dtype=np.float64)
        self.f_row_norms = np.linalg.norm(self.Fm, axis=1)
        self.conv = str(conv)
        self.fsc = float(fsc) if fsc is not None else self.vox_vol
        self.workers = int(fft_workers)

        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing='ij'), axis=-1).reshape(-1, 3)
        self.ops = []
        canon = np.arange(self.nvox, dtype=np.int64)
        for op in self.sg.operations():
            R = np.array(op.rot, dtype=np.float64) / op.DEN
            t = np.array(op.tran, dtype=np.float64) / op.DEN
            mp = ((ijk / self.Ns) @ R.T + t) * self.Ns
            if np.max(np.abs(mp - np.rint(mp))) >= 1e-6:
                raise ValueError(
                    f"Grid {self.shape} incompatible with symmetry operator {op.triplet()} of space group {self.sg.hm}"
                )
            mi = np.mod(np.rint(mp).astype(np.int64), self.shape)
            lin = (mi[:, 0] * self.shape[1] + mi[:, 1]) * self.shape[2] + mi[:, 2]
            self.ops.append((R, np.linalg.inv(R), t, lin))
            canon = np.minimum(canon, lin)
        self.nops = len(self.ops)

        self.asu_reps, inv = np.unique(canon, return_inverse=True)
        self.full_to_asu = inv.astype(np.int32)
        self.n_asu = len(self.asu_reps)
        self.orbit_mult = np.bincount(self.full_to_asu, minlength=self.n_asu).astype(np.float64)

        frac = ijk / self.Ns[None, :]
        self.xyz_full = frac @ self.Om.T
        self.asu_xyz = self.xyz_full[self.asu_reps]
        del ijk

        nu, nv, nw = self.shape
        fh = np.fft.fftfreq(nu, 1.0 / nu)[:, None, None]
        fk = np.fft.fftfreq(nv, 1.0 / nv)[None, :, None]
        fl = np.fft.rfftfreq(nw, 1.0 / nw)[None, None, :]
        hkl_r = np.stack(np.broadcast_arrays(fh, fk, fl), axis=-1)
        q_cart = hkl_r @ self.Fm
        self.stol2_r = 0.25 * np.sum(q_cart ** 2, axis=-1)
        stol_r = np.sqrt(self.stol2_r)

        o_it = gemmi.Element('O').it92
        h_it = gemmi.Element('H').it92
        fO = np.full_like(self.stol2_r, o_it.c)
        for ai, bi in zip(o_it.a, o_it.b):
            fO = fO + ai * np.exp(-bi * self.stol2_r)
        fH = np.full_like(self.stol2_r, h_it.c)
        for ai, bi in zip(h_it.a, h_it.b):
            fH = fH + ai * np.exp(-bi * self.stol2_r)
        kr = 4.0 * np.pi * stol_r * R_OH
        j0 = np.where(kr > 1e-8, np.sin(kr) / np.where(kr > 1e-8, kr, 1.0), 1.0)
        self.f_wat_r = fO + 2.0 * fH * j0   # ~10.008 e at s=0

        fh_nyq = np.fft.fftfreq(nu, 1.0 / nu)
        fk_nyq = np.fft.fftfreq(nv, 1.0 / nv)
        fl_nyq = np.fft.rfftfreq(nw, 1.0 / nw)
        if nu % 2 == 0:
            fh_nyq[nu // 2] = 0.0
        if nv % 2 == 0:
            fk_nyq[nv // 2] = 0.0
        if nw % 2 == 0:
            fl_nyq[-1] = 0.0
        self.ik_r = [
            2j * np.pi * (
                fh_nyq[:, None, None] * self.Fm[0, b]
                + fk_nyq[None, :, None] * self.Fm[1, b]
                + fl_nyq[None, None, :] * self.Fm[2, b]
            )
            for b in range(3)
        ]

    def sym_expand_xyz(self, xyz_arr, cutoff_A=3.5, occ_arr=None):
        """Expand Cartesian coordinates (and optional occupancies) across all symmetry & lattice images within `cutoff_A`."""
        xyz_arr = np.asarray(xyz_arr, dtype=np.float64).reshape(-1, 3)
        if len(xyz_arr) == 0:
            if occ_arr is not None:
                return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
            return np.zeros((0, 3), dtype=np.float64)

        m_vec = self.f_row_norms * (float(cutoff_A) + 0.05)
        k_max = [int(math.ceil(float(m_vec[a]))) for a in range(3)]
        shifts = np.stack(
            np.meshgrid(
                np.arange(-k_max[0], k_max[0] + 1, dtype=np.float64),
                np.arange(-k_max[1], k_max[1] + 1, dtype=np.float64),
                np.arange(-k_max[2], k_max[2] + 1, dtype=np.float64),
                indexing='ij',
            ),
            axis=-1,
        ).reshape(-1, 3)
        f = xyz_arr @ self.Fm.T
        lo = -m_vec[None, :]
        hi = 1.0 + m_vec[None, :]
        pts = []
        occs = [] if occ_arr is not None else None
        for R, _, t, _ in self.ops:
            g = f @ R.T + t
            g -= np.floor(g)
            for s in shifts:
                h = g + s
                m = np.all((h >= lo) & (h < hi), axis=1)
                if np.any(m):
                    pts.append(h[m] @ self.Om.T)
                    if occs is not None:
                        occs.append(occ_arr[m])
        if not pts:
            if occs is not None:
                return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
            return np.zeros((0, 3), dtype=np.float64)
        if occs is not None:
            return np.vstack(pts), np.concatenate(occs)
        return np.vstack(pts)

    def hkl_idx(self, H):
        H = np.asarray(H, dtype=np.int64)
        nu, nv, nw = self.shape
        in_nyq = (
            (H[:, 0] >= -(nu // 2)) & (H[:, 0] <= (nu - 1) // 2) &
            (H[:, 1] >= -(nv // 2)) & (H[:, 1] <= (nv - 1) // 2) &
            (H[:, 2] >= -(nw // 2)) & (H[:, 2] <= (nw - 1) // 2)
        )
        return (np.mod(H[:, 0], nu), np.mod(H[:, 1], nv), np.mod(H[:, 2], nw), in_nyq)

    def sf(self, rho, idx):
        nu, nv, nw = self.shape
        Ar = sp_fft.rfftn(np.asarray(rho, dtype=np.float64), s=self.shape, workers=self.workers)
        iu, iv, iw = idx[:3]
        neg_l = iw > (nw // 2)
        ru = np.where(neg_l, np.mod(nu - iu, nu), iu)
        rv = np.where(neg_l, np.mod(nv - iv, nv), iv)
        rw = np.where(neg_l, nw - iw, iw)
        af = np.where(neg_l, np.conj(Ar[ru, rv, rw]), Ar[ru, rv, rw])
        vals = self.fsc * (np.conj(af) if self.conv == 'plus' else af)
        if len(idx) == 4 and not np.all(idx[3]):
            vals = np.where(idx[3], vals, 0.0 + 0.0j)
        return vals

    def sf_adjoint(self, G, idx):
        nu, nv, nw = self.shape
        iu, iv, iw = idx[:3]
        g_use = np.asarray(G, dtype=np.complex128)
        if len(idx) == 4 and not np.all(idx[3]):
            m = idx[3]
            iu, iv, iw = iu[m], iv[m], iw[m]
            g_use = g_use[m]
        neg_l = iw > (nw // 2)
        ru = np.where(neg_l, np.mod(nu - iu, nu), iu)
        rv = np.where(neg_l, np.mod(nv - iv, nv), iv)
        rw = np.where(neg_l, nw - iw, iw)
        if self.conv == 'plus':
            w_in = self.fsc * np.where(neg_l, g_use, np.conj(g_use))
        else:
            w_in = self.fsc * np.where(neg_l, np.conj(g_use), g_use)
        Br = np.zeros((nu, nv, nw // 2 + 1), dtype=np.complex128)
        np.add.at(Br, (ru, rv, rw), w_in)
        C = 0.5 * Br
        neg_u = np.mod(-np.arange(nu), nu)[:, None]
        neg_v = np.mod(-np.arange(nv), nv)[None, :]
        C[:, :, 0] = 0.5 * (Br[:, :, 0] + np.conj(Br[neg_u, neg_v, 0]))
        if nw % 2 == 0:
            C[:, :, -1] = 0.5 * (Br[:, :, -1] + np.conj(Br[neg_u, neg_v, -1]))
        return self.nvox * sp_fft.irfftn(C, s=self.shape, workers=self.workers)

    def sym_s(self, f):
        fl = f.ravel()
        acc = np.zeros(self.nvox, dtype=np.float64)
        for _, _, _, lin in self.ops:
            acc += fl[lin]
        return (acc / self.nops).reshape(self.shape)

    def sym_s_adj(self, g):
        gl = g.ravel()
        out = np.zeros(self.nvox, dtype=np.float64)
        for _, _, _, lin in self.ops:
            out[lin] += gl
        return (out / self.nops).reshape(self.shape)

    def sym_u(self, u):
        ul = u.reshape(3, -1)
        out = np.zeros_like(ul)
        for _, Ri, _, lin in self.ops:
            out += Ri @ ul[:, lin]
        return (out / self.nops).reshape(u.shape)

    def sym_u_adj(self, g):
        gl = g.reshape(3, -1)
        out = np.zeros_like(gl)
        for _, Ri, _, lin in self.ops:
            out[:, lin] += Ri.T @ gl
        return (out / self.nops).reshape(g.shape)

    def spectral_grad_scalar(self, f_3d):
        F_r = sp_fft.rfftn(f_3d, s=self.shape, axes=(0, 1, 2), workers=self.workers)
        return np.stack([
            sp_fft.irfftn(ik * F_r, s=self.shape, axes=(0, 1, 2), workers=self.workers) for ik in self.ik_r
        ], axis=0)

    def spectral_div_vector(self, v_3d):
        acc_r = np.zeros(self.stol2_r.shape, dtype=np.complex128)
        for b in range(3):
            acc_r += self.ik_r[b] * sp_fft.rfftn(v_3d[b], s=self.shape, axes=(0, 1, 2), workers=self.workers)
        return sp_fft.irfftn(acc_r, s=self.shape, axes=(0, 1, 2), workers=self.workers)

    def gaussian_blur(self, arr_3d, B_blur):
        if B_blur <= 0.0:
            return arr_3d.copy()
        F_r = sp_fft.rfftn(arr_3d, s=self.shape, axes=(0, 1, 2), workers=self.workers)
        return sp_fft.irfftn(F_r * np.exp(-float(B_blur) * self.stol2_r), s=self.shape, axes=(0, 1, 2), workers=self.workers)


def build_open_solvent_envelope(cgrid: CrystalGridOps, pdb_path, r_vdw=2.15, width=0.18, smooth_B=8.0):
    """Compute occupancy-weighted open-solvent fraction f_open(x) in [0, 1] directly from PDB altlocs & occupancies."""
    st = gemmi.read_structure(pdb_path)
    by_alt = {}
    blank_items = []
    for ch in st[0]:
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            for a in r:
                if a.element.name in ('H', 'D') or a.occ <= 0.0:
                    continue
                pos = a.pos.tolist()
                if a.altloc in ('\0', ' '):
                    blank_items.append((pos, min(float(a.occ), 1.0)))
                else:
                    by_alt.setdefault(a.altloc, []).append((pos, min(float(a.occ), 1.0)))

    asu_pts = cgrid.asu_xyz
    blocked_asu = np.zeros(cgrid.n_asu, dtype=np.float64)
    cut_env = float(r_vdw) + 6.0 * float(width)

    if blank_items:
        xyz_b = np.array([it[0] for it in blank_items], dtype=np.float64)
        occ_b = np.array([it[1] for it in blank_items], dtype=np.float64)
        sx, so = cgrid.sym_expand_xyz(xyz_b, cutoff_A=cut_env, occ_arr=occ_b)
        if len(sx):
            d_b, idx_b = cKDTree(sx).query(asu_pts)
            blocked_asu += so[idx_b] * expit((r_vdw - d_b) / width)

    for alt, items in sorted(by_alt.items()):
        xyz_a = np.array([it[0] for it in items], dtype=np.float64)
        occ_a = np.array([it[1] for it in items], dtype=np.float64)
        sx, so = cgrid.sym_expand_xyz(xyz_a, cutoff_A=cut_env, occ_arr=occ_a)
        if len(sx):
            d_a, idx_a = cKDTree(sx).query(asu_pts)
            blocked_asu += so[idx_a] * expit((r_vdw - d_a) / width)

    f_open_asu = np.clip(1.0 - blocked_asu, 0.0, 1.0)
    f_open_3d = f_open_asu[cgrid.full_to_asu].reshape(cgrid.shape)
    if smooth_B > 0.0:
        f_open_3d = np.clip(cgrid.sym_s(cgrid.gaussian_blur(f_open_3d, smooth_B)), 0.0, 1.0)
    return f_open_3d


def build_solvent_support_mask(base_3d, open_mask_3d, base_thresh=0.010, open_floor=0.25, open_span=0.15):
    """Smooth [0, 1] support mask that is identically 0 deep inside the excluded protein core (`base == 0` & `open <= 0.25`)
    and 1 throughout hydration shells (`base >= 0.010 e/A^3`) and open cavities/channels (`open >= 0.40`)."""
    b = np.asarray(base_3d, dtype=np.float64)
    om = np.asarray(open_mask_3d, dtype=np.float64)
    return np.clip(np.maximum(b / float(base_thresh), (om - float(open_floor)) / float(open_span)), 0.0, 1.0)


def tricubic_bhat(shape):
    k = [np.cos(2.0 * np.pi * np.fft.fftfreq(n)) for n in shape[:-1]] + [
        np.cos(2.0 * np.pi * np.fft.rfftfreq(shape[-1]))
    ]
    b = [2.0 / 3.0 + kk / 3.0 for kk in k]
    return b[0][:, None, None] * b[1][None, :, None] * b[2][None, None, :]


def tricubic_prefilter(D, B_hat, workers=4):
    return sp_fft.irfftn(sp_fft.rfftn(D, s=D.shape, workers=workers) / B_hat, s=D.shape, workers=workers)


def _bspline3_weights(f):
    g = 1.0 - f
    f2 = f * f
    w = (
        (g * g * g) / 6.0,
        (3.0 * f2 * f - 6.0 * f2 + 4.0) / 6.0,
        (-3.0 * f2 * f + 3.0 * f2 + 3.0 * f + 1.0) / 6.0,
        (f2 * f) / 6.0,
    )
    dw = (
        -0.5 * g * g,
        0.5 * (3.0 * f2 - 4.0 * f),
        0.5 * (-3.0 * f2 + 2.0 * f + 1.0),
        0.5 * f2,
    )
    return w, dw


class PeriodicFourierFields:
    """Supports standard S(x)/W(x) AND Mass-Conserving Wall-Clamped Transport sym_s(det(I - grad u_eff) * base(x - u_eff))."""

    def __init__(self, cgrid: CrystalGridOps, base_3d, fitreso_s=6.0, fitreso_w=6.0,
                 use_scale=True, use_warp=True, mass_conserving=False, wall_mask_3d=None,
                 j_min=0.30, tau_j=0.08, lam_fold=0.05):
        self.cgrid = cgrid
        self.base = np.asarray(base_3d, dtype=np.float64)
        self.use_s = bool(use_scale)
        self.use_w = bool(use_warp)
        self.mass_conserving = bool(mass_conserving)
        self.wall_mask = np.asarray(wall_mask_3d, dtype=np.float64) if wall_mask_3d is not None else None
        self.j_min = float(j_min)
        self.tau_j = float(tau_j)
        self.lam_fold = float(lam_fold)

        self.B_hat = tricubic_bhat(cgrid.shape)
        self.C_base_flat = tricubic_prefilter(self.base, self.B_hat, workers=cgrid.workers).ravel()

        self.HS, self.HSi = self._make_half_space(fitreso_s) if self.use_s else (np.zeros((0, 3), int), None)
        self.HW, self.HWi = self._make_half_space(fitreso_w) if self.use_w else (np.zeros((0, 3), int), None)
        self.nS = 2 * len(self.HS)
        self.nW = 2 * len(self.HW)
        self.nx = (self.nS if self.use_s else 0) + (3 * self.nW if self.use_w else 0)

    def _make_half_space(self, dmin):
        cell = self.cgrid.cell
        n = [int(math.ceil(L / dmin)) for L in (cell.a, cell.b, cell.c)]
        H = [
            (h, k, l)
            for h in range(-n[0], n[0] + 1)
            for k in range(-n[1], n[1] + 1)
            for l in range(-n[2], n[2] + 1)
            if (h > 0 or (h == 0 and (k > 0 or (k == 0 and l > 0))))
            and cell.calculate_d([h, k, l]) >= dmin - 1e-9
        ]
        H.sort(key=lambda h: -cell.calculate_d(list(h)))
        Ha = np.array(H, dtype=int).reshape(-1, 3)
        hi_p = tuple(np.mod(Ha[:, a], self.cgrid.shape[a]) for a in range(3))
        hi_m = tuple(np.mod(-Ha[:, a], self.cgrid.shape[a]) for a in range(3))
        return Ha, (hi_p, hi_m)

    def _fourier(self, c, Hi):
        n = len(c) // 2
        a, b = c[:n], c[n:]
        Z = np.zeros(self.cgrid.shape, dtype=np.complex128)
        Z[Hi[0]] = 0.5 * (a - 1j * b)
        Z[Hi[1]] = 0.5 * (a + 1j * b)
        return self.cgrid.nvox * np.real(sp_fft.ifftn(Z, workers=self.cgrid.workers))

    def _fourier_adj(self, g, Hi):
        Gt = sp_fft.fftn(g, workers=self.cgrid.workers)[Hi[0]]
        return np.concatenate([Gt.real, -Gt.imag])

    def _tricubic_eval_fixed_base(self, uf):
        shape = self.cgrid.shape
        Ns = self.cgrid.Ns
        C = self.C_base_flat
        I, Wt, dWt = [], [], []
        for a in range(3):
            g = np.arange(shape[a], dtype=np.float64).reshape([-1 if b == a else 1 for b in range(3)])
            p = g - uf[a] * Ns[a]
            p0 = np.floor(p)
            f = p - p0
            p0i = p0.astype(np.int32)
            I.append([np.mod(p0i + o, shape[a]).astype(np.int32) for o in (-1, 0, 1, 2)])
            w, dw = _bspline3_weights(f)
            Wt.append(w)
            dWt.append(dw)

        W = np.zeros(shape, dtype=np.float64)
        dW0 = np.zeros(shape, dtype=np.float64)
        dW1 = np.zeros(shape, dtype=np.float64)
        dW2 = np.zeros(shape, dtype=np.float64)
        s1, s2 = np.int32(shape[1]), np.int32(shape[2])

        for ox in range(4):
            ix_s = I[0][ox] * s1
            wx0, dwx0 = Wt[0][ox], dWt[0][ox]
            for oy in range(4):
                lxy = (ix_s + I[1][oy]) * s2
                wxy = wx0 * Wt[1][oy]
                dxy0 = dwx0 * Wt[1][oy]
                dxy1 = wx0 * dWt[1][oy]
                for oz in range(4):
                    cv = C[lxy + I[2][oz]]
                    wz = Wt[2][oz]
                    W += (wxy * wz) * cv
                    dW0 += (dxy0 * wz) * cv
                    dW1 += (dxy1 * wz) * cv
                    dW2 += (wxy * dWt[2][oz]) * cv
        return W, np.stack([dW0, dW1, dW2], axis=0)

    def split(self, x):
        cs = x[:self.nS] if self.use_s else np.zeros(0)
        cw = x[(self.nS if self.use_s else 0):] if self.use_w else np.zeros(0)
        return cs, cw

    def forward_and_pullback(self, x, compute_grad=True):
        """Return (rho_3d, pen_fold, diag_dict, pullback_fn)."""
        cgrid = self.cgrid
        cs, cw = self.split(x)
        S = np.exp(cgrid.sym_s(self._fourier(cs, self.HSi))) if self.use_s else None

        pen_fold = 0.0
        jac_det = None
        cof_A = None
        dpen_dJ = None
        uf_raw = None
        uf_eff = None
        W_raw = self.base
        dW = None

        if self.use_w:
            raw_u = np.stack([self._fourier(cw[i * self.nW:(i + 1) * self.nW], self.HWi) / cgrid.Lc[i] for i in range(3)])
            uf_raw = cgrid.sym_u(raw_u)
            uf_eff = (uf_raw * self.wall_mask[None, :, :, :]) if self.wall_mask is not None else uf_raw
            W_raw, dW = self._tricubic_eval_fixed_base(uf_eff)

            if self.mass_conserving:
                u_orth = np.einsum('ij,j...->i...', cgrid.Om, uf_eff)
                grad_u = np.stack([cgrid.spectral_grad_scalar(u_orth[a]) for a in range(3)], axis=0)
                A = -grad_u
                for a in range(3):
                    A[a, a] += 1.0
                r0, r1, r2 = A[0], A[1], A[2]
                c0 = np.cross(r1, r2, axisa=0, axisb=0, axisc=0)
                c1 = np.cross(r2, r0, axisa=0, axisb=0, axisc=0)
                c2 = np.cross(r0, r1, axisa=0, axisb=0, axisc=0)
                cof_A = np.stack([c0, c1, c2], axis=0)
                jac_det = np.sum(r0 * c0, axis=0)

                z_b = (self.j_min - jac_det) / self.tau_j
                sp_b = np.where(z_b > 30.0, z_b, np.log1p(np.exp(np.clip(z_b, -60.0, 30.0))))
                pen_fold = 0.5 * self.lam_fold * float(np.mean((self.tau_j * sp_b) ** 2))
                dpen_dJ = -(self.lam_fold / float(cgrid.nvox)) * (self.tau_j * sp_b) * expit(z_b)

                W_eff_unsym = jac_det * W_raw
            else:
                W_eff_unsym = W_raw
            # Exact space-group symmetrization (removes separable B-spline anisotropy in trigonal/hexagonal groups)
            W_eff = cgrid.sym_s(W_eff_unsym)
        else:
            W_eff = self.base

        rho = (W_eff * S) if self.use_s else W_eff

        diag = {}
        if self.use_s:
            diag['sd_lnS'] = float(np.std(np.log(S)))
        if self.use_w:
            uo = np.einsum('ij,j...->i...', cgrid.Om, uf_eff)
            un = np.sqrt(np.sum(uo ** 2, axis=0))
            diag['u_rms'] = float(np.sqrt(np.mean(un ** 2)))
            diag['u_max'] = float(np.max(un))
            if jac_det is not None:
                diag['jac_min'] = float(np.min(jac_det))
                diag['folded_pct'] = float(100.0 * np.mean(jac_det <= 0.0))

        if not compute_grad:
            return rho, pen_fold, diag, None

        def pullback(grho):
            gcs = self._fourier_adj(cgrid.sym_s_adj(grho * rho), self.HSi) if self.use_s else np.zeros(0)
            if not self.use_w:
                return gcs
            gW_eff = cgrid.sym_s_adj((grho * S) if self.use_s else grho)
            if self.mass_conserving:
                gW_raw = gW_eff * jac_det
                gJ = gW_eff * W_raw + dpen_dJ
                gu_frac_warp = -gW_raw[None] * dW * cgrid.Ns[:, None, None, None]
                gu_orth_jac = np.stack([
                    cgrid.spectral_div_vector(gJ[None] * cof_A[a]) for a in range(3)
                ], axis=0)
                gu_frac_eff = gu_frac_warp + np.einsum('ji,j...->i...', cgrid.Om, gu_orth_jac)
            else:
                gu_frac_eff = -gW_eff[None] * dW * cgrid.Ns[:, None, None, None]

            gu_frac_raw = (gu_frac_eff * self.wall_mask[None, :, :, :]) if self.wall_mask is not None else gu_frac_eff
            gus = cgrid.sym_u_adj(gu_frac_raw)
            gcw = np.concatenate([self._fourier_adj(gus[i] / cgrid.Lc[i], self.HWi) for i in range(3)])
            return np.concatenate([gcs, gcw])

        return rho, pen_fold, diag, pullback


def pick_asu_difference_peaks(cgrid: CrystalGridOps, diff_maps_and_sigmas, open_mask_3d,
                              z_thresh=3.5, max_peaks=120, min_sep_A=1.45, open_min=0.18):
    """Pick symmetry-unique local 3D extrema across multi-scale open-solvent work-train mFo - DFc maps."""
    candidates = []

    for B_pick, dmap in diff_maps_and_sigmas:
        wmap = dmap * open_mask_3d
        open_vals = dmap[open_mask_3d > open_min]
        if len(open_vals) < 100:
            open_vals = dmap.ravel()
        sigma = max(1.4826 * float(np.median(np.abs(open_vals - np.median(open_vals)))), 1e-6)
        zmap = wmap / sigma
        az = np.abs(zmap)

        is_max = (az >= float(z_thresh)) & (open_mask_3d >= open_min)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nb = np.roll(az, shift=(dx, dy, dz), axis=(0, 1, 2))
                    is_max &= (az >= nb)

        asu_mask = np.zeros(cgrid.nvox, dtype=bool)
        asu_mask[cgrid.asu_reps] = True
        keep_flat = is_max.ravel() & asu_mask
        idxs = np.nonzero(keep_flat)[0]
        for idx in idxs:
            candidates.append((
                float(az.ravel()[idx]),
                float(zmap.ravel()[idx]),
                float(dmap.ravel()[idx]),
                float(B_pick),
                cgrid.xyz_full[idx].copy(),
            ))

    candidates.sort(key=lambda t: -t[0])
    chosen = []
    sym_pts = np.zeros((0, 3), dtype=np.float64)
    sym_tree = None

    for az_v, sz_v, val_e, b_init, xyz in candidates:
        if len(chosen) >= max_peaks:
            break
        if sym_tree is not None and sym_tree.query(xyz)[0] < min_sep_A:
            continue
        chosen.append({
            'xyz0': xyz,
            'z_score': sz_v,
            'val_eA3': val_e,
            'B_init': max(8.0, b_init + 4.0),
        })
        imgs = cgrid.sym_expand_xyz(xyz[None, :], cutoff_A=max(2.5, float(min_sep_A) + 0.5))
        if len(imgs):
            sym_pts = np.vstack([sym_pts, imgs])
            sym_tree = cKDTree(sym_pts)

    return chosen


class ContinuousPeakDict:
    """Continuous-coordinate space-group-symmetric IT92 Gaussian water/ion dictionary (zero grid-edge ripple)."""

    def __init__(self, cgrid: CrystalGridOps, peak_seeds, B_base=2.0, half_win=9, max_shift=0.75, b_min=None):
        self.cgrid = cgrid
        self.M = len(peak_seeds)
        self.B_base = float(B_base)
        self.b_min = float(b_min) if b_min is not None else 5.5
        self.half_win = int(half_win)
        self.max_shift = float(max_shift)
        self.xyz0 = np.array([p['xyz0'] for p in peak_seeds], dtype=np.float64) if self.M else np.zeros((0, 3))
        self.B_init = np.array([p['B_init'] for p in peak_seeds], dtype=np.float64) if self.M else np.zeros(0)
        self.z_init = np.array([p['z_score'] for p in peak_seeds], dtype=np.float64) if self.M else np.zeros(0)

        self.K_wat_r = (cgrid.f_wat_r * np.exp(-self.B_base * cgrid.stol2_r)) / cgrid.vox_vol
        self.offs = np.arange(-self.half_win, self.half_win + 1, dtype=np.int32)
        self.n_params = 5 * self.M

    def _eta_bounds(self):
        lo_dB = max(3.5, self.b_min - self.B_base)
        hi_dB = max(95.0, lo_dB + 60.0)
        return math.log(lo_dB), math.log(hi_dB)

    def init_params(self):
        if self.M == 0:
            return np.zeros(0, dtype=np.float64)
        p = np.zeros((self.M, 5), dtype=np.float64)
        lo_eta, hi_eta = self._eta_bounds()
        p[:, 4] = np.clip(np.log(np.clip(self.B_init - self.B_base, 4.0, 120.0)), lo_eta, hi_eta)
        return p.ravel()

    def bounds(self):
        b = []
        ms = self.max_shift
        lo_eta, hi_eta = self._eta_bounds()
        for _ in range(self.M):
            b.extend([(-1.8, 2.8), (-ms, ms), (-ms, ms), (-ms, ms), (lo_eta, hi_eta)])
        return b

    def unpack(self, theta_p):
        if self.M == 0:
            return np.zeros(0), np.zeros((0, 3)), np.zeros(0)
        P = np.asarray(theta_p, dtype=np.float64).reshape(self.M, 5)
        dq = P[:, 0]
        dxyz = P[:, 1:4]
        eta = P[:, 4]
        return dq, dxyz, eta

    def forward_and_pullback(self, theta_p, compute_grad=True):
        cgrid = self.cgrid
        if self.M == 0:
            z3 = np.zeros(cgrid.shape, dtype=np.float64)
            return z3, (lambda g: np.zeros(0)) if compute_grad else None

        dq, dxyz, eta = self.unpack(theta_p)
        xyz = self.xyz0 + dxyz
        dB = np.exp(eta)
        inv_2sig2 = (4.0 * (np.pi ** 2)) / dB

        u_raw = np.zeros(cgrid.shape, dtype=np.float64)
        stencils = []
        offs = self.offs
        Ns = cgrid.Ns
        Om = cgrid.Om
        Fm = cgrid.Fm
        shape = cgrid.shape

        for j in range(self.M):
            pos = xyz[j]
            i2s = inv_2sig2[j]
            c_idx = np.rint((pos @ Fm.T) * Ns).astype(np.int32)
            ia = c_idx[0] + offs
            ib = c_idx[1] + offs
            ic = c_idx[2] + offs
            fa = (ia / Ns[0])[:, None, None, None]
            fb = (ib / Ns[1])[None, :, None, None]
            fc = (ic / Ns[2])[None, None, :, None]
            d_vec = (fa * Om[:, 0] + fb * Om[:, 1] + fc * Om[:, 2]) - pos[None, None, None, :]
            r2 = np.sum(d_vec * d_vec, axis=-1)
            E_box = np.exp(-i2s * r2)
            W_box = E_box / np.sum(E_box)
            ix = np.ix_(np.mod(ia, shape[0]), np.mod(ib, shape[1]), np.mod(ic, shape[2]))
            if dq[j] != 0.0:
                u_raw[ix] += dq[j] * W_box
            if compute_grad:
                dbar_vec = np.sum(W_box[..., None] * d_vec, axis=(0, 1, 2))
                r2_bar = float(np.sum(W_box * r2))
                stencils.append((ix, d_vec, r2, dbar_vec, r2_bar, W_box, i2s))

        u_sym = cgrid.nops * cgrid.sym_s(u_raw)
        F_u = sp_fft.rfftn(u_sym, s=shape, workers=cgrid.workers)
        rho_peaks = sp_fft.irfftn(F_u * self.K_wat_r, s=shape, workers=cgrid.workers)

        if not compute_grad:
            return rho_peaks, None

        def pullback(grho):
            F_g = sp_fft.rfftn(grho, s=shape, workers=cgrid.workers)
            gu_sym = sp_fft.irfftn(F_g * self.K_wat_r, s=shape, workers=cgrid.workers)
            gu_raw = cgrid.nops * cgrid.sym_s_adj(gu_sym)

            grad = np.zeros((self.M, 5), dtype=np.float64)
            for j in range(self.M):
                ix, d_vec, r2, dbar_vec, r2_bar, W_box, i2s = stencils[j]
                g_box = gu_raw[ix]
                gw = g_box * W_box
                grad[j, 0] = float(np.sum(gw))
                dq_j = dq[j]
                if dq_j != 0.0:
                    grad[j, 1:4] = dq_j * 2.0 * i2s * np.sum(gw[..., None] * (d_vec - dbar_vec[None, None, None, :]), axis=(0, 1, 2))
                    grad[j, 4] = dq_j * i2s * float(np.sum(gw * (r2 - r2_bar)))
            return grad.ravel()

        return rho_peaks, pullback


class WatershedDivotDict:
    """Extracts Bader steepest-ascent watershed divots on open-solvent work-train mFo - DFc and refines weights."""

    def __init__(self, cgrid: CrystalGridOps, diff_map_3d, open_mask_3d,
                 z_peak=3.5, z_floor=1.8, rmax_A=2.2, max_divots=100, smooth_B=3.0, open_min=0.18):
        self.cgrid = cgrid
        wmap = cgrid.sym_s(diff_map_3d * open_mask_3d)
        open_vals = diff_map_3d[open_mask_3d > open_min]
        if len(open_vals) < 100:
            open_vals = diff_map_3d.ravel()
        sigma = max(1.4826 * float(np.median(np.abs(open_vals - np.median(open_vals)))), 1e-6)
        self.sigma = sigma

        templates_asu = []

        for sign in (+1.0, -1.0):
            smap = sign * wmap
            active = (smap >= z_floor * sigma) & (open_mask_3d >= open_min)
            if not np.any(active):
                continue

            best_slope = np.zeros(cgrid.shape, dtype=np.float64)
            parent = np.arange(cgrid.nvox, dtype=np.int64).reshape(cgrid.shape)
            lin_grid = parent.copy()

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        step_cart = (np.array([dx, dy, dz], dtype=np.float64) / cgrid.Ns) @ cgrid.Om.T
                        d_step = float(np.linalg.norm(step_cart))
                        nb_val = np.roll(smap, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
                        nb_lin = np.roll(lin_grid, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
                        slope = (nb_val - smap) / d_step
                        better = active & (slope > best_slope + 1e-12)
                        best_slope = np.where(better, slope, best_slope)
                        parent = np.where(better, nb_lin, parent)

            p_flat = parent.ravel()
            for _ in range(6):
                p_flat = p_flat[p_flat]

            canon_dest = cgrid.full_to_asu[p_flat]
            act_vox = np.nonzero(active.ravel())[0]
            act_orbits = canon_dest[act_vox]

            unique_orbits = np.unique(act_orbits)
            cand_divots = []
            for orb in unique_orbits:
                rep_vox = cgrid.asu_reps[orb]
                pk_z = float(smap.ravel()[rep_vox] / sigma)
                if pk_z < z_peak:
                    continue
                cand_divots.append((pk_z, int(orb), int(rep_vox)))

            cand_divots.sort(key=lambda t: -t[0])
            for pk_z, orb, rep_vox in cand_divots:
                if len(templates_asu) >= max_divots:
                    break
                vox_in_basin = act_vox[act_orbits == orb]
                asu_in_basin = np.unique(cgrid.full_to_asu[vox_in_basin])
                pk_xyz = cgrid.xyz_full[rep_vox]
                imgs = cgrid.sym_expand_xyz(pk_xyz[None, :], cutoff_A=float(rmax_A) + 0.5)
                if len(imgs) == 0:
                    continue
                d_to_pk = cKDTree(imgs).query(cgrid.asu_xyz[asu_in_basin])[0]
                keep_a = asu_in_basin[d_to_pk <= rmax_A]
                if len(keep_a) < 3:
                    continue

                tmpl_asu = np.zeros(cgrid.n_asu, dtype=np.float64)
                taper = np.clip(1.0 - (d_to_pk[d_to_pk <= rmax_A] / (rmax_A + 0.35)) ** 4, 0.0, 1.0)
                tmpl_asu[keep_a] = sign * smap.ravel()[cgrid.asu_reps[keep_a]] * taper
                tmpl_3d = tmpl_asu[cgrid.full_to_asu].reshape(cgrid.shape)
                if smooth_B > 0.0:
                    tmpl_3d = cgrid.sym_s(cgrid.gaussian_blur(tmpl_3d, smooth_B))
                tmpl_asu = tmpl_3d.ravel()[cgrid.asu_reps]
                mx = float(np.max(np.abs(tmpl_asu)))
                if mx > 1e-8:
                    templates_asu.append((pk_z, tmpl_asu / mx, sign, int(len(keep_a))))

        templates_asu.sort(key=lambda t: -t[0])
        templates_asu = templates_asu[:max_divots]
        self.K = len(templates_asu)
        self.Psi_asu = np.stack([t[1] for t in templates_asu], axis=0) if self.K else np.zeros((0, cgrid.n_asu))
        self.meta = [(t[0], t[2], t[3]) for t in templates_asu]

    def forward_and_pullback(self, weights, compute_grad=True):
        cgrid = self.cgrid
        if self.K == 0:
            z3 = np.zeros(cgrid.shape, dtype=np.float64)
            return z3, (lambda g: np.zeros(0)) if compute_grad else None
        w = np.asarray(weights, dtype=np.float64)
        asu_sum = w @ self.Psi_asu
        rho_div = asu_sum[cgrid.full_to_asu].reshape(cgrid.shape)
        if not compute_grad:
            return rho_div, None

        def pullback(grho):
            g_asu = np.bincount(cgrid.full_to_asu, grho.ravel(), minlength=cgrid.n_asu)
            return self.Psi_asu @ g_asu

        return rho_div, pullback


class SolventAmplitudeProblem:
    """Evaluates |Fo| amplitude loss & gradient strictly on train_mask, reporting work_val & free R cleanly."""

    # k_solv is floored above 0: at k_solv = 0 the gradient wrt rho (proportional to k_solv) vanishes identically,
    # so a start whose best scale is 0 could never move and would write an all-zero FPART
    K_SOLV_MIN = 0.05
    P_LO = np.array([0.50, K_SOLV_MIN], dtype=np.float64)
    P_HI = np.array([1.50, 5.00], dtype=np.float64)

    def __init__(self, cgrid: CrystalGridOps, H, Fo, free_mask, K_scale, F_calc, train_mask=None, val_mask=None,
                 k_solv_mode: str = 'free', f_model_ref=None):
        if k_solv_mode not in ('free', 'fixed_one'):
            raise ValueError(f"k_solv_mode must be 'free' or 'fixed_one', got {k_solv_mode!r}")
        self.k_solv_mode = str(k_solv_mode)
        self.cgrid = cgrid
        self.H = np.asarray(H, dtype=int)
        self.idx = cgrid.hkl_idx(self.H)
        if len(self.idx) == 4 and not np.all(self.idx[3]):
            n_out = int(np.sum(~self.idx[3]))
            print(
                f"surfnturf warning: {n_out}/{len(self.H)} reflections exceed base grid {cgrid.shape} Nyquist limit "
                f"and have FPART=0 during divot fitting",
                file=sys.stderr,
            )
        self.Fo = np.asarray(Fo, dtype=np.float64)
        self.free = np.asarray(free_mask, dtype=bool)
        self.work = ~self.free
        if int(np.sum(self.work)) < 2:
            raise ValueError(f"SolventAmplitudeProblem requires at least 2 work reflections (got {int(np.sum(self.work))})")
        self.K = np.asarray(K_scale, dtype=np.float64)
        self.Q = np.asarray(F_calc, dtype=np.complex128)
        self.f_model_ref = np.asarray(f_model_ref, dtype=np.float64) if f_model_ref is not None else None

        self.stol2 = 0.25 * np.asarray(cgrid.cell.calculate_1_d2_array(self.H), dtype=np.float64)
        self.d_arr = 1.0 / np.sqrt(np.maximum(4.0 * self.stol2, 1e-12))

        if train_mask is None or val_mask is None:
            from .bmob import make_work_train_val_split
            self.train, self.val = make_work_train_val_split(self.H, self.d_arr, self.free, spacegroup=cgrid.sg)
        else:
            self.train = np.asarray(train_mask, dtype=bool) & self.work
            self.val = np.asarray(val_mask, dtype=bool) & self.work

        assert not np.any(self.train & self.free), "FREE SET LEAK IN TRAIN!"
        assert not np.any(self.val & self.free), "FREE SET LEAK IN VAL!"

        # Precompute exact reciprocal-space stabilizer multiplicity nops / m_stab(h) so compute_work_diff_map is in exact e/A^3
        m_stab = np.zeros(len(self.H), dtype=np.float64)
        for R, _, _, _ in cgrid.ops:
            hR = self.H @ R
            m_stab += np.all(np.abs(hR - self.H) < 1e-6, axis=1)
            m_stab += np.all(np.abs(hR + self.H) < 1e-6, axis=1)
        self.diff_map_weight = float(cgrid.nops) / np.maximum(m_stab, 1.0)

        self.nrm_train = float(np.sum(self.Fo[self.train] ** 2))
        self.p_cache = np.array([1.0, 1.0], dtype=np.float64)

    def r_factor(self, sel, Fm):
        denom = float(np.sum(self.Fo[sel])) if np.any(sel) else 0.0
        if denom <= 0.0:
            return float('nan')
        return float(np.sum(np.abs(self.Fo[sel] - np.abs(Fm[sel]))) / denom)

    def eval_fmodel_ref_r(self, sel):
        if self.f_model_ref is None or not np.any(sel):
            return float('nan')
        denom = float(np.sum(self.Fo[sel]))
        if denom <= 0.0:
            return float('nan')
        return float(np.sum(np.abs(self.Fo[sel] - self.f_model_ref[sel])) / denom)

    def solve_inner_scales(self, P, sel_mask, p_init=None, nit=20):
        """Solve inner amplitude scales on `sel_mask`.

        In `'fixed_one'` mode (default for calibrated e/A^3 solvent maps), locks `k_solv == k_prot == k0` via
        closed-form projection onto [P_LO[0], P_HI[0]], preventing L1-regularized peak dictionaries from cheating by
        inflating `k_solv / k_prot`. In `'free'` mode, runs history-free 2D box-constrained Gauss-Newton.
        """
        lo, hi = self.P_LO, self.P_HI
        w = sel_mask
        Qw = (self.K * self.Q)[w]
        Pw = (self.K * P)[w]
        Fow = self.Fo[w]

        if self.k_solv_mode == 'fixed_one':
            au = np.abs(Qw + Pw)
            k0 = float(np.dot(au, Fow) / max(float(np.dot(au, au)), 1e-12))
            k0 = float(np.clip(k0, lo[0], hi[0]))
            return np.array([k0, k0], dtype=np.float64)

        def _obj(pv):
            res = Fow - np.abs(pv[0] * Qw + pv[1] * Pw)
            return float(np.dot(res, res))

        cand_inits = [
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.35], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        ]
        if p_init is not None:
            cand_inits.append(np.asarray(p_init, dtype=np.float64))
        best_p, best_f = None, float('inf')
        for p0 in cand_inits:
            p_run = self._gauss_newton_scales(np.clip(p0, lo, hi), _obj, Qw, Pw, Fow, lo, hi, nit)
            f_run = _obj(p_run)
            if f_run < best_f:
                best_p, best_f = p_run, f_run
        return best_p

    @staticmethod
    def _gauss_newton_scales(p, _obj, Qw, Pw, Fow, lo, hi, nit):
        p = np.array(p, dtype=np.float64)
        cur_f = _obj(p)
        for _ in range(nit):
            Fm = p[0] * Qw + p[1] * Pw
            a = np.maximum(np.abs(Fm), 1e-12)
            e = Fm / a
            r = Fow - a
            Jm = np.column_stack([np.real(np.conj(e) * Qw), np.real(np.conj(e) * Pw)])
            # Gradient of sum(r^2) is -2 * g_m; coordinate i is locked on bound if step pushes further out
            g_m = Jm.T @ r
            free_c = np.ones(2, dtype=bool)
            for c in range(2):
                if (p[c] <= lo[c] + 1e-12 and g_m[c] <= 0.0) or (p[c] >= hi[c] - 1e-12 and g_m[c] >= 0.0):
                    free_c[c] = False
            if not np.any(free_c):
                break
            Jf = Jm[:, free_c]
            Hf = Jf.T @ Jf
            gf = g_m[free_c]
            reg = 1e-10 * max(float(np.trace(Hf)), 1e-6) * np.eye(int(np.sum(free_c)))
            dp_f = np.linalg.solve(Hf + reg, gf)
            dp = np.zeros(2, dtype=np.float64)
            dp[free_c] = dp_f

            step_accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.02):
                p_try = np.clip(p + alpha * dp, lo, hi)
                f_try = _obj(p_try)
                if f_try <= cur_f:
                    step_sz = float(np.max(np.abs(p_try - p)))
                    p = p_try
                    cur_f = f_try
                    step_accepted = True
                    break
            if not step_accepted or step_sz < 1e-9:
                break
        return p

    def eval_rho(self, rho_3d, active_mask=None, compute_grad=True):
        use_mask = self.train if active_mask is None else active_mask
        assert not np.any(use_mask & self.free), "Attempted to evaluate loss on FREE reflections!"
        nrm = max(float(np.sum(self.Fo[use_mask] ** 2)), 1e-12)

        P = self.cgrid.sf(rho_3d, self.idx)
        p = self.solve_inner_scales(P, use_mask)
        self.p_cache = p.copy()
        Fm = self.K * (p[0] * self.Q + p[1] * P)
        a = np.maximum(np.abs(Fm), 1e-12)
        e = Fm / a
        r = self.Fo - a
        loss = float(np.sum(r[use_mask] ** 2) / nrm)

        grho = None
        if compute_grad:
            dLda = np.zeros(len(self.Fo), dtype=np.float64)
            dLda[use_mask] = -2.0 * r[use_mask] / nrm
            G = self.K * p[1] * dLda * e
            grho = self.cgrid.sf_adjoint(G, self.idx)

        return {
            'loss': loss,
            'grho': grho,
            'P': P,
            'Fm': Fm,
            'p': p,
            'r_train': self.r_factor(self.train, Fm),
            'r_val': self.r_factor(self.val, Fm),
            'r_work': self.r_factor(self.work, Fm),
            'r_free': self.r_factor(self.free, Fm),
        }

    def scaled_solvent_rho(self, rho_3d, active_mask=None):
        """Return `(rho_scaled, ev)` with `rho_scaled = (k_solv / k_prot) * rho_3d` matching `ev['Fm']`."""
        use_mask = self.work if active_mask is None else active_mask
        ev = self.eval_rho(rho_3d, active_mask=use_mask, compute_grad=False)
        scale = float(ev['p'][1] / max(ev['p'][0], 1e-8))
        return scale * np.asarray(rho_3d, dtype=np.float64), ev

    def compute_work_diff_map(self, rho_3d, active_mask=None, B_smooth=8.0):
        """Compute exact e/A^3 mFo - DFc difference map across all 230 space groups using ONLY active_mask."""
        use_mask = self.train if active_mask is None else active_mask
        assert not np.any(use_mask & self.free), "Attempted to include FREE reflections in difference map!"
        ev = self.eval_rho(rho_3d, active_mask=use_mask, compute_grad=False)
        Fm = ev['Fm']
        a = np.maximum(np.abs(Fm), 1e-12)
        phase = Fm / a
        dw = np.exp(-float(B_smooth) * self.stol2)

        dF = np.zeros(len(self.Fo), dtype=np.complex128)
        dF[use_mask] = (
            self.diff_map_weight[use_mask]
            * ((self.Fo[use_mask] - a[use_mask]) / np.maximum(ev['p'][0] * self.K[use_mask], 1e-6))
            * phase[use_mask]
            * dw[use_mask]
        )
        if len(self.idx) == 4 and not np.all(self.idx[3]):
            dF = np.where(self.idx[3], dF, 0.0 + 0.0j)

        Z = np.zeros(self.cgrid.shape, dtype=np.complex128)
        idx_p = self.idx[:3]
        idx_m = tuple(np.mod(-self.H[:, c], self.cgrid.shape[c]) for c in range(3))
        if self.cgrid.conv == 'plus':
            np.add.at(Z, idx_p, np.conj(dF))
            np.add.at(Z, idx_m, dF)
            dmap = (self.cgrid.nvox / float(self.cgrid.cell.volume)) * np.real(sp_fft.ifftn(Z, workers=self.cgrid.workers))
        else:
            np.add.at(Z, idx_p, dF)
            np.add.at(Z, idx_m, np.conj(dF))
            dmap = (1.0 / float(self.cgrid.cell.volume)) * np.real(sp_fft.fftn(Z, workers=self.cgrid.workers))
        return self.cgrid.sym_s(dmap)


def write_fpart_cgrid(cgrid: CrystalGridOps, rho_3d, hkl_mtz_path, out_mtz_path, col_names=('FPART', 'PHIPART')):
    """Write reciprocal-space solvent structure factors FPART/PHIPART onto the Miller indices of `hkl_mtz_path`."""
    from .io import check_crystal_compatibility
    m = gemmi.read_mtz_file(hkl_mtz_path)
    check_crystal_compatibility(cgrid.cell, cgrid.sg, "solvent grid", m.cell, m.spacegroup, str(hkl_mtz_path))
    H = np.array(m.make_miller_array(), dtype=int)
    idx = cgrid.hkl_idx(H)
    if len(idx) == 4 and not np.all(idx[3]):
        n_out = int(np.sum(~idx[3]))
        print(
            f"surfnturf warning: {n_out}/{len(H)} reflections in {hkl_mtz_path} exceed grid {cgrid.shape} Nyquist limit "
            f"and are set to FPART=0",
            file=sys.stderr,
        )
    v = cgrid.sf(rho_3d, idx)
    out = gemmi.Mtz(with_base=True)
    out.cell = m.cell
    out.spacegroup = m.spacegroup
    out.add_dataset('bulk')
    out.add_column(col_names[0], 'F')
    out.add_column(col_names[1], 'P')
    out.set_data(np.column_stack([H, np.abs(v), np.degrees(np.angle(v))]).astype(np.float32))
    out.write_to_file(out_mtz_path)
    return out_mtz_path


def extract_problem_from_fmodel(cgrid: CrystalGridOps, fmodel_mtz_path: str, data_mtz_path: str = None,
                                free_flag_col: str = 'FreeR_flag', free_flag_value=None,
                                k_solv_mode: str = 'fixed_one'):
    """Build SolventAmplitudeProblem from a Phenix `*_f_model.mtz` file (`export_final_f_model=True`)."""
    fm = gemmi.read_mtz_file(fmodel_mtz_path)
    lbls = fm.column_labels()
    arr = np.array(fm, copy=True)
    cmap = {l.upper().replace('-', '').replace('_', ''): arr[:, i] for i, l in enumerate(lbls)}

    def pick(*names, required=True):
        for nm in names:
            k = nm.upper().replace('-', '').replace('_', '')
            if k in cmap:
                return cmap[k]
        if required:
            raise ValueError(
                f"{fmodel_mtz_path} is missing required column ({'/'.join(names)}); "
                f"present columns: {lbls}. Export with `phenix.refine ... output.export_final_f_model=True`."
            )
        return None

    H = arr[:, :3].astype(int)
    Fo = pick('FOBS', 'F-obs-filtered', 'F-obs', 'FP').astype(np.float64)
    flags = pick('R_FREE_FLAGS', 'R-free-flags', 'FreeR_flag')
    valid = np.isfinite(Fo) & (Fo > 0.0) & np.isfinite(flags)
    if int(np.sum(valid)) < 2:
        raise ValueError(f"{fmodel_mtz_path}: fewer than 2 valid positive reflections found")

    from .bmob import free_mask_from_flags
    free = free_mask_from_flags(flags, valid, free_flag_value, source_desc=fmodel_mtz_path)

    fc_amp = pick('FCALC', 'FCALCATOMS')
    fc_phi = pick('PHIFCALC', 'PHIFCALCATOMS')
    fk_amp = pick('FMASK', 'FPART')
    fk_phi = pick('PHIFMASK', 'PHIFPART', 'PHIPART')
    fm_amp = pick('FMODEL', 'F-model')
    fm_phi = pick('PHIFMODEL', 'PHIF-model')
    km = pick('K_MASK', 'KMASK').astype(np.float64)

    k_iso = pick('K_ISOTROPIC', 'K_ISO', 'KISO', required=False)
    k_aniso = pick('K_ANISOTROPIC', 'K_ANISO', 'KANISO', required=False)
    k_total_col = pick('K_TOTAL', 'KTOTAL', required=False)
    if k_iso is not None and k_aniso is not None:
        kk = (k_iso * k_aniso).astype(np.float64)
    elif k_total_col is not None:
        kk = k_total_col.astype(np.float64)
    else:
        raise ValueError(
            f"{fmodel_mtz_path} is missing K_ISOTROPIC/K_ANISOTROPIC (or K_TOTAL); present columns: {lbls}."
        )

    Fc = fc_amp * np.exp(1j * np.deg2rad(fc_phi))
    Fk = fk_amp * np.exp(1j * np.deg2rad(fk_phi))
    Fmod = fm_amp * np.exp(1j * np.deg2rad(fm_phi))

    denom = kk * (Fc + km * Fk)
    nz = valid & (np.abs(denom) > 1e-6)
    cK = float(np.median(np.abs(Fmod[nz]) / np.abs(denom[nz]))) if np.any(nz) else 1.0
    K_scale = cK * kk

    from .bmob import make_work_train_val_split
    from .io import resolve_crystal_symmetry, check_crystal_compatibility
    sg_hm = fm.spacegroup.hm if fm.spacegroup is not None else ""
    mtz_cell, mtz_sg = resolve_crystal_symmetry(fm.cell, sg_hm, source_desc=str(fmodel_mtz_path))
    check_crystal_compatibility(cgrid.cell, cgrid.sg, "solvent grid", mtz_cell, mtz_sg, str(fmodel_mtz_path))
    if data_mtz_path is not None and os.path.isfile(data_mtz_path):
        dm_chk = gemmi.read_mtz_file(data_mtz_path)
        check_crystal_compatibility(mtz_cell, mtz_sg, str(fmodel_mtz_path), dm_chk.cell, dm_chk.spacegroup, str(data_mtz_path))

    free_v = free[valid]
    train_v, val_v = make_work_train_val_split(
        H[valid], None, free_v, data_mtz_path=data_mtz_path, free_flag_col=free_flag_col, spacegroup=mtz_sg
    )

    return SolventAmplitudeProblem(
        cgrid, H[valid], Fo[valid], free_v, K_scale[valid], Fc[valid],
        train_mask=train_v, val_mask=val_v, k_solv_mode=k_solv_mode,
        f_model_ref=np.abs(Fmod)[valid],
    )


def load_solvhook_base(npz_path: str, fft_workers: int = 4):
    """Load a saved `(cgrid, base_3d)` solvent base `.npz` bundle."""
    z = np.load(npz_path, allow_pickle=False)
    base = np.array(z['base'], dtype=np.float64)
    shape = tuple(int(n) for n in z['shape'])
    conv = str(z['conv'])
    fsc = float(z['fsc'])
    cell = gemmi.UnitCell(*[float(v) for v in z['cell']])
    sg_hm = str(z['sg'])
    sg = gemmi.find_spacegroup_by_name(sg_hm, alpha=cell.alpha, gamma=cell.gamma) or gemmi.SpaceGroup(sg_hm)
    cgrid = CrystalGridOps(cell, sg, shape, conv=conv, fsc=fsc, fft_workers=fft_workers)
    return cgrid, base


def fit_periodic_arm(prob: SolventAmplitudeProblem, pf: PeriodicFourierFields,
                     rs=1e-4, rw=1e-4, maxiter=160, bound=3.0):
    """Fit PeriodicFourierFields: first on prob.train (90%) for blind r_val, then warm-started on prob.work (95%)."""
    t0 = time.time()
    x0 = np.zeros(pf.nx, dtype=np.float64)
    bounds = [(-bound, bound)] * pf.nx

    def make_obj(mask):
        def obj(x):
            rho, pen_fold, _, pb = pf.forward_and_pullback(x, compute_grad=True)
            ev = prob.eval_rho(rho, active_mask=mask, compute_grad=True)
            cs, cw = pf.split(x)
            pen_reg = float(rs) * float(cs @ cs) + float(rw) * float(cw @ cw)
            g_reg = np.concatenate([2.0 * float(rs) * cs, 2.0 * float(rw) * cw])
            g = pb(ev['grho']) + g_reg
            return ev['loss'] + pen_fold + pen_reg, g
        return obj

    res_tr = minimize(make_obj(prob.train), x0, method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-11, 'gtol': 1e-8})
    rho_tr, _, _, _ = pf.forward_and_pullback(res_tr.x, compute_grad=False)
    r_val_blind = prob.eval_rho(rho_tr, active_mask=prob.train, compute_grad=False)['r_val']

    res = minimize(make_obj(prob.work), res_tr.x, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max(25, maxiter // 4), 'ftol': 1e-11, 'gtol': 1e-8})
    rho_opt, _, diag, _ = pf.forward_and_pullback(res.x, compute_grad=False)
    diag['dt_s'] = round(time.time() - t0, 1)
    diag['nit'] = int(res_tr.nit + res.nit)
    diag['r_val'] = float(r_val_blind)
    return rho_opt, res.x, diag


def fit_scale_divot_peak_given_hp(prob: SolventAmplitudeProblem, cgrid: CrystalGridOps, base_3d, open_mask_3d,
                                   active_mask, fitreso_s=12.0, use_divots=True, use_peaks=True,
                                   z_thresh=3.4, lam_l1=4e-5, rs=1e-4, n_passes=1, max_peaks_per_pass=90,
                                   maxiter_s0=60, maxiter_joint=130, precomputed_s0=None,
                                   warm_state=None, return_state: bool = False, use_scale: bool = True):
    """Fit S(x)*base(x) + WatershedDivots + ContinuousPeakDict strictly on active_mask (NEVER free!)."""
    d_min_p = float(np.min(prob.d_arr)) if len(prob.d_arr) else 1.0
    n_active = max(int(np.sum(active_mask)), 1)
    fitreso_eff = max(float(fitreso_s), round(4.0 * d_min_p, 2)) if d_min_p > 1.50 else float(fitreso_s)
    b_lo = max(6.0, round(3.0 * (d_min_p ** 2), 1)) if d_min_p > 1.45 else 6.0
    b_hi = max(22.0, round(8.0 * (d_min_p ** 2), 1)) if d_min_p > 1.45 else 22.0
    rmax_eff = max(2.2, round(1.3 * d_min_p, 2)) if d_min_p > 1.70 else 2.2
    min_sep_eff = max(1.35, round(0.85 * d_min_p, 2)) if d_min_p > 1.60 else 1.35
    b_peak_min = max(5.5, 4.0 * (d_min_p ** 2))
    rs_eff = float(rs) * max(1.0, (d_min_p / 1.0) ** 2)

    max_peaks_eff = min(int(max_peaks_per_pass), max(8, n_active // 200))
    max_divots_eff = min(180, max(10, n_active // 160))
    n_passes_eff = (1 if d_min_p > 2.0 else int(n_passes)) if int(n_passes) > 0 else 0

    supp_mask = build_solvent_support_mask(base_3d, open_mask_3d)
    pf_s = PeriodicFourierFields(cgrid, base_3d, fitreso_s=fitreso_eff, use_scale=bool(use_scale), use_warp=False)

    def _make_joint_obj(wdiv_cur, pdict_cur, n_s_cur, n_d_cur, n_p_cur, l1_cur):
        def obj_joint(vec):
            cs_v = vec[:n_s_cur]
            wd_v = vec[n_s_cur:n_s_cur + n_d_cur]
            pk_v = vec[n_s_cur + n_d_cur:]

            rho_s, _, _, pb_s = pf_s.forward_and_pullback(cs_v, compute_grad=True)
            tot_raw = rho_s
            pb_d = pb_p = None
            if n_d_cur:
                rho_d, pb_d = wdiv_cur.forward_and_pullback(wd_v, compute_grad=True)
                tot_raw = tot_raw + rho_d
            if n_p_cur:
                rho_p, pb_p = pdict_cur.forward_and_pullback(pk_v, compute_grad=True)
                tot_raw = tot_raw + rho_p

            sp_val, dpos = softplus_pos(tot_raw, tau=0.012)
            rho_pos = supp_mask * sp_val
            ev = prob.eval_rho(rho_pos, active_mask=active_mask, compute_grad=True)
            gtot = ev['grho'] * (supp_mask * dpos)

            pen = rs_eff * float(cs_v @ cs_v)
            gcs = pb_s(gtot) + 2.0 * rs_eff * cs_v

            if n_d_cur:
                l1_d, gl1_d = smooth_l1_penalty(wd_v, eps=0.02)
                pen += l1_cur * l1_d
                gwd = pb_d(gtot) + l1_cur * gl1_d
            else:
                gwd = np.zeros(0)

            if n_p_cur:
                dq_v, dxyz_v, _ = pdict_cur.unpack(pk_v)
                l1_p, gl1_dq = smooth_l1_penalty(dq_v, eps=0.015)
                pen_xyz = 2e-6 * float(np.sum(dxyz_v ** 2))
                pen += l1_cur * l1_p + pen_xyz
                gpk = pb_p(gtot).reshape(-1, 5)
                gpk[:, 0] += l1_cur * gl1_dq
                gpk[:, 1:4] += 4e-6 * dxyz_v
                gpk = gpk.ravel()
            else:
                gpk = np.zeros(0)

            return ev['loss'] + pen, np.concatenate([gcs, gwd, gpk])
        return obj_joint

    if warm_state is not None:
        wdiv, all_peak_seeds, x_warm = warm_state
        pdict = ContinuousPeakDict(cgrid, all_peak_seeds, b_min=b_peak_min) if use_peaks else ContinuousPeakDict(cgrid, [])
        n_s = pf_s.nx
        n_d = wdiv.K if (use_divots and wdiv is not None) else 0
        n_p = pdict.n_params
        bounds = [(-2.5, 2.5)] * n_s + [(-4.0, 6.0)] * n_d + pdict.bounds()
        if len(x_warm) > 0:
            res_j = minimize(
                _make_joint_obj(wdiv, pdict, n_s, n_d, n_p, float(lam_l1)),
                x_warm, method='L-BFGS-B', jac=True, bounds=bounds,
                options={'maxiter': maxiter_joint, 'ftol': 1e-11, 'gtol': 1e-8},
            )
            x_sol = res_j.x
        else:
            res_j = None
            x_sol = np.zeros(0, dtype=np.float64)
        cs_cur = x_sol[:n_s]
        w_div_cur = x_sol[n_s:n_s + n_d] if n_d else np.zeros(0, dtype=np.float64)
        all_peak_params = x_sol[n_s + n_d:] if n_p else np.zeros(0, dtype=np.float64)
        tot_raw = pf_s.forward_and_pullback(cs_cur, compute_grad=False)[0]
        if n_d:
            tot_raw = tot_raw + wdiv.forward_and_pullback(w_div_cur, compute_grad=False)[0]
        if n_p:
            tot_raw = tot_raw + pdict.forward_and_pullback(all_peak_params, compute_grad=False)[0]
        rho_cur = supp_mask * softplus_pos(tot_raw, tau=0.012)[0]
    else:
        if not use_scale:
            cs_cur = np.zeros(0, dtype=np.float64)
            rho_s_only = np.asarray(base_3d, dtype=np.float64)
            dmap0_8 = precomputed_s0[2] if precomputed_s0 is not None else None
            dmap0_22 = precomputed_s0[3] if precomputed_s0 is not None else None
        elif precomputed_s0 is not None:
            cs_cur, rho_s_only, dmap0_8, dmap0_22 = precomputed_s0
            cs_cur = cs_cur.copy()
        else:
            def obj_s0(cs):
                rho_s, _, _, pb_s = pf_s.forward_and_pullback(cs, compute_grad=True)
                ev = prob.eval_rho(rho_s, active_mask=active_mask, compute_grad=True)
                return ev['loss'] + rs_eff * float(cs @ cs), pb_s(ev['grho']) + 2.0 * rs_eff * cs

            res_s0 = minimize(obj_s0, np.zeros(pf_s.nx), method='L-BFGS-B', jac=True,
                              bounds=[(-2.5, 2.5)] * pf_s.nx, options={'maxiter': maxiter_s0, 'ftol': 1e-10})
            cs_cur = res_s0.x.copy()
            rho_s_only, _, _, _ = pf_s.forward_and_pullback(cs_cur, compute_grad=False)
            dmap0_8 = dmap0_22 = None

        rho_cur = (supp_mask * softplus_pos(rho_s_only, tau=0.012)[0]) if n_passes_eff == 0 else rho_s_only

        all_peak_seeds = []
        all_peak_params = np.zeros(0, dtype=np.float64)
        wdiv = None
        w_div_cur = np.zeros(0, dtype=np.float64)
        res_j = None

        for p_idx in range(n_passes_eff):
            if p_idx == 0 and dmap0_8 is not None and dmap0_22 is not None:
                dmap_8, dmap_22 = dmap0_8, dmap0_22
            else:
                dmap_8 = prob.compute_work_diff_map(rho_cur, active_mask=active_mask, B_smooth=b_lo)
                dmap_22 = prob.compute_work_diff_map(rho_cur, active_mask=active_mask, B_smooth=b_hi)

            if use_divots and p_idx == 0:
                wdiv = WatershedDivotDict(
                    cgrid, dmap_8, open_mask_3d,
                    z_peak=max(2.5, z_thresh - 0.3), z_floor=1.6, rmax_A=rmax_eff,
                    max_divots=max_divots_eff, smooth_B=2.5,
                )
                w_div_cur = np.zeros(wdiv.K, dtype=np.float64)

            if use_peaks:
                new_seeds = pick_asu_difference_peaks(
                    cgrid, [(b_lo, dmap_8), (b_hi, dmap_22)], open_mask_3d,
                    z_thresh=(z_thresh if p_idx == 0 else z_thresh + 0.2),
                    max_peaks=max_peaks_eff, min_sep_A=min_sep_eff,
                )
                for ns in new_seeds:
                    ns['B_init'] = max(float(ns['B_init']), b_peak_min + 2.0)

                if all_peak_seeds and new_seeds:
                    prev_pdict = ContinuousPeakDict(cgrid, all_peak_seeds, b_min=b_peak_min)
                    _, dxyz_prev, _ = prev_pdict.unpack(all_peak_params)
                    prev_xyz = prev_pdict.xyz0 + dxyz_prev
                    sym_prev = cgrid.sym_expand_xyz(prev_xyz, cutoff_A=2.5)
                    if len(sym_prev):
                        tree_prev = cKDTree(sym_prev)
                        filtered_new = [ns for ns in new_seeds if tree_prev.query(ns['xyz0'])[0] >= 1.25]
                    else:
                        filtered_new = new_seeds
                else:
                    filtered_new = new_seeds

                if filtered_new:
                    new_pdict = ContinuousPeakDict(cgrid, filtered_new, b_min=b_peak_min)
                    all_peak_seeds.extend(filtered_new)
                    all_peak_params = np.concatenate([all_peak_params, new_pdict.init_params()])

            pdict = ContinuousPeakDict(cgrid, all_peak_seeds, b_min=b_peak_min) if use_peaks else ContinuousPeakDict(cgrid, [])
            n_s = pf_s.nx
            n_d = wdiv.K if (use_divots and wdiv is not None) else 0
            n_p = pdict.n_params

            x_init = np.concatenate([cs_cur, w_div_cur if n_d else np.zeros(0), all_peak_params if n_p else np.zeros(0)])
            bounds = (
                [(-2.5, 2.5)] * n_s +
                [(-4.0, 6.0)] * n_d +
                pdict.bounds()
            )

            if len(x_init) > 0:
                res_j = minimize(
                    _make_joint_obj(wdiv, pdict, n_s, n_d, n_p, float(lam_l1)),
                    x_init, method='L-BFGS-B', jac=True, bounds=bounds,
                    options={'maxiter': maxiter_joint, 'ftol': 1e-11, 'gtol': 1e-8},
                )
                cs_cur = res_j.x[:n_s]
                if n_d:
                    w_div_cur = res_j.x[n_s:n_s + n_d]
                if n_p:
                    all_peak_params = res_j.x[n_s + n_d:]
            else:
                res_j = None

            rho_s, _, _, _ = pf_s.forward_and_pullback(cs_cur, compute_grad=False)
            tot_raw = rho_s
            if n_d:
                rd, _ = wdiv.forward_and_pullback(w_div_cur, compute_grad=False)
                tot_raw = tot_raw + rd
            if n_p:
                rp, _ = pdict.forward_and_pullback(all_peak_params, compute_grad=False)
                tot_raw = tot_raw + rp
            rho_cur = supp_mask * softplus_pos(tot_raw, tau=0.012)[0]

    n_active_peaks = int(np.sum(np.abs(all_peak_params[0::5]) > 0.05)) if len(all_peak_params) else 0
    n_active_divs = int(np.sum(np.abs(w_div_cur) > 0.05)) if len(w_div_cur) else 0
    ev_end = prob.eval_rho(rho_cur, active_mask=active_mask, compute_grad=False)
    info = {
        'z_thresh': z_thresh,
        'lam_l1': lam_l1,
        'n_divots_tot': int(len(w_div_cur)),
        'n_divots_act': n_active_divs,
        'n_peaks_tot': int(len(all_peak_seeds)),
        'n_peaks_act': n_active_peaks,
        'r_train': ev_end['r_train'],
        'r_val': ev_end['r_val'],
    }
    if return_state:
        x_end = res_j.x.copy() if res_j is not None else cs_cur.copy()
        return rho_cur, info, (wdiv, list(all_peak_seeds), x_end)
    return rho_cur, info


def fit_adaptive_divot_peak_blind(prob: SolventAmplitudeProblem, cgrid: CrystalGridOps, base_3d, open_mask_3d,
                                  fitreso_s=12.0, use_divots=True, use_peaks=True, n_passes=1,
                                  hp_grid=None, gate_on_val: bool = True, min_rval_gain='auto'):
    """Select (base vs scale_only vs (z_thresh, lam_l1)) strictly on work_val (blind to FreeR_flag==0!)."""
    t0 = time.time()
    d_min_p = float(np.min(prob.d_arr)) if len(prob.d_arr) else 1.0
    if hp_grid is None:
        if d_min_p >= 1.35:
            hp_grid = ((4.4, 3.0e-4), (3.8, 1.2e-4), (3.3, 4.5e-5), (2.9, 1.8e-5))
        else:
            hp_grid = ((3.8, 1.2e-4), (3.3, 4.5e-5), (2.9, 1.8e-5))

    ev_base = prob.eval_rho(base_3d, active_mask=prob.train, compute_grad=False)
    rval_base = ev_base['r_val'] if math.isfinite(ev_base['r_val']) else ev_base['r_train']
    rval_fmodel_in = prob.eval_fmodel_ref_r(prob.val)
    rval_gate_ref = min(float(rval_base), float(rval_fmodel_in)) if math.isfinite(rval_fmodel_in) else float(rval_base)

    if min_rval_gain is None or str(min_rval_gain).strip().lower() == 'auto':
        n_val = max(int(np.sum(prob.val)), 1)
        min_gain = max(1e-3, 0.80 * rval_gate_ref / math.sqrt(2.0 * n_val))
    else:
        min_gain = float(min_rval_gain)

    # Fit S(x) once on prob.train; only retain S(x) inside joint divot+peak if S(x) itself generalizes on work_val
    fitreso_eff = max(float(fitreso_s), round(4.0 * d_min_p, 2)) if d_min_p > 1.50 else float(fitreso_s)
    b_lo = max(6.0, round(3.0 * (d_min_p ** 2), 1)) if d_min_p > 1.45 else 6.0
    b_hi = max(22.0, round(8.0 * (d_min_p ** 2), 1)) if d_min_p > 1.45 else 22.0
    rs_eff = 1e-4 * max(1.0, (d_min_p / 1.0) ** 2)
    pf_s0 = PeriodicFourierFields(cgrid, base_3d, fitreso_s=fitreso_eff, use_scale=True, use_warp=False)
    supp_mask = build_solvent_support_mask(base_3d, open_mask_3d)

    def obj_s0_tr(cs):
        rho_s, _, _, pb_s = pf_s0.forward_and_pullback(cs, compute_grad=True)
        ev = prob.eval_rho(rho_s, active_mask=prob.train, compute_grad=True)
        return ev['loss'] + rs_eff * float(cs @ cs), pb_s(ev['grho']) + 2.0 * rs_eff * cs

    res_s0 = minimize(obj_s0_tr, np.zeros(pf_s0.nx), method='L-BFGS-B', jac=True,
                      bounds=[(-2.5, 2.5)] * pf_s0.nx, options={'maxiter': 45, 'ftol': 1e-10})
    cs_s0 = res_s0.x.copy()
    rho_s0_raw, _, _, _ = pf_s0.forward_and_pullback(cs_s0, compute_grad=False)
    rho_s0_pos = supp_mask * softplus_pos(rho_s0_raw, tau=0.012)[0]
    ev_s0 = prob.eval_rho(rho_s0_pos, active_mask=prob.train, compute_grad=False)
    rval_scale = ev_s0['r_val'] if math.isfinite(ev_s0['r_val']) else ev_s0['r_train']

    use_scale_joint = (not (gate_on_val and np.any(prob.val))) or (rval_scale < rval_gate_ref - min_gain)
    seed_rho0 = rho_s0_raw if use_scale_joint else np.asarray(base_3d, dtype=np.float64)
    dmap0_8 = prob.compute_work_diff_map(seed_rho0, active_mask=prob.train, B_smooth=b_lo)
    dmap0_22 = prob.compute_work_diff_map(seed_rho0, active_mask=prob.train, B_smooth=b_hi)
    pre_s0 = (cs_s0 if use_scale_joint else np.zeros(0, dtype=np.float64), seed_rho0, dmap0_8, dmap0_22)

    best_hp = tuple(hp_grid[0])
    best_hp_rval = float('inf')
    best_hp_state = None
    val_trace = []

    for z_t, l1_w in hp_grid:
        _, inf, st_cur = fit_scale_divot_peak_given_hp(
            prob, cgrid, base_3d, open_mask_3d, active_mask=prob.train,
            fitreso_s=fitreso_s, use_divots=use_divots, use_peaks=use_peaks,
            z_thresh=z_t, lam_l1=l1_w, n_passes=n_passes, max_peaks_per_pass=120,
            maxiter_s0=45, maxiter_joint=85, precomputed_s0=pre_s0, return_state=True,
            use_scale=use_scale_joint,
        )
        val_trace.append((z_t, l1_w, inf['r_train'], inf['r_val'], inf['n_divots_act'], inf['n_peaks_act']))
        score_val = inf['r_val'] if math.isfinite(inf['r_val']) else inf['r_train']
        if score_val < best_hp_rval:
            best_hp_rval = score_val
            best_hp = (z_t, l1_w)
            best_hp_state = st_cur

    if gate_on_val and np.any(prob.val):
        best_stage = 'base'
        best_rval = float(rval_gate_ref)
        if rval_scale < best_rval - min_gain:
            best_stage = 'scale_only'
            best_rval = float(rval_scale)
        if best_hp_rval < best_rval - min_gain:
            best_stage = 'divot_peak'
            best_rval = float(best_hp_rval)
    else:
        best_stage = 'divot_peak'
        best_rval = float(best_hp_rval)

    if best_stage == 'base':
        rho_final = np.asarray(base_3d, dtype=np.float64).copy()
        ev_w = prob.eval_rho(rho_final, active_mask=prob.work, compute_grad=False)
        info_final = {
            'selected_stage': 'base',
            'z_thresh': None,
            'lam_l1': None,
            'n_divots_tot': 0,
            'n_divots_act': 0,
            'n_peaks_tot': 0,
            'n_peaks_act': 0,
            'r_train': ev_w['r_train'],
            'r_val': float(best_rval),
        }
    elif best_stage == 'scale_only':
        def obj_s0_wk(cs):
            rho_s, _, _, pb_s = pf_s0.forward_and_pullback(cs, compute_grad=True)
            ev = prob.eval_rho(rho_s, active_mask=prob.work, compute_grad=True)
            return ev['loss'] + rs_eff * float(cs @ cs), pb_s(ev['grho']) + 2.0 * rs_eff * cs

        res_wk = minimize(obj_s0_wk, cs_s0, method='L-BFGS-B', jac=True,
                          bounds=[(-2.5, 2.5)] * pf_s0.nx, options={'maxiter': 25, 'ftol': 1e-10})
        rho_final = supp_mask * softplus_pos(pf_s0.forward_and_pullback(res_wk.x, compute_grad=False)[0], tau=0.012)[0]
        ev_w = prob.eval_rho(rho_final, active_mask=prob.work, compute_grad=False)
        info_final = {
            'selected_stage': 'scale_only',
            'z_thresh': None,
            'lam_l1': None,
            'n_divots_tot': 0,
            'n_divots_act': 0,
            'n_peaks_tot': 0,
            'n_peaks_act': 0,
            'r_train': ev_w['r_train'],
            'r_val': float(best_rval),
        }
    else:
        rho_final, info_final = fit_scale_divot_peak_given_hp(
            prob, cgrid, base_3d, open_mask_3d, active_mask=prob.work,
            fitreso_s=fitreso_s, use_divots=use_divots, use_peaks=use_peaks,
            z_thresh=best_hp[0], lam_l1=best_hp[1], n_passes=n_passes, max_peaks_per_pass=120,
            maxiter_s0=30, maxiter_joint=35, warm_state=best_hp_state,
            use_scale=use_scale_joint,
        )
        info_final['selected_stage'] = 'divot_peak'
        info_final['r_val'] = float(best_rval)

    info_final['use_scale_joint'] = bool(use_scale_joint)
    info_final['r_val_fmodel_in'] = float(rval_fmodel_in)
    info_final['r_val_base'] = float(rval_base)
    info_final['r_val_scale'] = float(rval_scale)
    info_final['r_val_best_hp'] = float(best_hp_rval)
    info_final['min_rval_gain_used'] = float(min_gain)
    info_final['dt_s'] = round(time.time() - t0, 1)
    info_final['val_trace'] = val_trace
    return rho_final, info_final
