"""surfNturf core differentiable hydration field:
PMF + Lone Pairs + Langmuir + Multi-Step Tetrahedral OZ + Rosenfeld 0D Cavity Cap + Optional Crystal E-Field + IT92 Rendering.

Implements:
  1. Exact crystal-symmetry Asymmetric Unit (ASU) orbit indexing on 3D CCP4 grids (zero train/test leak).
  2. 10-class real chemical atom typing keeping explicit protein hydrogens + 3 explicit lone-pair virtual-site classes
     (`use_lp=True`: `LP_sp2` at 120 deg on C=O / aromatic N, `LP_neg` on COO-/oxyanions, `LP_sp3` at 109.47 deg on OH)
     and multi-altloc conformer completion (`reconnect`, `letters`, or `quantiles`).
  3. C2-smooth cardinal cubic B-spline radial potentials u_t(r) on [1.4, 5.6 A] pinned to 0 in bulk.
  4. Grand-canonical Langmuir saturation g_O = (1/alpha) * sigmoid(log(alpha/(1-alpha)) - U).
  5. Multi-step (`n_oz_steps=4`) radial Ornstein-Zernike + directional tetrahedral (109.47 deg, l=0,1,2 spherical
     harmonics on the 2.80 A H-bond shell) self-consistent network closure (`use_tet_oz=True`) with exact analytical
     empty-bulk asymptote subtraction (`zero_bulk_offset=True`, guaranteeing g_O(inf) == 1.000 identically).
  6. Parameter-free Rosenfeld 0D hard-sphere cavity packing cap (`use_pauli_cap=True`, r_cav = 1.30 A) enforcing at
     most ~1 water molecule (10 e) per 2.60 A oxygen exclusion sphere while leaving 1-HB/2-HB & bulk water untouched.
  7. Optional screened, Thole-damped AMBER-charge crystal electric-field polarization (`use_epol=False` by default)
     capturing |E|, |E|^2, and 3-body cross-atom cooperativity |sum_a E_a|^2 - sum_a |E_a|^2.
  8. Acceptor-only water-H orientation term (strictly zero net charge, conserving F000) + exact IT92 rendering.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import glob
import math
import os
import numpy as np
import gemmi
from scipy import fft as sp_fft
from scipy.spatial import cKDTree
from scipy.special import expit

RHO_BULK = 0.3344          # pure liquid water electron density (e / A^3)
R_OH = 0.9572              # water O-H bond length (A)

ATOM_TYPES = [
    'O_carb',   # 0: neutral carbonyl/amide acceptor O (backbone O, ASN OD1, GLN OE1)
    'O_neg',    # 1: anionic carboxylate/oxyanion O + compact coordinating ions (F-, Mg2+, Zn2+, Ca2+, Na+, Fe, Cu, Mn)
    'O_hyd',    # 2: hydroxyl O (SER OG, THR OG1, TYR OH) + unprotonated aromatic N acceptor
    'N_don',    # 3: amide/amine/guanidinium/proline heavy N + large halides/cations (Cl-, Br-, I-, K+)
    'H_pol',    # 4: neutral polar H-bond donor H (bonded to neutral N or O)
    'H_pos',    # 5: cationic polar H-bond donor H (N-term, LYS HZ*, ARG HE/HH*, HIP)
    'C_sp2',    # 6: planar sp2 carbon + tetrahedral P/As core
    'C_sp3',    # 7: aliphatic sp3 carbon (CA, CB, etc.)
    'H_apol',   # 8: nonpolar carbon-bonded H
    'S_sul',    # 9: sulfur / selenium (CYS SG, MET SD, MSE SE)
]
LP_TYPES = [
    'LP_sp2',   # 10: sp2 lone-pair virtual site (carbonyl C=O 120 deg in-plane lobes; aromatic N external bisector)
    'LP_neg',   # 11: anionic carboxylate / oxyanion lone-pair virtual site
    'LP_sp3',   # 12: sp3 tetrahedral out-of-plane lone-pair virtual site (hydroxyl / ether O)
]
EXTENDED_ATOM_TYPES = ATOM_TYPES + LP_TYPES
ACCEPTOR_TYPES = (0, 1, 2)
TWO_TYPES = ['pol', 'apol']

_COMPACT_IONS = {
    'F', 'LI', 'BE', 'NA', 'MG', 'AL', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE',
    'CO', 'NI', 'CU', 'ZN', 'GA', 'SR', 'Y', 'ZR', 'MO', 'CD', 'IN', 'SN', 'LA', 'HG', 'PB',
}
_LARGE_POLAR_IONS = {'CL', 'BR', 'I', 'K', 'RB', 'CS', 'BA', 'TL'}

R_MIN = 1.40
DR = 0.30
N_KNOTS = 14               # knots at 1.4, 1.7, ..., 5.3 A; knot 14 at 5.6 A is pinned to 0.0
R_CUT = R_MIN + DR * N_KNOTS   # 5.60 A

# Acceptor water-H polarization spline knots: 1.3, 1.55, ..., 3.3 A (9 knots; pinned to 0 at 3.55 A)
RH_MIN = 1.30
DRH = 0.25
N_HKNOTS = 9
RH_CUT = RH_MIN + DRH * N_HKNOTS  # 3.55 A

# Water-water Ornstein-Zernike radial shell radii (A) and Gaussian shell thickness (A)
OZ_RADII = np.array([1.25, 1.90, 2.80, 3.60, 4.50], dtype=np.float64)
OZ_SIGMA = 0.32

# Lone-pair, tetrahedral OZ, Rosenfeld 0D cavity cap, and crystal electric-field constants
D_LP = 0.65          # Lone-pair virtual-site offset from acceptor O/N nucleus (A)
R_TET = 2.80         # H-bond network shell radius for tetrahedral l=0,1,2 moments (A)
SIG_TET = 0.30       # Radial Gaussian width of 2.80 A H-bond shell window (A)
OZ_DAMP = 0.65       # Under-relaxation factor for multi-step self-consistent OZ closure
R_CAV = 1.30         # Rosenfeld 0D oxygen exclusion cavity radius (diameter 2.60 A < d_OO_min)
A_CAV = 0.10         # Smooth Fermi skin of 0D exclusion cavity indicator (A)
N_CAV_MAX = 1.05     # Hard-sphere 1-water ceiling per 2.60 A cavity
BETA_CAV = 10.0      # Sharpness of 0D cavity softplus cap
E_CUT = 7.50         # Crystal electric-field cutoff (A)
LAMBDA_DEBYE = 5.50  # Screened crystal dielectric decay length (A)
A_THOLE = 0.65       # Short-range Thole polarization damping length (A)
E_UNIT = 0.10        # Characteristic electric field unit (e / A^2) for O(1) conditioning

_DEFAULT_CLASS_Q = np.array(
    [-0.57, -0.80, -0.60, -0.45, +0.31, +0.38, +0.15, -0.05, +0.05, -0.20, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

SP2_CARBONS = {
    ('ASP', 'CG'), ('GLU', 'CD'), ('ASN', 'CG'), ('GLN', 'CD'), ('ARG', 'CZ'),
    ('PHE', 'CG'), ('PHE', 'CD1'), ('PHE', 'CD2'), ('PHE', 'CE1'), ('PHE', 'CE2'), ('PHE', 'CZ'),
    ('TYR', 'CG'), ('TYR', 'CD1'), ('TYR', 'CD2'), ('TYR', 'CE1'), ('TYR', 'CE2'), ('TYR', 'CZ'),
    ('TRP', 'CG'), ('TRP', 'CD1'), ('TRP', 'CD2'), ('TRP', 'CE2'), ('TRP', 'CE3'),
    ('TRP', 'CZ2'), ('TRP', 'CZ3'), ('TRP', 'CH2'),
    ('HIS', 'CG'), ('HIS', 'CD2'), ('HIS', 'CE1'),
    ('HID', 'CG'), ('HID', 'CD2'), ('HID', 'CE1'),
    ('HIE', 'CG'), ('HIE', 'CD2'), ('HIE', 'CE1'),
    ('HIP', 'CG'), ('HIP', 'CD2'), ('HIP', 'CE1'),
}

_BACKBONE_Q = {
    'N': -0.4157, 'H': 0.2719, 'HN': 0.2719, 'CA': 0.0321, 'HA': 0.0823,
    'HA2': 0.0823, 'HA3': 0.0823, 'C': 0.5973, 'O': -0.5679, 'OXT': -0.7850,
}

_SIDECHAIN_Q = {
    'GLY': {'CA': -0.0502},
    'ASP': {'CG': 0.6028, 'OD1': -0.8014, 'OD2': -0.8014},
    'GLU': {'CD': 0.6376, 'OE1': -0.8188, 'OE2': -0.8188},
    'LYS': {'NZ': -0.3800, 'HZ1': 0.4600, 'HZ2': 0.4600, 'HZ3': 0.4600},
    'ARG': {'NE': -0.5300, 'HE': 0.3600, 'CZ': 0.8100,
            'NH1': -0.8600, 'HH11': 0.5200, 'HH12': 0.5200,
            'NH2': -0.8600, 'HH21': 0.5200, 'HH22': 0.5200},
    'ASN': {'CG': 0.6730, 'OD1': -0.5930, 'ND2': -0.9200, 'HD21': 0.4200, 'HD22': 0.4200},
    'GLN': {'CD': 0.6986, 'OE1': -0.6086, 'NE2': -0.9400, 'HE21': 0.4250, 'HE22': 0.4250},
    'SER': {'CB': 0.1271, 'OG': -0.6546, 'HG': 0.4275},
    'THR': {'CB': 0.2159, 'OG1': -0.6761, 'HG1': 0.4102},
    'TYR': {'CZ': 0.1587, 'OH': -0.5579, 'HH': 0.3992},
    'TRP': {'NE1': -0.3412, 'HE1': 0.3412},
    'CYS': {'CB': 0.0186, 'SG': -0.3119, 'HG': 0.1933},
    'CYX': {'CB': 0.0081, 'SG': -0.1081},
    'MET': {'CG': 0.0369, 'SD': -0.2738, 'CE': -0.0131},
    'MSE': {'CG': 0.0369, 'SE': -0.2738, 'CE': -0.0131},
    'PRO': {'N': -0.2548, 'CD': -0.0174, 'CA': 0.0605, 'C': 0.5973, 'O': -0.5679},
}


def softplus(x, beta=4.0):
    """Overflow-safe C-infinity softplus: (1/beta) * log(1 + exp(beta * x))."""
    bx = beta * np.asarray(x, dtype=np.float64)
    out = np.where(bx > 30.0, np.asarray(x, dtype=np.float64), np.log1p(np.exp(np.clip(bx, -60.0, 30.0))) / beta)
    return float(out) if np.ndim(x) == 0 else out


def _unit_vec(v):
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-8 else None


class TetrahedralNetworkKernels:
    """Exact discrete `rfftn` kernels on `ASUGrid` for 2.80 A H-bond coordination & tetrahedral (109.47 deg) closure."""

    def __init__(self, grid: 'ASUGrid', r_hb: float = R_TET, sig_hb: float = SIG_TET):
        self.grid = grid
        self.shape = grid.shape
        self.vox_vol = float(grid.cell.volume) / float(grid.nvox)
        vox_mol = self.vox_vol * float(grid.n_bulk)

        df = grid.frac - np.rint(grid.frac)
        dr = df @ grid.O.T
        r = np.linalg.norm(dr, axis=1)
        inv_r = 1.0 / np.maximum(r, 1e-12)
        u = dr * inv_r[:, None]
        u[r < 1e-10] = 0.0

        w_sh = np.exp(-0.5 * ((r - float(r_hb)) / float(sig_hb)) ** 2)
        w_sh[r < 1e-10] = 0.0

        self.K0 = np.fft.rfftn(w_sh.reshape(self.shape), axes=(0, 1, 2))
        self.K0_conj = np.conj(self.K0)

        self.K1 = np.empty((3,) + self.K0.shape, dtype=np.complex128)
        for a in range(3):
            self.K1[a] = np.fft.rfftn((w_sh * u[:, a]).reshape(self.shape), axes=(0, 1, 2))
        self.K1_conj = np.conj(self.K1)

        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        s2 = math.sqrt(2.0)
        is2 = 1.0 / s2
        is6 = 1.0 / math.sqrt(6.0)
        phis = [
            is2 * (ux * ux - uy * uy),
            is6 * (2.0 * uz * uz - ux * ux - uy * uy),
            s2 * ux * uy,
            s2 * ux * uz,
            s2 * uy * uz,
        ]
        self.K2 = np.empty((5,) + self.K0.shape, dtype=np.complex128)
        for p in range(5):
            self.K2[p] = np.fft.rfftn((w_sh * phis[p]).reshape(self.shape), axes=(0, 1, 2))
        self.K2_conj = np.conj(self.K2)

        # Exact empty-bulk (g == 1, F_hbsol == 0) baseline constants for `zero_bulk_offset=True`
        xi_0 = math.log(2.0) / 4.0
        hb_0 = vox_mol * xi_0
        nc_0 = hb_0 * float(np.real(self.K0[0, 0, 0]))
        j1_0_sq = sum((hb_0 * float(np.real(self.K1[a, 0, 0, 0]))) ** 2 for a in range(3))
        q2_0_sq = sum((hb_0 * float(np.real(self.K2[p, 0, 0, 0]))) ** 2 for p in range(5))
        self.A_frust_0 = (q2_0_sq + (2.0 / 3.0) * j1_0_sq) / (1.0 + nc_0 * nc_0)
        self.sat_nc_0 = (nc_0 * nc_0) / (1.0 + 0.25 * nc_0 * nc_0)
        self.T_coop_0 = self.sat_nc_0 * math.exp(-2.0 * self.A_frust_0)


class ASUGrid:
    """Periodic 3D crystal grid with exact SpaceGroup Asymmetric Unit (ASU) reduction and Fourier kernels."""

    def __init__(self, cell, sg, full_shape, step=2):
        if cell is None or not cell.is_crystal():
            raise ValueError("ASUGrid requires a valid crystallographic UnitCell")
        self.cell = cell
        self.sg = sg if sg is not None else gemmi.SpaceGroup('P 1')
        self.full_shape = tuple(int(k) for k in full_shape)
        self.step = int(step)
        if self.step < 1 or any(k < self.step or (k % self.step) != 0 for k in self.full_shape):
            raise ValueError(f"Grid full_shape {self.full_shape} is not divisible by step={self.step}")
        self.shape = tuple(k // self.step for k in self.full_shape)
        nu, nv, nw = self.shape
        self.nvox = nu * nv * nw
        self.O = np.array(cell.orth.mat.tolist(), dtype=np.float64)
        self.F = np.array(cell.frac.mat.tolist(), dtype=np.float64)
        self.f_row_norms = np.linalg.norm(self.F, axis=1)
        self.ops = list(self.sg.operations())

        iu = np.arange(0, self.full_shape[0], self.step)[:nu] / self.full_shape[0]
        iv = np.arange(0, self.full_shape[1], self.step)[:nv] / self.full_shape[1]
        iw = np.arange(0, self.full_shape[2], self.step)[:nw] / self.full_shape[2]
        fr = np.stack(np.meshgrid(iu, iv, iw, indexing='ij'), axis=-1).reshape(-1, 3)
        self.frac = fr
        self.xyz = fr @ self.O.T

        grid_dims = np.array([nu, nv, nw], dtype=np.float64)
        ijk = np.stack(np.meshgrid(np.arange(nu), np.arange(nv), np.arange(nw), indexing='ij'), axis=-1).reshape(-1, 3)
        canon = np.arange(self.nvox, dtype=np.int64)
        for op in self.ops:
            R = np.array(op.rot, dtype=np.float64) / op.DEN
            t = np.array(op.tran, dtype=np.float64) / op.DEN
            mapped = ((ijk / grid_dims) @ R.T + t) * grid_dims
            if np.max(np.abs(mapped - np.rint(mapped))) >= 1e-6:
                raise ValueError(
                    f"Grid shape {self.shape} (full_shape={self.full_shape}, step={self.step}) is incompatible "
                    f"with symmetry operator {op.triplet()} of space group {self.sg.hm}"
                )
            mapped_i = np.mod(np.rint(mapped).astype(np.int64), np.array([nu, nv, nw], dtype=np.int64))
            lin = (mapped_i[:, 0] * nv + mapped_i[:, 1]) * nw + mapped_i[:, 2]
            canon = np.minimum(canon, lin)
        asu_reps, inv = np.unique(canon, return_inverse=True)
        self.asu_reps = asu_reps
        self.full_to_asu = inv.astype(np.int32)
        self.n_asu = len(asu_reps)
        self.asu_xyz = self.xyz[asu_reps]
        self.orbit_mult = np.bincount(self.full_to_asu, minlength=self.n_asu).astype(np.float64)

        h = np.fft.fftfreq(nu, 1.0 / nu)[:, None, None]
        k = np.fft.fftfreq(nv, 1.0 / nv)[None, :, None]
        l = np.fft.rfftfreq(nw, 1.0 / nw)[None, None, :]
        hkl = np.stack(np.broadcast_arrays(h, k, l), axis=-1)
        q_cart = hkl @ self.F
        inv_d2 = np.sum(q_cart ** 2, axis=-1)
        self.stol2 = inv_d2 / 4.0
        self.stol = np.sqrt(self.stol2)

        o_it = gemmi.Element('O').it92
        h_it = gemmi.Element('H').it92
        fO = np.full_like(self.stol2, o_it.c)
        for ai, bi in zip(o_it.a, o_it.b):
            fO = fO + ai * np.exp(-bi * self.stol2)
        fH = np.full_like(self.stol2, h_it.c)
        for ai, bi in zip(h_it.a, h_it.b):
            fH = fH + ai * np.exp(-bi * self.stol2)
        kr = 4.0 * np.pi * self.stol * R_OH
        j0 = np.where(kr > 1e-8, np.sin(kr) / np.where(kr > 1e-8, kr, 1.0), 1.0)
        self.f_water_0 = fO + 2.0 * fH * j0
        self.f_H_0 = fH.copy()
        self.f_H_0[0, 0, 0] = 0.0
        self.n_bulk = RHO_BULK / float(self.f_water_0[0, 0, 0])

        k_rad = 4.0 * np.pi * self.stol
        gauss_env = np.exp(-0.5 * (k_rad * OZ_SIGMA) ** 2)
        oz = []
        for Rq in OZ_RADII:
            x = k_rad * Rq
            sinc = np.where(x > 1e-8, np.sin(x) / np.where(x > 1e-8, x, 1.0), 1.0)
            oz.append(sinc * gauss_env)
        self.oz_kernels = np.stack(oz, axis=0)
        self.oz_dc = np.real(self.oz_kernels[:, 0, 0, 0]).astype(np.float64)

        self._tet_kernels = None
        self._cav_cache = None

    def get_tet_kernels(self) -> TetrahedralNetworkKernels:
        if self._tet_kernels is None:
            self._tet_kernels = TetrahedralNetworkKernels(self)
        return self._tet_kernels

    def get_cavity_kernel(self):
        """Return lazy-cached `(K_cav, K_cav_conj, n_cav_bulk)` for Rosenfeld 0D 1-water exclusion cap."""
        if self._cav_cache is None:
            df = self.frac - np.rint(self.frac)
            r = np.linalg.norm(df @ self.O.T, axis=1)
            w_cav = expit((R_CAV - r) / A_CAV).reshape(self.shape)
            K_cav = np.fft.rfftn(w_cav, axes=(0, 1, 2))
            K_cav_conj = np.conj(K_cav)
            vox_mol = (float(self.cell.volume) / float(self.nvox)) * float(self.n_bulk)
            n_cav_bulk = vox_mol * float(np.real(K_cav[0, 0, 0]))
            self._cav_cache = (K_cav, K_cav_conj, n_cav_bulk)
        return self._cav_cache

    def water_kernel(self, B_map):
        dw = np.exp(-float(B_map) * self.stol2)
        return self.f_water_0 * dw, self.f_H_0 * dw

    def asu_to_full_3d(self, arr_asu):
        return arr_asu[self.full_to_asu].reshape(self.shape)

    def full_3d_to_asu_grad(self, grad_3d):
        return np.bincount(self.full_to_asu, grad_3d.ravel(), minlength=self.n_asu)

    def full_3d_to_asu_mean(self, arr_3d):
        return np.bincount(self.full_to_asu, arr_3d.ravel(), minlength=self.n_asu) / self.orbit_mult

    def sym_expand(self, xyz, margin=None, cutoff_A=R_CUT, dedup_self: bool = False):
        """Expand Cartesian coordinates by all space-group and lattice images within `cutoff_A` of [0, 1)^3.

        When `dedup_self=True`, solute atoms whose symmetry self-images fall within 0.25 A of one another (e.g. within
        0.125 A of a 2-fold rotation axis) emit only one image per coincident self-orbit so PMF/OZ/Epol spline
        potentials are never multiplied 2x-6x. Pre-filtering stabilizers in O(n_ops * N) keeps high-symmetry groups fast.
        """
        if len(xyz) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
        m_vec = self.f_row_norms * (float(cutoff_A) + 0.05)
        if margin is not None:
            m_vec = np.maximum(m_vec, float(margin))
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

        f = np.asarray(xyz, dtype=np.float64) @ self.F.T
        out_p, out_i = [], []
        lo = -m_vec[None, :]
        hi = 1.0 + m_vec[None, :]
        op_mats = [(np.array(op.rot, dtype=np.float64) / op.DEN, np.array(op.tran, dtype=np.float64) / op.DEN)
                   for op in self.ops]
        spec_idx = None
        prev_spec_gs = []
        if dedup_self and len(op_mats) > 1:
            is_spec = np.zeros(len(f), dtype=bool)
            for Rc, tc in op_mats:
                if np.all(np.abs(Rc - np.eye(3)) < 1e-9) and np.all(np.abs(np.mod(tc + 0.5, 1.0) - 0.5) < 1e-9):
                    continue
                df0 = np.mod((f @ Rc.T + tc) - f + 0.5, 1.0) - 0.5
                is_spec |= (np.sum((df0 @ self.O.T) ** 2, axis=1) < 0.0625)
            if np.any(is_spec):
                spec_idx = np.nonzero(is_spec)[0]

        for R, t in op_mats:
            g = f @ R.T + t
            g -= np.floor(g)
            active_atom = None
            if spec_idx is not None:
                gs = g[spec_idx]
                if prev_spec_gs:
                    dup_s = np.zeros(len(spec_idx), dtype=bool)
                    for gp in prev_spec_gs:
                        df = np.mod(gs - gp + 0.5, 1.0) - 0.5
                        dup_s |= (np.sum((df @ self.O.T) ** 2, axis=1) < 0.0625)
                    if np.any(dup_s):
                        active_atom = np.ones(len(f), dtype=bool)
                        active_atom[spec_idx[dup_s]] = False
                prev_spec_gs.append(gs)
            for s in shifts:
                h = g + s
                m = np.all((h >= lo) & (h < hi), axis=1)
                if active_atom is not None:
                    m = m & active_atom
                if np.any(m):
                    out_p.append(h[m] @ self.O.T)
                    out_i.append(np.nonzero(m)[0])
        if not out_p:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
        return np.vstack(out_p), np.concatenate(out_i).astype(np.int32)


def _rec_chain_icode(rec):
    if len(rec) == 7:
        return str(rec[5]).strip(), str(rec[6]).strip()
    if len(rec) >= 8:
        return str(rec[6]).strip(), str(rec[7]).strip()
    return '', ''


def _gemmi_chain_icode(ch_idx: int, ch: gemmi.Chain, r: gemmi.Residue):
    ch_tag = ch.name.strip() or f'#{ch_idx}'
    ic_tag = r.seqid.icode.strip() if r.seqid.icode != '\0' else ''
    return ch_tag, ic_tag


def assign_atom_partial_charges(atom_list, bonded_h_by_idx, bonded_hv_by_idx):
    """Assign electrostatically balanced AMBER-scale partial atomic charges `q_a` (in `e`) to ordered real atoms."""
    n = len(atom_list)
    q = np.zeros(n, dtype=np.float64)

    # Identify doubly protonated HIS/HIP rings and C-terminal COO- (C bonded to both O and OXT)
    res_his_nd1_h = {}
    res_his_ne2_h = {}
    cterm_c_indices = set()
    for i, rec in enumerate(atom_list):
        rseq, res, name, elem, _ = rec[:5]
        ch_id, ic_id = _rec_chain_icode(rec)
        rkey = (ch_id, rseq, ic_id, res.strip().upper())
        an = name.strip().upper()
        el = elem.strip().upper()
        if rkey[3] in ('HIS', 'HID', 'HIE', 'HIP') and el == 'N':
            if an == 'ND1':
                res_his_nd1_h[rkey] = len(bonded_h_by_idx[i]) > 0
            elif an == 'NE2':
                res_his_ne2_h[rkey] = len(bonded_h_by_idx[i]) > 0
        if an == 'OXT' and bonded_hv_by_idx[i]:
            cterm_c_indices.add(bonded_hv_by_idx[i][0])

    for i, rec in enumerate(atom_list):
        rseq, res, name, elem, _xyz = rec[:5]
        ch_id, ic_id = _rec_chain_icode(rec)
        rkey = (ch_id, rseq, ic_id, res.strip().upper())
        rn = rkey[3]
        an = name.strip().upper()
        el = elem.strip().upper()
        if el == 'D':
            el = 'H'

        if rn in ('CL', 'BR', 'I', 'F'):
            q[i] = -1.0
            continue
        if rn in ('NA', 'K', 'LI', 'RB', 'CS'):
            q[i] = +1.0
            continue
        if rn in ('MG', 'ZN', 'CA', 'MN', 'FE', 'CU', 'CO', 'NI', 'SR', 'BA', 'CD'):
            q[i] = +2.0
            continue
        if rn in ('SO4', 'SUL'):
            q[i] = 1.40 if el == 'S' else -0.850
            continue
        if rn in ('PO4', 'PI'):
            q[i] = 1.36 if el == 'P' else -0.840
            continue

        # C-terminal carboxylate (C + O + OXT sums to -0.9706 vs neutral C + O = +0.0294 -> exact -1.0000 shift)
        if i in cterm_c_indices and an == 'C':
            q[i] = 0.5994
            continue
        if an == 'O' and any(j in cterm_c_indices for j in bonded_hv_by_idx[i]):
            q[i] = -0.7850
            continue
        if an == 'OXT':
            q[i] = -0.7850
            continue

        # Topology-based N-terminal cationic amine (NH3+ / PRO NH2+), independent of H atom naming (H1/H2/H3 vs H/H2/H3)
        _na_names = ('A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'ADE', 'CYT', 'GUA', 'URI', 'THY')
        if an == 'N' and len(bonded_h_by_idx[i]) >= 2 and rn not in _na_names:
            q[i] = -0.2548 if rn == 'PRO' else -0.3000
            continue
        if el in ('H', 'D') and bonded_hv_by_idx[i] and rn not in _na_names:
            p_idx = bonded_hv_by_idx[i][0]
            if atom_list[p_idx][2].strip().upper() == 'N' and len(bonded_h_by_idx[p_idx]) >= 2:
                n_nh = len(bonded_h_by_idx[p_idx])
                q[i] = (1.0000 / n_nh) if rn == 'PRO' else (1.1562 / n_nh)
                continue

        if rn in ('HIS', 'HID', 'HIE', 'HIP'):
            is_hip = (rn == 'HIP') or (res_his_nd1_h.get(rkey, False) and res_his_ne2_h.get(rkey, False))
            if an in _BACKBONE_Q:
                q[i] = _BACKBONE_Q[an]
            elif an == 'CG':
                q[i] = 0.1600 if is_hip else 0.0442
            elif an == 'CD2':
                q[i] = 0.1400 if is_hip else 0.0792
            elif an == 'CE1':
                q[i] = 0.2400 if is_hip else 0.1557
            elif an in ('ND1', 'NE2'):
                if is_hip:
                    q[i] = -0.3400
                else:
                    q[i] = -0.3811 if len(bonded_h_by_idx[i]) > 0 else -0.3629
            elif an in ('HD1', 'HE2'):
                q[i] = 0.5200 if is_hip else 0.3649
            elif el == 'C':
                q[i] = -0.05 * len(bonded_h_by_idx[i])
            elif el in ('H', 'D'):
                q[i] = 0.05
            continue

        if rn == 'CYS' and an in ('CB', 'SG') and (
            (an == 'SG' and len(bonded_h_by_idx[i]) == 0)
            or (an == 'CB' and any(
                atom_list[j][2].strip().upper() == 'SG' and len(bonded_h_by_idx[j]) == 0
                for j in bonded_hv_by_idx[i]
            ))
        ):
            q[i] = _SIDECHAIN_Q['CYX'][an]
            continue

        if rn in _SIDECHAIN_Q and an in _SIDECHAIN_Q[rn]:
            q[i] = _SIDECHAIN_Q[rn][an]
        elif an in _BACKBONE_Q:
            q[i] = _BACKBONE_Q[an]
        elif el == 'O':
            q[i] = -0.55 if len(bonded_h_by_idx[i]) == 0 else -0.62
        elif el == 'N':
            q[i] = -0.45 if len(bonded_h_by_idx[i]) > 0 else -0.52
        elif el == 'P':
            q[i] = 1.16
        elif el == 'C':
            # Balance attached nonpolar C-H (+0.05 each) so CH / CH2 / CH3 groups are strictly neutral
            q[i] = -0.05 * len(bonded_h_by_idx[i])
        elif el == 'H' and bonded_hv_by_idx[i]:
            pel = atom_list[bonded_hv_by_idx[i][0]][3].strip().upper()
            q[i] = 0.33 if pel in ('N', 'O') else 0.05
    return q


def _build_bonds_and_lone_pairs(ordered_atoms, xyz_real, typ_real, need_lp: bool, need_q: bool):
    """Build covalent neighbor lists on `ordered_atoms`, optional lone-pair virtual sites (10..12), and AMBER charges."""
    n_real = len(xyz_real)
    if n_real == 0:
        return xyz_real, typ_real, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)

    elems = ['H' if a[3].strip().upper() == 'D' else a[3].strip().upper() for a in ordered_atoms]
    is_h = np.array([el == 'H' for el in elems], dtype=bool)

    tree = cKDTree(xyz_real)
    nb_all = tree.query_ball_point(xyz_real, 1.95)
    bonded_h = [[] for _ in range(n_real)]
    bonded_hv = [[] for _ in range(n_real)]
    for i, nbrs in enumerate(nb_all):
        for j in nbrs:
            if j == i:
                continue
            d = float(np.linalg.norm(xyz_real[i] - xyz_real[j]))
            if is_h[i] and (not is_h[j]) and d <= 1.30:
                bonded_hv[i].append((d, j))
            elif (not is_h[i]) and is_h[j] and d <= 1.30:
                bonded_h[i].append((d, j))
            elif (not is_h[i]) and (not is_h[j]) and d <= 1.92:
                bonded_hv[i].append((d, j))
    bonded_h_idx = [[j for _, j in sorted(x)] for x in bonded_h]
    bonded_hv_idx = [[j for _, j in sorted(x)] for x in bonded_hv]

    q_real = assign_atom_partial_charges(ordered_atoms, bonded_h_idx, bonded_hv_idx) if need_q else None

    if not need_lp:
        q_out = q_real if need_q else np.zeros(0, dtype=np.float64)
        return xyz_real, typ_real, q_out, np.zeros(0, dtype=np.int32)

    lp_xyz = []
    lp_typ = []
    lp_parent = []
    c60, s60 = 0.5, 0.5 * math.sqrt(3.0)
    c_tet, s_tet = 1.0 / math.sqrt(3.0), math.sqrt(2.0 / 3.0)

    for i in range(n_real):
        t = int(typ_real[i])
        el = elems[i]
        xi = xyz_real[i]

        if t in (0, 1) and el == 'O' and bonded_hv_idx[i]:
            c_idx = bonded_hv_idx[i][0]
            cel = elems[c_idx]
            u = _unit_vec(xi - xyz_real[c_idx])
            if u is None:
                continue
            lp_class = 10 if t == 0 else 11
            c_nbrs = [k for k in (bonded_hv_idx[c_idx] + bonded_h_idx[c_idx]) if k != i]
            if cel == 'C' and len(c_nbrs) >= 2:
                p_vec = xyz_real[c_nbrs[0]] - xyz_real[c_nbrs[1]]
                v = _unit_vec(p_vec - float(np.dot(p_vec, u)) * u)
                if v is not None:
                    lp_xyz.append(xi + D_LP * (c60 * u + s60 * v))
                    lp_typ.append(lp_class)
                    lp_parent.append(i)
                    lp_xyz.append(xi + D_LP * (c60 * u - s60 * v))
                    lp_typ.append(lp_class)
                    lp_parent.append(i)
                    continue
            lp_xyz.append(xi + D_LP * u)
            lp_typ.append(lp_class)
            lp_parent.append(i)

        elif t == 2 and el == 'O':
            nbrs = bonded_hv_idx[i] + bonded_h_idx[i]
            if len(nbrs) >= 2:
                a_u = _unit_vec(xyz_real[nbrs[0]] - xi)
                b_u = _unit_vec(xyz_real[nbrs[1]] - xi)
                if a_u is not None and b_u is not None:
                    u = _unit_vec(-(a_u + b_u))
                    w = _unit_vec(np.cross(a_u, b_u))
                    if u is not None and w is not None:
                        lp_xyz.append(xi + D_LP * (c_tet * u + s_tet * w))
                        lp_typ.append(12)
                        lp_parent.append(i)
                        lp_xyz.append(xi + D_LP * (c_tet * u - s_tet * w))
                        lp_typ.append(12)
                        lp_parent.append(i)
            elif len(nbrs) == 1:
                c_idx = nbrs[0]
                a_u = _unit_vec(xyz_real[c_idx] - xi)
                gp_nbrs = [k for k in bonded_hv_idx[c_idx] if k != i]
                if a_u is not None and gp_nbrs:
                    gp_vec = xyz_real[gp_nbrs[0]] - xyz_real[c_idx]
                    p_u = _unit_vec(gp_vec - float(np.dot(gp_vec, a_u)) * a_u)
                    if p_u is not None:
                        b_u = (-1.0 / 3.0) * a_u + (2.0 * math.sqrt(2.0) / 3.0) * p_u
                        u = _unit_vec(-(a_u + b_u))
                        w = _unit_vec(np.cross(a_u, b_u))
                        if u is not None and w is not None:
                            lp_xyz.append(xi + D_LP * (c_tet * u + s_tet * w))
                            lp_typ.append(12)
                            lp_parent.append(i)
                            lp_xyz.append(xi + D_LP * (c_tet * u - s_tet * w))
                            lp_typ.append(12)
                            lp_parent.append(i)

        elif t == 2 and el == 'N' and len(bonded_h_idx[i]) == 0 and len(bonded_hv_idx[i]) >= 2:
            c1, c2 = bonded_hv_idx[i][0], bonded_hv_idx[i][1]
            u1 = _unit_vec(xi - xyz_real[c1])
            u2 = _unit_vec(xi - xyz_real[c2])
            if u1 is not None and u2 is not None:
                u = _unit_vec(u1 + u2)
                if u is not None:
                    lp_xyz.append(xi + D_LP * u)
                    lp_typ.append(10)
                    lp_parent.append(i)

    if lp_xyz:
        xyz_ext = np.vstack([xyz_real, np.asarray(lp_xyz, dtype=np.float64)])
        typ_ext = np.concatenate([typ_real, np.asarray(lp_typ, dtype=np.int32)])
        q_ext = np.concatenate([q_real, np.zeros(len(lp_xyz), dtype=np.float64)]) if need_q else np.zeros(0)
        parent_arr = np.asarray(lp_parent, dtype=np.int32)
    else:
        xyz_ext = xyz_real
        typ_ext = typ_real
        q_ext = q_real if need_q else np.zeros(0)
        parent_arr = np.zeros(0, dtype=np.int32)
    return xyz_ext, typ_ext, q_ext, parent_arr


def classify_conformer_atoms(atoms_list, two_type_heavy_only=False, use_lp=False, return_charges=False, return_lp_parents=False):
    """Classify [(res_seq, res_name, atom_name, elem, xyz[, ...]), ...] into ATOM_TYPES (+ optional LP_TYPES 10..12)."""
    if two_type_heavy_only:
        xyz, typ = [], []
        for rec in atoms_list:
            el_u = rec[3].strip().upper()
            if el_u in ('H', 'D'):
                continue
            xyz.append(rec[4])
            is_pol = (el_u in ('N', 'O') or el_u in _COMPACT_IONS or el_u in _LARGE_POLAR_IONS)
            typ.append(0 if is_pol else 1)
        xyz_a = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        typ_a = np.array(typ, dtype=np.int32)
        q_a = np.zeros(len(typ_a), dtype=np.float64)
        p_a = np.zeros(0, dtype=np.int32)
        if return_charges and return_lp_parents:
            return xyz_a, typ_a, q_a, p_a
        if return_charges:
            return xyz_a, typ_a, q_a
        if return_lp_parents:
            return xyz_a, typ_a, p_a
        return xyz_a, typ_a

    heavy_idx = [i for i, a in enumerate(atoms_list) if a[3].strip().upper() not in ('H', 'D')]
    hyd_idx = [i for i, a in enumerate(atoms_list) if a[3].strip().upper() in ('H', 'D')]
    heavy_elems = [atoms_list[i][3].strip().upper() for i in heavy_idx]
    h_xyz = np.array([atoms_list[i][4] for i in heavy_idx], dtype=np.float64).reshape(-1, 3)
    tree_h = cKDTree(h_xyz) if len(h_xyz) else None

    parent_of_hyd = {}
    nh_on_heavy = np.zeros(len(heavy_idx), dtype=np.int32)
    if hyd_idx and tree_h is not None:
        hy_xyz = np.array([atoms_list[i][4] for i in hyd_idx], dtype=np.float64).reshape(-1, 3)
        d, pi = tree_h.query(hy_xyz)
        for k, idx_a in enumerate(hyd_idx):
            if d[k] < 1.30:
                parent_of_hyd[idx_a] = pi[k]
                nh_on_heavy[pi[k]] += 1

    his_nd1 = []
    his_ne2 = []
    for hi_local, idx_a in enumerate(heavy_idx):
        rec = atoms_list[idx_a]
        rseq, rname, aname, _, pos = rec[:5]
        ch_id, ic_id = _rec_chain_icode(rec)
        if rname in ('HIS', 'HID', 'HIE', 'HIP') and nh_on_heavy[hi_local] >= 1:
            if aname == 'ND1':
                his_nd1.append((hi_local, (ch_id, rseq, ic_id), pos))
            elif aname == 'NE2':
                his_ne2.append((hi_local, (ch_id, rseq, ic_id), pos))
    hip_heavy_locals = set()
    for hi1, rkey1, p1 in his_nd1:
        for hi2, rkey2, p2 in his_ne2:
            if rkey1 == rkey2:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dz = p1[2] - p2[2]
                if dx * dx + dy * dy + dz * dz < 2.65 * 2.65:
                    hip_heavy_locals.add(hi1)
                    hip_heavy_locals.add(hi2)

    has_any_h = len(hyd_idx) > 0
    heavy_is_cationic = np.zeros(len(heavy_idx), dtype=bool)
    xyz_out, typ_out = [], []

    for hi_local, idx_a in enumerate(heavy_idx):
        _, rname, aname, el_raw, pos = atoms_list[idx_a][:5]
        el = el_raw.strip().upper()
        nh = nh_on_heavy[hi_local]
        if el == 'N':
            if ((aname == 'N' and nh >= 2)
                    or (rname == 'LYS' and aname == 'NZ')
                    or (rname == 'ARG' and aname in ('NE', 'NH1', 'NH2'))
                    or (hi_local in hip_heavy_locals)
                    or (not has_any_h and rname == 'HIP' and aname in ('ND1', 'NE2'))):
                heavy_is_cationic[hi_local] = True

        if el == 'O':
            if (aname in ('OXT', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P')
                    or (rname == 'ASP' and aname in ('OD1', 'OD2'))
                    or (rname == 'GLU' and aname in ('OE1', 'OE2'))
                    or (rname in ('SO4', 'PO4') and aname.startswith('O'))):
                t = 1
            elif (nh >= 1
                  or (rname == 'SER' and aname == 'OG')
                  or (rname == 'THR' and aname == 'OG1')
                  or (rname == 'TYR' and aname == 'OH')):
                t = 2
            else:
                # Ether / glycosidic / ribose O4' / phosphodiester O3',O5' (2 covalent non-metal heavy bonds) -> sp3 O_hyd (2)
                nbrs = tree_h.query_ball_point(h_xyz[hi_local], 1.82) if tree_h is not None else ()
                n_cov = sum(1 for j in nbrs if j != hi_local and heavy_elems[j] in ('C', 'P', 'S', 'SI', 'B', 'AS', 'SE'))
                t = 2 if n_cov >= 2 else 0
        elif el == 'N':
            if has_any_h:
                t = 2 if (nh == 0 and rname in ('HIS', 'HID', 'HIE', 'HIP') and aname in ('ND1', 'NE2')) else 3
            else:
                t = 2 if (rname in ('HIS', 'HID') and aname == 'NE2') or (rname == 'HIE' and aname == 'ND1') else 3
        elif el in ('S', 'SE'):
            t = 9
        elif el in _COMPACT_IONS:
            t = 1
        elif el in _LARGE_POLAR_IONS:
            t = 3
        elif el in ('P', 'AS'):
            t = 6
        else:
            t = 6 if (aname == 'C' or (rname, aname) in SP2_CARBONS) else 7
        xyz_out.append(pos)
        typ_out.append(t)

    for idx_a in hyd_idx:
        pos = atoms_list[idx_a][4]
        p_loc = parent_of_hyd.get(idx_a, None)
        if p_loc is not None:
            p_el = heavy_elems[p_loc]
            if p_el in ('N', 'O'):
                t = 5 if heavy_is_cationic[p_loc] else 4
            else:
                t = 8
        else:
            t = 8
        xyz_out.append(pos)
        typ_out.append(t)

    xyz_real = np.array(xyz_out, dtype=np.float64).reshape(-1, 3)
    typ_real = np.array(typ_out, dtype=np.int32)

    if not use_lp and not return_charges and not return_lp_parents:
        return xyz_real, typ_real

    ordered_atoms = [atoms_list[i] for i in (heavy_idx + hyd_idx)]
    xyz_ext, typ_ext, q_ext, lp_parents = _build_bonds_and_lone_pairs(
        ordered_atoms, xyz_real, typ_real, need_lp=bool(use_lp), need_q=bool(return_charges)
    )
    if return_charges and return_lp_parents:
        return xyz_ext, typ_ext, q_ext, lp_parents
    if return_charges:
        return xyz_ext, typ_ext, q_ext
    if return_lp_parents:
        return xyz_ext, typ_ext, lp_parents
    return xyz_ext, typ_ext


def _validate_ensemble_mode(mode: str) -> str:
    m = str(mode).strip()
    if m in ('auto', 'letters', 'quantiles', 'reconnect', 'geometry'):
        return m
    for pfx in ('reconnect_flex', 'geometry_flex'):
        if m.startswith(pfx) and (m[len(pfx):] == '' or m[len(pfx):].isdigit()):
            return m
    raise ValueError(
        f"Unknown ensemble mode {mode!r}; choose 'auto', 'letters', 'reconnect', or 'quantiles'."
    )


def load_ensemble(path_or_dir, nconf=32, two_type_heavy_only=False, mode='auto',
                  return_weights=False, use_lp=False, return_charges=False, complete_h=True):
    """Load either a directory of member PDBs (m*.pdb) or a multi-altloc PDB/mmCIF."""
    from .hydrogens import complete_conformer_hydrogens
    from .reconnect import reconnect_pdb_to_conformers
    mode = _validate_ensemble_mode(mode)
    use_lp_eff = bool(use_lp) and (not two_type_heavy_only)

    def _finish(raw_atom_lists, wts):
        members = []
        charges = [] if return_charges else None
        for al in raw_atom_lists:
            if complete_h and not two_type_heavy_only:
                al = complete_conformer_hydrogens(al)
            if return_charges:
                x, t, q = classify_conformer_atoms(
                    al, two_type_heavy_only=two_type_heavy_only, use_lp=use_lp_eff, return_charges=True
                )
                members.append((x, t))
                charges.append(q)
            else:
                members.append(
                    classify_conformer_atoms(al, two_type_heavy_only=two_type_heavy_only, use_lp=use_lp_eff)
                )
        if return_charges and return_weights:
            return members, wts, charges
        if return_charges:
            return members, charges
        if return_weights:
            return members, wts
        return members

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
            for ch_idx, ch in enumerate(st[0]):
                for r in ch:
                    if r.name in ('HOH', 'WAT', 'DOD'):
                        continue
                    ch_tag, ic_tag = _gemmi_chain_icode(ch_idx, ch, r)
                    for a in r:
                        if a.occ <= 0:
                            continue
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), ch_tag, ic_tag))
            raw_lists.append(al)
        wts = np.full(len(raw_lists), 1.0 / max(len(raw_lists), 1), dtype=np.float64)
        return _finish(raw_lists, wts)

    st = gemmi.read_structure(path_or_dir)
    all_alts = set()
    any_h_split_on_blank_hv = False
    n_pos_hv = 0
    split_res_sets = []
    split_q_first = []
    for ch in st[0]:
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            r_blank_hv = False
            r_alt_hv = False
            r_alt_h_letters = set()
            r_hv_by_alt = {}
            for a in r:
                is_h = a.element.name in ('H', 'D')
                alt_s = a.altloc.strip() if a.altloc != '\0' else ''
                if not is_h and a.occ > 0:
                    n_pos_hv += 1
                    if alt_s:
                        all_alts.add(alt_s)
                        r_alt_hv = True
                        r_hv_by_alt.setdefault(alt_s, []).append(float(a.occ))
                    else:
                        r_blank_hv = True
                elif is_h and a.occ > 0 and alt_s:
                    r_alt_h_letters.add(alt_s)
                elif not is_h and alt_s:
                    r_alt_hv = True
            if r_hv_by_alt:
                split_res_sets.append(set(r_hv_by_alt.keys()))
                l0 = sorted(r_hv_by_alt.keys())[0]
                tot_q = sum(float(np.mean(v)) for v in r_hv_by_alt.values())
                if tot_q > 1e-6:
                    split_q_first.append((l0, float(np.mean(r_hv_by_alt[l0])) / tot_q))
            if r_blank_hv and not r_alt_hv and r_alt_h_letters:
                any_h_split_on_blank_hv = True
                all_alts.update(r_alt_h_letters)

    if n_pos_hv == 0:
        raise ValueError(f"{path_or_dir}: no non-water heavy atoms with occupancy > 0 found")

    if not all_alts and not any_h_split_on_blank_hv:
        al = []
        for ch_idx, ch in enumerate(st[0]):
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                ch_tag, ic_tag = _gemmi_chain_icode(ch_idx, ch, r)
                for a in r:
                    if a.occ > 0:
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), ch_tag, ic_tag))
        return _finish([al], np.array([1.0], dtype=np.float64))

    first_alt = sorted(all_alts)[0] if all_alts else ''
    q_first_vals = [qf for (l0, qf) in split_q_first if l0 == first_alt]
    letters_auto_safe = (
        all_alts.issubset({'A', 'B'})
        and all(s == all_alts for s in split_res_sets)
        and (len(q_first_vals) <= 1 or (max(q_first_vals) - min(q_first_vals)) <= 0.08)
    )
    use_letters = (mode == 'letters') or (mode == 'auto' and letters_auto_safe)
    if use_letters:
        letters = sorted(all_alts)
        mem_lists = {L: [] for L in letters}
        occ_lists = {L: [] for L in letters}
        for ch_idx, ch in enumerate(st[0]):
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                ch_tag, ic_tag = _gemmi_chain_icode(ch_idx, ch, r)
                blank_atoms = []
                blank_hv = False
                any_alt_hv = False
                by_alt_raw = {}
                for a in r:
                    rec = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), ch_tag, ic_tag)
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
            [(a['seqnum'], a['resname'], a['name'], a['elem'], list(a['pos']),
              str(a.get('chain', '')).strip(), str(a.get('icode', '')).strip()) for a in c]
            for c in confs
        ]
        wts = np.full(len(raw_lists), 1.0 / len(raw_lists), dtype=np.float64)
        return _finish(raw_lists, wts)

    quantiles = (np.arange(nconf, dtype=np.float64) + 0.5) / nconf
    members_raw = [[] for _ in range(nconf)]
    for ch_idx, ch in enumerate(st[0]):
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            ch_tag, ic_tag = _gemmi_chain_icode(ch_idx, ch, r)
            blank_atoms = []
            blank_hv = False
            any_alt_hv = False
            by_alt_raw = {}
            for a in r:
                rec7 = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), ch_tag, ic_tag)
                if a.altloc == '\0' or not a.altloc.strip():
                    if a.occ > 0:
                        blank_atoms.append(rec7)
                        if a.element.name not in ('H', 'D'):
                            blank_hv = True
                else:
                    if a.element.name not in ('H', 'D'):
                        any_alt_hv = True
                    by_alt_raw.setdefault(a.altloc.strip(), []).append((rec7, a.occ))
            by_alt = {}
            hv_weights = {}
            for L, recs in by_alt_raw.items():
                hv_occs = [occ for (r7, occ) in recs if r7[3] not in ('H', 'D') and occ > 0]
                if hv_occs:
                    by_alt[L] = [r7 for (r7, occ) in recs if occ > 0 or r7[3] in ('H', 'D')]
                    hv_weights[L] = float(np.mean(hv_occs))
                elif blank_hv and not any_alt_hv:
                    h_occs = [occ for (_, occ) in recs if occ > 0]
                    if h_occs:
                        by_alt[L] = [r7 for (r7, occ) in recs if occ > 0]
                        hv_weights[L] = float(np.mean(h_occs))
            if not by_alt:
                for m in range(nconf):
                    members_raw[m].extend(blank_atoms)
                continue
            letters = sorted(by_alt.keys())
            weights = np.array([hv_weights[L] for L in letters], dtype=np.float64)
            weights = weights / weights.sum()
            cum = np.cumsum(weights)
            for m, q in enumerate(quantiles):
                members_raw[m].extend(blank_atoms)
                k = min(int(np.searchsorted(cum, q, side='right')), len(letters) - 1)
                members_raw[m].extend(by_alt[letters[k]])
    wts = np.full(nconf, 1.0 / nconf, dtype=np.float64)
    return _finish(members_raw, wts)


def _bspline_weights(d, r_min, dr, n_knots):
    """Cardinal cubic B-spline basis weights for distances d on knots r_min + k*dr (k=0..n_knots-1), pinned 0 at n_knots."""
    s = np.clip((d - r_min) / dr, 0.0, float(n_knots))
    i = np.minimum(np.floor(s).astype(np.int32), n_knots - 1)
    f = s - i
    omf = 1.0 - f
    b0 = (omf ** 3) * (1.0 / 6.0)
    b1 = (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) * (1.0 / 6.0)
    b2 = (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) * (1.0 / 6.0)
    b3 = (f ** 3) * (1.0 / 6.0)
    k0 = np.maximum(i - 1, 0)
    k1 = i
    k2 = i + 1
    k3 = i + 2
    return (k0, b0), (k1, b1), (k2, b2), (k3, b3)


def _compute_epol_additive_terms(grid: ASUGrid, tree_grid: cKDTree, xyz_real, typ_real, q_real):
    """Evaluate additive electric-field vector `(ex, ey, ez)`, `e_self_sq`, and `log_sgate` on ASU for a real-atom subset."""
    n_asu = grid.n_asu
    ex = np.zeros(n_asu, dtype=np.float64)
    ey = np.zeros(n_asu, dtype=np.float64)
    ez = np.zeros(n_asu, dtype=np.float64)
    e_self = np.zeros(n_asu, dtype=np.float64)
    lgate = np.zeros(n_asu, dtype=np.float64)
    if len(xyz_real) == 0:
        return ex, ey, ez, e_self, lgate

    SE, Sei = grid.sym_expand(xyz_real, cutoff_A=E_CUT, dedup_self=True)
    if len(SE) == 0:
        return ex, ey, ez, e_self, lgate
    St_e = typ_real[Sei]
    Sq_e = q_real[Sei]
    nb_e = tree_grid.query_ball_point(SE, E_CUT)
    le = np.array([len(x) for x in nb_e], dtype=np.int32)
    if le.sum() == 0:
        return ex, ey, ez, e_self, lgate

    pts = grid.asu_xyz
    ae = np.repeat(np.arange(len(SE), dtype=np.int32), le)
    pe = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb_e if len(x)])
    rij = pts[pe] - SE[ae]
    d_e = np.maximum(np.linalg.norm(rij, axis=1), 1e-6)
    t_e = St_e[ae]
    q_e = Sq_e[ae]

    is_he = (t_e == 4) | (t_e == 5) | (t_e == 8)
    r_st = np.where(is_he, 1.35, 2.05)
    cl_e = d_e < (r_st + 0.75)
    if np.any(cl_e):
        lg = -np.logaddexp(0.0, -(d_e[cl_e] - r_st[cl_e]) / 0.14)
        lgate = np.bincount(pe[cl_e], weights=lg, minlength=n_asu)

    q_nz = np.abs(q_e) > 1e-4
    if np.any(q_nz):
        pe_q, d_q, rij_q, qq = pe[q_nz], d_e[q_nz], rij[q_nz], q_e[q_nz]
        u_th = d_q / A_THOLE
        f_thole = 1.0 - (1.0 + u_th + 0.5 * u_th * u_th) * np.exp(-u_th)
        eyuk = np.exp(-d_q / LAMBDA_DEBYE) * (1.0 / (d_q * d_q) + 1.0 / (LAMBDA_DEBYE * d_q))
        e_sc = (qq * f_thole * eyuk) / (E_UNIT * d_q)
        ex = np.bincount(pe_q, weights=e_sc * rij_q[:, 0], minlength=n_asu)
        ey = np.bincount(pe_q, weights=e_sc * rij_q[:, 1], minlength=n_asu)
        ez = np.bincount(pe_q, weights=e_sc * rij_q[:, 2], minlength=n_asu)
        e_self = np.bincount(pe_q, weights=(e_sc * d_q) ** 2, minlength=n_asu)

    return ex, ey, ez, e_self, lgate


def _epol_features_from_additive(ex, ey, ez, e_self, lgate):
    """Convert `(ex, ey, ez, e_self, lgate)` into `(N_asu, 3)` `[|E|, |E|^2, E_cross_coop]`."""
    s_gate = np.exp(lgate)
    g2 = s_gate * s_gate
    raw_sq = ex * ex + ey * ey + ez * ez
    etot_sq = g2 * raw_sq
    ecross = g2 * (raw_sq - e_self)
    f_mag = np.sqrt(etot_sq + 0.04) - 0.20
    f_sq = etot_sq / (1.0 + 0.08 * etot_sq)
    f_coop = np.tanh(0.25 * ecross)
    return np.column_stack([f_mag, f_sq, f_coop])


def build_asu_features(grid: ASUGrid, members, n_types=None, include_hpol=True,
                       include_tet_oz=False, include_epol=False, charges=None,
                       member_weights=None, return_extra=False, max_workers=8):
    """Precompute per-member cubic B-spline tensors (and optional TetOZ / Epol features) on the ASU grid."""
    M = len(members)
    if n_types is None:
        max_t = max((int(np.max(t)) for _, t in members if len(t) > 0), default=9)
        n_types = 2 if max_t <= 1 else (13 if max_t >= 10 else 10)
    n_asu = grid.n_asu
    n_feat = int(n_types) * N_KNOTS
    two_type = (int(n_types) == 2)
    Phi_O = np.zeros((M, n_asu, n_feat), dtype=np.float32)
    Phi_H = np.zeros((M, n_asu, len(ACCEPTOR_TYPES) * N_HKNOTS), dtype=np.float32) if include_hpol else None
    steric_H = np.ones((M, n_asu), dtype=np.float32) if include_hpol else None
    E_feats = np.zeros((M, n_asu, 3), dtype=np.float32) if include_epol else None
    sol_hb_m = np.zeros((M, n_asu), dtype=np.float64) if include_tet_oz else None

    pts = grid.asu_xyz
    tree_grid = cKDTree(pts)
    vox_vol = float(grid.cell.volume) / float(grid.nvox)

    def _work_m(m):
        xyz, typ = members[m]
        if len(xyz) == 0:
            return
        if int(n_types) == 10 and np.any(typ >= 10):
            keep = typ < 10
            xyz_use, typ_use = xyz[keep], typ[keep]
        else:
            xyz_use, typ_use = xyz, typ

        SX, Si = grid.sym_expand(xyz_use, cutoff_A=R_CUT, dedup_self=True)
        if len(SX) > 0:
            St = typ_use[Si]
            nb = tree_grid.query_ball_point(SX, R_CUT)
            lens = np.array([len(x) for x in nb], dtype=np.int32)
            if lens.sum() > 0:
                ai = np.repeat(np.arange(len(SX), dtype=np.int32), lens)
                pi = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb if len(x)])
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
                            nhf = len(ACCEPTOR_TYPES) * N_HKNOTS
                            buf_h = np.zeros(n_asu * nhf, dtype=np.float64)
                            for kk, bw in _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS):
                                valid = kk < N_HKNOTS
                                if np.any(valid):
                                    flat_col = ti_a[valid] * N_HKNOTS + kk[valid]
                                    flat_idx = pi_a[valid].astype(np.int64) * nhf + flat_col
                                    buf_h += np.bincount(flat_idx, weights=bw[valid], minlength=n_asu * nhf)
                            Phi_H[m] = buf_h.reshape(n_asu, nhf).astype(np.float32)

        real_atoms = np.ones(len(typ), dtype=bool) if two_type else (typ < 10)
        xyz_r = xyz[real_atoms]
        typ_r = typ[real_atoms]

        if include_tet_oz and len(xyz_r) > 0:
            pol_hv = (typ_r == 0) if two_type else ((typ_r == 0) | (typ_r == 1) | (typ_r == 2) | (typ_r == 3))
            if np.any(pol_hv):
                SP, _ = grid.sym_expand(xyz_r[pol_hv], cutoff_A=1.8, dedup_self=True)
                if len(SP) > 0:
                    nb_p = tree_grid.query_ball_point(SP, 1.5)
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
            ex, ey, ez, es, lg = _compute_epol_additive_terms(grid, tree_grid, xyz_r, typ_r, q_r)
            E_feats[m] = _epol_features_from_additive(ex, ey, ez, es, lg).astype(np.float32)

    if M == 1 or max_workers <= 1:
        for m in range(M):
            _work_m(m)
    else:
        with ThreadPoolExecutor(max_workers=min(int(max_workers), M)) as ex_pool:
            list(ex_pool.map(_work_m, range(M)))

    if not return_extra and not include_tet_oz and not include_epol:
        return Phi_O, Phi_H, steric_H

    mw = np.asarray(member_weights if member_weights is not None else np.full(M, 1.0 / max(M, 1)), dtype=np.float64)
    mw = mw / mw.sum()
    F_hbsol = np.fft.rfftn(grid.asu_to_full_3d(mw @ sol_hb_m), axes=(0, 1, 2)) if include_tet_oz else None
    extra = {'F_hbsol': F_hbsol, 'E_feats': E_feats}
    return (Phi_O, Phi_H, steric_H, extra) if (return_extra or include_tet_oz or include_epol) else (Phi_O, Phi_H, steric_H)


def _partition_common_and_diff_members(members, charges=None):
    """Split `members = [(xyz, typ), ...]` into `(com_xyz, com_typ, com_q), diff_list` for ~10x M > 1 speedup."""
    M = len(members)
    has_q = charges is not None
    if M <= 1:
        q_list = [np.asarray(charges[0], dtype=np.float64)] if has_q else [None]
        return (np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)), [
            (members[m][0], members[m][1], q_list[m]) for m in range(M)
        ]

    key_lists = []
    common_counts = None
    for m, (xyz, typ) in enumerate(members):
        q = np.rint(xyz * 10000.0).astype(np.int64)
        if has_q and charges[m] is not None:
            qq = np.rint(np.asarray(charges[m], dtype=np.float64) * 10000.0).astype(np.int64)
            ks = [(int(q[i, 0]), int(q[i, 1]), int(q[i, 2]), int(typ[i]), int(qq[i])) for i in range(len(typ))]
        else:
            ks = [(int(q[i, 0]), int(q[i, 1]), int(q[i, 2]), int(typ[i]), 0) for i in range(len(typ))]
        key_lists.append(ks)
        cnt = Counter(ks)
        common_counts = cnt if common_counts is None else (common_counts & cnt)

    if not common_counts:
        return (np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)), [
            (members[m][0], members[m][1], (np.asarray(charges[m], dtype=np.float64) if has_q and charges[m] is not None else None))
            for m in range(M)
        ]

    rem0 = Counter(common_counts)
    com_xyz, com_typ, com_q = [], [], []
    xyz0, typ0 = members[0]
    q0 = np.asarray(charges[0], dtype=np.float64) if (has_q and charges[0] is not None) else None
    for i, k in enumerate(key_lists[0]):
        if rem0[k] > 0:
            rem0[k] -= 1
            com_xyz.append(xyz0[i])
            com_typ.append(typ0[i])
            if q0 is not None:
                com_q.append(q0[i])

    diff_members = []
    for m, ((xyz, typ), ks) in enumerate(zip(members, key_lists)):
        qm = np.asarray(charges[m], dtype=np.float64) if (has_q and charges[m] is not None) else None
        rem = Counter(common_counts)
        d_idx = []
        for i, k in enumerate(ks):
            if rem[k] > 0:
                rem[k] -= 1
            else:
                d_idx.append(i)
        if d_idx:
            idx_arr = np.asarray(d_idx, dtype=np.int32)
            diff_members.append((xyz[idx_arr], typ[idx_arr], (qm[idx_arr] if qm is not None else None)))
        else:
            diff_members.append((np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32),
                                 (np.zeros(0, dtype=np.float64) if qm is not None else None)))

    return (
        np.asarray(com_xyz, dtype=np.float64),
        np.asarray(com_typ, dtype=np.int32),
        np.asarray(com_q, dtype=np.float64) if com_q else np.zeros(0, dtype=np.float64),
    ), diff_members


class DatasetBundle:
    """Holds one MD dataset's ASU target, pre-flattened float64 BLAS feature matrices, and rendering kernel."""

    def __init__(self, name, grid: ASUGrid, rho_full, B_map, Phi_O, Phi_H=None, steric_H=None,
                 mask_asu=None, member_weights=None, F_hbsol=None, E_feats=None):
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
        self.Phi_O_flat = np.ascontiguousarray(Phi_O, dtype=np.float64).reshape(self.M * grid.n_asu, -1)
        self.Phi_H_flat = (np.ascontiguousarray(Phi_H, dtype=np.float64).reshape(self.M * grid.n_asu, -1)
                           if Phi_H is not None else None)
        self.steric_H_64 = (np.ascontiguousarray(steric_H, dtype=np.float64)
                            if steric_H is not None else None)
        self.F_hbsol = F_hbsol
        self.E_feats_2d = (np.ascontiguousarray(E_feats, dtype=np.float64).reshape(self.M * grid.n_asu, 3)
                           if E_feats is not None else None)
        self.g_full = (rho_full / RHO_BULK).astype(np.float64) if rho_full is not None else None
        self.g_asu = grid.full_3d_to_asu_mean(self.g_full) if self.g_full is not None else None
        self.weights_asu = grid.orbit_mult.copy()
        if mask_asu is not None:
            self.weights_asu = self.weights_asu * mask_asu.astype(np.float64)
        self.w_sum = float(self.weights_asu.sum())


class WaterFieldSpec:
    """Configuration & parameter packing for the differentiable hydration field."""

    def __init__(self, n_types=None, use_langmuir=True, use_oz=True, use_hpol=True, use_blur=True,
                 fit_box_scale=False, n_datasets=1, use_lp=None, use_tet_oz=None, n_oz_steps=None,
                 use_epol=False, use_pauli_cap=None, zero_bulk_offset=None):
        is_default_v2 = (n_types is None)
        if is_default_v2:
            self.use_lp = True if use_lp is None else bool(use_lp)
            self.n_types = 13 if self.use_lp else 10
        else:
            nt = int(n_types)
            if nt == 2:
                self.use_lp = False
                self.n_types = 2
            elif nt == 13 or bool(use_lp):
                self.use_lp = True
                self.n_types = 13
            else:
                self.use_lp = False
                self.n_types = nt

        self.use_langmuir = bool(use_langmuir)
        self.use_oz = bool(use_oz)
        tet_default = is_default_v2 if use_tet_oz is None else bool(use_tet_oz)
        self.use_tet_oz = tet_default and self.use_oz
        if n_oz_steps is None:
            self.n_oz_steps = 4 if self.use_tet_oz else 1
        else:
            self.n_oz_steps = max(1, int(n_oz_steps)) if self.use_oz else 0
        self.use_epol = bool(use_epol)
        self.use_pauli_cap = is_default_v2 if use_pauli_cap is None else bool(use_pauli_cap)
        self.zero_bulk_offset = is_default_v2 if zero_bulk_offset is None else bool(zero_bulk_offset)
        self.use_hpol = bool(use_hpol)
        self.use_blur = bool(use_blur)
        self.fit_box_scale = bool(fit_box_scale)
        self.n_datasets = int(n_datasets)

        self.n_pmf = self.n_types * N_KNOTS
        self.n_alpha = 1 if self.use_langmuir else 0
        self.n_oz = len(OZ_RADII) if self.use_oz else 0
        self.n_tet = 4 if self.use_tet_oz else 0
        self.n_epol = 3 if self.use_epol else 0
        self.n_hpol = (len(ACCEPTOR_TYPES) * N_HKNOTS) if self.use_hpol else 0
        self.n_scale = self.n_datasets if self.fit_box_scale else 0
        self.n_params = (
            self.n_pmf + self.n_alpha + self.n_oz + self.n_tet + self.n_epol + self.n_hpol + self.n_scale
        )

    def init_params(self):
        w_pmf = np.zeros((self.n_types, N_KNOTS), dtype=np.float64)
        knots = R_MIN + DR * np.arange(N_KNOTS)
        n_real_t = 2 if self.n_types == 2 else min(10, self.n_types)
        for t in range(n_real_t):
            if self.n_types != 2 and t in (4, 5, 8):
                r0 = 1.75 if t in (4, 5) else 2.25
            elif self.n_types != 2 and t == 9:
                r0 = 3.25
            elif (self.n_types != 2 and t in (0, 1, 2, 3)) or (self.n_types == 2 and t == 0):
                r0 = 2.65
            else:
                r0 = 3.05
            w_pmf[t] = 4.0 * expit((r0 - knots) / 0.22)
        # LP virtual sites (types 10..12) initialize to 0.0 (exact neutral superset of 10-type PMF)
        parts = [w_pmf.ravel()]
        if self.use_langmuir:
            parts.append(np.array([-4.5 if (self.use_tet_oz or self.use_pauli_cap) else -3.0]))
        if self.use_oz:
            parts.append(np.zeros(self.n_oz))
        if self.use_tet_oz:
            parts.append(np.array([0.05, -0.05, 0.0, 0.50], dtype=np.float64))
        if self.use_epol:
            parts.append(np.zeros(3, dtype=np.float64))
        if self.use_hpol:
            parts.append(np.zeros(self.n_hpol))
        if self.fit_box_scale:
            parts.append(np.zeros(self.n_datasets))
        return np.concatenate(parts)

    def unpack_all(self, theta):
        idx = 0
        w_pmf = theta[idx:idx + self.n_pmf]; idx += self.n_pmf
        logit_alpha = float(theta[idx]) if self.use_langmuir else None; idx += self.n_alpha
        v_oz = theta[idx:idx + self.n_oz] if self.use_oz else None; idx += self.n_oz
        w_tet = theta[idx:idx + self.n_tet] if self.use_tet_oz else None; idx += self.n_tet
        w_epol = theta[idx:idx + self.n_epol] if self.use_epol else None; idx += self.n_epol
        w_hpol = theta[idx:idx + self.n_hpol] if self.use_hpol else None; idx += self.n_hpol
        log_k = theta[idx:idx + self.n_scale] if self.fit_box_scale else None
        return w_pmf, logit_alpha, v_oz, w_tet, w_epol, w_hpol, log_k

    def unpack(self, theta):
        """Legacy 5-tuple unpack `(w_pmf, logit_alpha, v_oz, w_hpol, log_k)`."""
        w_pmf, logit_alpha, v_oz, _, _, w_hpol, log_k = self.unpack_all(theta)
        return w_pmf, logit_alpha, v_oz, w_hpol, log_k


def _activate_g_from_U(U_arr, use_langmuir: bool, z0: float, alpha: float):
    if use_langmuir:
        return expit(z0 - U_arr) / alpha
    return np.exp(-np.clip(U_arr, -4.5, 40.0))


def _dg_dU_and_dz0(g_arr, U_arr, use_langmuir: bool, alpha: float):
    if use_langmuir:
        s_arr = alpha * g_arr
        return -g_arr * (1.0 - s_arr), g_arr * (alpha - s_arr)
    in_r = (U_arr > -4.5) & (U_arr < 40.0)
    return np.where(in_r, -g_arr, 0.0), None


def _run_oz_and_pauli_forward(spec: WaterFieldSpec, grid: ASUGrid, U_sol, mw, logit_alpha,
                              v_oz, w_tet, F_hbsol, need_grad: bool = False, fft_workers: int = 1):
    """Shared multi-step OZ + tetrahedral 3-body closure + zero-bulk-offset + Rosenfeld 0D cavity cap."""
    n_asu = grid.n_asu
    alpha = float(expit(logit_alpha)) if spec.use_langmuir else 0.0
    z0 = float(logit_alpha) if spec.use_langmuir else 0.0

    g0_m = _activate_g_from_U(U_sol, spec.use_langmuir, z0, alpha)
    g_cur_bar = mw @ g0_m

    step_tape = []
    if spec.use_oz:
        K_steps = max(1, int(spec.n_oz_steps))
        eta = 1.0 if K_steps == 1 else OZ_DAMP
        vox_mol = (float(grid.cell.volume) / float(grid.nvox)) * float(grid.n_bulk)
        xi_bulk = math.log(2.0) / 4.0

        if spec.use_tet_oz:
            w_frust, w_tcoop, w_nc, raw_bsol = [float(x) for x in w_tet]
            beta_sol = float(softplus(raw_bsol, beta=1.0))
            tk = grid.get_tet_kernels()
            F_hbs = F_hbsol if F_hbsol is not None else np.zeros(tk.K0.shape, dtype=np.complex128)
        else:
            w_frust = w_tcoop = w_nc = beta_sol = 0.0
            tk = None
            F_hbs = None

        c_bulk = 0.0
        if spec.zero_bulk_offset:
            c_bulk = xi_bulk * float(np.dot(v_oz, grid.oz_dc[:spec.n_oz]))
            if spec.use_tet_oz:
                c_bulk += (
                    w_frust * tk.A_frust_0
                    + w_tcoop * tk.T_coop_0
                    + w_nc * tk.sat_nc_0
                )

        dU_cum = np.zeros(n_asu, dtype=np.float64)
        g_m_cur = g0_m
        for _s in range(K_steps):
            g_in_bar = g_cur_bar
            xi_asu = softplus(g_in_bar - 1.0, beta=4.0)
            xi_3d = grid.asu_to_full_3d(xi_asu)
            F_xi = sp_fft.rfftn(xi_3d, axes=(0, 1, 2), workers=fft_workers)

            psi_xi_3d = sp_fft.irfftn(
                grid.oz_kernels[:spec.n_oz] * F_xi[None, ...], s=grid.shape, axes=(1, 2, 3), workers=fft_workers
            )
            dU_step_3d = np.tensordot(v_oz, psi_xi_3d, axes=(0, 0))

            tet_cache = None
            if spec.use_tet_oz:
                F_hb = beta_sol * F_hbs + vox_mol * F_xi
                Nc = sp_fft.irfftn(tk.K0 * F_hb, s=grid.shape, axes=(0, 1, 2), workers=fft_workers)
                J1 = sp_fft.irfftn(tk.K1 * F_hb[None, ...], s=grid.shape, axes=(1, 2, 3), workers=fft_workers)
                Q2 = sp_fft.irfftn(tk.K2 * F_hb[None, ...], s=grid.shape, axes=(1, 2, 3), workers=fft_workers)

                j1_sq = np.sum(J1 * J1, axis=0)
                q2_sq = np.sum(Q2 * Q2, axis=0)
                num_f = q2_sq + (2.0 / 3.0) * j1_sq
                den_f = 1.0 + Nc * Nc
                A_frust = num_f / den_f

                nc2 = Nc * Nc
                den_c = 1.0 + 0.25 * nc2
                sat_nc = nc2 / den_c
                exp_af = np.exp(-2.0 * A_frust)
                T_coop = sat_nc * exp_af

                dU_step_3d = dU_step_3d + w_frust * A_frust + w_tcoop * T_coop + w_nc * sat_nc
                if need_grad:
                    tet_cache = (Nc, J1, Q2, den_f, A_frust, den_c, sat_nc, exp_af, T_coop)

            if spec.zero_bulk_offset:
                dU_step_3d = dU_step_3d - c_bulk

            dU_step_asu = grid.full_3d_to_asu_mean(dU_step_3d)
            dU_cum_next = (1.0 - eta) * dU_cum + eta * dU_step_asu
            U_tot_next = U_sol + dU_cum_next[None, :]
            g_m_next = _activate_g_from_U(U_tot_next, spec.use_langmuir, z0, alpha)
            g_next_bar = mw @ g_m_next

            if need_grad:
                step_tape.append((g_in_bar, psi_xi_3d, tet_cache, U_tot_next, g_m_next))

            dU_cum = dU_cum_next
            g_m_cur = g_m_next
            g_cur_bar = g_next_bar

        g_raw_m = g_m_cur
        g_raw_bar = g_cur_bar
    else:
        g_raw_m = g0_m
        g_raw_bar = g_cur_bar

    cav_cache = None
    if spec.use_pauli_cap:
        K_cav, K_cav_conj, n_cav_bulk = grid.get_cavity_kernel()
        vox_mol = (float(grid.cell.volume) / float(grid.nvox)) * float(grid.n_bulk)
        n_cav_3d = sp_fft.irfftn(
            K_cav * sp_fft.rfftn(vox_mol * grid.asu_to_full_3d(g_raw_bar), axes=(0, 1, 2), workers=fft_workers),
            s=grid.shape, axes=(0, 1, 2), workers=fft_workers,
        )
        n_cav_asu = grid.full_3d_to_asu_mean(n_cav_3d)
        c_cap_0 = float(softplus(n_cav_bulk - N_CAV_MAX, beta=BETA_CAV))
        D_cav = 1.0 + softplus(n_cav_asu - N_CAV_MAX, beta=BETA_CAV) - c_cap_0
        inv_D = 1.0 / D_cav
        gO_m = g_raw_m * inv_D[None, :]
        gO_bar = g_raw_bar * inv_D
        if need_grad:
            cav_cache = (g_raw_m, n_cav_asu, inv_D, K_cav_conj, vox_mol)
    else:
        gO_m = g_raw_m
        gO_bar = g_raw_bar

    tape = (g0_m, alpha, step_tape, cav_cache, F_hbsol) if need_grad else None
    return gO_m, gO_bar, tape


def compute_pmf_density_with_pullback(spec: WaterFieldSpec, ds, theta, ds_index=0,
                                      need_grad=True, use_f32_blas=True):
    """Evaluate per-conformer `(gO_m, gH_m, scale_k)` and exact reverse-mode pullback to `theta`."""
    w_pmf, logit_alpha, v_oz, w_tet, w_epol, w_hpol, log_k = spec.unpack_all(theta)
    grid = ds.grid
    M = ds.M
    n_asu = grid.n_asu
    mw = ds.mw
    w_fft = max(1, min(8, int(os.environ.get('OMP_NUM_THREADS', '4'))))

    if getattr(ds, 'Phi_O_flat', None) is not None:
        PhiO = ds.Phi_O_flat if ds.Phi_O_flat.shape[1] == spec.n_pmf else ds.Phi_O_flat[:, :spec.n_pmf]
        U_sol = (PhiO @ w_pmf).reshape(M, n_asu)
    elif use_f32_blas and getattr(ds, 'Phi_O_2d_f32', None) is not None:
        PhiO32 = ds.Phi_O_2d_f32 if ds.Phi_O_2d_f32.shape[1] == spec.n_pmf else ds.Phi_O_2d_f32[:, :spec.n_pmf]
        U_sol = (PhiO32 @ w_pmf.astype(np.float32)).reshape(M, n_asu).astype(np.float64)
    else:
        U_sol = np.empty((M, n_asu), dtype=np.float64)
        for m in range(M):
            U_sol[m] = ds.Phi_O_f32[m, :, :spec.n_pmf].astype(np.float64) @ w_pmf

    E2d = getattr(ds, 'E_feats_2d', None)
    if spec.use_epol and E2d is not None and w_epol is not None:
        U_sol = U_sol + (E2d @ w_epol).reshape(M, n_asu)

    F_hbsol = getattr(ds, 'F_hbsol', None)
    gO_m, _, tape = _run_oz_and_pauli_forward(
        spec, grid, U_sol, mw, logit_alpha, v_oz, w_tet, F_hbsol, need_grad=need_grad, fft_workers=w_fft
    )

    has_hpol = spec.use_hpol and w_hpol is not None and (
        getattr(ds, 'Phi_H_flat', None) is not None or getattr(ds, 'Phi_H_f32', None) is not None
    )
    if has_hpol:
        if getattr(ds, 'Phi_H_flat', None) is not None:
            raw_h = (ds.Phi_H_flat @ w_hpol).reshape(M, n_asu)
        elif use_f32_blas and getattr(ds, 'Phi_H_2d_f32', None) is not None:
            raw_h = (ds.Phi_H_2d_f32 @ w_hpol.astype(np.float32)).reshape(M, n_asu).astype(np.float64)
        else:
            raw_h = np.empty((M, n_asu), dtype=np.float64)
            for m in range(M):
                raw_h[m] = ds.Phi_H_f32[m].astype(np.float64) @ w_hpol
        th_h = np.tanh(raw_h)
        gH_m = ds.steric_H_64 * th_h
    else:
        th_h = None
        gH_m = None

    scale_k = float(np.exp(log_k[ds_index])) if (spec.fit_box_scale and log_k is not None) else 1.0
    if not need_grad:
        return gO_m, gH_m, scale_k, None

    g0_m, alpha, step_tape, cav_cache, F_hbs = tape

    def pullback_theta(dL_dgO_m, dL_dgH_m=None, dL_dlogk=0.0):
        grad_theta = np.zeros_like(theta)
        if spec.fit_box_scale:
            grad_theta[-spec.n_scale + ds_index] = float(dL_dlogk)

        if has_hpol and dL_dgH_m is not None:
            dL_draw = dL_dgH_m * ds.steric_H_64 * (1.0 - th_h ** 2)
            i0 = spec.n_pmf + spec.n_alpha + spec.n_oz + spec.n_tet + spec.n_epol
            if getattr(ds, 'Phi_H_flat', None) is not None:
                grad_theta[i0:i0 + spec.n_hpol] = dL_draw.ravel() @ ds.Phi_H_flat
            elif use_f32_blas and getattr(ds, 'Phi_H_2d_f32', None) is not None:
                grad_theta[i0:i0 + spec.n_hpol] = (dL_draw.ravel().astype(np.float32) @ ds.Phi_H_2d_f32).astype(np.float64)
            else:
                acc_h = np.zeros(spec.n_hpol, dtype=np.float64)
                for m in range(M):
                    acc_h += dL_draw[m] @ ds.Phi_H_f32[m].astype(np.float64)
                grad_theta[i0:i0 + spec.n_hpol] = acc_h

        # 1. Backprop through Rosenfeld 0D cavity cap if enabled
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

        # 2. Backprop through K_steps of OZ + TetOZ + zero_bulk_offset
        dL_dUsol = np.zeros((M, n_asu), dtype=np.float64)
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

        # 3. Backprop through initial solute step g0_m
        dg0_dU, dg0_dz = _dg_dU_and_dz0(g0_m, U_sol, spec.use_langmuir, alpha)
        dL_dUsol += dL_dg_cur * dg0_dU
        if spec.use_langmuir:
            grad_logit_alpha += float(np.sum(dL_dg_cur * dg0_dz))

        if getattr(ds, 'Phi_O_flat', None) is not None:
            PhiO = ds.Phi_O_flat if ds.Phi_O_flat.shape[1] == spec.n_pmf else ds.Phi_O_flat[:, :spec.n_pmf]
            grad_theta[:spec.n_pmf] = dL_dUsol.ravel() @ PhiO
        elif use_f32_blas and getattr(ds, 'Phi_O_2d_f32', None) is not None:
            PhiO32 = ds.Phi_O_2d_f32 if ds.Phi_O_2d_f32.shape[1] == spec.n_pmf else ds.Phi_O_2d_f32[:, :spec.n_pmf]
            grad_theta[:spec.n_pmf] = (dL_dUsol.ravel().astype(np.float32) @ PhiO32).astype(np.float64)
        else:
            acc_o = np.zeros(spec.n_pmf, dtype=np.float64)
            for m in range(M):
                acc_o += dL_dUsol[m] @ ds.Phi_O_f32[m, :, :spec.n_pmf].astype(np.float64)
            grad_theta[:spec.n_pmf] = acc_o

        idx = spec.n_pmf
        if spec.use_langmuir:
            grad_theta[idx] = grad_logit_alpha
            idx += 1
        if spec.use_oz:
            grad_theta[idx:idx + spec.n_oz] = grad_voz
            idx += spec.n_oz
        if spec.use_tet_oz:
            grad_theta[idx:idx + 4] = grad_wtet
            idx += 4
        if spec.use_epol and E2d is not None:
            grad_theta[idx:idx + 3] = dL_dUsol.ravel() @ E2d

        return grad_theta

    return gO_m, gH_m, scale_k, pullback_theta


def forward_direct(spec: WaterFieldSpec, grid: ASUGrid, members, member_weights, theta, B_map: float = 5.0,
                   return_per_member: bool = False, charges=None):
    """Memory-lean, zero-feature-tensor forward evaluation `(g_hat_asu, gO_bar_asu)` for inference (`surfnturf predict`)."""
    w_pmf, logit_alpha, v_oz, w_tet, w_epol, w_hpol, _ = spec.unpack_all(theta)
    M = len(members)
    n_asu = grid.n_asu
    mw = np.asarray(member_weights, dtype=np.float64)
    mw = mw / mw.sum()
    two_type = (spec.n_types == 2)
    w_fft = max(1, min(8, int(os.environ.get('OMP_NUM_THREADS', '4'))))

    W_pmf_pad = np.zeros((spec.n_types, N_KNOTS + 3), dtype=np.float64)
    W_pmf_pad[:, :N_KNOTS] = w_pmf.reshape(spec.n_types, N_KNOTS)

    include_hpol = bool(spec.use_hpol and w_hpol is not None)
    if include_hpol:
        W_h_pad = np.zeros((len(ACCEPTOR_TYPES), N_HKNOTS + 3), dtype=np.float64)
        W_h_pad[:, :N_HKNOTS] = w_hpol.reshape(len(ACCEPTOR_TYPES), N_HKNOTS)
        gH_m = np.zeros((M, n_asu), dtype=np.float64) if return_per_member else None
        gH_bar = np.zeros(n_asu, dtype=np.float64)
    else:
        W_h_pad = None
        gH_m = None
        gH_bar = None

    include_epol = bool(spec.use_epol and w_epol is not None)
    pts = grid.asu_xyz
    tree_grid = cKDTree(pts)
    vox_vol = float(grid.cell.volume) / float(grid.nvox)

    def _eval_atom_subset(xyz_sub, typ_sub, q_sub=None):
        u_vec = np.zeros(n_asu, dtype=np.float64)
        lst_vec = np.zeros(n_asu, dtype=np.float64) if include_hpol else None
        rh_vec = np.zeros(n_asu, dtype=np.float64) if include_hpol else None
        hb_vec = np.zeros(n_asu, dtype=np.float64) if spec.use_tet_oz else None
        ep_tup = (
            np.zeros(n_asu), np.zeros(n_asu), np.zeros(n_asu), np.zeros(n_asu), np.zeros(n_asu)
        ) if include_epol else None

        if len(xyz_sub) == 0:
            return u_vec, lst_vec, rh_vec, hb_vec, ep_tup

        if spec.n_types == 10 and np.any(typ_sub >= 10):
            keep = typ_sub < 10
            xyz_use, typ_use = xyz_sub[keep], typ_sub[keep]
        else:
            xyz_use, typ_use = xyz_sub, typ_sub

        SX, Si = grid.sym_expand(xyz_use, cutoff_A=R_CUT, dedup_self=True)
        if len(SX) > 0:
            St = typ_use[Si]
            nb = tree_grid.query_ball_point(SX, R_CUT)
            lens = np.array([len(x) for x in nb], dtype=np.int32)
            if lens.sum() > 0:
                ai = np.repeat(np.arange(len(SX), dtype=np.int32), lens)
                pi = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb if len(x)])
                d = np.linalg.norm(pts[pi] - SX[ai], axis=1)
                ti = St[ai]

                (k0, b0), (k1, b1), (k2, b2), (k3, b3) = _bspline_weights(d, R_MIN, DR, N_KNOTS)
                u_pair = (
                    W_pmf_pad[ti, k0] * b0
                    + W_pmf_pad[ti, k1] * b1
                    + W_pmf_pad[ti, k2] * b2
                    + W_pmf_pad[ti, k3] * b3
                )
                u_vec = np.bincount(pi, weights=u_pair, minlength=n_asu)

                if include_hpol:
                    real_m = np.ones_like(ti, dtype=bool) if two_type else (ti < 10)
                    if np.any(real_m):
                        pr, dr, tr = pi[real_m], d[real_m], ti[real_m]
                        is_h_atom = np.zeros_like(tr, dtype=bool) if two_type else ((tr == 4) | (tr == 5) | (tr == 8))
                        r_clash = np.where(is_h_atom, 1.25, 1.55)
                        close = dr < (r_clash + 0.80)
                        if np.any(close):
                            l_excl = -np.logaddexp(0.0, -(dr[close] - r_clash[close]) / 0.12)
                            lst_vec = np.bincount(pr[close], weights=l_excl, minlength=n_asu)
                        acc_mask = ((tr == 0) if two_type else ((tr == 0) | (tr == 1) | (tr == 2))) & (dr < RH_CUT)
                        if np.any(acc_mask):
                            pi_a, d_a, ti_a = pr[acc_mask], dr[acc_mask], tr[acc_mask]
                            (hk0, hb0), (hk1, hb1), (hk2, hb2), (hk3, hb3) = _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS)
                            h_pair = (
                                W_h_pad[ti_a, hk0] * hb0
                                + W_h_pad[ti_a, hk1] * hb1
                                + W_h_pad[ti_a, hk2] * hb2
                                + W_h_pad[ti_a, hk3] * hb3
                            )
                            rh_vec = np.bincount(pi_a, weights=h_pair, minlength=n_asu)

        real_atoms = np.ones(len(typ_sub), dtype=bool) if two_type else (typ_sub < 10)
        xyz_r = xyz_sub[real_atoms]
        typ_r = typ_sub[real_atoms]

        if spec.use_tet_oz and len(xyz_r) > 0:
            pol_hv = (typ_r == 0) if two_type else ((typ_r == 0) | (typ_r == 1) | (typ_r == 2) | (typ_r == 3))
            if np.any(pol_hv):
                SP, _ = grid.sym_expand(xyz_r[pol_hv], cutoff_A=1.8, dedup_self=True)
                if len(SP) > 0:
                    nb_p = tree_grid.query_ball_point(SP, 1.5)
                    lp = np.array([len(x) for x in nb_p], dtype=np.int32)
                    if lp.sum() > 0:
                        ap = np.repeat(np.arange(len(SP), dtype=np.int32), lp)
                        pp = np.concatenate([np.asarray(x, dtype=np.int32) for x in nb_p if len(x)])
                        dp = np.linalg.norm(pts[pp] - SP[ap], axis=1)
                        sig_b = 0.40
                        norm_vox = vox_vol / ((2.0 * math.pi * sig_b * sig_b) ** 1.5)
                        hb_vec = np.bincount(pp, weights=norm_vox * np.exp(-0.5 * (dp / sig_b) ** 2), minlength=n_asu)

        if include_epol and len(xyz_r) > 0:
            if q_sub is not None and len(q_sub) == len(typ_sub):
                q_r = q_sub[real_atoms]
            elif q_sub is not None and len(q_sub) == len(xyz_r):
                q_r = q_sub
            else:
                q_r = _DEFAULT_CLASS_Q[np.clip(typ_r, 0, len(_DEFAULT_CLASS_Q) - 1)]
            ep_tup = _compute_epol_additive_terms(grid, tree_grid, xyz_r, typ_r, q_r)

        return u_vec, lst_vec, rh_vec, hb_vec, ep_tup

    (com_xyz, com_typ, com_q), diff_members = _partition_common_and_diff_members(members, charges=charges)
    u_com, lst_com, rh_com, hb_com, ep_com = _eval_atom_subset(
        com_xyz, com_typ, (com_q if len(com_q) else None)
    )

    U_sol = np.empty((M, n_asu), dtype=np.float64)
    sol_hb_asu = hb_com.copy() if spec.use_tet_oz else None
    for m, (xyz_d, typ_d, q_d) in enumerate(diff_members):
        u_d, lst_d, rh_d, hb_d, ep_d = _eval_atom_subset(xyz_d, typ_d, q_d)
        U_sol[m] = u_com + u_d
        if include_epol:
            ef = _epol_features_from_additive(
                ep_com[0] + ep_d[0], ep_com[1] + ep_d[1], ep_com[2] + ep_d[2],
                ep_com[3] + ep_d[3], ep_com[4] + ep_d[4],
            )
            U_sol[m] += ef @ w_epol
        if spec.use_tet_oz:
            sol_hb_asu += mw[m] * hb_d
        if include_hpol:
            hm = np.exp(lst_com + lst_d) * np.tanh(rh_com + rh_d)
            if return_per_member:
                gH_m[m] = hm
            gH_bar += mw[m] * hm

    F_hbsol = sp_fft.rfftn(grid.asu_to_full_3d(sol_hb_asu), axes=(0, 1, 2), workers=w_fft) if spec.use_tet_oz else None
    gO_m, gO_bar, _ = _run_oz_and_pauli_forward(
        spec, grid, U_sol, mw, logit_alpha, v_oz, w_tet, F_hbsol, need_grad=False, fft_workers=w_fft
    )

    if spec.use_blur:
        Kw, Kh = grid.water_kernel(B_map)
        gO_3d = grid.asu_to_full_3d(gO_bar)
        F_elec = Kw * sp_fft.rfftn(grid.n_bulk * gO_3d, axes=(0, 1, 2), workers=w_fft)
        if gH_bar is not None:
            gH_3d = grid.asu_to_full_3d(gH_bar)
            F_elec = F_elec + Kh * sp_fft.rfftn(grid.n_bulk * gH_3d, axes=(0, 1, 2), workers=w_fft)
        rho_3d = sp_fft.irfftn(F_elec, s=grid.shape, axes=(0, 1, 2), workers=w_fft)
        g_hat = grid.full_3d_to_asu_mean(rho_3d) / RHO_BULK
    else:
        g_hat = gO_bar

    if return_per_member:
        return g_hat, gO_bar, gO_m, gH_m
    return g_hat, gO_bar


def forward_and_grad(spec: WaterFieldSpec, ds: DatasetBundle, theta, ds_index=0,
                     loss_mode='l2', compute_grad=True, ext_dL_dghat=None, return_grad_B=False):
    """Evaluate predicted `g_hat` on ASU (and loss + exact analytical gradient w.r.t. `theta` and optionally `B_map`)."""
    grid = ds.grid
    mw = ds.mw
    gO_m, gH_m, scale_k, pb_theta = compute_pmf_density_with_pullback(
        spec, ds, theta, ds_index=ds_index, need_grad=compute_grad, use_f32_blas=False
    )
    gO_bar = mw @ gO_m
    gH_bar = (mw @ gH_m) if gH_m is not None else None

    if spec.use_blur:
        gO_3d = grid.asu_to_full_3d(gO_bar)
        F_elec = ds.Kw * np.fft.rfftn(grid.n_bulk * gO_3d, axes=(0, 1, 2))
        if gH_bar is not None:
            gH_3d = grid.asu_to_full_3d(gH_bar)
            F_elec = F_elec + ds.Kh * np.fft.rfftn(grid.n_bulk * gH_3d, axes=(0, 1, 2))
        rho_3d = np.fft.irfftn(F_elec, s=grid.shape, axes=(0, 1, 2))
        g_hat_base = grid.full_3d_to_asu_mean(rho_3d) / RHO_BULK
    else:
        F_elec = None
        g_hat_base = gO_bar
    g_hat = scale_k * g_hat_base

    if not compute_grad:
        return g_hat, gO_bar

    if ext_dL_dghat is not None:
        if callable(ext_dL_dghat):
            loss, dL_dghat = ext_dL_dghat(g_hat)
            loss = float(loss)
            dL_dghat = np.asarray(dL_dghat, dtype=np.float64)
        else:
            loss = 0.0
            dL_dghat = np.asarray(ext_dL_dghat, dtype=np.float64)
    else:
        w_asu = ds.weights_asu
        inv_w = 1.0 / ds.w_sum
        target = ds.g_asu
        if loss_mode == 'kl':
            eps = 1e-4
            rad = np.hypot(g_hat, eps)
            g_tilde = 0.5 * (g_hat + rad)
            dg_tilde = 0.5 * (1.0 + g_hat / rad)
            t_pos = np.maximum(target, 0.0)
            ref = np.where(t_pos > eps, t_pos * np.log(np.maximum(t_pos, eps)) - t_pos, 0.0)
            loss_vec = (g_tilde - t_pos * np.log(g_tilde)) + ref
            loss = float(np.sum(w_asu * loss_vec) * inv_w)
            dL_dghat = w_asu * inv_w * (1.0 - t_pos / g_tilde) * dg_tilde
        else:
            diff = g_hat - target
            loss = float(0.5 * np.sum(w_asu * (diff ** 2)) * inv_w)
            dL_dghat = w_asu * inv_w * diff

    dL_dlogk = float(np.sum(dL_dghat * g_hat))
    dL_dbase = dL_dghat * scale_k

    if spec.use_blur:
        g_rho_3d = grid.asu_to_full_3d((dL_dbase / RHO_BULK) / grid.orbit_mult)
        F_grho = np.fft.rfftn(g_rho_3d, axes=(0, 1, 2))
        grad_gO_3d = grid.n_bulk * np.fft.irfftn(ds.Kw * F_grho, s=grid.shape, axes=(0, 1, 2))
        dL_dgO_bar = grid.full_3d_to_asu_grad(grad_gO_3d)
        if gH_bar is not None:
            grad_gH_3d = grid.n_bulk * np.fft.irfftn(ds.Kh * F_grho, s=grid.shape, axes=(0, 1, 2))
            dL_dgH_bar = grid.full_3d_to_asu_grad(grad_gH_3d)
        else:
            dL_dgH_bar = None
        if return_grad_B:
            nw = grid.shape[2]
            rfft_w = np.full(nw // 2 + 1, 2.0, dtype=np.float64)
            rfft_w[0] = 1.0
            if nw % 2 == 0:
                rfft_w[-1] = 1.0
            grad_B = -float(np.sum(rfft_w * grid.stol2 * np.real(np.conj(F_grho) * F_elec))) / float(grid.nvox)
        else:
            grad_B = 0.0
    else:
        dL_dgO_bar = dL_dbase
        dL_dgH_bar = None
        grad_B = 0.0

    dL_dgO_m = np.outer(mw, dL_dgO_bar)
    dL_dgH_m = np.outer(mw, dL_dgH_bar) if dL_dgH_bar is not None else None
    grad_theta = pb_theta(dL_dgO_m, dL_dgH_m, dL_dlogk)

    if return_grad_B:
        return loss, grad_theta, grad_B, g_hat, gO_bar
    return loss, grad_theta, g_hat, gO_bar


def solv_to_fpart(solv_3d, cell_volume, hkl):
    """Direct FFT structure factors F_part(h) matching gemmi.transform_map_to_f_phi (+2*pi*i convention)."""
    nu, nv, nw = solv_3d.shape
    hkl = np.asarray(hkl, dtype=np.int64)
    F_grid = np.fft.ifftn(np.asarray(solv_3d, dtype=np.float64), axes=(0, 1, 2))
    out = float(cell_volume) * F_grid[hkl[:, 0] % nu, hkl[:, 1] % nv, hkl[:, 2] % nw]
    in_nyq = (
        (hkl[:, 0] >= -(nu // 2)) & (hkl[:, 0] <= (nu - 1) // 2) &
        (hkl[:, 1] >= -(nv // 2)) & (hkl[:, 1] <= (nv - 1) // 2) &
        (hkl[:, 2] >= -(nw // 2)) & (hkl[:, 2] <= (nw - 1) // 2)
    )
    if not np.all(in_nyq):
        out = np.where(in_nyq, out, 0.0 + 0.0j)
    return out


def fpart_grad_to_asu(grid: ASUGrid, g_hat, hkl, dL_dFpart, clamp_nonneg=True):
    """Exact adjoint of `solv_3d = maximum(asu_to_full_3d(g_hat) * RHO_BULK, 0) -> solv_to_fpart(...)`."""
    nu, nv, nw = grid.shape
    hkl = np.asarray(hkl, dtype=np.int64)
    in_nyq = (
        (hkl[:, 0] >= -(nu // 2)) & (hkl[:, 0] <= (nu - 1) // 2) &
        (hkl[:, 1] >= -(nv // 2)) & (hkl[:, 1] <= (nv - 1) // 2) &
        (hkl[:, 2] >= -(nw // 2)) & (hkl[:, 2] <= (nw - 1) // 2)
    )
    Q = np.zeros((nu, nv, nw), dtype=np.complex128)
    dF = np.asarray(dL_dFpart, dtype=np.complex128)
    if not np.all(in_nyq):
        hkl = hkl[in_nyq]
        dF = dF[in_nyq]
    np.add.at(Q, (hkl[:, 0] % nu, hkl[:, 1] % nv, hkl[:, 2] % nw), dF)
    dL_dsolv_3d = (float(grid.cell.volume) / float(grid.nvox)) * np.real(np.fft.fftn(Q, axes=(0, 1, 2)))
    if clamp_nonneg:
        rho_3d = grid.asu_to_full_3d(g_hat) * RHO_BULK
        dL_drho_3d = np.where(rho_3d > 0.0, dL_dsolv_3d, 0.0)
    else:
        dL_drho_3d = dL_dsolv_3d
    return grid.full_3d_to_asu_grad(dL_drho_3d) * RHO_BULK


def fpart_forward_and_grad(spec: WaterFieldSpec, ds: DatasetBundle, theta, B_map, hkl,
                           fpart_loss_and_grad_fn, ds_index=0, clamp_nonneg=True):
    """Single-pass reciprocal-space F_part evaluation + exact analytical backprop to (theta, B_map)."""
    grid = ds.grid
    B_val = float(B_map)
    if ds.B_map != B_val or ds.Kw is None:
        ds.B_map = B_val
        ds.Kw, ds.Kh = grid.water_kernel(B_val)

    saved_fpart = [None]

    def _asu_hook(g_hat):
        rho_3d = grid.asu_to_full_3d(g_hat) * RHO_BULK
        solv_3d = np.maximum(rho_3d, 0.0) if clamp_nonneg else rho_3d
        f_part = solv_to_fpart(solv_3d, grid.cell.volume, hkl)
        saved_fpart[0] = f_part
        loss_val, dL_dF = fpart_loss_and_grad_fn(f_part)
        dL_dghat = fpart_grad_to_asu(grid, g_hat, hkl, dL_dF, clamp_nonneg=clamp_nonneg)
        return loss_val, dL_dghat

    loss, grad_theta, grad_B, _, _ = forward_and_grad(
        spec, ds, theta, ds_index=ds_index, compute_grad=True,
        ext_dL_dghat=_asu_hook, return_grad_B=True
    )
    return loss, grad_theta, grad_B, saved_fpart[0]


def fit_kmask_per_bin(f_calc, f_part, k_tot, f_obs, work_mask, bin_idx, n_bins,
                      loss_type='l2', huber_delta=1.0, k_bounds=(0.0, 3.0), n_newton=8):
    """Variable-Projection per-bin k_mask >= 0 minimizing the exact work-set amplitude loss (dL / dk_mask = 0)."""
    k_min, k_max = float(k_bounds[0]), float(k_bounds[1])
    k_bins = np.zeros(n_bins, dtype=np.float64)
    grid_k = np.linspace(k_min, k_max, 61)
    eps_abs = 1e-12

    for b in range(n_bins):
        sel = work_mask & (bin_idx == b)
        if not np.any(sel):
            continue
        fc = f_calc[sel]
        fp = f_part[sel]
        kt = k_tot[sel]
        fo = f_obs[sel]

        fm_grid = kt[:, None] * (fc[:, None] + fp[:, None] * grid_k[None, :])
        res_grid = np.abs(fm_grid) - fo[:, None]
        if loss_type == 'huber':
            obj_grid = np.sum((huber_delta ** 2) * (np.sqrt(1.0 + (res_grid / huber_delta) ** 2) - 1.0), axis=0)
        else:
            obj_grid = 0.5 * np.sum(res_grid ** 2, axis=0)
        kb = float(grid_k[int(np.argmin(obj_grid))])

        for _ in range(n_newton):
            fm = kt * (fc + kb * fp)
            afm = np.maximum(np.abs(fm), eps_abs)
            u = fm / afm
            kt_fp = kt * fp
            d_afm = np.real(np.conj(u) * kt_fp)
            d2_afm = (np.abs(kt_fp) ** 2 - d_afm ** 2) / afm
            r = afm - fo
            if loss_type == 'huber':
                s = np.sqrt(1.0 + (r / huber_delta) ** 2)
                psi1 = r / s
                psi2 = 1.0 / (s ** 3)
            else:
                psi1 = r
                psi2 = np.ones_like(r)
            g1 = float(np.sum(psi1 * d_afm))
            h2 = float(np.sum(psi2 * (d_afm ** 2) + np.maximum(psi1, 0.0) * d2_afm))
            if abs(g1) < 1e-12 or h2 <= 0.0:
                break
            kb_new = float(np.clip(kb - g1 / h2, k_min, k_max))
            if abs(kb_new - kb) < 1e-12:
                kb = kb_new
                break
            kb = kb_new
        k_bins[b] = kb

    return k_bins, k_bins[bin_idx]


def smoothness_penalty_and_grad(spec: WaterFieldSpec, theta, lam_pmf=2e-3, lam_hpol=2e-3, lam_oz=1e-4, lam_extra=2e-4):
    """Second-difference spline curvature & L2 regularization across all active parameter blocks."""
    w_pmf, _, v_oz, w_tet, w_epol, w_hpol, _ = spec.unpack_all(theta)
    grad = np.zeros_like(theta)
    pen = 0.0

    W = w_pmf.reshape(spec.n_types, N_KNOTS)
    Wp = np.hstack([W, np.zeros((spec.n_types, 1))])
    d2 = Wp[:, 2:] - 2.0 * Wp[:, 1:-1] + Wp[:, :-2]
    pen += 0.5 * lam_pmf * float(np.sum(d2 ** 2))
    gW = np.zeros_like(Wp)
    gW[:, :-2] += lam_pmf * d2
    gW[:, 1:-1] -= 2.0 * lam_pmf * d2
    gW[:, 2:] += lam_pmf * d2
    pen += 0.5 * lam_pmf * float(np.sum(W[:, -1] ** 2))
    gW[:, -2] += lam_pmf * W[:, -1]
    if spec.use_lp and spec.n_types > 10:
        pen += 0.5 * (0.25 * lam_pmf) * float(np.sum(W[10:] ** 2))
        gW[10:, :N_KNOTS] += (0.25 * lam_pmf) * W[10:]
    grad[:spec.n_pmf] = gW[:, :N_KNOTS].ravel()

    idx = spec.n_pmf + spec.n_alpha
    if spec.use_oz and v_oz is not None:
        pen += 0.5 * lam_oz * float(np.sum(v_oz ** 2))
        grad[idx:idx + spec.n_oz] = lam_oz * v_oz
        idx += spec.n_oz

    if spec.use_tet_oz and w_tet is not None:
        reg_t = np.array([w_tet[0], w_tet[1], w_tet[2], w_tet[3] - 0.50], dtype=np.float64)
        pen += 0.5 * lam_extra * float(np.sum(reg_t ** 2))
        grad[idx:idx + 4] = lam_extra * reg_t
        idx += 4

    if spec.use_epol and w_epol is not None:
        pen += 0.5 * lam_extra * float(np.sum(w_epol ** 2))
        grad[idx:idx + 3] = lam_extra * w_epol
        idx += 3

    if spec.use_hpol and w_hpol is not None:
        H = w_hpol.reshape(len(ACCEPTOR_TYPES), N_HKNOTS)
        Hp = np.hstack([H, np.zeros((len(ACCEPTOR_TYPES), 1))])
        dh2 = Hp[:, 2:] - 2.0 * Hp[:, 1:-1] + Hp[:, :-2]
        pen += 0.5 * lam_hpol * float(np.sum(dh2 ** 2) + 0.2 * np.sum(H ** 2))
        gH = np.zeros_like(Hp)
        gH[:, :-2] += lam_hpol * dh2
        gH[:, 1:-1] -= 2.0 * lam_hpol * dh2
        gH[:, 2:] += lam_hpol * dh2
        gH[:, :N_HKNOTS] += 0.2 * lam_hpol * H
        grad[idx:idx + spec.n_hpol] = gH[:, :N_HKNOTS].ravel()

    return pen, grad


def param_bounds(spec: WaterFieldSpec):
    """Physical box bounds for L-BFGS-B."""
    n_base = 2 if spec.n_types == 2 else min(10, spec.n_types)
    bounds = [(-6.0, 25.0)] * (n_base * N_KNOTS)
    if spec.use_lp and spec.n_types > 10:
        bounds.extend([(-5.0, 8.0)] * ((spec.n_types - 10) * N_KNOTS))
    if spec.use_langmuir:
        bounds.append((-6.5 if (spec.use_tet_oz or spec.use_pauli_cap) else -6.0, -0.8))
    if spec.use_oz:
        bounds.extend([(-5.0, 5.0)] * spec.n_oz)
    if spec.use_tet_oz:
        bounds.extend([(0.0, 4.0), (-4.0, 0.5), (-2.0, 2.0), (-3.0, 3.0)])
    if spec.use_epol:
        bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)])
    if spec.use_hpol:
        bounds.extend([(-4.0, 4.0)] * spec.n_hpol)
    if spec.fit_box_scale:
        bounds.extend([(-0.35, 0.35)] * spec.n_scale)
    return bounds
