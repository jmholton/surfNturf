"""surfNturf core differentiable hydration field (PMF + Langmuir saturation + Ornstein-Zernike shells + IT92 rendering).

Implements:
  1. Exact crystal-symmetry Asymmetric Unit (ASU) orbit indexing on 3D CCP4 grids (zero train/test leak).
  2. 10-class chemical atom typing keeping explicit protein hydrogens (polar donor H, cationic H, apolar H,
     carbonyl/carboxylate/hydroxyl O, donor N, sp2/sp3 C, chalcogen S/Se, plus ions/metals/phosphorus) and
     multi-altloc conformer completion (`reconnect`, `letters`, or `quantiles`).
  3. C2-smooth cardinal cubic B-spline radial potentials u_t(r) on [1.4, 5.6 A] pinned to 0 in bulk.
  4. Grand-canonical Langmuir / lattice-gas saturation g_O = (1/alpha) * sigmoid(log(alpha/(1-alpha)) - U) so
     multi-H-bond groove bridges cooperate multiplicatively while saturating smoothly at physical site occupancy.
  5. 1-FFT Ornstein-Zernike water-water radial shell feedback (annular exclusion moat 1.2-2.3 A + 2.8 A H-bond bridge).
  6. Acceptor-only water-H orientation term (zero net charge, never placed on N-H donors).
  7. Exact IT92 (O + spherical 2H shell at r_OH = 0.9572 A) x exp(-B_map * stol^2) Fourier rendering kernel.
"""
import glob
import math
import os
import numpy as np
import gemmi
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


def softplus(x, beta=4.0):
    """Overflow-safe C-infinity softplus: (1/beta) * log(1 + exp(beta * x))."""
    bx = beta * x
    return np.where(bx > 30.0, x, np.log1p(np.exp(np.clip(bx, -60.0, 30.0))) / beta)


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

        # Fractional and orthogonal coordinates of the (nu, nv, nw) subgrid matching [::step, ::step, ::step]
        iu = np.arange(0, self.full_shape[0], self.step)[:nu] / self.full_shape[0]
        iv = np.arange(0, self.full_shape[1], self.step)[:nv] / self.full_shape[1]
        iw = np.arange(0, self.full_shape[2], self.step)[:nw] / self.full_shape[2]
        fr = np.stack(np.meshgrid(iu, iv, iw, indexing='ij'), axis=-1).reshape(-1, 3)
        self.frac = fr
        self.xyz = fr @ self.O.T

        # Exact symmetry orbit mapping on the grid (verified in fractional coordinates)
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
        self.asu_reps = asu_reps                      # indices into full flattened grid (N_asu,)
        self.full_to_asu = inv.astype(np.int32)       # map full grid -> 0..N_asu-1
        self.n_asu = len(asu_reps)
        self.asu_xyz = self.xyz[asu_reps]
        self.orbit_mult = np.bincount(self.full_to_asu, minlength=self.n_asu).astype(np.float64)

        # Reciprocal-space stol^2 = 1 / (4 d^2) on rfftn grid
        h = np.fft.fftfreq(nu, 1.0 / nu)[:, None, None]
        k = np.fft.fftfreq(nv, 1.0 / nv)[None, :, None]
        l = np.fft.rfftfreq(nw, 1.0 / nw)[None, None, :]
        hkl = np.stack(np.broadcast_arrays(h, k, l), axis=-1)
        q_cart = hkl @ self.F
        inv_d2 = np.sum(q_cart ** 2, axis=-1)
        self.stol2 = inv_d2 / 4.0
        self.stol = np.sqrt(self.stol2)

        # IT92 scattering factors for O and H
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
        self.f_water_0 = fO + 2.0 * fH * j0           # ~10.0 e at s=0
        self.f_H_0 = fH.copy()
        self.f_H_0[0, 0, 0] = 0.0                     # strictly zero net electrons (conserves F000)
        self.n_bulk = RHO_BULK / float(self.f_water_0[0, 0, 0])

        # Ornstein-Zernike radial shell kernels in Fourier space
        k_rad = 4.0 * np.pi * self.stol               # 2*pi / d
        gauss_env = np.exp(-0.5 * (k_rad * OZ_SIGMA) ** 2)
        oz = []
        for Rq in OZ_RADII:
            x = k_rad * Rq
            sinc = np.where(x > 1e-8, np.sin(x) / np.where(x > 1e-8, x, 1.0), 1.0)
            oz.append(sinc * gauss_env)
        self.oz_kernels = np.stack(oz, axis=0)        # (Q, nu, nv, nw//2+1)

    def water_kernel(self, B_map):
        dw = np.exp(-float(B_map) * self.stol2)
        return self.f_water_0 * dw, self.f_H_0 * dw

    def asu_to_full_3d(self, arr_asu):
        return arr_asu[self.full_to_asu].reshape(self.shape)

    def full_3d_to_asu_grad(self, grad_3d):
        return np.bincount(self.full_to_asu, grad_3d.ravel(), minlength=self.n_asu)

    def full_3d_to_asu_mean(self, arr_3d):
        return np.bincount(self.full_to_asu, arr_3d.ravel(), minlength=self.n_asu) / self.orbit_mult

    def sym_expand(self, xyz, margin=None, cutoff_A=R_CUT):
        """Expand Cartesian coordinates by all space-group and lattice images within `cutoff_A` of [0, 1)^3."""
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
        for op in self.ops:
            R = np.array(op.rot, dtype=np.float64) / op.DEN
            t = np.array(op.tran, dtype=np.float64) / op.DEN
            g = f @ R.T + t
            g -= np.floor(g)
            for s in shifts:
                h = g + s
                m = np.all((h >= lo) & (h < hi), axis=1)
                if np.any(m):
                    out_p.append(h[m] @ self.O.T)
                    out_i.append(np.nonzero(m)[0])
        if not out_p:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
        return np.vstack(out_p), np.concatenate(out_i).astype(np.int32)


def classify_conformer_atoms(atoms_list, two_type_heavy_only=False):
    """Classify [(res_seq, res_name, atom_name, elem, xyz), ...] into ATOM_TYPES (or TWO_TYPES)."""
    if two_type_heavy_only:
        xyz, typ = [], []
        for _, _, _, el, pos in atoms_list:
            el_u = el.upper()
            if el_u in ('H', 'D'):
                continue
            xyz.append(pos)
            is_pol = (el_u in ('N', 'O') or el_u in _COMPACT_IONS or el_u in _LARGE_POLAR_IONS)
            typ.append(0 if is_pol else 1)
        return np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(typ, dtype=np.int32)

    heavy_idx = [i for i, a in enumerate(atoms_list) if a[3].upper() not in ('H', 'D')]
    hyd_idx = [i for i, a in enumerate(atoms_list) if a[3].upper() in ('H', 'D')]
    h_xyz = np.array([atoms_list[i][4] for i in heavy_idx], dtype=np.float64).reshape(-1, 3)
    tree_h = cKDTree(h_xyz) if len(h_xyz) else None

    # Map each H/D to its bonded heavy atom (within 1.30 A) and count bonded H/D per heavy atom
    parent_of_hyd = {}
    nh_on_heavy = np.zeros(len(heavy_idx), dtype=np.int32)
    if hyd_idx and tree_h is not None:
        hy_xyz = np.array([atoms_list[i][4] for i in hyd_idx], dtype=np.float64).reshape(-1, 3)
        d, pi = tree_h.query(hy_xyz)
        for k, idx_a in enumerate(hyd_idx):
            if d[k] < 1.30:
                parent_of_hyd[idx_a] = pi[k]
                nh_on_heavy[pi[k]] += 1

    # Detect doubly-protonated histidines (HIP: intra-ring ND1 & NE2 within 2.65 A both carry >=1 H/D)
    his_nd1 = []
    his_ne2 = []
    for hi_local, idx_a in enumerate(heavy_idx):
        rseq, rname, aname, _, pos = atoms_list[idx_a]
        if rname in ('HIS', 'HID', 'HIE', 'HIP') and nh_on_heavy[hi_local] >= 1:
            if aname == 'ND1':
                his_nd1.append((hi_local, rseq, pos))
            elif aname == 'NE2':
                his_ne2.append((hi_local, rseq, pos))
    hip_heavy_locals = set()
    for hi1, rseq1, p1 in his_nd1:
        for hi2, rseq2, p2 in his_ne2:
            if rseq1 == rseq2:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dz = p1[2] - p2[2]
                if dx * dx + dy * dy + dz * dz < 2.65 * 2.65:
                    hip_heavy_locals.add(hi1)
                    hip_heavy_locals.add(hi2)

    has_any_h = len(hyd_idx) > 0
    heavy_is_cationic = np.zeros(len(heavy_idx), dtype=bool)
    xyz_out, typ_out = [], []

    # Classify heavy atoms first
    for hi_local, idx_a in enumerate(heavy_idx):
        _, rname, aname, el_raw, pos = atoms_list[idx_a]
        el = el_raw.upper()
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
                t = 1  # O_neg
            elif (nh >= 1
                  or (rname == 'SER' and aname == 'OG')
                  or (rname == 'THR' and aname == 'OG1')
                  or (rname == 'TYR' and aname == 'OH')):
                t = 2  # O_hyd
            else:
                t = 0  # O_carb
        elif el == 'N':
            if has_any_h:
                t = 2 if (nh == 0 and rname in ('HIS', 'HID', 'HIE', 'HIP') and aname in ('ND1', 'NE2')) else 3
            else:
                t = 2 if (rname in ('HIS', 'HID') and aname == 'NE2') or (rname == 'HIE' and aname == 'ND1') else 3
        elif el in ('S', 'SE'):
            t = 9      # S_sul
        elif el in _COMPACT_IONS:
            t = 1      # O_neg (2.2-2.8 A hydration shell)
        elif el in _LARGE_POLAR_IONS:
            t = 3      # N_don (3.1-3.5 A hydration shell, polar tether)
        elif el in ('P', 'AS'):
            t = 6      # C_sp2
        else:
            t = 6 if (aname == 'C' or (rname, aname) in SP2_CARBONS) else 7
        xyz_out.append(pos)
        typ_out.append(t)

    # Classify hydrogen / deuterium atoms
    for idx_a in hyd_idx:
        _, _, _, _, pos = atoms_list[idx_a]
        p_loc = parent_of_hyd.get(idx_a, None)
        if p_loc is not None:
            p_el = atoms_list[heavy_idx[p_loc]][3].upper()
            if p_el in ('N', 'O'):
                t = 5 if heavy_is_cationic[p_loc] else 4
            else:
                t = 8
        else:
            t = 8
        xyz_out.append(pos)
        typ_out.append(t)

    return np.array(xyz_out, dtype=np.float64).reshape(-1, 3), np.array(typ_out, dtype=np.int32)


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


def load_ensemble(path_or_dir, nconf=32, two_type_heavy_only=False, mode='auto', return_weights=False):
    """Load either a directory of member PDBs (m*.pdb) or a multi-altloc PDB/mmCIF.

    mode:
      - 'auto': if active heavy-atom altlocs are a subset of {'A', 'B'}, build exact letter conformers
                weighted by mean heavy-atom occupancy; otherwise build nconf geometry-reconnected
                complete conformers ('reconnect').
      - 'letters': one conformer per active altloc letter (blank + letter), weighted by occupancy.
      - 'reconnect' (or 'reconnect_flex1..3'; alias 'geometry'): nconf geometry-reconnected complete
                conformers via surfnturf.reconnect.reconnect_pdb_to_conformers (equal weights 1/nconf).
      - 'quantiles': nconf alphabetical occupancy-quantile conformers (H-only ghost altlocs dropped, equal weights).
    """
    from .reconnect import reconnect_pdb_to_conformers
    mode = _validate_ensemble_mode(mode)

    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, 'm*.pdb')))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_dir, '*.pdb')))
        if not files:
            raise ValueError(f"No .pdb files found in ensemble directory {path_or_dir!r}")
        members = []
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
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist()))
            members.append(classify_conformer_atoms(al, two_type_heavy_only=two_type_heavy_only))
        wts = np.full(len(members), 1.0 / max(len(members), 1), dtype=np.float64)
        return (members, wts) if return_weights else members

    st = gemmi.read_structure(path_or_dir)
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
                        al.append((r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist()))
        members = [classify_conformer_atoms(al, two_type_heavy_only=two_type_heavy_only)]
        wts = np.array([1.0], dtype=np.float64)
        return (members, wts) if return_weights else members

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
                    rec = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist())
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
        members = [classify_conformer_atoms(mem_lists[L], two_type_heavy_only=two_type_heavy_only) for L in letters]
        wts = np.array([np.mean(occ_lists[L]) if occ_lists[L] else 1.0 for L in letters], dtype=np.float64)
        wts = wts / wts.sum()
        return (members, wts) if return_weights else members

    if mode == 'auto' or mode.startswith(('reconnect', 'geometry')):
        occ_flex = 0
        for pfx in ('reconnect_flex', 'geometry_flex'):
            if mode.startswith(pfx):
                occ_flex = int(mode[len(pfx):] or '0')
        _, confs, _ = reconnect_pdb_to_conformers(
            path_or_dir, nconf=nconf, occ_flex=occ_flex, quotas='hamilton', rebuild_h=True, add_missing_h=True
        )
        out_m = [
            classify_conformer_atoms(
                [(a['seqnum'], a['resname'], a['name'], a['elem'], list(a['pos'])) for a in c],
                two_type_heavy_only=two_type_heavy_only
            )
            for c in confs
        ]
        wts = np.full(len(out_m), 1.0 / len(out_m), dtype=np.float64)
        return (out_m, wts) if return_weights else out_m

    quantiles = (np.arange(nconf, dtype=np.float64) + 0.5) / nconf
    members = [[] for _ in range(nconf)]
    for ch in st[0]:
        for r in ch:
            if r.name in ('HOH', 'WAT', 'DOD'):
                continue
            blank_atoms = []
            blank_hv = False
            any_alt_hv = False
            by_alt_raw = {}
            for a in r:
                rec = (r.seqid.num, r.name, a.name, a.element.name, a.pos.tolist(), a.occ)
                if a.altloc == '\0' or not a.altloc.strip():
                    if a.occ > 0:
                        blank_atoms.append(rec[:5])
                        if a.element.name not in ('H', 'D'):
                            blank_hv = True
                else:
                    if a.element.name not in ('H', 'D'):
                        any_alt_hv = True
                    by_alt_raw.setdefault(a.altloc.strip(), []).append(rec)
            by_alt = {}
            hv_weights = {}
            for L, recs in by_alt_raw.items():
                hv_occs = [x[5] for x in recs if x[3] not in ('H', 'D') and x[5] > 0]
                if hv_occs:
                    by_alt[L] = [x[:5] for x in recs if x[5] > 0 or x[3] in ('H', 'D')]
                    hv_weights[L] = float(np.mean(hv_occs))
                elif blank_hv and not any_alt_hv:
                    h_occs = [x[5] for x in recs if x[5] > 0]
                    if h_occs:
                        by_alt[L] = [x[:5] for x in recs if x[5] > 0]
                        hv_weights[L] = float(np.mean(h_occs))
            if not by_alt:
                for m in range(nconf):
                    members[m].extend(blank_atoms)
                continue
            letters = sorted(by_alt.keys())
            weights = np.array([hv_weights[L] for L in letters], dtype=np.float64)
            weights = weights / weights.sum()
            cum = np.cumsum(weights)
            for m, q in enumerate(quantiles):
                members[m].extend(blank_atoms)
                k = min(int(np.searchsorted(cum, q, side='right')), len(letters) - 1)
                members[m].extend(by_alt[letters[k]])
    out_m = [classify_conformer_atoms(al, two_type_heavy_only=two_type_heavy_only) for al in members]
    wts = np.full(nconf, 1.0 / nconf, dtype=np.float64)
    return (out_m, wts) if return_weights else out_m


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


def build_asu_features(grid: ASUGrid, members, n_types=10, include_hpol=True):
    """Precompute per-member cubic B-spline tensors on the ASU grid.

    Returns:
      Phi_O: float32 array (M, N_asu, n_types * N_KNOTS)
      Phi_H: float32 array (M, N_asu, len(ACCEPTOR_TYPES) * N_HKNOTS) or None
      steric_H: float32 array (M, N_asu) smooth steric gate preventing water-H inside any atom (<1.50 A)
    """
    M = len(members)
    n_asu = grid.n_asu
    n_feat = n_types * N_KNOTS
    Phi_O = np.zeros((M, n_asu, n_feat), dtype=np.float32)
    Phi_H = np.zeros((M, n_asu, len(ACCEPTOR_TYPES) * N_HKNOTS), dtype=np.float32) if include_hpol else None
    steric_H = np.ones((M, n_asu), dtype=np.float32) if include_hpol else None

    pts = grid.asu_xyz
    tree_grid = cKDTree(pts)
    for m, (xyz, typ) in enumerate(members):
        if len(xyz) == 0:
            continue
        SX, Si = grid.sym_expand(xyz, cutoff_A=R_CUT)
        if len(SX) == 0:
            continue
        St = typ[Si]
        nb = tree_grid.query_ball_point(SX, R_CUT)
        lens = np.array([len(x) for x in nb], dtype=np.int32)
        if lens.sum() == 0:
            continue
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
            is_h_atom = (ti == 4) | (ti == 5) | (ti == 8)
            r_clash = np.where(is_h_atom, 1.25, 1.55)
            close = d < (r_clash + 0.80)
            if np.any(close):
                l_excl = -np.logaddexp(0.0, -(d[close] - r_clash[close]) / 0.12)
                steric_H[m] = np.exp(np.bincount(pi[close], weights=l_excl, minlength=n_asu)).astype(np.float32)

            acc_mask = ((ti == 0) | (ti == 1) | (ti == 2)) & (d < RH_CUT)
            if np.any(acc_mask):
                pi_a = pi[acc_mask]
                d_a = d[acc_mask]
                ti_a = ti[acc_mask]
                nhf = len(ACCEPTOR_TYPES) * N_HKNOTS
                buf_h = np.zeros(n_asu * nhf, dtype=np.float64)
                for kk, bw in _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS):
                    valid = kk < N_HKNOTS
                    if np.any(valid):
                        flat_col = ti_a[valid] * N_HKNOTS + kk[valid]
                        flat_idx = pi_a[valid].astype(np.int64) * nhf + flat_col
                        buf_h += np.bincount(flat_idx, weights=bw[valid], minlength=n_asu * nhf)
                Phi_H[m] = buf_h.reshape(n_asu, nhf).astype(np.float32)

    return Phi_O, Phi_H, steric_H


def _partition_common_and_diff_members(members):
    """Split `members = [(xyz, typ), ...]` into `(common_xyz, common_typ), diff_members` for M > 1 speedup."""
    M = len(members)
    if M <= 1:
        return (np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32)), members

    from collections import Counter
    key_lists = []
    common_counts = None
    for xyz, typ in members:
        q = np.rint(xyz * 10000.0).astype(np.int64)
        ks = [(int(q[i, 0]), int(q[i, 1]), int(q[i, 2]), int(typ[i])) for i in range(len(typ))]
        key_lists.append(ks)
        cnt = Counter(ks)
        common_counts = cnt if common_counts is None else (common_counts & cnt)

    if not common_counts:
        return (np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32)), members

    rem0 = Counter(common_counts)
    com_xyz, com_typ = [], []
    xyz0, typ0 = members[0]
    for i, k in enumerate(key_lists[0]):
        if rem0[k] > 0:
            rem0[k] -= 1
            com_xyz.append(xyz0[i])
            com_typ.append(typ0[i])

    diff_members = []
    for (xyz, typ), ks in zip(members, key_lists):
        rem = Counter(common_counts)
        d_idx = []
        for i, k in enumerate(ks):
            if rem[k] > 0:
                rem[k] -= 1
            else:
                d_idx.append(i)
        if d_idx:
            idx_arr = np.asarray(d_idx, dtype=np.int32)
            diff_members.append((xyz[idx_arr], typ[idx_arr]))
        else:
            diff_members.append((np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int32)))

    return (np.asarray(com_xyz, dtype=np.float64), np.asarray(com_typ, dtype=np.int32)), diff_members


def forward_direct(spec: 'WaterFieldSpec', grid: ASUGrid, members, member_weights, theta, B_map: float = 5.0,
                   return_per_member: bool = False):
    """Memory-lean, zero-feature-tensor forward evaluation `(g_hat_asu, gO_bar_asu)` for inference (`surfnturf predict`).

    Automatically factors atoms shared identically across all `M` conformers so single-conformer core regions are
    evaluated once rather than `M` times (~10x additional speedup on multi-altloc structures).
    """
    w_pmf, logit_alpha, v_oz, w_hpol, _ = spec.unpack(theta)
    M = len(members)
    n_asu = grid.n_asu
    mw = np.asarray(member_weights, dtype=np.float64)
    mw = mw / mw.sum()

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

    pts = grid.asu_xyz
    tree_grid = cKDTree(pts)

    def _eval_atom_subset(xyz_sub, typ_sub):
        u_vec = np.zeros(n_asu, dtype=np.float64)
        lst_vec = np.zeros(n_asu, dtype=np.float64) if include_hpol else None
        rh_vec = np.zeros(n_asu, dtype=np.float64) if include_hpol else None
        if len(xyz_sub) == 0:
            return u_vec, lst_vec, rh_vec
        SX, Si = grid.sym_expand(xyz_sub, cutoff_A=R_CUT)
        if len(SX) == 0:
            return u_vec, lst_vec, rh_vec
        St = typ_sub[Si]
        nb = tree_grid.query_ball_point(SX, R_CUT)
        lens = np.array([len(x) for x in nb], dtype=np.int32)
        if lens.sum() == 0:
            return u_vec, lst_vec, rh_vec
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
            is_h_atom = (ti == 4) | (ti == 5) | (ti == 8)
            r_clash = np.where(is_h_atom, 1.25, 1.55)
            close = d < (r_clash + 0.80)
            if np.any(close):
                l_excl = -np.logaddexp(0.0, -(d[close] - r_clash[close]) / 0.12)
                lst_vec = np.bincount(pi[close], weights=l_excl, minlength=n_asu)
            acc_mask = ((ti == 0) | (ti == 1) | (ti == 2)) & (d < RH_CUT)
            if np.any(acc_mask):
                pi_a = pi[acc_mask]
                d_a = d[acc_mask]
                ti_a = ti[acc_mask]
                (hk0, hb0), (hk1, hb1), (hk2, hb2), (hk3, hb3) = _bspline_weights(d_a, RH_MIN, DRH, N_HKNOTS)
                h_pair = (
                    W_h_pad[ti_a, hk0] * hb0
                    + W_h_pad[ti_a, hk1] * hb1
                    + W_h_pad[ti_a, hk2] * hb2
                    + W_h_pad[ti_a, hk3] * hb3
                )
                rh_vec = np.bincount(pi_a, weights=h_pair, minlength=n_asu)
        return u_vec, lst_vec, rh_vec

    (com_xyz, com_typ), diff_members = _partition_common_and_diff_members(members)
    u_com, lst_com, rh_com = _eval_atom_subset(com_xyz, com_typ)

    U0 = np.empty((M, n_asu), dtype=np.float64)
    for m, (xyz_d, typ_d) in enumerate(diff_members):
        u_d, lst_d, rh_d = _eval_atom_subset(xyz_d, typ_d)
        U0[m] = u_com + u_d
        if include_hpol:
            hm = np.exp(lst_com + lst_d) * np.tanh(rh_com + rh_d)
            if return_per_member:
                gH_m[m] = hm
            gH_bar += mw[m] * hm

    if spec.use_langmuir:
        alpha = float(expit(logit_alpha))
        z0 = float(logit_alpha)
        g1_m = expit(z0 - U0) / alpha
    else:
        alpha = 0.0
        z0 = 0.0
        g1_m = np.exp(-np.clip(U0, -4.5, 40.0))

    g1_bar = mw @ g1_m

    if spec.use_oz:
        xi_asu = softplus(g1_bar - 1.0, beta=4.0)
        xi_3d = grid.asu_to_full_3d(xi_asu)
        F_xi = np.fft.rfftn(xi_3d, axes=(0, 1, 2))
        psi_xi_3d = np.empty((spec.n_oz,) + grid.shape, dtype=np.float64)
        for q in range(spec.n_oz):
            psi_xi_3d[q] = np.fft.irfftn(grid.oz_kernels[q] * F_xi, s=grid.shape, axes=(0, 1, 2))
        dU_ww_3d = np.tensordot(v_oz, psi_xi_3d, axes=(0, 0))
        dU_ww_asu = grid.full_3d_to_asu_mean(dU_ww_3d)
        U_tot = U0 + dU_ww_asu[None, :]
        if spec.use_langmuir:
            gO_m = expit(z0 - U_tot) / alpha
        else:
            gO_m = np.exp(-np.clip(U_tot, -4.5, 40.0))
        gO_bar = mw @ gO_m
    else:
        gO_m = g1_m
        gO_bar = g1_bar

    if spec.use_blur:
        Kw, Kh = grid.water_kernel(B_map)
        gO_3d = grid.asu_to_full_3d(gO_bar)
        F_elec = Kw * np.fft.rfftn(grid.n_bulk * gO_3d, axes=(0, 1, 2))
        if gH_bar is not None:
            gH_3d = grid.asu_to_full_3d(gH_bar)
            F_elec = F_elec + Kh * np.fft.rfftn(grid.n_bulk * gH_3d, axes=(0, 1, 2))
        rho_3d = np.fft.irfftn(F_elec, s=grid.shape, axes=(0, 1, 2))
        g_hat = grid.full_3d_to_asu_mean(rho_3d) / RHO_BULK
    else:
        g_hat = gO_bar

    if return_per_member:
        return g_hat, gO_bar, gO_m, gH_m
    return g_hat, gO_bar


class DatasetBundle:
    """Holds one MD dataset's ASU target, pre-flattened float64 BLAS feature matrices, and rendering kernel."""

    def __init__(self, name, grid: ASUGrid, rho_full, B_map, Phi_O, Phi_H=None, steric_H=None,
                 mask_asu=None, member_weights=None):
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
        self.g_full = (rho_full / RHO_BULK).astype(np.float64)
        self.g_asu = grid.full_3d_to_asu_mean(self.g_full)
        self.weights_asu = grid.orbit_mult.copy()
        if mask_asu is not None:
            self.weights_asu = self.weights_asu * mask_asu.astype(np.float64)
        self.w_sum = float(self.weights_asu.sum())


class WaterFieldSpec:
    """Configuration & parameter packing for the differentiable hydration field."""

    def __init__(self, n_types=10, use_langmuir=True, use_oz=True, use_hpol=True, use_blur=True, fit_box_scale=False, n_datasets=1):
        self.n_types = int(n_types)
        self.use_langmuir = bool(use_langmuir)
        self.use_oz = bool(use_oz)
        self.use_hpol = bool(use_hpol)
        self.use_blur = bool(use_blur)
        self.fit_box_scale = bool(fit_box_scale)
        self.n_datasets = int(n_datasets)

        self.n_pmf = self.n_types * N_KNOTS
        self.n_oz = len(OZ_RADII) if self.use_oz else 0
        self.n_hpol = (len(ACCEPTOR_TYPES) * N_HKNOTS) if self.use_hpol else 0
        self.n_alpha = 1 if self.use_langmuir else 0
        self.n_scale = self.n_datasets if self.fit_box_scale else 0
        self.n_params = self.n_pmf + self.n_alpha + self.n_oz + self.n_hpol + self.n_scale

    def init_params(self):
        w_pmf = np.zeros((self.n_types, N_KNOTS), dtype=np.float64)
        knots = R_MIN + DR * np.arange(N_KNOTS)
        for t in range(self.n_types):
            if self.n_types == 10 and t in (4, 5, 8):
                r0 = 1.75 if t in (4, 5) else 2.25
            elif self.n_types == 10 and t == 9:
                r0 = 3.25
            elif (self.n_types == 10 and t in (0, 1, 2, 3)) or (self.n_types == 2 and t == 0):
                r0 = 2.65
            else:
                r0 = 3.05
            w_pmf[t] = 4.0 * expit((r0 - knots) / 0.22)
        parts = [w_pmf.ravel()]
        if self.use_langmuir:
            parts.append(np.array([-3.0]))
        if self.use_oz:
            parts.append(np.zeros(self.n_oz))
        if self.use_hpol:
            parts.append(np.zeros(self.n_hpol))
        if self.fit_box_scale:
            parts.append(np.zeros(self.n_datasets))
        return np.concatenate(parts)

    def unpack(self, theta):
        idx = 0
        w_pmf = theta[idx:idx + self.n_pmf]; idx += self.n_pmf
        logit_alpha = theta[idx] if self.use_langmuir else None; idx += self.n_alpha
        v_oz = theta[idx:idx + self.n_oz] if self.use_oz else None; idx += self.n_oz
        w_hpol = theta[idx:idx + self.n_hpol] if self.use_hpol else None; idx += self.n_hpol
        log_k = theta[idx:idx + self.n_scale] if self.fit_box_scale else None
        return w_pmf, logit_alpha, v_oz, w_hpol, log_k


def forward_and_grad(spec: WaterFieldSpec, ds: DatasetBundle, theta, ds_index=0,
                     loss_mode='l2', compute_grad=True, ext_dL_dghat=None, return_grad_B=False):
    """Evaluate predicted g_hat on ASU (and loss + exact analytical gradient w.r.t. theta and optionally B_map)."""
    w_pmf, logit_alpha, v_oz, w_hpol, log_k = spec.unpack(theta)
    grid = ds.grid
    M = ds.M
    mw = ds.mw

    U0 = (ds.Phi_O_flat @ w_pmf).reshape(M, grid.n_asu)

    if spec.use_langmuir:
        alpha = float(expit(logit_alpha))
        z0 = logit_alpha
        sig1 = expit(z0 - U0)
        g1_m = sig1 / alpha
    else:
        alpha = 0.0
        z0 = 0.0
        U0_clip = np.clip(U0, -4.5, 40.0)
        g1_m = np.exp(-U0_clip)

    g1_bar = mw @ g1_m

    if spec.use_oz:
        xi_asu = softplus(g1_bar - 1.0, beta=4.0)
        xi_3d = grid.asu_to_full_3d(xi_asu)
        F_xi = np.fft.rfftn(xi_3d, axes=(0, 1, 2))
        psi_xi_3d = np.empty((spec.n_oz,) + grid.shape, dtype=np.float64)
        for q in range(spec.n_oz):
            psi_xi_3d[q] = np.fft.irfftn(grid.oz_kernels[q] * F_xi, s=grid.shape, axes=(0, 1, 2))
        dU_ww_3d = np.tensordot(v_oz, psi_xi_3d, axes=(0, 0))
        dU_ww_asu = grid.full_3d_to_asu_mean(dU_ww_3d)
        U_tot = U0 + dU_ww_asu[None, :]
        if spec.use_langmuir:
            sig2 = expit(z0 - U_tot)
            gO_m = sig2 / alpha
        else:
            U_clip = np.clip(U_tot, -4.5, 40.0)
            gO_m = np.exp(-U_clip)
        gO_bar = mw @ gO_m
    else:
        U_tot = U0
        gO_m = g1_m
        gO_bar = g1_bar

    if spec.use_hpol and ds.Phi_H_flat is not None:
        raw_h = (ds.Phi_H_flat @ w_hpol).reshape(M, grid.n_asu)
        th_h = np.tanh(raw_h)
        gH_m = ds.steric_H_64 * th_h
        gH_bar = mw @ gH_m
    else:
        th_h = None
        gH_bar = None

    scale_k = float(np.exp(log_k[ds_index])) if spec.fit_box_scale else 1.0
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

    grad_theta = np.zeros_like(theta)
    if spec.fit_box_scale:
        idx_sc = spec.n_pmf + spec.n_alpha + spec.n_oz + spec.n_hpol + ds_index
        grad_theta[idx_sc] = float(np.sum(dL_dghat * g_hat))

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

    if spec.use_hpol and dL_dgH_bar is not None and ds.Phi_H_flat is not None:
        dL_draw = np.outer(mw, dL_dgH_bar) * ds.steric_H_64 * (1.0 - th_h ** 2)
        grad_wh = dL_draw.ravel() @ ds.Phi_H_flat
        i0 = spec.n_pmf + spec.n_alpha + spec.n_oz
        grad_theta[i0:i0 + spec.n_hpol] = grad_wh

    def _dg_dU_and_dlogit(g_arr):
        if spec.use_langmuir:
            dg_dU = -g_arr * (1.0 - alpha * g_arr)
            dg_dz0 = alpha * g_arr * (1.0 - g_arr)
            return dg_dU, dg_dz0
        else:
            in_range = (U_tot > -4.5) & (U_tot < 40.0) if (g_arr is gO_m) else ((U0 > -4.5) & (U0 < 40.0))
            return np.where(in_range, -g_arr, 0.0), None

    dg2_dU, dg2_dz0 = _dg_dU_and_dlogit(gO_m)
    mw_gO = np.outer(mw, dL_dgO_bar)
    dL_dUtot = mw_gO * dg2_dU
    grad_logit_alpha = float(np.sum(mw_gO * dg2_dz0)) if spec.use_langmuir else 0.0

    if spec.use_oz:
        dL_ddU_asu = dL_dUtot.sum(axis=0)
        dL_ddU_3d = grid.asu_to_full_3d(dL_ddU_asu / grid.orbit_mult)
        grad_voz = np.array([np.sum(dL_ddU_3d * psi_xi_3d[q]) for q in range(spec.n_oz)], dtype=np.float64)
        i_oz = spec.n_pmf + spec.n_alpha
        grad_theta[i_oz:i_oz + spec.n_oz] = grad_voz

        W_hat = np.tensordot(v_oz, grid.oz_kernels, axes=(0, 0))
        dL_dxi_3d = np.fft.irfftn(W_hat * np.fft.rfftn(dL_ddU_3d, axes=(0, 1, 2)), s=grid.shape, axes=(0, 1, 2))
        dL_dxi_asu = grid.full_3d_to_asu_grad(dL_dxi_3d)
        dL_dg1_bar = dL_dxi_asu * expit(4.0 * (g1_bar - 1.0))

        dg1_dU0, dg1_dz0 = _dg_dU_and_dlogit(g1_m)
        mw_g1 = np.outer(mw, dL_dg1_bar)
        dL_dU0 = dL_dUtot + mw_g1 * dg1_dU0
        if spec.use_langmuir:
            grad_logit_alpha += float(np.sum(mw_g1 * dg1_dz0))
    else:
        dL_dU0 = dL_dUtot

    if spec.use_langmuir:
        grad_theta[spec.n_pmf] = grad_logit_alpha

    grad_theta[:spec.n_pmf] = dL_dU0.ravel() @ ds.Phi_O_flat

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


def smoothness_penalty_and_grad(spec: WaterFieldSpec, theta, lam_pmf=2e-3, lam_hpol=2e-3, lam_oz=1e-4):
    """Second-difference spline curvature regularization (including pinned-0 boundary knot at R_CUT)."""
    w_pmf, _, v_oz, w_hpol, _ = spec.unpack(theta)
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
    grad[:spec.n_pmf] = gW[:, :N_KNOTS].ravel()

    if spec.use_oz and v_oz is not None:
        pen += 0.5 * lam_oz * float(np.sum(v_oz ** 2))
        i_oz = spec.n_pmf + spec.n_alpha
        grad[i_oz:i_oz + spec.n_oz] = lam_oz * v_oz

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
        i0 = spec.n_pmf + spec.n_alpha + spec.n_oz
        grad[i0:i0 + spec.n_hpol] = gH[:, :N_HKNOTS].ravel()

    return pen, grad


def param_bounds(spec: WaterFieldSpec):
    """Physical box bounds for L-BFGS-B."""
    bounds = [(-6.0, 25.0)] * spec.n_pmf
    if spec.use_langmuir:
        bounds.append((-6.0, -0.8))
    if spec.use_oz:
        bounds.extend([(-5.0, 5.0)] * spec.n_oz)
    if spec.use_hpol:
        bounds.extend([(-4.0, 4.0)] * spec.n_hpol)
    if spec.fit_box_scale:
        bounds.extend([(-0.35, 0.35)] * spec.n_scale)
    return bounds
