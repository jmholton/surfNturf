"""Standalone crystallographic riding-hydrogen completion for surfNturf solute conformers.

Many X-ray PDB depositions omit hydrogen atoms entirely or deposit only a subset of hydrogens,
whereas surfNturf's 10-class hydration PMF+OZ model resolves directional N-H...O_w and O-H...O_w
hydrogen-bond donor peaks via explicit `H_pol`, `H_pos`, and `H_apol` spline channels.

This module reconstructs missing riding hydrogens across standard protein and nucleic-acid
residues in pure Python/NumPy (< 5 ms per conformer, zero external binary dependencies) while
leaving all existing H/D coordinates untouched and never modifying the atomic model used for F_calc.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "MSE",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

_STD_NA = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "ADE", "CYT", "GUA", "URI", "THY"}

_METALS = {
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE",
    "CO", "NI", "CU", "ZN", "GA", "RB", "SR", "Y", "ZR", "NB", "MO", "RU", "RH",
    "PD", "AG", "CD", "IN", "SN", "CS", "BA", "LA", "HG", "PB",
}

# Nitrogens that always carry H in standard residues (donors, never H-bond acceptors for OH/HIS scoring)
_DONOR_N = {
    "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "TRP": {"NE1"},
    "A": {"N6"}, "DA": {"N6"}, "ADE": {"N6"}, "C": {"N4"}, "DC": {"N4"}, "CYT": {"N4"},
    "G": {"N1", "N2"}, "DG": {"N1", "N2"}, "GUA": {"N1", "N2"},
    "U": {"N3"}, "URI": {"N3"}, "DT": {"N3"}, "THY": {"N3"},
}


def _is_donor_n(rname, aname):
    """True for backbone amide N and side-chain/base N that carry H in every standard protonation state."""
    return (aname == "N" and rname in _STD_AA) or aname in _DONOR_N.get(rname, ())


# Standard X-ray electron-cloud riding bond lengths (matching phenix.reduce / cctbx defaults)
D_NH_SP2 = 0.86
D_NH_SP3 = 0.89
D_OH = 0.84
D_SH = 1.20
D_CH_AR = 0.93
D_CH_1 = 0.98
D_CH_2 = 0.97
D_CH_3 = 0.96

_C_TET = 1.0 / math.sqrt(3.0)
_S_TET = math.sqrt(2.0 / 3.0)
_COS_TET = -1.0 / 3.0
_SIN_TET = 2.0 * math.sqrt(2.0) / 3.0
_SIN_120 = 0.5 * math.sqrt(3.0)

# Sidechain CH2 (methylene) definitions: resname -> list of (C_atom, nbr1, nbr2, (h_name1, h_name2))
_SIDECHAIN_CH2 = {
    "SER": [("CB", "CA", "OG", ("HB2", "HB3"))],
    "CYS": [("CB", "CA", "SG", ("HB2", "HB3"))],
    "CYX": [("CB", "CA", "SG", ("HB2", "HB3"))],
    "ASP": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "ASN": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "HIS": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "HID": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "HIE": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "HIP": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "PHE": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "TYR": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "TRP": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "LEU": [("CB", "CA", "CG", ("HB2", "HB3"))],
    "ILE": [("CG1", "CB", "CD1", ("HG12", "HG13"))],
    "GLU": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "CD", ("HG2", "HG3"))],
    "GLN": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "CD", ("HG2", "HG3"))],
    "MET": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "SD", ("HG2", "HG3"))],
    "MSE": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "SE", ("HG2", "HG3"))],
    "PRO": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "CD", ("HG2", "HG3")), ("CD", "CG", "N", ("HD2", "HD3"))],
    "ARG": [("CB", "CA", "CG", ("HB2", "HB3")), ("CG", "CB", "CD", ("HG2", "HG3")), ("CD", "CG", "NE", ("HD2", "HD3"))],
    "LYS": [
        ("CB", "CA", "CG", ("HB2", "HB3")),
        ("CG", "CB", "CD", ("HG2", "HG3")),
        ("CD", "CG", "CE", ("HD2", "HD3")),
        ("CE", "CD", "NZ", ("HE2", "HE3")),
    ],
}

# Sidechain CH3 (methyl) definitions: resname -> list of (C_atom, parent_atom, stagger_ref_atom, (h1, h2, h3))
_SIDECHAIN_CH3 = {
    "ALA": [("CB", "CA", "N", ("HB1", "HB2", "HB3"))],
    "THR": [("CG2", "CB", "CA", ("HG21", "HG22", "HG23"))],
    "VAL": [
        ("CG1", "CB", "CA", ("HG11", "HG12", "HG13")),
        ("CG2", "CB", "CA", ("HG21", "HG22", "HG23")),
    ],
    "LEU": [
        ("CD1", "CG", "CB", ("HD11", "HD12", "HD13")),
        ("CD2", "CG", "CB", ("HD21", "HD22", "HD23")),
    ],
    "ILE": [
        ("CG2", "CB", "CA", ("HG21", "HG22", "HG23")),
        ("CD1", "CG1", "CB", ("HD11", "HD12", "HD13")),
    ],
    "MET": [("CE", "SD", "CG", ("HE1", "HE2", "HE3"))],
    "MSE": [("CE", "SE", "CG", ("HE1", "HE2", "HE3"))],
}

# Sidechain aromatic C-H definitions: resname -> list of (C_atom, nbr1, nbr2, h_name)
_SIDECHAIN_AR_CH = {
    "PHE": [
        ("CD1", "CG", "CE1", "HD1"), ("CD2", "CG", "CE2", "HD2"),
        ("CE1", "CD1", "CZ", "HE1"), ("CE2", "CD2", "CZ", "HE2"),
        ("CZ", "CE1", "CE2", "HZ"),
    ],
    "TYR": [
        ("CD1", "CG", "CE1", "HD1"), ("CD2", "CG", "CE2", "HD2"),
        ("CE1", "CD1", "CZ", "HE1"), ("CE2", "CD2", "CZ", "HE2"),
    ],
    "TRP": [
        ("CD1", "CG", "NE1", "HD1"), ("CE3", "CD2", "CZ3", "HE3"),
        ("CZ2", "CE2", "CH2", "HZ2"), ("CZ3", "CE3", "CH2", "HZ3"),
        ("CH2", "CZ2", "CZ3", "HH2"),
    ],
    "HIS": [("CD2", "CG", "NE2", "HD2"), ("CE1", "ND1", "NE2", "HE1")],
    "HID": [("CD2", "CG", "NE2", "HD2"), ("CE1", "ND1", "NE2", "HE1")],
    "HIE": [("CD2", "CG", "NE2", "HD2"), ("CE1", "ND1", "NE2", "HE1")],
    "HIP": [("CD2", "CG", "NE2", "HD2"), ("CE1", "ND1", "NE2", "HE1")],
}


def _uvec(v):
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-8 else None


def _orth_unit(axis_u):
    cand = np.array([1.0, 0.0, 0.0]) if abs(axis_u[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    p = cand - float(np.dot(cand, axis_u)) * axis_u
    return _uvec(p)


def _bonded(p1, p2, max_d=1.85):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return (dx * dx + dy * dy + dz * dz) <= (max_d * max_d)


def _sp2_bisector(center, nbr1, nbr2, d):
    if not (_bonded(center, nbr1, 1.90) and _bonded(center, nbr2, 1.90)):
        return None
    u1 = _uvec(nbr1 - center)
    u2 = _uvec(nbr2 - center)
    if u1 is None or u2 is None:
        return None
    b = _uvec(-(u1 + u2))
    return (center + d * b) if b is not None else None


def _sp2_nh2(n_pos, c_pos, plane_ref_pos, d=D_NH_SP2):
    if not (_bonded(n_pos, c_pos, 1.75) and _bonded(c_pos, plane_ref_pos, 1.75)):
        return None
    u_nc = _uvec(c_pos - n_pos)
    if u_nc is None:
        return None
    v_ref = plane_ref_pos - c_pos
    p = _uvec(v_ref - float(np.dot(v_ref, u_nc)) * u_nc)
    if p is None:
        p = _orth_unit(u_nc)
    h1 = n_pos + d * (-0.5 * u_nc + _SIN_120 * p)
    h2 = n_pos + d * (-0.5 * u_nc - _SIN_120 * p)
    return h1, h2


def _sp3_methine(c_pos, n1, n2, n3, d=D_CH_1):
    if not (_bonded(c_pos, n1, 1.85) and _bonded(c_pos, n2, 1.85) and _bonded(c_pos, n3, 1.85)):
        return None
    u1, u2, u3 = _uvec(n1 - c_pos), _uvec(n2 - c_pos), _uvec(n3 - c_pos)
    if u1 is None or u2 is None or u3 is None:
        return None
    b = _uvec(-(u1 + u2 + u3))
    return (c_pos + d * b) if b is not None else None


def _sp3_methylene(c_pos, n1, n2, d=D_CH_2):
    if not (_bonded(c_pos, n1, 1.95) and _bonded(c_pos, n2, 1.95)):
        return None
    u1, u2 = _uvec(n1 - c_pos), _uvec(n2 - c_pos)
    if u1 is None or u2 is None:
        return None
    b = _uvec(-(u1 + u2))
    w = _uvec(np.cross(u1, u2))
    if b is None or w is None:
        return None
    return (
        c_pos + d * (_C_TET * b + _S_TET * w),
        c_pos + d * (_C_TET * b - _S_TET * w),
    )


def _sp3_rotor3(x_pos, parent_pos, stagger_ref_pos, d, max_bond=2.05):
    if not _bonded(x_pos, parent_pos, max_bond):
        return None
    a = _uvec(parent_pos - x_pos)
    if a is None:
        return None
    p1 = None
    if stagger_ref_pos is not None and _bonded(parent_pos, stagger_ref_pos, 2.05):
        v_ref = stagger_ref_pos - parent_pos
        p1 = _uvec(-(v_ref - float(np.dot(v_ref, a)) * a))
    if p1 is None:
        p1 = _orth_unit(a)
    p2 = np.cross(a, p1)
    out = []
    for phi_deg in (0.0, 120.0, 240.0):
        ph = math.radians(phi_deg)
        u = _COS_TET * a + _SIN_TET * (math.cos(ph) * p1 + math.sin(ph) * p2)
        out.append(x_pos + d * u)
    return out


def _best_rotatable_xh(x_pos, parent_pos, stagger_ref_pos, d, angle_deg, acc_tree, acc_xyz, hv_tree, planar_bias=False):
    if not _bonded(x_pos, parent_pos, 2.05):
        return None
    a = _uvec(parent_pos - x_pos)
    if a is None:
        return None
    p1 = None
    if stagger_ref_pos is not None and _bonded(parent_pos, stagger_ref_pos, 2.05):
        v_ref = stagger_ref_pos - parent_pos
        p1 = _uvec(-(v_ref - float(np.dot(v_ref, a)) * a))
    if p1 is None:
        p1 = _orth_unit(a)
    p2 = np.cross(a, p1)

    ang_rad = math.radians(angle_deg)
    ca = math.cos(ang_rad)  # note: `a` points X -> parent, so X-H at bond angle `theta` has dot(u_H, a) = cos(theta)
    sa = math.sin(ang_rad)

    near_acc = acc_xyz[acc_tree.query_ball_point(x_pos, 3.45)] if acc_tree is not None else np.zeros((0, 3))
    best_score = -1e9
    best_h = None
    for k in range(12):
        ph = math.radians(30.0 * k)
        u = ca * a + sa * (math.cos(ph) * p1 + math.sin(ph) * p2)
        h = x_pos + d * u
        score = 0.05 * math.cos(2.0 * ph if planar_bias else 3.0 * ph)
        if hv_tree is not None:
            d_hv, _ = hv_tree.query(h)
            if d_hv < 1.85 and float(np.linalg.norm(h - x_pos)) < d + 0.05:
                # Ignore X itself (`d_hv == d` when X is nearest, since d <= 1.20 < 1.85 -> query k=3!)
                dists_3, _ = hv_tree.query(h, k=min(4, hv_tree.n))
                dists_arr = np.atleast_1d(dists_3)
                other_min = float(np.min(dists_arr[dists_arr > d + 0.15])) if np.any(dists_arr > d + 0.15) else 9.9
                if other_min < 1.95:
                    score -= 4.0 * ((1.95 - other_min) ** 2)
        for ax in near_acc:
            if np.linalg.norm(ax - x_pos) < 1.2:
                continue
            dh = float(np.linalg.norm(ax - h))
            if 1.55 <= dh <= 2.55:
                v_ha = _uvec(ax - h)
                v_hx = _uvec(x_pos - h)
                if v_ha is not None and v_hx is not None:
                    cos_xha = float(np.dot(v_hx, v_ha))  # -1.0 is linear 180 deg H-bond
                    if cos_xha < -0.35:
                        score += (-cos_xha) * math.exp(-((dh - 1.90) / 0.35) ** 2)
        if score > best_score:
            best_score = score
            best_h = h
    return best_h


def _unpack_atom_rec(rec):
    """Extract `(rseq, rname, aname, elem, pos, biso, chain, icode)` from 5-, 6-, 7-, or 8-tuples."""
    n = len(rec)
    rseq, rname, aname, elem, pos = rec[0], rec[1], rec[2], rec[3], rec[4]
    if n == 5:
        return rseq, rname, aname, elem, pos, 15.0, "", ""
    if n == 6:
        return rseq, rname, aname, elem, pos, float(rec[5]), "", ""
    if n == 7:
        return rseq, rname, aname, elem, pos, 15.0, str(rec[5]).strip(), str(rec[6]).strip()
    return rseq, rname, aname, elem, pos, float(rec[5]), str(rec[6]).strip(), str(rec[7]).strip()


def complete_conformer_hydrogens(atoms_list, mode="auto"):
    """Add missing riding hydrogens to a single-conformer atom list (preserving tuple schema).

    Parameters
    ----------
    atoms_list : list of tuples
        Supports `(seqnum, resname, atom_name, elem, pos)` (5-tuple),
        `(seqnum, resname, atom_name, elem, pos, biso)` (6-tuple),
        `(seqnum, resname, atom_name, elem, pos, chain, icode)` (7-tuple), or
        `(seqnum, resname, atom_name, elem, pos, biso, chain, icode)` (8-tuple).
    mode : str
        `'auto'` or `'1'`/`True` completes missing standard riding hydrogens while preserving all existing H/D;
        `'0'`/`False`/`'none'` returns `atoms_list` unchanged.
    """
    if mode in (False, "0", "false", "False", "none", "None") or not atoms_list:
        return atoms_list

    rec_len = len(atoms_list[0])
    has_biso = rec_len in (6, 8) or rec_len > 8
    has_chain = rec_len >= 7
    hv_indices = []
    hy_indices = []
    n_h = 0
    n_d = 0
    for idx, rec in enumerate(atoms_list):
        el = rec[3].strip().upper()
        if el in ("H", "D"):
            hy_indices.append(idx)
            if el == "D":
                n_d += 1
            else:
                n_h += 1
        else:
            hv_indices.append(idx)

    if len(hv_indices) < 3:
        return atoms_list

    h_sym = "D" if (n_d > 0 and n_h == 0) else "H"
    hv_xyz = np.array([atoms_list[i][4] for i in hv_indices], dtype=np.float64)
    hv_tree = cKDTree(hv_xyz)

    nh_on_hv = np.zeros(len(hv_indices), dtype=np.int32)
    if hy_indices:
        hy_xyz = np.array([atoms_list[i][4] for i in hy_indices], dtype=np.float64)
        d_hy, pi_hy = hv_tree.query(hy_xyz)
        for d_val, p_i in zip(d_hy, pi_hy):
            if d_val < 1.32:
                nh_on_hv[p_i] += 1

    # Acceptor & sulfur/metal trees for rotatable OH/SH and HIS tautomer / CYS disulfide checks
    acc_pts = []
    sm_pts = []
    metal_pts = []
    hv_global_to_local = {}
    for loc_i, glob_i in enumerate(hv_indices):
        hv_global_to_local[glob_i] = loc_i
        el = atoms_list[glob_i][3].strip().upper()
        p = hv_xyz[loc_i]
        if el in ("O", "S") or (el == "N" and nh_on_hv[loc_i] == 0 and not _is_donor_n(
                atoms_list[glob_i][1].strip().upper(), atoms_list[glob_i][2].strip().upper())):
            acc_pts.append(p)
        if el in ("S", "SE") or el in _METALS:
            sm_pts.append(p)
        if el in _METALS:
            metal_pts.append(p)

    acc_xyz = np.array(acc_pts, dtype=np.float64).reshape(-1, 3)
    acc_tree = cKDTree(acc_xyz) if len(acc_xyz) else None
    sm_xyz = np.array(sm_pts, dtype=np.float64).reshape(-1, 3)
    sm_tree = cKDTree(sm_xyz) if len(sm_xyz) else None
    metal_xyz = np.array(metal_pts, dtype=np.float64).reshape(-1, 3)
    metal_tree = cKDTree(metal_xyz) if len(metal_xyz) else None

    # Partition atoms_list into ordered residue blocks keyed by (chain, seqnum, icode, resname)
    res_blocks = []
    cur_key = None
    cur_seen = set()
    cur_block = []
    for idx, rec in enumerate(atoms_list):
        rseq, rname_raw, aname_raw, _, _, _, chain_id, icode = _unpack_atom_rec(rec)
        rname = rname_raw.strip().upper()
        aname = aname_raw.strip().upper()
        key = (chain_id, rseq, icode, rname)
        if key != cur_key or aname in cur_seen:
            if cur_block:
                res_blocks.append((cur_key[0], cur_key[1], cur_key[2], cur_key[3], cur_block))
            cur_key = key
            cur_seen = {aname}
            cur_block = [idx]
        else:
            cur_seen.add(aname)
            cur_block.append(idx)
    if cur_block:
        res_blocks.append((cur_key[0], cur_key[1], cur_key[2], cur_key[3], cur_block))

    # Build per-residue heavy-atom lookup: aname -> (pos_np, hv_local_idx, biso)
    res_maps = []
    for chain_id, rseq, icode, rname, idxs in res_blocks:
        amap = {}
        for gi in idxs:
            _, _, an_raw, el_raw, _, biso, _, _ = _unpack_atom_rec(atoms_list[gi])
            if el_raw.strip().upper() in ("H", "D"):
                continue
            amap[an_raw.strip().upper()] = (hv_xyz[hv_global_to_local[gi]], hv_global_to_local[gi], biso)
        res_maps.append((chain_id, rseq, icode, rname, idxs, amap))

    def _make_rec(rseq, rname, hname, pos_arr, parent_biso, chain_id, icode):
        p_list = [float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2])]
        h_biso = min(999.0, 1.2 * float(parent_biso))
        if has_biso and has_chain:
            return (rseq, rname, hname, h_sym, p_list, h_biso, chain_id, icode)
        if has_chain:
            return (rseq, rname, hname, h_sym, p_list, chain_id, icode)
        if has_biso:
            return (rseq, rname, hname, h_sym, p_list, h_biso)
        return (rseq, rname, hname, h_sym, p_list)

    out_atoms = []
    for b_idx, (chain_id, rseq, icode, rname, idxs, amap) in enumerate(res_maps):
        for gi in idxs:
            out_atoms.append(atoms_list[gi])

        def _mrec(hname, pos_arr, parent_biso):
            return _make_rec(rseq, rname, hname, pos_arr, parent_biso, chain_id, icode)

        def _has_h(aname):
            return (aname not in amap) or (nh_on_hv[amap[aname][1]] > 0)

        def _near_metal(pos_arr, cutoff=2.45):
            return (metal_tree is not None) and (metal_tree.query(pos_arr)[0] <= cutoff)

        if rname in _STD_AA:
            # 1. Backbone N
            if "N" in amap and "CA" in amap and not _has_h("N") and not _near_metal(amap["N"][0], 2.35):
                n_pos, n_li, n_b = amap["N"]
                ca_pos = amap["CA"][0]
                same_chain = (b_idx > 0 and chain_id == res_maps[b_idx - 1][0])
                c_prev_pos = None
                if same_chain and "C" in res_maps[b_idx - 1][5]:
                    cand_c = res_maps[b_idx - 1][5]["C"][0]
                    if _bonded(cand_c, n_pos, 1.68):
                        c_prev_pos = cand_c
                if c_prev_pos is None:
                    # Check if any other heavy atom besides CA/CD is covalently bonded to N (e.g. N-cap)
                    d_nbrs, i_nbrs = hv_tree.query(n_pos, k=min(4, len(hv_xyz)))
                    extra_cov = [
                        i_n for d_n, i_n in zip(np.atleast_1d(d_nbrs), np.atleast_1d(i_nbrs))
                        if 0.4 < d_n <= 1.65 and i_n != amap["CA"][1] and (rname != "PRO" or "CD" not in amap or i_n != amap["CD"][1])
                    ]
                    if extra_cov:
                        c_prev_pos = hv_xyz[extra_cov[0]]
                # Without C(i-1), a true chain start is NH3+: the first block, a new chain ID, a non-amino-acid
                # predecessor, or restarted (seqnum, icode) numbering. An internal gap within the same chain stays
                # an amide N with no H, since its in-plane H direction is undefined without C(i-1).
                n_terminus = False
                if c_prev_pos is None:
                    if b_idx == 0 or (not same_chain) or res_maps[b_idx - 1][3] not in _STD_AA:
                        n_terminus = True
                    else:
                        try:
                            n_terminus = (int(rseq), icode) <= (int(res_maps[b_idx - 1][1]), res_maps[b_idx - 1][2])
                        except (TypeError, ValueError):
                            n_terminus = False

                if rname == "PRO":
                    if c_prev_pos is None and n_terminus and "CD" in amap:
                        pair = _sp3_methylene(n_pos, ca_pos, amap["CD"][0], d=D_NH_SP3)
                        if pair is not None:
                            out_atoms.append(_mrec("H2", pair[0], n_b))
                            out_atoms.append(_mrec("H3", pair[1], n_b))
                else:
                    if c_prev_pos is not None:
                        h_am = _sp2_bisector(n_pos, c_prev_pos, ca_pos, d=D_NH_SP2)
                        if h_am is not None:
                            out_atoms.append(_mrec(h_sym, h_am, n_b))
                    elif n_terminus:
                        ref_p = amap["C"][0] if "C" in amap else (amap["CB"][0] if "CB" in amap else None)
                        tri = _sp3_rotor3(n_pos, ca_pos, ref_p, d=D_NH_SP3)
                        if tri is not None:
                            for k_h, hp in enumerate(tri, 1):
                                out_atoms.append(_mrec(f"H{k_h}", hp, n_b))

            # 2. Backbone CA (HA or GLY HA2/HA3)
            if "CA" in amap and "N" in amap and "C" in amap and not _has_h("CA"):
                ca_pos, _, ca_b = amap["CA"]
                if rname == "GLY":
                    pair = _sp3_methylene(ca_pos, amap["N"][0], amap["C"][0], d=D_CH_2)
                    if pair is not None:
                        out_atoms.append(_mrec("HA2", pair[0], ca_b))
                        out_atoms.append(_mrec("HA3", pair[1], ca_b))
                elif "CB" in amap:
                    ha = _sp3_methine(ca_pos, amap["N"][0], amap["C"][0], amap["CB"][0], d=D_CH_1)
                    if ha is not None:
                        out_atoms.append(_mrec("HA", ha, ca_b))

            # 3. Sidechain methines (VAL/ILE/THR CB, LEU CG)
            if rname in ("VAL", "ILE", "THR") and "CB" in amap and not _has_h("CB"):
                tri_names = {"VAL": ("CA", "CG1", "CG2"), "ILE": ("CA", "CG1", "CG2"), "THR": ("CA", "OG1", "CG2")}[rname]
                if all(k in amap for k in tri_names):
                    hb = _sp3_methine(amap["CB"][0], amap[tri_names[0]][0], amap[tri_names[1]][0], amap[tri_names[2]][0], d=D_CH_1)
                    if hb is not None:
                        out_atoms.append(_mrec("HB", hb, amap["CB"][2]))
            if rname == "LEU" and "CG" in amap and not _has_h("CG") and all(k in amap for k in ("CB", "CD1", "CD2")):
                hg = _sp3_methine(amap["CG"][0], amap["CB"][0], amap["CD1"][0], amap["CD2"][0], d=D_CH_1)
                if hg is not None:
                    out_atoms.append(_mrec("HG", hg, amap["CG"][2]))

            # 4. Sidechain methylenes (CH2)
            for c_nm, n1_nm, n2_nm, (hn1, hn2) in _SIDECHAIN_CH2.get(rname, ()):
                if c_nm in amap and n1_nm in amap and n2_nm in amap and not _has_h(c_nm):
                    pair = _sp3_methylene(amap[c_nm][0], amap[n1_nm][0], amap[n2_nm][0], d=D_CH_2)
                    if pair is not None:
                        out_atoms.append(_mrec(hn1, pair[0], amap[c_nm][2]))
                        out_atoms.append(_mrec(hn2, pair[1], amap[c_nm][2]))

            # 5. Sidechain methyls (CH3)
            for c_nm, p_nm, ref_nm, (hn1, hn2, hn3) in _SIDECHAIN_CH3.get(rname, ()):
                if c_nm in amap and p_nm in amap and not _has_h(c_nm):
                    ref_p = amap[ref_nm][0] if ref_nm in amap else None
                    tri = _sp3_rotor3(amap[c_nm][0], amap[p_nm][0], ref_p, d=D_CH_3)
                    if tri is not None:
                        for hn, hp in zip((hn1, hn2, hn3), tri):
                            out_atoms.append(_mrec(hn, hp, amap[c_nm][2]))

            # 6. Aromatic ring C-H
            for c_nm, n1_nm, n2_nm, hn in _SIDECHAIN_AR_CH.get(rname, ()):
                if c_nm in amap and n1_nm in amap and n2_nm in amap and not _has_h(c_nm):
                    hp = _sp2_bisector(amap[c_nm][0], amap[n1_nm][0], amap[n2_nm][0], d=D_CH_AR)
                    if hp is not None:
                        out_atoms.append(_mrec(hn, hp, amap[c_nm][2]))

            # 7. Sidechain polar & cationic N-H / O-H / S-H
            if rname == "ASN" and "ND2" in amap and "CG" in amap and "OD1" in amap and not _has_h("ND2"):
                nh2 = _sp2_nh2(amap["ND2"][0], amap["CG"][0], amap["OD1"][0], d=D_NH_SP2)
                if nh2 is not None:
                    out_atoms.append(_mrec("HD21", nh2[0], amap["ND2"][2]))
                    out_atoms.append(_mrec("HD22", nh2[1], amap["ND2"][2]))

            elif rname == "GLN" and "NE2" in amap and "CD" in amap and "OE1" in amap and not _has_h("NE2"):
                nh2 = _sp2_nh2(amap["NE2"][0], amap["CD"][0], amap["OE1"][0], d=D_NH_SP2)
                if nh2 is not None:
                    out_atoms.append(_mrec("HE21", nh2[0], amap["NE2"][2]))
                    out_atoms.append(_mrec("HE22", nh2[1], amap["NE2"][2]))

            elif rname == "LYS" and "NZ" in amap and "CE" in amap and not _has_h("NZ") and not _near_metal(amap["NZ"][0]):
                ref_p = amap["CD"][0] if "CD" in amap else None
                tri = _sp3_rotor3(amap["NZ"][0], amap["CE"][0], ref_p, d=D_NH_SP3)
                if tri is not None:
                    for hn, hp in zip(("HZ1", "HZ2", "HZ3"), tri):
                        out_atoms.append(_mrec(hn, hp, amap["NZ"][2]))

            elif rname == "ARG":
                if "NE" in amap and "CD" in amap and "CZ" in amap and not _has_h("NE"):
                    he = _sp2_bisector(amap["NE"][0], amap["CD"][0], amap["CZ"][0], d=D_NH_SP2)
                    if he is not None:
                        out_atoms.append(_mrec("HE", he, amap["NE"][2]))
                for nh_nm, h1_nm, h2_nm in (("NH1", "HH11", "HH12"), ("NH2", "HH21", "HH22")):
                    if nh_nm in amap and "CZ" in amap and "NE" in amap and not _has_h(nh_nm):
                        nh2 = _sp2_nh2(amap[nh_nm][0], amap["CZ"][0], amap["NE"][0], d=D_NH_SP2)
                        if nh2 is not None:
                            out_atoms.append(_mrec(h1_nm, nh2[0], amap[nh_nm][2]))
                            out_atoms.append(_mrec(h2_nm, nh2[1], amap[nh_nm][2]))

            elif rname == "TRP" and "NE1" in amap and "CD1" in amap and "CE2" in amap and not _has_h("NE1"):
                he1 = _sp2_bisector(amap["NE1"][0], amap["CD1"][0], amap["CE2"][0], d=D_NH_SP2)
                if he1 is not None:
                    out_atoms.append(_mrec("HE1", he1, amap["NE1"][2]))

            elif rname in ("HIS", "HID", "HIE", "HIP"):
                has_nd1_h = _has_h("ND1")
                has_ne2_h = _has_h("NE2")
                hd1_cand = (
                    _sp2_bisector(amap["ND1"][0], amap["CG"][0], amap["CE1"][0], d=D_NH_SP2)
                    if ("ND1" in amap and "CG" in amap and "CE1" in amap and not has_nd1_h and not _near_metal(amap["ND1"][0]))
                    else None
                )
                he2_cand = (
                    _sp2_bisector(amap["NE2"][0], amap["CD2"][0], amap["CE1"][0], d=D_NH_SP2)
                    if ("NE2" in amap and "CD2" in amap and "CE1" in amap and not has_ne2_h and not _near_metal(amap["NE2"][0]))
                    else None
                )
                if rname == "HIP":
                    if hd1_cand is not None:
                        out_atoms.append(_mrec("HD1", hd1_cand, amap["ND1"][2]))
                    if he2_cand is not None:
                        out_atoms.append(_mrec("HE2", he2_cand, amap["NE2"][2]))
                elif rname == "HID":
                    if hd1_cand is not None:
                        out_atoms.append(_mrec("HD1", hd1_cand, amap["ND1"][2]))
                elif rname == "HIE":
                    if he2_cand is not None:
                        out_atoms.append(_mrec("HE2", he2_cand, amap["NE2"][2]))
                elif not has_nd1_h and not has_ne2_h:
                    # Generic HIS with 0 ring N-H: choose tautomer from local H-bond acceptors (default HID,
                    # matching field.py's heavy-atom-only typing of HIS so completion does not flip the tautomer)
                    def _hb_score(h_pt, n_pt):
                        if h_pt is None or acc_tree is None:
                            return -1.0
                        sc = 0.0
                        for ax in acc_xyz[acc_tree.query_ball_point(n_pt, 3.35)]:
                            if np.linalg.norm(ax - n_pt) < 2.2:
                                continue
                            dh = float(np.linalg.norm(ax - h_pt))
                            if 1.55 <= dh <= 2.45:
                                sc = max(sc, math.exp(-((dh - 1.90) / 0.30) ** 2))
                        return sc

                    s_d1 = _hb_score(hd1_cand, amap["ND1"][0]) if hd1_cand is not None else -1.0
                    s_e2 = _hb_score(he2_cand, amap["NE2"][0]) if he2_cand is not None else -1.0
                    if s_e2 > s_d1 + 0.15 and he2_cand is not None:
                        out_atoms.append(_mrec("HE2", he2_cand, amap["NE2"][2]))
                    elif hd1_cand is not None:
                        out_atoms.append(_mrec("HD1", hd1_cand, amap["ND1"][2]))
                    elif he2_cand is not None:
                        out_atoms.append(_mrec("HE2", he2_cand, amap["NE2"][2]))

            elif rname == "SER" and "OG" in amap and "CB" in amap and not _has_h("OG") and not _near_metal(amap["OG"][0]):
                ref_p = amap["CA"][0] if "CA" in amap else None
                hg = _best_rotatable_xh(amap["OG"][0], amap["CB"][0], ref_p, D_OH, 109.5, acc_tree, acc_xyz, hv_tree)
                if hg is not None:
                    out_atoms.append(_mrec("HG", hg, amap["OG"][2]))

            elif rname == "THR" and "OG1" in amap and "CB" in amap and not _has_h("OG1") and not _near_metal(amap["OG1"][0]):
                ref_p = amap["CA"][0] if "CA" in amap else None
                hg1 = _best_rotatable_xh(amap["OG1"][0], amap["CB"][0], ref_p, D_OH, 109.5, acc_tree, acc_xyz, hv_tree)
                if hg1 is not None:
                    out_atoms.append(_mrec("HG1", hg1, amap["OG1"][2]))

            elif rname == "TYR" and "OH" in amap and "CZ" in amap and not _has_h("OH") and not _near_metal(amap["OH"][0]):
                ref_p = amap["CE1"][0] if "CE1" in amap else None
                hh = _best_rotatable_xh(amap["OH"][0], amap["CZ"][0], ref_p, D_OH, 109.5, acc_tree, acc_xyz, hv_tree, planar_bias=True)
                if hh is not None:
                    out_atoms.append(_mrec("HH", hh, amap["OH"][2]))

            elif rname == "CYS" and "SG" in amap and "CB" in amap and not _has_h("SG"):
                sg_pos = amap["SG"][0]
                # Only protonate free thiol SG when no disulfide/metal partner sits within 2.50 A
                near_sm = sm_xyz[sm_tree.query_ball_point(sg_pos, 2.50)] if sm_tree is not None else np.zeros((0, 3))
                is_bonded_sm = any(float(np.linalg.norm(p - sg_pos)) > 0.35 for p in near_sm)
                if not is_bonded_sm:
                    ref_p = amap["CA"][0] if "CA" in amap else None
                    hg = _best_rotatable_xh(sg_pos, amap["CB"][0], ref_p, D_SH, 96.0, acc_tree, acc_xyz, hv_tree)
                    if hg is not None:
                        out_atoms.append(_mrec("HG", hg, amap["SG"][2]))

        elif rname in _STD_NA:
            # Nucleobase polar NH / NH2 + aromatic/methyl C-H + ribose/deoxyribose C-H + RNA 2'-OH
            if rname in ("A", "DA", "ADE") and "N6" in amap and "C6" in amap and "N1" in amap and not _has_h("N6"):
                nh2 = _sp2_nh2(amap["N6"][0], amap["C6"][0], amap["N1"][0], d=D_NH_SP2)
                if nh2 is not None:
                    out_atoms.append(_mrec("H61", nh2[0], amap["N6"][2]))
                    out_atoms.append(_mrec("H62", nh2[1], amap["N6"][2]))
            elif rname in ("C", "DC", "CYT") and "N4" in amap and "C4" in amap and "N3" in amap and not _has_h("N4"):
                nh2 = _sp2_nh2(amap["N4"][0], amap["C4"][0], amap["N3"][0], d=D_NH_SP2)
                if nh2 is not None:
                    out_atoms.append(_mrec("H41", nh2[0], amap["N4"][2]))
                    out_atoms.append(_mrec("H42", nh2[1], amap["N4"][2]))
            elif rname in ("G", "DG", "GUA"):
                if "N1" in amap and "C6" in amap and "C2" in amap and not _has_h("N1"):
                    h1 = _sp2_bisector(amap["N1"][0], amap["C6"][0], amap["C2"][0], d=D_NH_SP2)
                    if h1 is not None:
                        out_atoms.append(_mrec("H1", h1, amap["N1"][2]))
                if "N2" in amap and "C2" in amap and "N3" in amap and not _has_h("N2"):
                    nh2 = _sp2_nh2(amap["N2"][0], amap["C2"][0], amap["N3"][0], d=D_NH_SP2)
                    if nh2 is not None:
                        out_atoms.append(_mrec("H21", nh2[0], amap["N2"][2]))
                        out_atoms.append(_mrec("H22", nh2[1], amap["N2"][2]))
            elif rname in ("U", "DT", "URI", "THY") and "N3" in amap and "C2" in amap and "C4" in amap and not _has_h("N3"):
                h3 = _sp2_bisector(amap["N3"][0], amap["C2"][0], amap["C4"][0], d=D_NH_SP2)
                if h3 is not None:
                    out_atoms.append(_mrec("H3", h3, amap["N3"][2]))

            # Purine / pyrimidine aromatic C-H & thymine methyl
            if rname in ("A", "DA", "ADE", "G", "DG", "GUA") and "C8" in amap and "N7" in amap and "N9" in amap and not _has_h("C8"):
                h8 = _sp2_bisector(amap["C8"][0], amap["N7"][0], amap["N9"][0], d=D_CH_AR)
                if h8 is not None:
                    out_atoms.append(_mrec("H8", h8, amap["C8"][2]))
            if rname in ("A", "DA", "ADE") and "C2" in amap and "N1" in amap and "N3" in amap and not _has_h("C2"):
                h2 = _sp2_bisector(amap["C2"][0], amap["N1"][0], amap["N3"][0], d=D_CH_AR)
                if h2 is not None:
                    out_atoms.append(_mrec("H2", h2, amap["C2"][2]))
            if rname in ("C", "DC", "CYT", "U", "DT", "URI", "THY"):
                if "C6" in amap and "N1" in amap and "C5" in amap and not _has_h("C6"):
                    h6 = _sp2_bisector(amap["C6"][0], amap["N1"][0], amap["C5"][0], d=D_CH_AR)
                    if h6 is not None:
                        out_atoms.append(_mrec("H6", h6, amap["C6"][2]))
                if rname in ("C", "DC", "CYT", "U", "URI") and "C5" in amap and "C4" in amap and "C6" in amap and not _has_h("C5"):
                    h5 = _sp2_bisector(amap["C5"][0], amap["C4"][0], amap["C6"][0], d=D_CH_AR)
                    if h5 is not None:
                        out_atoms.append(_mrec("H5", h5, amap["C5"][2]))
                if rname in ("DT", "THY"):
                    cm_nm = "C7" if "C7" in amap else ("C5M" if "C5M" in amap else None)
                    if cm_nm is not None and "C5" in amap and not _has_h(cm_nm):
                        ref_c4 = amap["C4"][0] if "C4" in amap else None
                        hm3 = _sp3_rotor3(amap[cm_nm][0], amap["C5"][0], ref_c4, d=D_CH_3)
                        if hm3 is not None:
                            h_lbls = ("H71", "H72", "H73") if cm_nm == "C7" else ("H5M1", "H5M2", "H5M3")
                            for h_nm, h_p in zip(h_lbls, hm3):
                                out_atoms.append(_mrec(h_nm, h_p, amap[cm_nm][2]))

            # Ribose / deoxyribose sugar C-H
            gly_n = "N9" if rname in ("A", "DA", "ADE", "G", "DG", "GUA") else "N1"
            if "C1'" in amap and "O4'" in amap and "C2'" in amap and gly_n in amap and not _has_h("C1'"):
                h1p = _sp3_methine(amap["C1'"][0], amap["O4'"][0], amap["C2'"][0], amap[gly_n][0], d=D_CH_1)
                if h1p is not None:
                    out_atoms.append(_mrec("H1'", h1p, amap["C1'"][2]))
            if "C2'" in amap and "C1'" in amap and "C3'" in amap and not _has_h("C2'"):
                if "O2'" in amap:
                    h2p = _sp3_methine(amap["C2'"][0], amap["C1'"][0], amap["C3'"][0], amap["O2'"][0], d=D_CH_1)
                    if h2p is not None:
                        out_atoms.append(_mrec("H2'", h2p, amap["C2'"][2]))
                else:
                    h2_pair = _sp3_methylene(amap["C2'"][0], amap["C1'"][0], amap["C3'"][0], d=D_CH_2)
                    if h2_pair is not None:
                        out_atoms.append(_mrec("H2'", h2_pair[0], amap["C2'"][2]))
                        out_atoms.append(_mrec("H2''", h2_pair[1], amap["C2'"][2]))
            if "C3'" in amap and "C2'" in amap and "C4'" in amap and "O3'" in amap and not _has_h("C3'"):
                h3p = _sp3_methine(amap["C3'"][0], amap["C2'"][0], amap["C4'"][0], amap["O3'"][0], d=D_CH_1)
                if h3p is not None:
                    out_atoms.append(_mrec("H3'", h3p, amap["C3'"][2]))
            if "C4'" in amap and "C3'" in amap and "O4'" in amap and "C5'" in amap and not _has_h("C4'"):
                h4p = _sp3_methine(amap["C4'"][0], amap["C3'"][0], amap["O4'"][0], amap["C5'"][0], d=D_CH_1)
                if h4p is not None:
                    out_atoms.append(_mrec("H4'", h4p, amap["C4'"][2]))
            if "C5'" in amap and "C4'" in amap and "O5'" in amap and not _has_h("C5'"):
                h5_pair = _sp3_methylene(amap["C5'"][0], amap["C4'"][0], amap["O5'"][0], d=D_CH_2)
                if h5_pair is not None:
                    out_atoms.append(_mrec("H5'", h5_pair[0], amap["C5'"][2]))
                    out_atoms.append(_mrec("H5''", h5_pair[1], amap["C5'"][2]))

            if "O2'" in amap and "C2'" in amap and not _has_h("O2'"):
                ref_p = amap["C1'"][0] if "C1'" in amap else None
                ho2 = _best_rotatable_xh(amap["O2'"][0], amap["C2'"][0], ref_p, D_OH, 109.5, acc_tree, acc_xyz, hv_tree)
                if ho2 is not None:
                    out_atoms.append(_mrec("HO2'", ho2, amap["O2'"][2]))

    return out_atoms
