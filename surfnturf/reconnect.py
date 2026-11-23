"""Geometry-guided conformer reconnector for multi-altloc crystallographic models.

Fast heuristic conformer completion ("poor-man's untangler") for heterogeneous multi-altloc structures:
  surfnturf reconnect model.pdb outdir [--nconf 32] [--occ-flex 0] [--quotas hamilton|quantile]
                                       [--no-rebuild-h] [--pseudo pseudo.pdb]
  (standalone `python reconnect.py` takes the same flags, or the legacy key=value form: nconf=32 occ_flex=0 ...)

For exact global integer-linear-programming (ILP) conformer assignment, pass a directory of member PDBs from
Passmore's Untangler directly to `surfnturf predict <member_dir> ...`.

What `reconnect` fixes vs naive alphabetical occupancy-quantile assignment:
  1. Ghost H-only altlocs: phenix.refine ties riding amide H(i, L) occupancy to C(i-1, L), leaving lone amide H
     atoms at occ > 0 on altlocs whose heavy atoms refined to occ = 0 (12 altlocs -> 39 missing-residue holes across
     32 conformers in 1aho_best_so_far_trimmed.pdb) and zeroing amide H on 82 valid heavy-atom altlocs where i-1
     lacks letter L. Active altlocs and occupancies are determined from heavy atoms (occ > 0) while preserving
     genuine H/D-only rotamer splits on single-conformer heavy residues and re-bisecting cross-letter amide H/D.
  2. Geometry-guided Hungarian + multi-cut block matching: solves exact O(M^3) Kuhn-Munkres assignments along the
     chain followed by iterative single-residue and sub-chain interval-cut sweeps minimizing peptide C-N strain,
     nucleic-acid O3'-P strain, SG-SG disulfide strain, inter-residue covalent/metal coordination strain, and
     crystal-symmetry heavy-atom clashes (< 2.2 A hard + VDW ramp).
  3. Optional bounded occupancy relaxation (occ_flex=k allows +-k/nconf occupancy adjustment per state; default 0
     strictly preserves per-residue integer quotas).

Pure Python 3 standard library (uses gemmi automatically if available for general space-group symmetry).
"""
import math
import os
import sys
from collections import defaultdict

_TWO_LETTER_ELEMENTS = {
    'HE', 'LI', 'BE', 'NE', 'NA', 'MG', 'AL', 'SI', 'CL', 'AR', 'CA', 'SC', 'TI', 'CR', 'MN', 'FE',
    'CO', 'NI', 'CU', 'ZN', 'GA', 'GE', 'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'ZR', 'NB', 'MO', 'TC',
    'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'SB', 'TE', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND',
    'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'RE', 'OS', 'IR',
    'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'NP', 'PU',
}

_METALS = {
    'LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'GA', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'CS', 'BA', 'LA',
    'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE',
    'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'U',
}

_LARGE_METALS = {'NA', 'K', 'RB', 'CS', 'CA', 'SR', 'BA', 'LA', 'PB', 'TL'}

_R_COV = {
    'H': 0.32, 'D': 0.32, 'B': 0.85, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.64,
    'SI': 1.15, 'P': 1.07, 'S': 1.05, 'CL': 0.99, 'AS': 1.20, 'SE': 1.20, 'BR': 1.14, 'I': 1.33,
}

_HY36_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def _parse_hy36_4(s: str) -> int:
    """Parse a 4-column PDB integer or hybrid-36 residue sequence number."""
    st = s.strip()
    if not st:
        return 0
    if st.lstrip('-+').isdigit():
        return int(st)
    c0 = s[0]
    if 'A' <= c0 <= 'Z':
        val = 0
        for ch in s:
            idx = _HY36_DIGITS.find(ch)
            if idx < 0 or idx >= 36:
                raise ValueError(f'Invalid hybrid-36 residue number {s!r}')
            val = val * 36 + idx
        return val - 10 * (36 ** 3) + 10000
    if 'a' <= c0 <= 'z':
        val = 0
        for ch in s:
            idx = _HY36_DIGITS.find(ch.upper()) if ch.isdigit() else (_HY36_DIGITS.find(ch) - 26)
            if idx < 0 or idx >= 36:
                raise ValueError(f'Invalid hybrid-36 residue number {s!r}')
            val = val * 36 + idx
        return val - 10 * (36 ** 3) + 10000 + 26 * (36 ** 3)
    return int(st)


def _format_hy36_4(n: int) -> str:
    """Format an integer residue sequence number into 4 PDB columns (hybrid-36 outside -999..9999)."""
    if -999 <= n <= 9999:
        return '%4d' % n
    if n >= 10000:
        v = n - 10000 + 10 * (36 ** 3)
        alphabet = _HY36_DIGITS[:36]
        if v >= 36 ** 4:
            v = n - 10000 - 26 * (36 ** 3)
            out = []
            for _ in range(3):
                v, r = divmod(v, 36)
                out.append(alphabet[r].lower() if r >= 10 else alphabet[r])
            out.append(_HY36_DIGITS[36 + v])
            return ''.join(reversed(out))
        out = []
        for _ in range(4):
            v, r = divmod(v, 36)
            out.append(alphabet[r])
        return ''.join(reversed(out))
    raise ValueError(f'Residue number {n} cannot be formatted in 4 PDB columns')


def _infer_pdb_element(name_col4: str, resname: str, is_het: bool) -> str:
    """Infer uppercase chemical element symbol when PDB columns 77-78 are blank."""
    nm = name_col4.strip().upper()
    if not nm:
        return 'C'
    rname = resname.strip().upper()
    # Left-justified column 13 letter (or digit-prefixed H like '1HG2')
    if name_col4[0].isdigit():
        rest = nm.lstrip('0123456789')
        return rest[0] if rest else 'H'
    if name_col4[0].isalpha():
        two = nm[:2]
        if len(two) == 2 and two in _TWO_LETTER_ELEMENTS:
            return two
        return nm[0]
    # Space in column 13 (' CA ', ' ZN ', ' CL '): in HETATM where residue name matches a 2-letter ion/metal, keep 2-letter
    letters = ''.join(ch for ch in nm if ch.isalpha())
    if not letters:
        return 'C'
    if is_het and len(letters) >= 2 and letters[:2] in _TWO_LETTER_ELEMENTS and rname.startswith(letters[:2]):
        return letters[:2]
    return letters[0]


def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def angle_deg(a, b, c):
    ux, uy, uz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    vx, vy, vz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    nu = math.sqrt(ux * ux + uy * uy + uz * uz)
    nv = math.sqrt(vx * vx + vy * vy + vz * vz)
    if nu < 1e-8 or nv < 1e-8:
        return 120.0
    cs = max(-1.0, min(1.0, (ux * vx + uy * vy + uz * vz) / (nu * nv)))
    return math.degrees(math.acos(cs))


def dihedral_deg(p0, p1, p2, p3):
    b0 = (p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])
    b1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    b2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
    n1 = math.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
    if n1 < 1e-8:
        return 180.0
    b1u = (b1[0] / n1, b1[1] / n1, b1[2] / n1)
    d0 = b0[0] * b1u[0] + b0[1] * b1u[1] + b0[2] * b1u[2]
    d2 = b2[0] * b1u[0] + b2[1] * b1u[1] + b2[2] * b1u[2]
    v = (b0[0] - d0 * b1u[0], b0[1] - d0 * b1u[1], b0[2] - d0 * b1u[2])
    w = (b2[0] - d2 * b1u[0], b2[1] - d2 * b1u[1], b2[2] - d2 * b1u[2])
    x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
    m = (b1u[1] * v[2] - b1u[2] * v[1], b1u[2] * v[0] - b1u[0] * v[2], b1u[0] * v[1] - b1u[1] * v[0])
    y = m[0] * w[0] + m[1] * w[1] + m[2] * w[2]
    return math.degrees(math.atan2(y, x))


def hungarian_min_cost(cost):
    """Exact O(N^3) Kuhn-Munkres / Jonker-Volgenant min-cost bipartite matching on N x N matrix."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            rc_i0 = cost[i0 - 1]
            u_i0 = u[i0]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = rc_i0[j - 1] - u_i0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = [0] * n
    for j in range(1, n + 1):
        ans[p[j] - 1] = j - 1
    return ans


def _parse_cell_and_symops(cryst1_line, source_label='<input>'):
    """Return (orth_mat, frac_mat, sym_ops) from CRYST1 line; uses gemmi if available, else general triclinic stdlib."""
    if not cryst1_line:
        return None, None, [(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))]

    a = float(cryst1_line[6:15])
    b = float(cryst1_line[15:24])
    c = float(cryst1_line[24:33])
    al = float(cryst1_line[33:40])
    be = float(cryst1_line[40:47])
    ga = float(cryst1_line[47:54])
    raw_sg = cryst1_line[55:66].strip()
    if not raw_sg:
        print(f"reconnect: {source_label} CRYST1 has no space group; assuming 'P 1'", file=sys.stderr)
        sg_hm = 'P 1'
    else:
        sg_hm = raw_sg

    try:
        import gemmi
        cell = gemmi.UnitCell(a, b, c, al, be, ga)
        sg = gemmi.find_spacegroup_by_name(sg_hm, alpha=al, gamma=ga) or gemmi.SpaceGroup(sg_hm)
        O = tuple(tuple(float(v) for v in row) for row in cell.orth.mat.tolist())
        F = tuple(tuple(float(v) for v in row) for row in cell.frac.mat.tolist())
        ops = []
        for op in sg.operations():
            R = tuple(tuple(op.rot[r][k] / op.DEN for k in range(3)) for r in range(3))
            t = tuple(op.tran[r] / op.DEN for r in range(3))
            ops.append((R, t))
        return O, F, ops
    except Exception as e:
        gemmi_err = 'gemmi not available' if isinstance(e, ImportError) else f'gemmi: {e}'

    # Pure-stdlib triclinic orthogonalization / fractionalization + common crystallographic space groups
    ar, br, gr = math.radians(al), math.radians(be), math.radians(ga)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    sg_g = math.sin(gr)
    cz = math.sqrt(max(1e-12, 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg)) / sg_g
    cy = (ca - cb * cg) / sg_g
    O = (
        (a, b * cg, c * cb),
        (0.0, b * sg_g, c * cy),
        (0.0, 0.0, c * cz),
    )
    F = (
        (1.0 / a, -cg / (a * sg_g), (cg * cy - cb * sg_g) / (a * sg_g * cz)),
        (0.0, 1.0 / (b * sg_g), -cy / (b * sg_g * cz)),
        (0.0, 0.0, 1.0 / (c * cz)),
    )
    sg_key = ' '.join(sg_hm.upper().split())
    is_rhomb_axes = (abs(a - b) < 1e-2 and abs(b - c) < 1e-2 and abs(al - be) < 1e-2 and abs(be - ga) < 1e-2 and abs(ga - 120.0) > 1.0)
    if sg_key in ('P 21 21 21', 'P212121'):
        ops = [
            (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
            (((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), (0.5, 0.0, 0.5)),
            (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), (0.0, 0.5, 0.5)),
            (((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), (0.5, 0.5, 0.0)),
        ]
    elif sg_key in ('P 1 21 1', 'P 21', 'P21'):
        ops = [
            (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
            (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), (0.0, 0.5, 0.0)),
        ]
    elif sg_key in ('P 21 21 2', 'P21212'):
        ops = [
            (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
            (((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
            (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), (0.5, 0.5, 0.0)),
            (((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), (0.5, 0.5, 0.0)),
        ]
    elif sg_key in ('R 3 :R', 'R3:R') or (sg_key in ('R 3', 'R3') and is_rhomb_axes):
        ops = [
            (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0)),
            (((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (0.0, 0.0, 0.0)),
            (((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), (0.0, 0.0, 0.0)),
        ]
    else:
        ops = [(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))]
        if sg_key not in ('P 1', 'P1'):
            print(f'reconnect: space group {sg_hm!r} not supported without gemmi ({gemmi_err}); '
                  f'only lattice translations are used for symmetry-mate clashes', file=sys.stderr)
    return O, F, ops


def _mat_vec(M, v):
    return (
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    )


def parse_multi_altloc_pdb(pdb_path):
    """Parse non-water residues keeping all raw altloc records (supports both PDB and mmCIF via gemmi)."""
    if str(pdb_path).lower().endswith(('.cif', '.mmcif')):
        import gemmi
        st = gemmi.read_structure(pdb_path)
        if st.cell.is_crystal():
            a, b, c, al, be, ga = st.cell.parameters
            raw_sg = (st.spacegroup_hm or '').strip()
            if not raw_sg:
                print(f"reconnect: {pdb_path} has no space group; assuming 'P 1'", file=sys.stderr)
                sg_hm = 'P 1'
            else:
                sg_hm = raw_sg
            try:
                sg_obj = gemmi.find_spacegroup_by_name(sg_hm, alpha=al, gamma=ga) or gemmi.SpaceGroup(sg_hm)
                z = len(sg_obj.operations())
            except Exception:
                z = 1
            cryst1_line = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d' % (a, b, c, al, be, ga, sg_hm[:11], z)
        else:
            cryst1_line = None
            print(f'reconnect: {pdb_path} has no unit cell; symmetry-mate clashes are not checked and no CRYST1 is '
                  f'written', file=sys.stderr)
        res_order = []
        residues = {}
        for ch in st[0]:
            ch_id = ch.name or 'A'
            for r in ch:
                if r.name in ('HOH', 'WAT', 'DOD'):
                    continue
                seqnum = int(r.seqid.num)
                icode = r.seqid.icode if r.seqid.icode != '\0' else ' '
                rkey = (ch_id, seqnum, icode)
                if rkey not in residues:
                    res_order.append(rkey)
                    residues[rkey] = {
                        'rkey': rkey,
                        'chain': ch_id,
                        'seqnum': seqnum,
                        'icode': icode,
                        'resname': r.name,
                        'blank_raw': [],
                        'alts_raw': defaultdict(list),
                    }
                for at_g in r:
                    alt = at_g.altloc if (at_g.altloc != '\0' and at_g.altloc.strip()) else ' '
                    biso = float(at_g.b_iso)
                    if biso <= 0.0 and at_g.aniso.nonzero():
                        biso = (8.0 * (math.pi ** 2) / 3.0) * float(at_g.aniso.u11 + at_g.aniso.u22 + at_g.aniso.u33)
                    at = {
                        'name': at_g.name.strip(),
                        'alt': alt,
                        'chain': ch_id,
                        'seqnum': seqnum,
                        'icode': icode,
                        'resname': r.name,
                        'pos': (float(at_g.pos.x), float(at_g.pos.y), float(at_g.pos.z)),
                        'occ': float(at_g.occ),
                        'biso': biso,
                        'elem': at_g.element.name.upper(),
                        'het': r.het_flag == 'H',
                    }
                    if alt == ' ':
                        residues[rkey]['blank_raw'].append(at)
                    else:
                        residues[rkey]['alts_raw'][alt].append(at)
        return cryst1_line, res_order, residues

    cryst1_line = None
    res_order = []
    residues = {}

    with open(pdb_path, 'r') as fh:
        for raw_line in fh:
            if raw_line.startswith('ENDMDL'):
                break
            if raw_line.startswith('CRYST1') and cryst1_line is None:
                cryst1_line = raw_line.rstrip('\r\n')
            elif raw_line.startswith(('ATOM  ', 'HETATM')):
                resname = raw_line[17:20].strip()
                if resname in ('HOH', 'WAT', 'DOD'):
                    continue
                is_het = raw_line.startswith('HETATM')
                name_col4 = raw_line[12:16]
                name = name_col4.strip()
                alt = raw_line[16]
                chain = raw_line[20:22].strip() or ' '   # 2-character chain IDs use columns 21-22
                seqnum = _parse_hy36_4(raw_line[22:26])
                icode = raw_line[26]
                pos = (float(raw_line[30:38]), float(raw_line[38:46]), float(raw_line[46:54]))
                occ = float(raw_line[54:60])
                biso = float(raw_line[60:66])
                raw_el = raw_line[76:78].strip() if len(raw_line) >= 78 else ''
                elem = raw_el.upper() if raw_el else _infer_pdb_element(name_col4, resname, is_het)

                rkey = (chain, seqnum, icode)
                if rkey not in residues:
                    res_order.append(rkey)
                    residues[rkey] = {
                        'rkey': rkey,
                        'chain': chain,
                        'seqnum': seqnum,
                        'icode': icode,
                        'resname': resname,
                        'blank_raw': [],
                        'alts_raw': defaultdict(list),
                    }
                at = {
                    'name': name,
                    'alt': alt,
                    'chain': chain,
                    'seqnum': seqnum,
                    'icode': icode,
                    'resname': resname,
                    'pos': pos,
                    'occ': occ,
                    'biso': biso,
                    'elem': elem,
                    'het': is_het,
                }
                if alt in (' ', ''):
                    residues[rkey]['blank_raw'].append(at)
                else:
                    residues[rkey]['alts_raw'][alt].append(at)

    if cryst1_line is None:
        print(f'reconnect: {pdb_path} has no CRYST1; symmetry-mate clashes are not checked and no CRYST1 is written',
              file=sys.stderr)
    return cryst1_line, res_order, residues


def build_clean_states(res_order, residues, nconf=32):
    """Build active conformer states per residue, dropping riding-H ghosts while preserving H/D-only rotamer splits."""
    states = {}
    alpha_slots = {}
    hamilton_slots = {}
    buggy_alpha_holes = 0
    dropped_ghosts = []
    restored_zero_h = 0
    q_grid = [(m + 0.5) / nconf for m in range(nconf)]

    for idx, rkey in enumerate(res_order):
        r = residues[rkey]
        # Count how many missing-heavy-atom holes naive occ>0 (rq_members.py) would have created here
        alts_naive = {L: [a for a in ats if a['occ'] > 0] for L, ats in r['alts_raw'].items()}
        alts_naive = {L: ats for L, ats in alts_naive.items() if ats}
        blank_pos = [a for a in r['blank_raw'] if a['occ'] > 0]
        if alts_naive:
            Ls_n = sorted(alts_naive.keys())
            ws_n = [sum(a['occ'] for a in alts_naive[L]) / len(alts_naive[L]) for L in Ls_n]
            tot_n = sum(ws_n)
            if tot_n >= 0.9:
                ws_n = [w / tot_n for w in ws_n]
            cum_n = []
            s_n = 0.0
            for w in ws_n:
                s_n += w
                cum_n.append(s_n)
            for q in q_grid:
                k = 0
                while k < len(cum_n) and cum_n[k] <= q:
                    k += 1
                if k >= len(Ls_n):
                    buggy_alpha_holes += 1
                else:
                    chosen_hv = [a for a in (blank_pos + alts_naive[Ls_n[k]]) if a['elem'] not in ('H', 'D')]
                    if not chosen_hv:
                        buggy_alpha_holes += 1

        blank_hv_pos = [a for a in r['blank_raw'] if a['elem'] not in ('H', 'D') and a['occ'] > 0]
        if blank_hv_pos:
            blank_clean = [a for a in r['blank_raw'] if a['occ'] > 0 or a['elem'] in ('H', 'D')]
        else:
            blank_clean = [a for a in r['blank_raw'] if a['occ'] > 0 and a['elem'] not in ('H', 'D')]

        any_alt_has_heavy = any(
            a['elem'] not in ('H', 'D') for ats in r['alts_raw'].values() for a in ats
        )

        valid_letters = []
        ws_raw = []
        alt_atom_lists = {}
        if blank_hv_pos and not any_alt_has_heavy and r['alts_raw']:
            # Pure H/D rotamer split on a single-conformer heavy residue (e.g. Ser HG A/B): keep H/D states!
            for L in sorted(r['alts_raw'].keys()):
                ats = r['alts_raw'][L]
                pos_h = [dict(a) for a in ats if a['occ'] > 0]
                if not pos_h:
                    continue
                valid_letters.append(L)
                ws_raw.append(sum(a['occ'] for a in pos_h) / len(pos_h))
                alt_atom_lists[L] = pos_h
        else:
            for L in sorted(r['alts_raw'].keys()):
                ats = r['alts_raw'][L]
                hv_pos = [a for a in ats if a['elem'] not in ('H', 'D') and a['occ'] > 0]
                if not hv_pos:
                    pos_h = [a for a in ats if a['occ'] > 0]
                    if pos_h:
                        dropped_ghosts.append((r['seqnum'], r['resname'], L, [(a['name'], a['occ']) for a in pos_h]))
                    continue
                valid_letters.append(L)
                ws_raw.append(sum(a['occ'] for a in hv_pos) / len(hv_pos))
                kept = []
                for a in ats:
                    if a['elem'] not in ('H', 'D'):
                        if a['occ'] > 0:
                            kept.append(dict(a))
                    else:
                        if a['occ'] <= 0:
                            restored_zero_h += 1
                        kept.append(dict(a))
                alt_atom_lists[L] = kept

        if not valid_letters:
            comb = [dict(a) for a in blank_clean]
            amap = {a['name']: a for a in comb}
            hv = [(a['name'], a['elem'], a['pos']) for a in comb if a['elem'] not in ('H', 'D')]
            states[idx] = [{'label': '_', 'occ': 1.0, 'raw_occ': 1.0, 'atoms': amap, 'heavy': hv, 'all_atoms': comb}]
            alpha_slots[idx] = [0] * nconf
            hamilton_slots[idx] = [0] * nconf
        else:
            tot = sum(ws_raw)
            ws_norm = [w / tot for w in ws_raw]
            st_list = []
            for L, wn, wr in zip(valid_letters, ws_norm, ws_raw):
                comb = [dict(a) for a in blank_clean] + alt_atom_lists[L]
                amap = {a['name']: a for a in comb}
                hv = [(a['name'], a['elem'], a['pos']) for a in comb if a['elem'] not in ('H', 'D')]
                st_list.append({'label': L, 'occ': wn, 'raw_occ': wr, 'atoms': amap, 'heavy': hv, 'all_atoms': comb})
            states[idx] = st_list

            # Stratified quantile slots
            cum = []
            s = 0.0
            for w in ws_norm:
                s += w
                cum.append(s)
            slots = []
            for q in q_grid:
                k = 0
                while k < len(cum) and cum[k] <= q:
                    k += 1
                slots.append(min(k, len(st_list) - 1))
            alpha_slots[idx] = slots

            # Largest-remainder (Hamilton) integer counts summing to nconf
            exact = [w * nconf for w in ws_norm]
            cnts = [int(math.floor(x)) for x in exact]
            rem = nconf - sum(cnts)
            for k in sorted(range(len(exact)), key=lambda k: -(exact[k] - cnts[k]))[:rem]:
                cnts[k] += 1
            hslots = []
            for k, c in enumerate(cnts):
                hslots.extend([k] * c)
            hamilton_slots[idx] = hslots

    meta = {
        'buggy_alpha_holes': buggy_alpha_holes,
        'dropped_ghosts': dropped_ghosts,
        'restored_zero_h': restored_zero_h,
    }
    return states, alpha_slots, hamilton_slots, meta


def detect_disulfide_pairs(res_order, residues, states):
    """Auto-detect disulfide-capable Cys residue index pairs (u < v) where min_{a,b} d(SG_u, SG_v) <= 2.50 A."""
    cys_indices = [idx for idx, rk in enumerate(res_order) if residues[rk]['resname'] in ('CYS', 'CYX')]
    candidates = []
    for ii in range(len(cys_indices)):
        u = cys_indices[ii]
        for jj in range(ii + 1, len(cys_indices)):
            v = cys_indices[jj]
            if v == u + 1 and residues[res_order[u]]['chain'] == residues[res_order[v]]['chain']:
                continue
            dmin = float('inf')
            for sa in states[u]:
                if 'SG' not in sa['atoms']:
                    continue
                pa = sa['atoms']['SG']['pos']
                for sb in states[v]:
                    if 'SG' not in sb['atoms']:
                        continue
                    d = dist(pa, sb['atoms']['SG']['pos'])
                    if d < dmin:
                        dmin = d
            if dmin <= 2.50:
                candidates.append((dmin, u, v))
    candidates.sort()
    used = set()
    ss_pairs = []
    for _, u, v in candidates:
        if u not in used and v not in used:
            ss_pairs.append((u, v))
            used.add(u)
            used.add(v)
    return sorted(ss_pairs)


def _build_intra_heavy_bonds(states):
    """Return intra_nbrs[(res_idx, state_idx)][atom_name] = set of covalently bonded heavy atom names in same residue."""
    intra = {}
    for i, st_list in states.items():
        for a, st in enumerate(st_list):
            nbrs = defaultdict(set)
            hv = st['heavy']
            for u in range(len(hv)):
                nu, eu, pu = hv[u]
                ru = _R_COV.get(eu, 0.80)
                for v in range(u + 1, len(hv)):
                    nv, ev, pv = hv[v]
                    if eu in _METALS or ev in _METALS:
                        continue
                    rv = _R_COV.get(ev, 0.80)
                    if dist(pu, pv) <= ru + rv + 0.42:
                        nbrs[nu].add(nv)
                        nbrs[nv].add(nu)
            intra[(i, a)] = nbrs
    return intra


def precompute_energies(cryst1_line, res_order, residues, states, ss_pairs, source_label='<input>'):
    """Precompute exact peptide, phosphodiester, disulfide, metal/covalent, and crystal-symmetry clash matrices."""
    O_mat, F_mat, sym_ops = _parse_cell_and_symops(cryst1_line, source_label=source_label)
    n_res = len(res_order)
    indices = list(range(n_res))
    intra_nbrs = _build_intra_heavy_bonds(states)

    # 1. Consecutive peptide links i -> i+1 (C -> N) and nucleic-acid phosphodiester links (O3' -> P)
    E_pep = {}
    pep_bad = {}
    E_na = {}
    for i in range(n_res - 1):
        ri = residues[res_order[i]]
        rj = residues[res_order[i + 1]]
        if ri['chain'] != rj['chain'] or rj['seqnum'] - ri['seqnum'] not in (0, 1):
            continue
        na, nb = len(states[i]), len(states[i + 1])
        d_closest = min(
            (dist(states[i][a]['atoms']['C']['pos'], states[i + 1][b]['atoms']['N']['pos'])
             for a in range(na) for b in range(nb)
             if 'C' in states[i][a]['atoms'] and 'N' in states[i + 1][b]['atoms']),
            default=99.0
        )
        if d_closest <= 4.5:
            mat = [[0.0] * nb for _ in range(na)]
            bad = [[0] * nb for _ in range(na)]
            is_pro_next = (rj['resname'] == 'PRO')
            for a in range(na):
                sa = states[i][a]['atoms']
                la = states[i][a]['label']
                for b in range(nb):
                    sb = states[i + 1][b]['atoms']
                    lb = states[i + 1][b]['label']
                    cost = 0.0
                    is_b = 0
                    if 'C' in sa and 'N' in sb:
                        d_cn = dist(sa['C']['pos'], sb['N']['pos'])
                        if not (1.20 <= d_cn <= 1.50):
                            is_b = 1
                            cost += 5000.0 + 2000.0 * ((d_cn - 1.329) ** 2)
                        else:
                            cost += ((d_cn - 1.329) / 0.025) ** 2
                        if 'CA' in sa:
                            ang1 = angle_deg(sa['CA']['pos'], sa['C']['pos'], sb['N']['pos'])
                            cost += 0.5 * ((ang1 - 116.2) / 3.0) ** 2
                        if 'O' in sa:
                            ang_o = angle_deg(sa['O']['pos'], sa['C']['pos'], sb['N']['pos'])
                            cost += 0.3 * ((ang_o - 123.0) / 3.5) ** 2
                        if 'CA' in sb:
                            ang2 = angle_deg(sa['C']['pos'], sb['N']['pos'], sb['CA']['pos'])
                            cost += 0.5 * ((ang2 - 121.7) / 3.0) ** 2
                        if 'CA' in sa and 'CA' in sb:
                            om = dihedral_deg(sa['CA']['pos'], sa['C']['pos'], sb['N']['pos'], sb['CA']['pos'])
                            d_trans = min(abs(om - 180.0), abs(om + 180.0))
                            d_om = min(d_trans, abs(om)) if is_pro_next else d_trans
                            cost += 0.5 * (d_om / 6.0) ** 2
                            if d_om > 35.0:
                                cost += 200.0 * ((d_om - 35.0) / 10.0) ** 2
                    if la != '_' and la == lb:
                        cost -= 0.15
                    mat[a][b] = cost
                    bad[a][b] = is_b
            E_pep[i] = mat
            pep_bad[i] = bad
            continue

        # Check nucleic-acid O3' -> P phosphodiester backbone link
        d_na = min(
            (dist(states[i][a]['atoms']["O3'"]['pos'], states[i + 1][b]['atoms']['P']['pos'])
             for a in range(na) for b in range(nb)
             if "O3'" in states[i][a]['atoms'] and 'P' in states[i + 1][b]['atoms']),
            default=99.0
        )
        if d_na <= 2.50:
            mat = [[0.0] * nb for _ in range(na)]
            for a in range(na):
                sa = states[i][a]['atoms']
                la = states[i][a]['label']
                for b in range(nb):
                    sb = states[i + 1][b]['atoms']
                    lb = states[i + 1][b]['label']
                    cost = 0.0
                    if "O3'" in sa and 'P' in sb:
                        d_op = dist(sa["O3'"]['pos'], sb['P']['pos'])
                        if not (1.35 <= d_op <= 1.90):
                            cost += 5000.0 + 2000.0 * ((d_op - 1.600) ** 2)
                        else:
                            cost += ((d_op - 1.600) / 0.035) ** 2
                        if "C3'" in sa:
                            ang_c = angle_deg(sa["C3'"]['pos'], sa["O3'"]['pos'], sb['P']['pos'])
                            cost += 0.4 * ((ang_c - 120.0) / 4.5) ** 2
                        if "O5'" in sb:
                            ang_p = angle_deg(sa["O3'"]['pos'], sb['P']['pos'], sb["O5'"]['pos'])
                            cost += 0.4 * ((ang_p - 104.0) / 4.5) ** 2
                    if la != '_' and la == lb:
                        cost -= 0.15
                    mat[a][b] = cost
            E_na[i] = mat

    # 2. Disulfide bridges (u, v)
    E_ss = {}
    ss_bad = {}
    for u, v in ss_pairs:
        na, nb = len(states[u]), len(states[v])
        mat = [[0.0] * nb for _ in range(na)]
        bad = [[0] * nb for _ in range(na)]
        for a in range(na):
            sa = states[u][a]['atoms']
            la = states[u][a]['label']
            for b in range(nb):
                sb = states[v][b]['atoms']
                lb = states[v][b]['label']
                cost = 0.0
                is_b = 0
                if 'SG' in sa and 'SG' in sb:
                    d_ss = dist(sa['SG']['pos'], sb['SG']['pos'])
                    if not (1.90 <= d_ss <= 2.30):
                        is_b = 1
                        cost += 5000.0 + 2000.0 * ((d_ss - 2.031) ** 2)
                    else:
                        cost += ((d_ss - 2.031) / 0.030) ** 2
                    if 'CB' in sa:
                        a1 = angle_deg(sa['CB']['pos'], sa['SG']['pos'], sb['SG']['pos'])
                        cost += 0.4 * ((a1 - 104.2) / 3.5) ** 2
                    if 'CB' in sb:
                        a2 = angle_deg(sa['SG']['pos'], sb['SG']['pos'], sb['CB']['pos'])
                        cost += 0.4 * ((a2 - 104.2) / 3.5) ** 2
                if la != '_' and la == lb:
                    cost -= 0.15
                mat[a][b] = cost
                bad[a][b] = is_b
        E_ss[(u, v)] = mat
        ss_bad[(u, v)] = bad

    # 3. Crystal-symmetry nonbonded heavy-atom clash graph + auto-detected metal/glycan/ligand covalent bonds
    inv_c = 1.0 / 2.8
    all_x, all_y, all_z = [], [], []
    for i in indices:
        for st in states[i]:
            for _, _, (x, y, z) in st['heavy']:
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
    if not all_x:
        raise ValueError(f'{source_label}: no heavy atoms with occupancy > 0 found')

    xmin, xmax = min(all_x) - 3.0, max(all_x) + 3.0
    ymin, ymax = min(all_y) - 3.0, max(all_y) + 3.0
    zmin, zmax = min(all_z) - 3.0, max(all_z) + 3.0

    # Auto-detect home-cell non-peptide covalent & metal-coordination links (e.g. His-Zn, glycans, covalent ligands)
    ss_set = set(ss_pairs) | {(v, u) for u, v in ss_pairs}
    home_grid = defaultdict(list)
    for j in indices:
        for b, st in enumerate(states[j]):
            for name, elem, (px, py, pz) in st['heavy']:
                ck = (int(math.floor(px * inv_c)), int(math.floor(py * inv_c)), int(math.floor(pz * inv_c)))
                home_grid[ck].append((j, b, name, elem, px, py, pz))

    cov_candidates = defaultdict(list)  # (i, j, name_a, name_b, is_metal) -> list of (a, b, d)
    for i in indices:
        for a, st_a in enumerate(states[i]):
            for name_a, elem_a, (ax, ay, az) in st_a['heavy']:
                cx, cy, cz = int(math.floor(ax * inv_c)), int(math.floor(ay * inv_c)), int(math.floor(az * inv_c))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for (j, b, name_b, elem_b, bx, by, bz) in home_grid.get((cx + dx, cy + dy, cz + dz), ()):
                                if j <= i:
                                    continue
                                if j == i + 1 and (i in E_pep or i in E_na):
                                    continue
                                if (i, j) in ss_set and name_a == 'SG' and name_b == 'SG':
                                    continue
                                d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
                                a_met = elem_a in _METALS
                                b_met = elem_b in _METALS
                                if a_met != b_met:
                                    met_el = elem_a if a_met else elem_b
                                    lig_el = elem_b if a_met else elem_a
                                    if lig_el in ('N', 'O', 'S', 'SE', 'F', 'CL', 'BR', 'I'):
                                        d_hi = 2.85 if met_el in _LARGE_METALS else 2.55
                                        if 1.55 <= d <= d_hi:
                                            cov_candidates[(i, j, name_a, name_b, True)].append((a, b, d, d_hi))
                                elif not a_met and not b_met:
                                    d_cov_hi = _R_COV.get(elem_a, 0.80) + _R_COV.get(elem_b, 0.80) + 0.30
                                    if 1.10 <= d <= d_cov_hi:
                                        cov_candidates[(i, j, name_a, name_b, False)].append((a, b, d, d_cov_hi))

    exempt_12 = set()   # (i, j, name_a, name_b)
    exempt_13 = set()   # (i, j, name_a, name_b)
    E_cov = defaultdict(lambda: None)
    for (i, j, name_a, name_b, is_metal), hits in cov_candidates.items():
        # Verify at least one hit is sterically clean outside its 1-3 bonded neighborhood (rules out side-chain crashes)
        clean_ds = []
        for a, b, d, d_hi in hits:
            if is_metal:
                clean_ds.append(d)
                continue
            nb_a = intra_nbrs[(i, a)].get(name_a, set())
            nb_b = intra_nbrs[(j, b)].get(name_b, set())
            ok = True
            for na2, _, pa2 in states[i][a]['heavy']:
                for nb2, _, pb2 in states[j][b]['heavy']:
                    if na2 == name_a and nb2 == name_b:
                        continue
                    if (na2 == name_a and nb2 in nb_b) or (nb2 == name_b and na2 in nb_a):
                        continue
                    if dist(pa2, pb2) < 2.05:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                clean_ds.append(d)
        if not clean_ds:
            continue

        d_ref = sorted(clean_ds)[len(clean_ds) // 2]
        d_hi = hits[0][3]
        exempt_12.add((i, j, name_a, name_b))
        for a in range(len(states[i])):
            for na2 in intra_nbrs[(i, a)].get(name_a, set()):
                exempt_13.add((i, j, na2, name_b))
        for b in range(len(states[j])):
            for nb2 in intra_nbrs[(j, b)].get(name_b, set()):
                exempt_13.add((i, j, name_a, nb2))

        na_cnt, nb_cnt = len(states[i]), len(states[j])
        if E_cov[(i, j)] is None:
            E_cov[(i, j)] = [[0.0] * nb_cnt for _ in range(na_cnt)]
        cmat = E_cov[(i, j)]
        for a in range(na_cnt):
            sa = states[i][a]['atoms']
            la = states[i][a]['label']
            for b in range(nb_cnt):
                sb = states[j][b]['atoms']
                lb = states[j][b]['label']
                if name_a in sa and name_b in sb:
                    d_ab = dist(sa[name_a]['pos'], sb[name_b]['pos'])
                    if d_ab <= d_hi + 0.15:
                        sig = 0.12 if is_metal else 0.05
                        cmat[a][b] += ((d_ab - d_ref) / sig) ** 2
                    else:
                        if is_metal:
                            cmat[a][b] += 400.0 + 100.0 * ((d_ab - d_ref) ** 2)
                        else:
                            cmat[a][b] += 2500.0 + 800.0 * ((d_ab - d_ref) ** 2)
                    if la != '_' and la == lb:
                        cmat[a][b] -= 0.15

    # Fractional bounding box of primary molecule + 3 A margin
    if O_mat is not None and F_mat is not None and O_mat[0][0] > 2.0:
        corners = [(x, y, z) for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)]
        fcorners = [_mat_vec(F_mat, c) for c in corners]
        fmin = [min(fc[k] for fc in fcorners) for k in range(3)]
        fmax = [max(fc[k] for fc in fcorners) for k in range(3)]
    else:
        O_mat = None

    grid = defaultdict(list)
    for j in indices:
        for b, st in enumerate(states[j]):
            for name, elem, pos in st['heavy']:
                if O_mat is None:
                    px, py, pz = pos
                    ck = (int(math.floor(px * inv_c)), int(math.floor(py * inv_c)), int(math.floor(pz * inv_c)))
                    grid[ck].append((j, b, name, elem, px, py, pz, True, True))
                else:
                    f0 = _mat_vec(F_mat, pos)
                    for op_idx, (R, t) in enumerate(sym_ops):
                        rf = _mat_vec(R, f0)
                        ox, oy, oz = rf[0] + t[0], rf[1] + t[1], rf[2] + t[2]
                        ox -= math.floor(ox)
                        oy -= math.floor(oy)
                        oz -= math.floor(oz)
                        kx0, kx1 = int(math.floor(fmin[0] - ox)), int(math.ceil(fmax[0] - ox))
                        ky0, ky1 = int(math.floor(fmin[1] - oy)), int(math.ceil(fmax[1] - oy))
                        kz0, kz1 = int(math.floor(fmin[2] - oz)), int(math.ceil(fmax[2] - oz))
                        for kx in range(kx0, kx1 + 1):
                            fx = ox + kx
                            for ky in range(ky0, ky1 + 1):
                                fy = oy + ky
                                for kz in range(kz0, kz1 + 1):
                                    fz = oz + kz
                                    px, py, pz = _mat_vec(O_mat, (fx, fy, fz))
                                    if px < xmin or px > xmax or py < ymin or py > ymax or pz < zmin or pz > zmax:
                                        continue
                                    is_op0 = (op_idx == 0)
                                    is_ident = is_op0 and (abs(px - pos[0]) + abs(py - pos[1]) + abs(pz - pos[2]) < 1e-4)
                                    ck = (int(math.floor(px * inv_c)), int(math.floor(py * inv_c)), int(math.floor(pz * inv_c)))
                                    grid[ck].append((j, b, name, elem, px, py, pz, is_op0, is_ident))

    self_cost = {i: [0.0] * len(states[i]) for i in indices}
    self_clash = {i: [0] * len(states[i]) for i in indices}
    pair_raw_cost = defaultdict(float)
    pair_raw_clash = defaultdict(int)
    na_o5_set = {'OP1', 'OP2', 'O1P', 'O2P', "O5'"}

    for i in indices:
        for a, st in enumerate(states[i]):
            for name_a, elem_a, (ax, ay, az) in st['heavy']:
                cx, cy, cz = int(math.floor(ax * inv_c)), int(math.floor(ay * inv_c)), int(math.floor(az * inv_c))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for (j, b, name_b, elem_b, bx, by, bz, is_op0, is_ident) in grid.get((cx + dx, cy + dy, cz + dz), ()):
                                if j < i:
                                    continue
                                if j == i and b != a:
                                    continue
                                if j == i and is_ident:
                                    continue
                                dd2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
                                if dd2 >= 7.5625:
                                    continue
                                if is_ident and j == i + 1 and i in E_pep and name_a == 'C' and name_b == 'N':
                                    continue
                                if is_ident and j == i + 1 and i in E_na and name_a == "O3'" and name_b == 'P':
                                    continue
                                d = math.sqrt(dd2)
                                if name_a == 'SG' and name_b == 'SG' and (
                                    (is_ident and (i, j) in ss_set) or 1.90 <= d <= 2.30
                                ):
                                    continue
                                # Symmetry-mate metal coordination is also not a steric clash
                                if (elem_a in _METALS) != (elem_b in _METALS):
                                    met_el = elem_a if (elem_a in _METALS) else elem_b
                                    lig_el = elem_b if (elem_a in _METALS) else elem_a
                                    if lig_el in ('N', 'O', 'S', 'SE', 'F', 'CL', 'BR', 'I'):
                                        if 1.55 <= d <= (2.85 if met_el in _LARGE_METALS else 2.55):
                                            continue
                                if is_ident and ((i, j, name_a, name_b) in exempt_12 or (i, j, name_a, name_b) in exempt_13):
                                    continue
                                is_13_pep = is_ident and (j == i + 1 and i in E_pep) and (
                                    (name_a in ('CA', 'O') and name_b == 'N') or
                                    (name_a == 'C' and name_b in ('CA', 'CD'))
                                )
                                is_13_na = is_ident and (j == i + 1 and i in E_na) and (
                                    (name_a == "C3'" and name_b == 'P') or
                                    (name_a == "O3'" and name_b in na_o5_set)
                                )
                                is_13_ss = ((i, j) in ss_set) and ({name_a, name_b} == {'CB', 'SG'})
                                if is_13_pep or is_13_na or is_13_ss:
                                    continue
                                is_hard = 1 if d < 2.20 else 0
                                c_val = 0.0
                                if is_hard:
                                    c_val += 1500.0 + 3000.0 * ((2.20 - d) ** 2)
                                else:
                                    d0 = 2.52 if (elem_a in ('N', 'O') and elem_b in ('N', 'O')) else 2.72
                                    if d < d0:
                                        c_val += 8.0 * ((d0 - d) / (d0 - 2.20)) ** 2

                                if j == i:
                                    self_cost[i][a] += 0.5 * c_val
                                    self_clash[i][a] += is_hard
                                else:
                                    pair_raw_cost[(i, a, j, b)] += c_val
                                    if is_hard:
                                        pair_raw_clash[(i, a, j, b)] += 1

    for i in indices:
        for a in range(len(states[i])):
            self_clash[i][a] = (self_clash[i][a] + 1) // 2

    all_ij = set((k[0], k[2]) for k in pair_raw_cost.keys()) | set((k[0], k[2]) for k in pair_raw_clash.keys())
    for i in E_pep:
        all_ij.add((i, i + 1))
    for i in E_na:
        all_ij.add((i, i + 1))
    for u, v in ss_pairs:
        all_ij.add((min(u, v), max(u, v)))
    for (i, j), cm in E_cov.items():
        if cm is not None:
            all_ij.add((i, j))

    pair_neighbors = defaultdict(list)
    total_pair_cost = {}
    clash_pair_cnt = {}
    for (i, j) in sorted(all_ij):
        na, nb = len(states[i]), len(states[j])
        cmat = [[0.0] * nb for _ in range(na)]
        kmat = [[0] * nb for _ in range(na)]
        cov_m = E_cov.get((i, j))
        for a in range(na):
            for b in range(nb):
                c = pair_raw_cost.get((i, a, j, b), 0.0)
                k = pair_raw_clash.get((i, a, j, b), 0)
                if j == i + 1 and i in E_pep:
                    c += E_pep[i][a][b]
                if j == i + 1 and i in E_na:
                    c += E_na[i][a][b]
                if (i, j) in E_ss:
                    c += E_ss[(i, j)][a][b]
                elif (j, i) in E_ss:
                    c += E_ss[(j, i)][b][a]
                if cov_m is not None:
                    c += cov_m[a][b]
                cmat[a][b] = c
                kmat[a][b] = k
        if any(cmat[a][b] != 0.0 or kmat[a][b] != 0 for a in range(na) for b in range(nb)):
            total_pair_cost[(i, j)] = cmat
            clash_pair_cnt[(i, j)] = kmat
            if na > 1 or nb > 1:
                pair_neighbors[i].append((j, cmat, False))
                pair_neighbors[j].append((i, cmat, True))

    E_chain = dict(E_pep)
    for i, m in E_na.items():
        if i not in E_chain:
            E_chain[i] = m

    return E_pep, pep_bad, E_ss, ss_bad, self_cost, self_clash, total_pair_cost, clash_pair_cnt, pair_neighbors, E_chain


def score_members(n_res, assign, pep_bad, ss_bad, self_clash, clash_pair_cnt, self_cost, total_pair_cost):
    M = len(assign)
    t_pep, t_ss, t_cl, t_E = 0, 0, 0, 0.0
    for m in range(M):
        am = assign[m]
        for i, bmat in pep_bad.items():
            t_pep += bmat[am[i]][am[i + 1]]
        for (u, v), bmat in ss_bad.items():
            t_ss += bmat[am[u]][am[v]]
        cl_m = 0
        for i in range(n_res):
            cl_m += self_clash[i][am[i]]
            t_E += self_cost[i][am[i]]
        for (i, j), kmat in clash_pair_cnt.items():
            cl_m += kmat[am[i]][am[j]]
        for (i, j), cmat in total_pair_cost.items():
            t_E += cmat[am[i]][am[j]]
        t_cl += cl_m
    return t_pep / M, t_ss / M, t_cl / M, t_E / M


def reconnect_strict_quotas(n_res, slots_per_res, ss_pairs, E_chain, self_cost, pair_neighbors):
    """Exact Hungarian sequential initialization + iterative single-site & multi-cut block-matching sweeps."""
    M = len(slots_per_res[0])
    assign = [{0: slots_per_res[0][m]} for m in range(M)]
    for i in range(n_res - 1):
        j = i + 1
        slots_j = slots_per_res[j]
        cost_mat = [[0.0] * M for _ in range(M)]
        has_link = (i in E_chain)
        for m in range(M):
            ai = assign[m][i]
            for s, bj in enumerate(slots_j):
                c = (E_chain[i][ai][bj] if has_link else 0.0) + self_cost[j][bj]
                for (u, cmat, transp) in pair_neighbors[j]:
                    if u in assign[m] and (not has_link or u != i):
                        au = assign[m][u]
                        c += cmat[au][bj] if transp else cmat[bj][au]
                cost_mat[m][s] = c
        perm = hungarian_min_cost(cost_mat)
        for m in range(M):
            assign[m][j] = slots_j[perm[m]]

    candidate_intervals = []
    for k in range(n_res - 1):
        candidate_intervals.append((0, k))
    for length in (1, 2, 3, 4, 6, 10, 15):
        for s in range(max(0, n_res - length + 1)):
            candidate_intervals.append((s, s + length - 1))
    for u, v in ss_pairs:
        if u < v:
            candidate_intervals.append((u, v - 1))
            candidate_intervals.append((u + 1, v))

    for round_idx in range(12):
        improved = False
        # 1. Single-residue exact Hungarian updates
        for i in range(n_res):
            slots_i = slots_per_res[i]
            if len(set(slots_i)) <= 1:
                continue
            nbrs = pair_neighbors[i]
            cost_mat = [[0.0] * M for _ in range(M)]
            cur_e = 0.0
            for m in range(M):
                am = assign[m]
                ai_cur = am[i]
                c_cur = self_cost[i][ai_cur]
                for (j, cmat, transp) in nbrs:
                    aj = am[j]
                    c_cur += cmat[aj][ai_cur] if transp else cmat[ai_cur][aj]
                cur_e += c_cur
                for s, ai in enumerate(slots_i):
                    c = self_cost[i][ai]
                    for (j, cmat, transp) in nbrs:
                        aj = am[j]
                        c += cmat[aj][ai] if transp else cmat[ai][aj]
                    cost_mat[m][s] = c
            perm = hungarian_min_cost(cost_mat)
            new_e = sum(cost_mat[m][perm[m]] for m in range(M))
            if new_e < cur_e - 1e-6:
                for m in range(M):
                    assign[m][i] = slots_i[perm[m]]
                improved = True

        # 2. Sub-chain interval-cut exact Hungarian swaps
        has_var = [len(set(slots_per_res[r])) > 1 for r in range(n_res)]
        for (ia, ib) in candidate_intervals:
            if not any(has_var[ia:ib + 1]):
                continue
            inside = set(range(ia, ib + 1))
            outside = [r for r in range(n_res) if r not in inside]
            if not outside:
                continue
            cross_edges = []
            for u in outside:
                for (v, cmat, transp) in pair_neighbors[u]:
                    if v in inside:
                        cross_edges.append((u, v, cmat, transp))
            if not cross_edges:
                continue
            subchains = [{v: assign[s][v] for v in inside} for s in range(M)]
            cost_mat = [[0.0] * M for _ in range(M)]
            cur_e = 0.0
            for m in range(M):
                am = assign[m]
                for s in range(M):
                    sub = subchains[s]
                    c = 0.0
                    for (u, v, cmat, transp) in cross_edges:
                        au, bv = am[u], sub[v]
                        c += cmat[bv][au] if transp else cmat[au][bv]
                    cost_mat[m][s] = c
                cur_e += cost_mat[m][m]
            perm = hungarian_min_cost(cost_mat)
            new_e = sum(cost_mat[m][perm[m]] for m in range(M))
            if new_e < cur_e - 1e-6:
                for m in range(M):
                    sub = subchains[perm[m]]
                    for v in inside:
                        assign[m][v] = sub[v]
                improved = True

        if not improved:
            break
    return assign, round_idx + 1


def relax_orphan_slots(n_res, states, assign, self_cost, pair_neighbors, occ_pen=80.0, occ_flex=0):
    """Bounded occupancy relaxation (at most +-occ_flex conformer slots from initial integer quota per state)."""
    M = len(assign)
    target_occ = {i: [st['occ'] for st in states[i]] for i in range(n_res)}
    init_counts = {i: [0] * len(states[i]) for i in range(n_res)}
    for m in range(M):
        for i in range(n_res):
            init_counts[i][assign[m][i]] += 1
    counts = {i: list(init_counts[i]) for i in range(n_res)}

    if occ_flex > 0:
        for _ in range(6):
            changed = False
            for m in range(M):
                am = assign[m]
                for i in range(n_res):
                    na = len(states[i])
                    if na <= 1:
                        continue
                    cur_a = am[i]
                    nbrs = pair_neighbors[i]

                    def member_res_e(a_cand):
                        c = self_cost[i][a_cand]
                        for (j, cmat, transp) in nbrs:
                            aj = am[j]
                            c += cmat[aj][a_cand] if transp else cmat[a_cand][aj]
                        return c

                    cur_geom = member_res_e(cur_a)
                    if cur_geom < 50.0:
                        continue
                    best_a = cur_a
                    best_delta = 0.0
                    for a_cand in range(na):
                        if a_cand == cur_a:
                            continue
                        if abs((counts[i][cur_a] - 1) - init_counts[i][cur_a]) > occ_flex:
                            continue
                        if abs((counts[i][a_cand] + 1) - init_counts[i][a_cand]) > occ_flex:
                            continue
                        new_occ_cur = (counts[i][cur_a] - 1) / M
                        new_occ_cand = (counts[i][a_cand] + 1) / M
                        old_op = ((counts[i][cur_a] / M - target_occ[i][cur_a]) ** 2 +
                                  (counts[i][a_cand] / M - target_occ[i][a_cand]) ** 2) * occ_pen * M
                        new_op = ((new_occ_cur - target_occ[i][cur_a]) ** 2 +
                                  (new_occ_cand - target_occ[i][a_cand]) ** 2) * occ_pen * M
                        delta = (member_res_e(a_cand) - cur_geom) + (new_op - old_op)
                        if delta < best_delta - 1e-6:
                            best_delta = delta
                            best_a = a_cand
                    if best_a != cur_a:
                        counts[i][cur_a] -= 1
                        counts[i][best_a] += 1
                        am[i] = best_a
                        changed = True
            if not changed:
                break

    devs = [abs(counts[i][a] / M - target_occ[i][a]) for i in range(n_res) for a in range(len(states[i]))]
    slot_devs = [abs(counts[i][a] - init_counts[i][a]) for i in range(n_res) for a in range(len(states[i]))]
    max_d = max(devs) if devs else 0.0
    rms_d = math.sqrt(sum(d * d for d in devs) / max(len(devs), 1))
    return assign, max_d, rms_d, max(slot_devs, default=0)


def _ideal_amide_h_pos(pos_c_prev, pos_n, pos_ca, d_nh=0.860):
    """Planar sp2 riding amide H/D position along negative bisector of N->C_{i-1} and N->CA_i."""
    ux, uy, uz = pos_c_prev[0] - pos_n[0], pos_c_prev[1] - pos_n[1], pos_c_prev[2] - pos_n[2]
    vx, vy, vz = pos_ca[0] - pos_n[0], pos_ca[1] - pos_n[1], pos_ca[2] - pos_n[2]
    nu = math.sqrt(ux * ux + uy * uy + uz * uz)
    nv = math.sqrt(vx * vx + vy * vy + vz * vz)
    if nu < 1e-6 or nv < 1e-6:
        return None
    bx = -(ux / nu + vx / nv)
    by = -(uy / nu + vy / nv)
    bz = -(uz / nu + vz / nv)
    nb = math.sqrt(bx * bx + by * by + bz * bz)
    if nb < 1e-6:
        return None
    return (pos_n[0] + d_nh * bx / nb, pos_n[1] + d_nh * by / nb, pos_n[2] + d_nh * bz / nb)


def materialize_conformers(res_order, residues, states, assign, E_pep, rebuild_h=True, add_missing_h=False):
    """Build per-conformer atom lists (blanking altlocs, setting occ=1.0, and re-bisecting cross-letter amide H/D).

    add_missing_h: also add backbone amide H to a heavy-atom-only model (the solvent predictors type atoms with H).
    """
    M = len(assign)
    n_res = len(res_order)
    nh_samples = []
    n_h_atoms = 0
    n_d_atoms = 0
    for i in range(n_res):
        for st in states[i]:
            for a in st['all_atoms']:
                if a['elem'] == 'H':
                    n_h_atoms += 1
                elif a['elem'] == 'D':
                    n_d_atoms += 1
            if 'N' in st['atoms']:
                for h_nm in ('H', 'D'):
                    if h_nm in st['atoms']:
                        d = dist(st['atoms']['N']['pos'], st['atoms'][h_nm]['pos'])
                        if 0.75 <= d <= 1.15:
                            nh_samples.append(d)
                        break
    d_nh = sorted(nh_samples)[len(nh_samples) // 2] if nh_samples else 0.860
    default_amide_sym = 'D' if (n_d_atoms > 0 and n_h_atoms == 0) else 'H'
    fill_missing_h = add_missing_h or (n_h_atoms + n_d_atoms > 0)

    conformers = [[] for _ in range(M)]
    rebuilt_h_count = 0
    for m in range(M):
        am = assign[m]
        for i in range(n_res):
            st_cur = states[i][am[i]]
            atoms_cur = [dict(a) for a in st_cur['all_atoms']]
            if rebuild_h and (i - 1) in E_pep and residues[res_order[i]]['resname'] != 'PRO':
                st_prev = states[i - 1][am[i - 1]]
                if 'C' in st_prev['atoms'] and 'N' in st_cur['atoms'] and 'CA' in st_cur['atoms']:
                    p_c = st_prev['atoms']['C']['pos']
                    p_n = st_cur['atoms']['N']['pos']
                    p_ca = st_cur['atoms']['CA']['pos']
                    if dist(p_c, p_n) <= 1.65:
                        h_idx = next((k for k, a in enumerate(atoms_cur) if a['name'] in ('H', 'D')), None)
                        if h_idx is None:
                            need_bisect = fill_missing_h
                        else:
                            need_bisect = st_prev['label'] != st_cur['label'] or atoms_cur[h_idx]['occ'] <= 0.0
                        if need_bisect:
                            new_h = _ideal_amide_h_pos(p_c, p_n, p_ca, d_nh=d_nh)
                            if new_h is not None:
                                rebuilt_h_count += 1
                                if h_idx is not None:
                                    atoms_cur[h_idx]['pos'] = new_h
                                else:
                                    n_at = st_cur['atoms']['N']
                                    atoms_cur.append({
                                        'name': default_amide_sym,
                                        'alt': ' ',
                                        'chain': n_at['chain'],
                                        'seqnum': n_at['seqnum'],
                                        'icode': n_at['icode'],
                                        'resname': n_at['resname'],
                                        'pos': new_h,
                                        'occ': 1.0,
                                        'biso': n_at['biso'],
                                        'elem': default_amide_sym,
                                        'het': n_at.get('het', False),
                                    })
            conformers[m].extend(atoms_cur)
    return conformers, rebuilt_h_count


def reconnect_pdb_to_conformers(pdb_path, nconf=32, occ_flex=0, quotas='hamilton', rebuild_h=True, add_missing_h=False):
    """High-level entry point returning (cryst1_line, conformers, stats)."""
    if nconf < 1:
        raise ValueError(f'nconf must be >= 1 (got {nconf})')
    cryst1_line, res_order, residues = parse_multi_altloc_pdb(pdb_path)
    n_res = len(res_order)
    if n_res == 0:
        raise ValueError(f'{pdb_path}: no non-water residues found')
    states, alpha_slots, hamilton_slots, meta = build_clean_states(res_order, residues, nconf=nconf)
    ss_pairs = detect_disulfide_pairs(res_order, residues, states)
    (E_pep, pep_bad, E_ss, ss_bad, self_cost, self_clash,
     total_pair_cost, clash_pair_cnt, pair_neighbors, E_chain) = precompute_energies(
        cryst1_line, res_order, residues, states, ss_pairs, source_label=str(pdb_path)
    )

    alpha_assign = [{i: alpha_slots[i][m] for i in range(n_res)} for m in range(nconf)]
    a_pep, a_ss, a_cl, _ = score_members(
        n_res, alpha_assign, pep_bad, ss_bad, self_clash, clash_pair_cnt, self_cost, total_pair_cost
    )

    base_slots = alpha_slots if quotas == 'quantile' else hamilton_slots
    assign, n_sweeps = reconnect_strict_quotas(n_res, base_slots, ss_pairs, E_chain, self_cost, pair_neighbors)
    assign, max_docc, rms_docc, max_dslot = relax_orphan_slots(
        n_res, states, assign, self_cost, pair_neighbors, occ_pen=80.0, occ_flex=occ_flex
    )
    u_pep, u_ss, u_cl, _ = score_members(
        n_res, assign, pep_bad, ss_bad, self_clash, clash_pair_cnt, self_cost, total_pair_cost
    )

    conformers, rebuilt_h = materialize_conformers(
        res_order, residues, states, assign, E_pep, rebuild_h=rebuild_h, add_missing_h=add_missing_h
    )
    stats = {
        'n_res': n_res,
        'n_pep_links': len(E_pep),
        'n_ss_links': len(ss_pairs),
        'ss_residue_pairs': [(residues[res_order[u]]['seqnum'], residues[res_order[v]]['seqnum']) for u, v in ss_pairs],
        'nconf': nconf,
        'n_sweeps': n_sweeps,
        'dropped_ghosts': meta['dropped_ghosts'],
        'buggy_alpha_holes': meta['buggy_alpha_holes'],
        'restored_zero_h': meta['restored_zero_h'],
        'rebuilt_amide_h': rebuilt_h,
        'alpha_clean_geom': (a_pep, a_ss, a_cl),
        'reconnected_geom': (u_pep, u_ss, u_cl),
        'max_docc': max_docc,
        'rms_docc': rms_docc,
        'max_dslot': max_dslot,
    }
    return cryst1_line, conformers, stats


def _format_pdb_atom(serial, a, alt_char=' ', occ_val=1.0, occ_decimals=2, warned_long_res=None):
    nm = a['name']
    name_col = nm[:4] if len(nm) >= 4 else (' ' + nm).ljust(4)
    if len(a['chain']) > 2:
        raise ValueError(f"chain ID {a['chain']!r} is longer than the 2 characters a PDB file can hold")
    rname = a['resname']
    if len(rname) > 3:
        if warned_long_res is not None and rname not in warned_long_res:
            warned_long_res.add(rname)
            print(f"reconnect: truncating residue name {rname!r} to 3 characters ({rname[:3]!r}) for PDB output",
                  file=sys.stderr)
        rname = rname[:3]
    seq_col = _format_hy36_4(int(a['seqnum']))
    biso_val = min(max(float(a['biso']), -99.99), 999.99)
    return '%-6s%5d %-4s%1s%3s%2s%4s%1s   %8.3f%8.3f%8.3f%6.*f%6.2f          %2s' % (
        'HETATM' if a.get('het') else 'ATOM', serial % 100000, name_col, alt_char, rname, a['chain'].rjust(2),
        seq_col, a['icode'], a['pos'][0], a['pos'][1], a['pos'][2], occ_decimals, occ_val, biso_val,
        a['elem'][:2].rjust(2)
    )


def write_member_pdbs(cryst1_line, conformers, out_dir, pseudo_path=None):
    os.makedirs(out_dir, exist_ok=True)
    nconf = len(conformers)
    header = cryst1_line + '\n' if cryst1_line else ''
    warned_long_res = set()
    for m, atoms in enumerate(conformers):
        lines = [_format_pdb_atom(k + 1, a, alt_char=' ', occ_val=1.0, warned_long_res=warned_long_res)
                 for k, a in enumerate(atoms)]
        out_p = os.path.join(out_dir, 'm%02d.pdb' % (m + 1))
        with open(out_p, 'w', newline='\n') as f:
            f.write(header + '\n'.join(lines) + '\nEND\n')

    if pseudo_path:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        if nconf > len(letters):
            raise ValueError(f'pseudo.pdb supports at most {len(letters)} conformers (got {nconf})')
        base, extra = divmod(10000, nconf)
        w_list = [(base + (1 if m < extra else 0)) / 10000.0 for m in range(nconf)]
        plines = []
        serial = 0
        for m, atoms in enumerate(conformers):
            for a in atoms:
                serial += 1
                plines.append(_format_pdb_atom(
                    serial, a, alt_char=letters[m], occ_val=w_list[m], occ_decimals=4, warned_long_res=warned_long_res
                ))
        with open(pseudo_path, 'w', newline='\n') as f:
            f.write(header + '\n'.join(plines) + '\nEND\n')


def main(argv=None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = {'nconf': '--nconf', 'occ_flex': '--occ-flex', 'quotas': '--quotas', 'pseudo': '--pseudo'}
    conv = []
    for tok in argv:
        k, eq, v = tok.partition('=')
        if eq and k in legacy:
            conv += [legacy[k], v]
        elif eq and k == 'rebuild_h':
            if v.lower() in ('0', 'false', 'no'):
                conv.append('--no-rebuild-h')
        else:
            conv.append(tok)
    p = argparse.ArgumentParser(prog='reconnect.py', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pdb', help='Input multi-altloc PDB/mmCIF file.')
    p.add_argument('out_dir', help='Output directory for m01.pdb .. mNN.pdb.')
    p.add_argument('--nconf', type=int, default=32, help='Number of complete conformers (default: 32).')
    p.add_argument('--occ-flex', type=int, default=0, help='Bounded quota slack +-k/nconf (default: 0).')
    p.add_argument('--quotas', default='hamilton', choices=['hamilton', 'quantile'], type=str.lower,
                   help='Quota rounding rule.')
    p.add_argument('--no-rebuild-h', action='store_true', help='Disable cross-letter amide H/D bisector rebuilding.')
    p.add_argument('--pseudo', default=None, help='Optional combined multi-altloc pseudo-PDB output path.')
    a = p.parse_args(conv)
    pdb_path, out_dir = a.pdb, a.out_dir
    nconf, occ_flex, quotas = a.nconf, a.occ_flex, a.quotas
    rebuild_h = not a.no_rebuild_h
    pseudo_path = a.pseudo

    cryst1_line, conformers, st = reconnect_pdb_to_conformers(
        pdb_path, nconf=nconf, occ_flex=occ_flex, quotas=quotas, rebuild_h=rebuild_h
    )
    write_member_pdbs(cryst1_line, conformers, out_dir, pseudo_path=pseudo_path)

    na = [len(c) for c in conformers]
    a_pep, a_ss, a_cl = st['alpha_clean_geom']
    u_pep, u_ss, u_cl = st['reconnected_geom']
    print(f"{pdb_path}: {st['n_res']} residues, {nconf} conformers, atoms/conformer {min(na)}-{max(na)} "
          f"(dropped {len(st['dropped_ghosts'])} H-only ghost altlocs fixing {st['buggy_alpha_holes']} missing-residue holes; "
          f"restored {st['restored_zero_h']} zero-occ H, bisected {st['rebuilt_amide_h']} cross-letter amide H)")
    print(f"  alphabetical quantiles (clean): broken peptide {a_pep:.2f}/{st['n_pep_links']}, "
          f"broken SS {a_ss:.2f}/{st['n_ss_links']}, clashes <2.2A {a_cl:.2f}")
    print(f"  geometry-reconnected (flex={occ_flex}): broken peptide {u_pep:.2f}/{st['n_pep_links']}, "
          f"broken SS {u_ss:.2f}/{st['n_ss_links']}, clashes <2.2A {u_cl:.2f} "
          f"(max|d_slot|={st['max_dslot']}, max|d_occ|={st['max_docc']:.3f}, rms|d_occ|={st['rms_docc']:.4f})")
    if pseudo_path:
        print(f"  wrote {pseudo_path}: {nconf} members, {sum(na)} atoms")


if __name__ == '__main__':
    main()
