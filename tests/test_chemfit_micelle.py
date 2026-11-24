"""Verification tests for differentiable chemical feedback (`surfnturf.chemfit`) and amphiphilic micelles (`surfnturf.micelle`):
  1. Exact C-infinity analytical gradients vs central finite differences (< 1e-6) for CrystalChemAdapter + ChemicalSiteBasis
     (simultaneous 6-knob mother-liquor `gamma` + per-site `zeta` + IT92 `Delta F_calc` under `global_scalar` k_mask);
  2. Exact equivalence of sparse ChemicalSiteBasis `ASN` flip deltas against full `build_asu_features` rebuild;
  3. Blind recovery of a backwards `ASN` 180-deg amide flip from X-ray amplitudes + hydration, noise rejection on `val_mask`,
     and bit-for-bit zero `R-free` leakage (`f_obs[free_mask] *= 100` changes 0 bits of fitted parameters or maps);
  4. Exact analytical gradients (< 1e-6), zero protein-core leak, interior beta-barrel aqueous-pore protection, blind
     alkane/headgroup density recovery, soluble-protein rejection, and bit-for-bit zero `R-free` leakage in `micelle.py`.
"""
import io
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import gemmi

from surfnturf.field import (
    RHO_BULK,
    ASUGrid,
    WaterFieldSpec,
    build_asu_features,
    compute_pmf_density_with_pullback,
    load_ensemble,
    solv_to_fpart,
)
from surfnturf.io import load_saved_spec_and_theta
from surfnturf.bmob import (
    MultiBRenderer,
    ReciprocalTarget,
    SlimDatasetBundle,
    fast_build_asu_and_mobility,
    load_ensemble_with_bfac,
)
from surfnturf.divot import (
    CrystalGridOps,
    SolventAmplitudeProblem,
    build_open_solvent_envelope,
)
from surfnturf.chemfit import (
    CrystalChemAdapter,
    apply_confident_chemistry_to_pdb,
    build_chemical_site_basis,
    chem_loss_and_grad,
    fit_chemistry_blind_gated,
)
from surfnturf.micelle import (
    AmphiphilicMicelleModel,
    detect_transmembrane_belt,
    fit_micelle_blind,
)
from surfnturf.cli import main as cli_main


def _tetramer_pdb_text(flip_asn: bool = False) -> str:
    """Compact P 1 21 1 tetrapeptide containing ASN 1, HIS 2, ASP 3, SER 4 with standard covalent geometry."""
    od1 = (10.390, 8.959, 9.323)
    nd2 = (9.785, 6.882, 8.798)
    if flip_asn:
        od1, nd2 = nd2, od1
    lines = [
        "CRYST1   22.000   22.000   24.000  90.00  90.00  90.00 P 1 21 1      2",
        "ATOM      1  N   ASN A   1       6.200   9.200   8.200  1.00 10.00           N",
        "ATOM      2  CA  ASN A   1       7.550   9.700   8.400  1.00 10.00           C",
        "ATOM      3  C   ASN A   1       7.750  11.100   7.850  1.00 10.00           C",
        "ATOM      4  O   ASN A   1       6.850  11.920   7.800  1.00 10.50           O",
        "ATOM      5  CB  ASN A   1       8.580   8.780   7.750  1.00 11.00           C",
        "ATOM      6  CG  ASN A   1       9.650   8.200   8.680  1.00 11.50           C",
        f"ATOM      7  OD1 ASN A   1    {od1[0]:8.3f}{od1[1]:8.3f}{od1[2]:8.3f}  1.00 12.00           O",
        f"ATOM      8  ND2 ASN A   1    {nd2[0]:8.3f}{nd2[1]:8.3f}{nd2[2]:8.3f}  1.00 12.00           N",
        "ATOM      9  N   HIS A   2       8.980  11.350   7.420  1.00 10.00           N",
        "ATOM     10  CA  HIS A   2       9.350  12.650   6.900  1.00 10.50           C",
        "ATOM     11  C   HIS A   2      10.820  12.880   7.200  1.00 10.50           C",
        "ATOM     12  O   HIS A   2      11.650  12.000   7.050  1.00 11.00           O",
        "ATOM     13  CB  HIS A   2       9.080  12.750   5.390  1.00 11.50           C",
        "ATOM     14  CG  HIS A   2       7.650  12.550   5.020  1.00 12.00           C",
        "ATOM     15  ND1 HIS A   2       6.650  13.480   5.150  1.00 12.50           N",
        "ATOM     16  CE1 HIS A   2       5.520  12.980   4.700  1.00 12.50           C",
        "ATOM     17  NE2 HIS A   2       5.750  11.750   4.280  1.00 12.50           N",
        "ATOM     18  CD2 HIS A   2       7.080  11.460   4.480  1.00 12.00           C",
        "ATOM     19  N   ASP A   3      11.120  14.080   7.680  1.00 10.50           N",
        "ATOM     20  CA  ASP A   3      12.480  14.450   8.020  1.00 11.00           C",
        "ATOM     21  C   ASP A   3      12.920  15.720   7.300  1.00 11.00           C",
        "ATOM     22  O   ASP A   3      12.120  16.620   7.080  1.00 11.50           O",
        "ATOM     23  CB  ASP A   3      12.600  14.620   9.540  1.00 11.50           C",
        "ATOM     24  CG  ASP A   3      13.980  14.980  10.020  1.00 12.50           C",
        "ATOM     25  OD1 ASP A   3      14.920  14.220   9.750  1.00 13.50           O",
        "ATOM     26  OD2 ASP A   3      14.100  16.020  10.700  1.00 13.50           O",
        "ATOM     27  N   SER A   4      14.200  15.780   6.950  1.00 11.00           N",
        "ATOM     28  CA  SER A   4      14.780  16.950   6.300  1.00 11.50           C",
        "ATOM     29  C   SER A   4      16.250  16.720   6.020  1.00 12.00           C",
        "ATOM     30  O   SER A   4      16.700  15.580   5.900  1.00 12.50           O",
        "ATOM     31  OXT SER A   4      16.950  17.750   5.920  1.00 13.00           O",
        "ATOM     32  CB  SER A   4      14.580  18.180   7.180  1.00 12.00           C",
        "ATOM     33  OG  SER A   4      13.220  18.550   7.250  1.00 13.00           O",
        "END",
    ]
    return "\n".join(lines) + "\n"


def _synthetic_beta_barrel_pdb() -> str:
    """Build an 88-residue hollow transmembrane barrel aligned along Z:
    - Outer lipid-facing wall (R ~ 11.8 A, |z - 25| <= 11 A): LEU / ILE / VAL + TRP girdles at |z - 25| ~ 10.5 A;
    - Inner lumen wall (R ~ 6.2 A, |z - 25| <= 11 A): hydrophilic ASN / SER / ASP facing the central aqueous pore;
    - Extramembranous top/bottom caps (|z - 25| ~ 14.5 A): charged ASP / LYS / GLU loops.
    """
    lines = ["CRYST1   54.000   54.000   52.000  90.00  90.00  90.00 P 1           1"]
    serial = 1
    resseq = 1
    cx, cy, cz = 27.0, 27.0, 26.0

    def _add_atom(rn, an, el, x, y, z):
        nonlocal serial
        lines.append(
            f"ATOM  {serial:5d} {an:<4s} {rn:3s} A{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 18.00          {el:>2s}"
        )
        serial += 1

    # 8 transmembrane beta-strands around the cylinder
    for s in range(8):
        ang = 2.0 * math.pi * s / 8.0
        ux, uy = math.cos(ang), math.sin(ang)
        tx, ty = -uy, ux
        for k, dz in enumerate((-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5)):
            z = cz + dz
            # Alternating outward (lipid-facing) vs inward (pore-facing) residues along each beta strand
            if k in (0, 6):
                # Aromatic TRP girdle facing outward
                r_bb = 9.4
                bx, by = cx + r_bb * ux, cy + r_bb * uy
                _add_atom("TRP", "N", "N", bx - 0.6 * tx, by - 0.6 * ty, z - 0.5)
                _add_atom("TRP", "CA", "C", bx, by, z)
                _add_atom("TRP", "C", "C", bx + 0.6 * tx, by + 0.6 * ty, z + 0.5)
                _add_atom("TRP", "O", "O", bx + 1.2 * tx, by + 1.2 * ty, z + 0.5)
                _add_atom("TRP", "CB", "C", cx + 10.9 * ux, cy + 10.9 * uy, z)
                _add_atom("TRP", "CG", "C", cx + 11.9 * ux, cy + 11.9 * uy, z + 0.3)
                _add_atom("TRP", "NE1", "N", cx + 12.6 * ux, cy + 12.6 * uy, z + 0.9)
                _add_atom("TRP", "CZ2", "C", cx + 13.2 * ux, cy + 13.2 * uy, z - 0.2)
                _add_atom("TRP", "CH2", "C", cx + 13.5 * ux, cy + 13.5 * uy, z - 1.0)
                resseq += 1
            elif k % 2 == 1:
                # Exterior hydrophobic LEU
                r_bb = 9.5
                bx, by = cx + r_bb * ux, cy + r_bb * uy
                _add_atom("LEU", "N", "N", bx - 0.6 * tx, by - 0.6 * ty, z - 0.5)
                _add_atom("LEU", "CA", "C", bx, by, z)
                _add_atom("LEU", "C", "C", bx + 0.6 * tx, by + 0.6 * ty, z + 0.5)
                _add_atom("LEU", "O", "O", bx + 1.2 * tx, by + 1.2 * ty, z + 0.5)
                _add_atom("LEU", "CB", "C", cx + 11.0 * ux, cy + 11.0 * uy, z)
                _add_atom("LEU", "CG", "C", cx + 12.2 * ux, cy + 12.2 * uy, z + 0.4)
                _add_atom("LEU", "CD1", "C", cx + 13.1 * ux + 0.6 * tx, cy + 13.1 * uy + 0.6 * ty, z)
                _add_atom("LEU", "CD2", "C", cx + 13.1 * ux - 0.6 * tx, cy + 13.1 * uy - 0.6 * ty, z + 0.8)
                resseq += 1
            else:
                # Interior lumen-lining polar ASN facing inward toward (cx, cy)
                r_bb = 8.4
                bx, by = cx + r_bb * ux, cy + r_bb * uy
                _add_atom("ASN", "N", "N", bx - 0.6 * tx, by - 0.6 * ty, z - 0.5)
                _add_atom("ASN", "CA", "C", bx, by, z)
                _add_atom("ASN", "C", "C", bx + 0.6 * tx, by + 0.6 * ty, z + 0.5)
                _add_atom("ASN", "O", "O", bx + 1.2 * tx, by + 1.2 * ty, z + 0.5)
                _add_atom("ASN", "CB", "C", cx + 7.0 * ux, cy + 7.0 * uy, z)
                _add_atom("ASN", "CG", "C", cx + 6.0 * ux, cy + 6.0 * uy, z + 0.3)
                _add_atom("ASN", "OD1", "O", cx + 5.3 * ux + 0.5 * tx, cy + 5.3 * uy + 0.5 * ty, z + 0.6)
                _add_atom("ASN", "ND2", "N", cx + 5.3 * ux - 0.5 * tx, cy + 5.3 * uy - 0.5 * ty, z)
                resseq += 1

        # Top & bottom hydrophilic extramembranous cap residues (ASP / LYS) at dz = +-14.5 A
        for dz_cap, cap_rn in ((-14.5, "ASP"), (14.5, "LYS")):
            z = cz + dz_cap
            bx, by = cx + 9.2 * ux, cy + 9.2 * uy
            _add_atom(cap_rn, "N", "N", bx - 0.5 * tx, by - 0.5 * ty, z - 0.4)
            _add_atom(cap_rn, "CA", "C", bx, by, z)
            _add_atom(cap_rn, "C", "C", bx + 0.5 * tx, by + 0.5 * ty, z + 0.4)
            _add_atom(cap_rn, "O", "O", bx + 1.1 * tx, by + 1.1 * ty, z + 0.4)
            if cap_rn == "ASP":
                _add_atom("ASP", "CB", "C", cx + 10.5 * ux, cy + 10.5 * uy, z + 0.3)
                _add_atom("ASP", "CG", "C", cx + 11.4 * ux, cy + 11.4 * uy, z + 0.8)
                _add_atom("ASP", "OD1", "O", cx + 12.1 * ux + 0.6 * tx, cy + 12.1 * uy + 0.6 * ty, z + 1.1)
                _add_atom("ASP", "OD2", "O", cx + 12.1 * ux - 0.6 * tx, cy + 12.1 * uy - 0.6 * ty, z + 0.9)
            else:
                _add_atom("LYS", "CB", "C", cx + 10.5 * ux, cy + 10.5 * uy, z)
                _add_atom("LYS", "NZ", "N", cx + 12.2 * ux, cy + 12.2 * uy, z + 0.9)
            resseq += 1

    lines.append("END")
    return "\n".join(lines) + "\n"


class TestChemfitAndMicelle(unittest.TestCase):
    def test_chemfit_exact_sparse_equivalence_and_joint_fd_gradients(self):
        rng = np.random.default_rng(20261124)
        spec, theta0 = load_saved_spec_and_theta("l2")
        adapter = CrystalChemAdapter(spec, theta0)
        th_zero, _ = adapter.theta_and_pullback(np.zeros(6))
        self.assertTrue(np.array_equal(th_zero, theta0))

        with tempfile.TemporaryDirectory() as td:
            p_orig = os.path.join(td, "orig.pdb")
            p_flip = os.path.join(td, "flip.pdb")
            with open(p_orig, "w", encoding="utf-8", newline="\n") as f:
                f.write(_tetramer_pdb_text(flip_asn=False))
            with open(p_flip, "w", encoding="utf-8", newline="\n") as f:
                f.write(_tetramer_pdb_text(flip_asn=True))

            cell = gemmi.UnitCell(22.0, 22.0, 24.0, 90.0, 90.0, 90.0)
            sg = gemmi.SpaceGroup("P 1 21 1")
            grid = ASUGrid(cell, sg, (24, 24, 24), step=1)

            mem0, bis0, wts0 = load_ensemble_with_bfac(p_orig, nconf=1, use_lp=spec.use_lp)
            mem1, _ = load_ensemble(p_flip, nconf=1, use_lp=spec.use_lp, return_weights=True)
            Phi_O0, Phi_H0, st_H0, _, ext0 = fast_build_asu_and_mobility(
                grid, mem0, bis0, n_types=spec.n_types, include_hpol=spec.use_hpol,
                include_tet_oz=spec.use_tet_oz, member_weights=wts0, return_extra=True,
            )
            Phi_O1, Phi_H1, st_H1, ext1 = build_asu_features(
                grid, mem1, n_types=spec.n_types, include_hpol=spec.use_hpol,
                include_tet_oz=spec.use_tet_oz, member_weights=wts0, return_extra=True,
            )

            H = np.array([[h, k, l] for h in range(-5, 6) for k in range(0, 6) for l in range(0, 6)
                          if (h, k, l) != (0, 0, 0)], dtype=np.int64)
            d_hkl = 1.0 / np.sqrt(np.maximum(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64), 1e-12))
            basis = build_chemical_site_basis(p_orig, grid, spec, hkl=H, d_hkl=d_hkl)
            self.assertEqual(basis.n_sites, 4)  # ASN 1, HIS 2, ASP 3, SER 4

            # Verify sparse ASN 1 flip_180 delta reproduces full from-scratch build_asu_features on flipped PDB!
            ds0 = SlimDatasetBundle("c0", grid, 5.0, Phi_O0, Phi_H0, st_H0, member_weights=wts0, F_hbsol=ext0["F_hbsol"])
            ds1 = SlimDatasetBundle("c1", grid, 5.0, Phi_O1, Phi_H1, st_H1, member_weights=wts0, F_hbsol=ext1["F_hbsol"])
            rend = MultiBRenderer(grid, fft_workers=1)
            z_mob = np.full((1, grid.n_asu), np.log(12.0), dtype=np.float64)

            gO_ref1, gH_ref1, _, _ = compute_pmf_density_with_pullback(spec, ds1, theta0, need_grad=False)
            ghat_ref1, _, _ = rend.render_and_pullback(gO_ref1, gH_ref1, wts0, z_mob, compute_grad=False)
            solv_ref1 = np.maximum(grid.asu_to_full_3d(ghat_ref1) * RHO_BULK, 0.0)

            p_asn_flip = np.zeros(basis.n_zeta, dtype=np.float64)
            asn_site = [s for s in basis.sites if s["family"] == "asn_gln"][0]
            p_asn_flip[asn_site["zeta_slice"]] = 1.0

            dummy_targ = ReciprocalTarget(
                H, np.ones(len(H)) * 10.0, np.ones(len(H), dtype=bool), np.zeros(len(H), dtype=bool),
                np.ones(len(H), dtype=np.complex128) * 5.0, np.ones(len(H)), np.zeros(len(H), dtype=np.int32), 1, d_hkl,
                spacegroup=sg,
            )
            ev_sparse1 = chem_loss_and_grad(
                spec, ds0, rend, dummy_targ, z_mob, adapter, np.zeros(6),
                site_basis=basis, p_alt_override=p_asn_flip, include_atomic_dfc=False,
                kmask_mode="fixed_one", compute_grad=False,
            )
            self.assertLess(float(np.max(np.abs(ev_sparse1["solv_3d"] - solv_ref1))), 2e-5)

            # Simultaneous central FD gradient check on (gamma, zeta) with include_atomic_dfc=True & global_scalar
            free = (np.arange(len(H)) % 20 == 0)
            fc0 = rng.normal(0.0, 25.0, len(H)) + 1j * rng.normal(0.0, 25.0, len(H))
            fo = np.abs(1.05 * (fc0 + 1.15 * ev_sparse1["f_part"])) + rng.uniform(0.3, 1.5, len(H))
            targ = ReciprocalTarget(H, fo, ~free, free, fc0, np.ones(len(H)),
                                    np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg).for_training_subset()

            gamma_pt = rng.uniform(-0.25, 0.25, size=6)
            zeta_pt = rng.uniform(-0.8, 0.8, size=basis.n_zeta)
            ev = chem_loss_and_grad(
                spec, ds0, rend, targ, z_mob, adapter, gamma_pt,
                site_basis=basis, zeta=zeta_pt, include_atomic_dfc=True,
                kmask_mode="global_scalar", lam_gamma=3e-4, lam_kl=2e-4,
                compute_grad=True, strict_val_blind=True,
            )
            g_an = np.concatenate([ev["grad_gamma"], ev["grad_zeta"]])
            d_vec = rng.standard_normal(len(g_an))
            d_vec /= np.linalg.norm(d_vec)
            dg, dz = d_vec[:6], d_vec[6:]
            eps = 1e-4
            lp = chem_loss_and_grad(
                spec, ds0, rend, targ, z_mob, adapter, gamma_pt + eps * dg,
                site_basis=basis, zeta=zeta_pt + eps * dz, include_atomic_dfc=True,
                kmask_mode="global_scalar", lam_gamma=3e-4, lam_kl=2e-4,
                compute_grad=False, strict_val_blind=True,
            )["loss"]
            lm = chem_loss_and_grad(
                spec, ds0, rend, targ, z_mob, adapter, gamma_pt - eps * dg,
                site_basis=basis, zeta=zeta_pt - eps * dz, include_atomic_dfc=True,
                kmask_mode="global_scalar", lam_gamma=3e-4, lam_kl=2e-4,
                compute_grad=False, strict_val_blind=True,
            )["loss"]
            fd = (lp - lm) / (2.0 * eps)
            an = float(np.dot(g_an, d_vec))
            self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

    def test_chemfit_recovers_backwards_asn_and_zero_rfree_leakage(self):
        spec, theta0 = load_saved_spec_and_theta("l2")
        with tempfile.TemporaryDirectory() as td:
            p_true = os.path.join(td, "true.pdb")
            p_wrong = os.path.join(td, "wrong.pdb")
            with open(p_true, "w", encoding="utf-8", newline="\n") as f:
                f.write(_tetramer_pdb_text(flip_asn=False))
            with open(p_wrong, "w", encoding="utf-8", newline="\n") as f:
                f.write(_tetramer_pdb_text(flip_asn=True))

            cell = gemmi.UnitCell(22.0, 22.0, 24.0, 90.0, 90.0, 90.0)
            sg = gemmi.SpaceGroup("P 1 21 1")
            grid = ASUGrid(cell, sg, (24, 24, 24), step=1)
            rend = MultiBRenderer(grid, fft_workers=1)
            z_mob = np.full((1, grid.n_asu), np.log(10.0), dtype=np.float64)

            def _solute_fc_and_fpart(pdb_path, dmin=1.8):
                st_g = gemmi.read_structure(pdb_path)
                dc = gemmi.DensityCalculatorX()
                dc.d_min = dmin
                dc.set_grid_cell_and_spacegroup(st_g)
                dc.put_model_density_on_grid(st_g[0])
                asu = gemmi.transform_map_to_f_phi(dc.grid, half_l=True).prepare_asu_data(dmin=dmin)
                H_a = np.asarray(asu.miller_array, dtype=np.int64)
                Fc_a = np.asarray(asu.value_array, dtype=np.complex128)

                mem, bis, wts = load_ensemble_with_bfac(pdb_path, nconf=1, use_lp=spec.use_lp)
                Phi_O, Phi_H, st_H, _, ext = fast_build_asu_and_mobility(
                    grid, mem, bis, n_types=spec.n_types, include_hpol=spec.use_hpol,
                    include_tet_oz=spec.use_tet_oz, member_weights=wts, return_extra=True,
                )
                ds = SlimDatasetBundle("s", grid, 5.0, Phi_O, Phi_H, st_H, member_weights=wts, F_hbsol=ext["F_hbsol"])
                gO, gH, _, _ = compute_pmf_density_with_pullback(spec, ds, theta0, need_grad=False)
                gh, _, _ = rend.render_and_pullback(gO, gH, wts, z_mob, compute_grad=False)
                s3 = np.maximum(grid.asu_to_full_3d(gh) * RHO_BULK, 0.0)
                return H_a, Fc_a, solv_to_fpart(s3, cell.volume, H_a), ds

            H, Fc_true, Fp_true, _ = _solute_fc_and_fpart(p_true)
            _, Fc_wrong, _, ds_wrong = _solute_fc_and_fpart(p_wrong)
            Fo = np.maximum(np.abs(Fc_true + Fp_true), 0.05)
            free = (np.arange(len(H)) % 20 == 0)
            d_hkl = 1.0 / np.sqrt(np.maximum(np.asarray(cell.calculate_1_d2_array(H), dtype=np.float64), 1e-12))

            basis_w = build_chemical_site_basis(p_wrong, grid, spec, hkl=H, d_hkl=d_hkl)
            targ1 = ReciprocalTarget(H, Fo, ~free, free, Fc_wrong, np.ones(len(H)),
                                     np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg)
            res1 = fit_chemistry_blind_gated(
                spec, theta0, ds_wrong, rend, targ1, z_mob, site_basis=basis_w,
                mode="auto", kmask_mode="fixed_one", include_atomic_dfc=True, maxiter=35,
            )
            self.assertIn(res1["selected_stage"], ("sites", "joint"))
            self.assertLess(res1["r_val"], res1["r_val_table"]["base"] - 0.005)

            asn_shifts = [s for s in res1["site_shifts"] if s["family"] == "asn_gln"]
            self.assertEqual(len(asn_shifts), 1)
            self.assertEqual(asn_shifts[0]["favored"], "flip_180")
            self.assertGreater(asn_shifts[0]["p_favored"], 0.80)

            p_fixed = os.path.join(td, "fixed.pdb")
            edits = apply_confident_chemistry_to_pdb(p_wrong, p_fixed, basis_w, res1["site_probs"])
            self.assertEqual(len(edits), 1)
            st_fixed = gemmi.read_structure(p_fixed)
            od1_fixed = st_fixed[0][0][0].find_atom("OD1", "\0").pos
            self.assertLess(od1_fixed.dist(gemmi.Position(10.390, 8.959, 9.323)), 0.02)

            # IRONCLAD ZERO R-FREE LEAKAGE CHECK: corrupting f_obs[free_mask] by 100x changes ZERO bits!
            Fo_corrupt = Fo.copy()
            Fo_corrupt[free] *= 100.0
            targ2 = ReciprocalTarget(H, Fo_corrupt, ~free, free, Fc_wrong, np.ones(len(H)),
                                     np.zeros(len(H), dtype=np.int32), 1, d_hkl, spacegroup=sg)
            res2 = fit_chemistry_blind_gated(
                spec, theta0, ds_wrong, rend, targ2, z_mob, site_basis=basis_w,
                mode="auto", kmask_mode="fixed_one", include_atomic_dfc=True, maxiter=35,
            )
            self.assertEqual(res1["selected_stage"], res2["selected_stage"])
            self.assertTrue(np.array_equal(res1["gamma"], res2["gamma"]))
            self.assertTrue(np.array_equal(res1["p_alt"], res2["p_alt"]))
            self.assertTrue(np.array_equal(res1["solv_3d"], res2["solv_3d"]))

            # Also verify fit_bmob_blind_gated populates z_mob_m across every stage (including 'spatial')
            from surfnturf.bmob import fit_bmob_blind_gated, reduce_mobility_features_to_asu
            mem_w, bis_w, wts_w = load_ensemble_with_bfac(p_wrong, nconf=1, use_lp=spec.use_lp)
            _, _, _, mob_w, _ = fast_build_asu_and_mobility(
                grid, mem_w, bis_w, n_types=spec.n_types, include_hpol=spec.use_hpol,
                include_tet_oz=spec.use_tet_oz, member_weights=wts_w, return_extra=True,
            )
            mob_asu_w = reduce_mobility_features_to_asu(mob_w, wts_w)
            gO_w, gH_w, sk_w, _ = compute_pmf_density_with_pullback(spec, ds_wrong, theta0, need_grad=False)
            bm_res = fit_bmob_blind_gated(
                rend, targ1, ds_wrong, mob_asu_w, (gO_w, gH_w, sk_w), kmask_mode="fixed_one", maxiter=6,
            )
            self.assertIn("z_mob_m", bm_res)
            self.assertEqual(bm_res["z_mob_m"].shape, (1, grid.n_asu))
            if bm_res["selected_stage"] == "spatial":
                self.assertIn("psi", bm_res)
                self.assertIn("z_phys", bm_res)

    def test_micelle_barrel_pore_protection_fd_grad_and_zero_rfree_leak(self):
        rng = np.random.default_rng(7788)
        with tempfile.TemporaryDirectory() as td:
            p_bar = os.path.join(td, "barrel.pdb")
            p_sol = os.path.join(td, "soluble.pdb")
            with open(p_bar, "w", encoding="utf-8", newline="\n") as f:
                f.write(_synthetic_beta_barrel_pdb())
            with open(p_sol, "w", encoding="utf-8", newline="\n") as f:
                f.write(_tetramer_pdb_text(flip_asn=False))

            # 1. Soluble peptide is rejected cleanly; transmembrane beta-barrel is detected with aqueous pore protected
            belt_sol = detect_transmembrane_belt(p_sol)
            self.assertFalse(belt_sol["is_membrane"])

            belt_bar = detect_transmembrane_belt(p_bar)
            self.assertTrue(belt_bar["is_membrane"])
            self.assertGreater(abs(float(belt_bar["normal"][2])), 0.88)
            self.assertGreaterEqual(belt_bar["n_tm_atoms"], 25)
            self.assertGreaterEqual(belt_bar["n_pore_atoms"], 8)

            cell = gemmi.UnitCell(54.0, 54.0, 52.0, 90.0, 90.0, 90.0)
            sg = gemmi.SpaceGroup("P 1")
            shape = (36, 36, 36)
            cg = CrystalGridOps(cell, sg, shape, conv="plus", fft_workers=1)
            asu_g = ASUGrid(cell, sg, shape, step=1)
            open_3d = build_open_solvent_envelope(cg, p_bar, smooth_B=0.0)
            base_3d = RHO_BULK * open_3d

            mic = AmphiphilicMicelleModel(asu_g, base_3d, open_3d, belt_bar)
            p_true = np.array([9.5, 4.2, 3.3, 0.90, 1.40, 0.255, 0.415, 1.00], dtype=np.float64)
            rho_true, _, _ = mic.forward_and_pullback(p_true, compute_grad=False)

            # Central lumen voxel (27, 27, 26) = index (18, 18, 18) MUST stay purely aqueous (~0.3344 e/A^3),
            # whereas exterior transmembrane belt at x = 27 + 16 = 43 -> index (29, 18, 18) is low-density alkane!
            self.assertAlmostEqual(float(rho_true[18, 18, 18]), RHO_BULK, places=3)
            self.assertLess(float(rho_true[29, 18, 18]), 0.275)
            # Protein wall core stays identically 0.0 e/A^3
            core_vox = open_3d < 1e-4
            self.assertTrue(np.any(core_vox))
            self.assertEqual(float(np.max(rho_true[core_vox])), 0.0)

            # 2. Central finite-difference check across all 8 micelle parameters (< 1e-6)
            H = np.array([[h, k, l] for h in range(-6, 7) for k in range(-6, 7) for l in range(0, 7)
                          if (h, k, l) != (0, 0, 0)], dtype=np.int64)
            idx = cg.hkl_idx(H)
            Q = cg.sf(1.35 * (1.0 - open_3d), idx)
            Fo = np.abs(1.06 * (Q + cg.sf(rho_true, idx)))
            free = (np.arange(len(H)) % 20 == 0)
            prob = SolventAmplitudeProblem(cg, H, Fo, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")

            p_test = np.array([10.2, 3.8, 3.5, 1.05, 1.70, 0.272, 0.398, 0.98], dtype=np.float64)
            rho_t, _, pb_t = mic.forward_and_pullback(p_test, compute_grad=True)
            ev_t = prob.eval_rho(rho_t, active_mask=prob.train, compute_grad=True)
            g_an = pb_t(ev_t["grho"])

            dp = rng.standard_normal(8) * np.array([0.5, 0.3, 0.2, 0.1, 0.15, 0.02, 0.02, 0.03])
            dp /= np.linalg.norm(dp)
            eps = 1e-5
            rp, _, _ = mic.forward_and_pullback(p_test + eps * dp, compute_grad=False)
            rm, _, _ = mic.forward_and_pullback(p_test - eps * dp, compute_grad=False)
            lp = prob.eval_rho(rp, active_mask=prob.train, compute_grad=False)["loss"]
            lm = prob.eval_rho(rm, active_mask=prob.train, compute_grad=False)["loss"]
            fd = (lp - lm) / (2.0 * eps)
            an = float(np.dot(g_an, dp))
            self.assertLess(abs(an - fd) / max(abs(an), abs(fd), 1e-12), 1e-6)

            # 3. Blind micelle fit recovers alkane divot & headgroup collar + zero R-free leak
            rho_fit1, info1 = fit_micelle_blind(prob, cg, base_3d, p_bar, open_3d=open_3d, maxiter=30)
            self.assertEqual(info1["selected_stage"], "micelle")
            self.assertLess(info1["r_val"], info1["r_val_base"] - 0.005)
            mp1 = info1["micelle"]["params"]
            self.assertLess(mp1["rho_tail_eA3"], 0.300)
            self.assertGreater(mp1["rho_head_eA3"], 0.370)

            Fo_corrupt = Fo.copy()
            Fo_corrupt[free] *= 100.0
            prob_corrupt = SolventAmplitudeProblem(cg, H, Fo_corrupt, free, np.ones(len(H)), Q, k_solv_mode="fixed_one")
            rho_fit2, info2 = fit_micelle_blind(prob_corrupt, cg, base_3d, p_bar, open_3d=open_3d, maxiter=30)
            self.assertEqual(info1["params"], info2["params"])
            self.assertTrue(np.array_equal(rho_fit1, rho_fit2))

            # CLI smoke check for fit-chem and fit-micelle --help
            s_help = io.StringIO()
            with redirect_stdout(s_help), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cli_main(["fit-micelle", "--help"])
                self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
