# surfNturf

**Differentiable multi-conformer crystal hydration (*Surf*), conformer reconnection (*Turf*), 3D water mobility $B(\mathbf{x})$, and X-ray bulk-solvent modelling.**

Traditional crystallographic bulk-solvent models place a flat step mask at an arbitrary van der Waals offset around the macromolecule (`k_sol`, `b_sol`). Real crystal solvent is neither flat nor binary: polar hydrogen-bond donors and acceptors carve sharp first-shell hydration peaks, hydrophobic patches create depletion moats, water–water packing generates oscillatory second shells around `2.8 Å` and `4.5 Å`, and alternate macromolecular conformers (*Turf*) displace solvent (*Surf*) in mutually exclusive pockets.

**surfNturf** replaces the flat step mask with a fast, analytically differentiable, space-group-exact hydration field trained once on explicit-water crystal MD and transferable in seconds to any single-conformer, multi-altloc, or multi-model crystal structure — with **zero MD simulation required at application time**.

---

## Physical Architecture (*Surf* & *Turf*)

Given any crystallographic structure (single-conformer, `A/B`, heterogeneous multi-altloc PDB/mmCIF, or a directory of `m*.pdb` conformers from an external linear-programming untangler such as Passmore's `Untangler`), **surfNturf** evaluates five physically coupled layers on the exact crystal Asymmetric Unit (`ASUGrid`):

1. **Conformer Completion — *Turf* (`surfnturf.reconnect` or external `m*.pdb` ensembles):**
   Accepts pre-untangled conformer directories (`m01.pdb .. mNN.pdb`) directly, or expands heterogeneous multi-altloc PDB/mmCIF files on the fly into $M$ (default `32`) sterically compatible complete conformers via a fast Kuhn–Munkres Hungarian + interval-cut heuristic (`surfnturf reconnect`), filtering riding-H ghost altlocs, preserving DNA/disulfide/metal/glycan bonds, and re-bisecting cross-letter amide `H/D` atoms.
2. **10-Class Protonation-Aware Solute–Water PMF (`140` spline parameters):**
   Separates explicit polar donor hydrogens (`H_pol` at `1.85 Å`, cationic `H_pos` at `1.78 Å`), nonpolar `C–H` (`H_apol`), anionic carboxylate/coordinating ions (`O_neg`), carbonyl (`O_carb`), hydroxyl/aromatic-acceptor (`O_hyd`), donor `N` / large polar ions (`N_don`), `C_sp2` / `P`, `C_sp3`, and `S_sul` / `Se` using $C^2$ cardinal cubic B-splines on `[1.40, 5.60 Å]`.
3. **Grand-Canonical Langmuir Saturation (`1` parameter):**
   $$g_1^{(m)}(\mathbf{x}) = \frac{1}{\alpha}\,\sigma\!\left(\ln\frac{\alpha}{1-\alpha} - U_0^{(m)}(\mathbf{x})\right)$$
   Multi-tethered groove waters cooperate multiplicatively while saturating smoothly at physical single-site occupancy (`1/\alpha`).
4. **1-FFT Ornstein–Zernike Water–Water Packing (`5` parameters) & Acceptor Water-H Polarization (`27` parameters):**
   Convolves ensemble-averaged first-shell excess density with radial Gaussian shells (`1.25, 1.90, 2.80, 3.60, 4.50 Å`) to create the `1.9 Å` exclusion moat around pinned waters and `2.8 Å` second-shell H-bond bridges, plus a strictly zero-net-charge (`f_H(0) = 0`) acceptor-directed water-hydrogen polarization term.
5. **Exact `IT92` `(O + 2H)` Fourier Rendering, 3D Mobility $B(\mathbf{x})$, and Sparse `PeakDict` / `WatershedDivotDict`:**
   * Renders molecular water number density into electron density (`e/Å³`) with International Tables `IT92` oxygen + orientational `2H` (`r_OH = 0.9572 Å`) form factors, conserving `F000` identically.
   * **`surfnturf.bmob`**: Renders spatially varying water mobility $B(\mathbf{x}) \in [0.625, 2560]\text{ \AA}^2$ via a `13`-level Catmull–Rom partition-of-unity basis on $\ln B$, anchored to local protein $B_{\text{iso}}$ and H-bond tether count or refined against work-set $|F_{\text{obs}}|$.
   * **`surfnturf.divot`**: Captures crystal-specific interstitial cavities, cryoprotectant/ion pockets, and sub-voxel hydration peaks from exact `e/Å³` work-set `mFo - DFc` maps via continuous-coordinate `IT92` peaks (`ContinuousPeakDict`), Bader steepest-ascent basins (`WatershedDivotDict`), or wall-clamped mass-conserving transport (`W_mc`), with strict blind hyperparameter selection on an internal `5%` work-validation split (`FreeR_flag == 0` is never touched).

Every layer has **exact hand-derived analytical reverse-mode adjoints** (BLAS `DGEMV` + Parseval 3D FFTs, verified against central finite differences to `1e-10`–`5e-7`).

---

## Installation

`surfNturf` requires Python `>= 3.9` with `numpy`, `scipy`, and `gemmi` (all pre-installed in modern Phenix `2.x` and CCP4 `9` Python environments):

```bash
git clone https://github.com/jmholton/surfNturf.git
cd surfNturf
pip install -e .
```

On read-only synchrotron/cluster installations (`phenix.python` or `ccp4-python`), `surfNturf` also runs **directly from the git checkout without `pip install`**:

```bash
phenix.python -m surfnturf --help
phenix.python -m surfnturf test
```

---

## Quickstart

### 1. Zero-MD Hydration Maps & `FPART/PHIPART` Structure Factors
Predict de-blurred water oxygen relative probability (`_want.ccp4`), steric exclusion (`_cannot.ccp4` in `[0, 1]`), rendered total solvent density (`_solvent.ccp4` in `e/Å³`), ordered-water-displaced bulk solvent (`_bulk.ccp4`), and reciprocal-space `FPART/PHIPART` structure factors (`_fpart.mtz`):

```bash
# Onto the exact Miller indices of an experimental diffraction MTZ:
surfnturf predict model.pdb out/solv --hkl refme.mtz

# Or from a directory of Untangler / ensemble member PDBs (m01.pdb .. m32.pdb):
surfnturf predict untangled_members/ out/solv --hkl refme.mtz

# Or directly via phenix.python without pip install:
phenix.python -m surfnturf predict model.pdb out/solv hkl=refme.mtz B=5.0

# Dry prediction (ignoring any existing HOH records in model.pdb):
surfnturf predict model.pdb out/solv_dry --hkl refme.mtz --ignore-waters

# With zero-shot 6-parameter physical 3D mobility B(x):
surfnturf predict model.pdb out/solv_phys --hkl refme.mtz --bmob dry_a4 --ignore-waters
```

Use the resulting `out/solv_fpart.mtz` in `phenix.refine` (user bulk-solvent build):

```bash
phenix.refine model.pdb refme.mtz \
  refinement.input.bulk_solvent_map.file_name=out/solv_fpart.mtz \
  refinement.input.bulk_solvent_map.amplitudes_label=FPART \
  refinement.input.bulk_solvent_map.phases_label=PHIPART
```

### 2. Heuristic Multi-Altloc Conformer Reconnection (`surfnturf reconnect`)
Expand a heterogeneous multi-altloc PDB/mmCIF into `32` clash-free reconnected conformer PDBs (`m01.pdb .. m32.pdb`):

```bash
surfnturf reconnect model_multialt.pdb out/members --nconf 32
```

### 3. Work-Set X-Ray Mobility $B(\mathbf{x})$ & Sparse Peak/Divot Refinement
First export a 0-cycle `*_f_model.mtz` from `phenix.refine` with `output.export_final_f_model=True` (providing `FOBS`, `R_FREE_FLAGS`, `FCALC`, `PHIFCALC`, `FMASK`, `PHIFMASK`, `FMODEL`, `PHIFMODEL`, `K_MASK`, and `K_ISOTROPIC`/`K_ANISOTROPIC`):

```bash
# Joint 3.2 A-coupled 3D water mobility B(x) + PMF refinement on work |Fo|:
surfnturf fit-bmob --pdb model.pdb --fmodel-mtz score_001_f_model.mtz --out out/bmob --mode joint

# Blind work_val-selected sparse PeakDict + WatershedDivotDict refinement (FreeR_flag==0 untouched):
surfnturf fit-divot --pdb model.pdb --fmodel-mtz score_001_f_model.mtz --data-mtz refme.mtz --out out/hybrid --arm hybrid
```

### 4. Python API

```python
import surfnturf

cell, sg, grid, members, want_3d, cannot_3d, solv_3d = surfnturf.predict_maps_for_target(
    "model.pdb", weights="l2", B_map=5.0, grid_spacing=0.45, nconf=32
)
surfnturf.write_ccp4("solvent.ccp4", solv_3d, cell, sg)
surfnturf.write_fpart_mtz_to_dmin(solv_3d, cell, sg, dmin=1.0, out_mtz_path="solvent_fpart.mtz")
```

---

## Benchmarks

1. **Zero-shot cross-trajectory & cross-$B$ generalization against 32-conformer explicit-water AMBER MD maps (`1aho`, `1.2–6.0 Å` shell):**
   * Increases unseen-trajectory real-space correlation from `CC = 0.72` (2-type `N/O` vs `C/S` Gaussian baseline) to **`CC = 0.875–0.881`** across all 4 trajectories (`RMSE = 0.068–0.095 e/Å³`), outperforming periodic 32-conformer `AMBER22 3D-RISM KH` (`< 3 Å` shell `CCv = +0.70` vs `-0.16`).
2. **Ground-truth diffraction benchmark (`1aho`, `FreeR_flag == 0` untouched):**
   * Zero-MD `surfNturf` + `3.2 Å B(x)` + `Hybrid (Divot + PeakDict)` reaches **`R_true_free = 7.12%`** (`E_solv = 0.336`), matching a 50 ns explicit-water MD start + `Hybrid` (`7.09%`, `E_solv = 0.332`).
   * Over 30 macrocycles of `phenix.refine`, zero-MD `surfNturf` + `PeakDict` achieves `R_work / R_free = 7.03% / 8.91%`, `R_true_free = 8.45%`, and heavy-atom `RMSD = 0.0606 Å` (`max 0.337 Å`), outperforming 50 ns MD + Fourier shift-and-scale (`7.84% / 10.44%`, `R_true_free = 10.16%`).
3. **Real `1aho` experimental diffraction (`0.96 Å`, `P 21 21 21`, `0`-cycle common ruler on `1aho_best_so_far_trimmed.pdb`, `DRY` = `0` coordinate waters, `0` MD):**
   * Drops `R_work / R_free` from `17.22% / 17.93%` (Phenix flat mask) → `13.26% / 14.21%` (`surfNturf` uniform `B = 5 Å²`) → `11.18% / 13.08%` (`+ 3.2 Å B(x)`) → **`9.25% / 11.09%`** (`+ 3.2 Å B(x)` + blind `PeakDict`), surpassing `DRY` water-restrained 2019 MD (`10.03% / 11.79%`) and `DRY` iterative map solvent (`9.98% / 11.96%`).
4. **Cross-protein, cross-space-group, sub-atomic generalization on `8C5N` (`0.75 Å`, monoclinic `P 1 21 1`, $\beta = 99.25^\circ$, `186` residues, `198,461` reflections, `9,968` `FreeR_flag == 0` reflections untouched; `1aho` MD weights frozen):**
   * **Zero-shot water-site recall (`161` experimental `0.75 Å` waters):** `98.8%` sterically allowed (`cannot < 0.2`), `95.0%` at `want > 1.2`, with median $g_O = 6.42$ (`mean = 11.04`, `90th percentile = 24.82` vs open-solvent mean `1.40`).
   * **`DRY` (`0` coordinate `HOH` in PDB):** `21.22% / 21.72%` (Phenix flat mask) → `19.26% / 19.77%` (0-shot `surfNturf`, `-4.04%` in `3.5–1.8 Å`) → `17.69% / 18.49%` (`+ 3.2 Å B(x)`) → **`12.18% / 13.49%`** (`+ blind Hybrid`), **beating the deposited `161`-water `WET` model + flat mask (`14.60% / 15.53%`) by `-2.04%` `R_free` with zero coordinate waters**.
   * **`WET` (`161` deposited `HOH` + `CL` + `ANISOU` untouched):** `15.40% / 16.30%` (PDB header) → `14.60% / 15.53%` (Phenix flat mask) → `13.96% / 14.86%` (0-shot `surfNturf`) → **`11.08% / 12.69%`** (`+ 3.2 Å B(x) + blind Hybrid`, `-2.84%` overall `R_free` vs Phenix flat mask and `-5.19%` in `3.5–1.8 Å`).

---

## Verification Suite

Run all analytical vs. central finite-difference gradient and symmetry/mass-conservation checks:

```bash
surfnturf test
# or:
pytest
```

## License

MIT License — Copyright (c) 2026 James Holton. See [LICENSE](LICENSE).
