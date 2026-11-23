# Physical Principles of `surfNturf`

`surfNturf` models the coupled boundary between **macromolecular conformers (*Turf*)** and **crystal hydration & bulk solvent (*Surf*)** in X-ray and neutron crystallography. Every equation in the package is chosen to obey liquid-state statistical mechanics, crystallographic space-group symmetry, International Tables atomic scattering physics, and exact conservation of unit-cell electrons (`F000`), while remaining fast and analytically differentiable end-to-end.

---

## 1. From the Shrunken Continuum Mask to Structured Hydration

For over three decades, macromolecular refinement has modeled disordered crystal solvent with the rolled-probe and shrunken continuum mask (Jiang & Brünger 1994; Fokine & Urzhumtsev 2002; Afonine et al. 2005, 2013):
$$F_{\text{model}}(\mathbf{h}) = k_{\text{total}}(\mathbf{h})\Big[F_{\text{calc}}(\mathbf{h}) + k_{\text{mask}}(s)\,F_{\text{mask}}(\mathbf{h})\Big],$$
where a probe sphere of radius $r_{\text{probe}} \approx 1.1\text{ \AA}$ defines the solvent-accessible boundary around heavy atoms and a morphological shrink radius $r_{\text{shrink}} \approx 0.9\text{ \AA}$ pulls convex boundaries back to $r_{\text{vdW}} + (r_{\text{probe}} - r_{\text{shrink}}) \approx r_{\text{vdW}} + 0.2\text{ \AA}$, scaled across resolution shells by $k_{\text{mask}}(s)$ (or $k_{\text{sol}} e^{-B_{\text{sol}} s^2}$).

At low resolution ($d \gtrsim 4.5\text{--}6\text{ \AA}$), this time-honored formulation is remarkably effective, fast, and parameter-frugal: $r_{\text{shrink}} \approx 0.9\text{ \AA}$ acts as a well-tuned single-surface geometric compromise between close polar H-bonding contacts and wider nonpolar van der Waals exclusion. Between `~4 Å` and atomic resolution—where individual hydration shells and alternate conformers contribute strongly to structure factors—extending the uniform continuum introduces five physical refinements:

1. **Chemically specific onset & first-shell peaks:** Because a single `(r_probe, r_shrink)` offset must balance all surface elements simultaneously, shrinking to `r_vdW + 0.2 Å` (`~1.7 Å` on `O`, `~2.15 Å` on `C`) brings continuum density inside the `2.1–3.4 Å` hydrophobic exclusion zone around aliphatic carbons while still capping `5–20×` polar H-bond peaks (`O...O_w ≈ 2.65 Å`, `H...O_w ≈ 1.85 Å`) at $1\times$ bulk (`0.334 e/Å³`).
2. **Liquid water–water packing (second-shell oscillations):** Finite water molecular size (`~2.8 Å`) creates an annular exclusion trough (`1.2–2.3 Å`) around pinned first-shell waters together with cooperative `2.8 Å` and `4.5 Å` second- and third-shell bridges.
3. **Molecular `(O + 2H)` charge distribution:** Near hydrogen-bond acceptors (`C=O`, `COO-`), water molecules orient a proton inward by `r_OH = 0.9572 Å`, shifting electron density ~`1 Å` closer to the acceptor without changing total electron count (`F000`).
4. **Ensemble averaging across alternate conformers (*Turf*):** A single union mask treats minor conformers (`occ < 1`) with full steric exclusion, leaving zero continuum density at partial-occupancy water sites displaced by minor side-chain rotamers, whereas per-conformer hydration weights solvent naturally in proportion to `1 - occ`.
5. **Spatially varying 3D mobility $B(\mathbf{x})$:** Rather than applying one global $B_{\text{sol}}$ (or isotropic shell profile $k_{\text{mask}}(s)$) everywhere in the unit cell, local water mobility varies smoothly from `B ≈ 4–10 Å²` in multi-tethered clefts to `B ≈ 150–800 Å²` in wide solvent channels.

`surfNturf` builds on the `FPART / PHIPART` bulk-solvent architecture to model these effects through six coupled, analytically differentiable layers.

---

## 2. Layer 0 — *Turf*: Conformer Ensembles & Nonlinear Solvent Exclusion (`reconnect.py`)

X-ray diffraction amplitudes measure the coherent space-and-time average across unit cells in the crystal mosaic. When a macromolecule populates $M$ complete conformational states with weights $w_m$ ($\sum_{m=1}^M w_m = 1$), each conformer $m$ carves its own steric and electrostatic cavity and equilibrates with its own conditional hydration distribution $\rho_{\text{solv}}^{(m)}(\mathbf{x})$:
$$\langle \rho_{\text{solv}}(\mathbf{x}) \rangle = \sum_{m=1}^M w_m\,\rho_{\text{solv}}^{(m)}(\mathbf{x}).$$

Because steric exclusion $\exp(-U_0^{(m)}(\mathbf{x}))$ is strongly nonlinear, **averaging density across complete conformers differs fundamentally from evaluating one hydration field on occupancy-weighted atoms**:
$$\sum_{m=1}^M w_m\,e^{-U_0^{(m)}(\mathbf{x})} \neq \exp\!\left(-\sum_{m=1}^M w_m\,U_0^{(m)}(\mathbf{x})\right).$$
At a pocket occupied by a side-chain rotamer in `30%` of unit cells ($U_0^{(A)} \approx +20$, $e^{-U_0^{(A)}} \approx 0$) and open to water in `70%` of unit cells ($U_0^{(B)} \approx -1.8$, $g_O^{(B)} \approx 6$), the ensemble average yields a partial-occupancy water peak $0.70 \times 6 = 4.2\times$ bulk—whereas a potential averaged prior to exponentiation ($0.30(+20) + 0.70(-1.8) = +4.74$) would suppress water occupancy ($e^{-4.74} \approx 0.009$).

To construct the $M$ conformers:
* **External ILP ensembles:** `surfNturf` directly ingests directories of complete conformer PDBs (`m01.pdb .. mNN.pdb`), such as those produced by global integer-linear-programming untanglers (Passmore's `Untangler`).
* **Fast heuristic reconnection (`surfnturf.reconnect`):** When given a raw multi-altloc PDB/mmCIF directly, `reconnect.py` expands partial-chain altlocs into $M$ (default `32`) complete conformers:
  1. **Riding-H ghost elimination:** Refinement programs tie backbone amide `H(i, L)` occupancy to `C(i-1, L)`, leaving positive-occupancy lone `H` atoms on altlocs whose heavy atoms refined to zero occupancy, and zeroing `H` on valid heavy-atom altlocs across cross-letter joins. `reconnect` determines active altloc states strictly from heavy atoms (`occ > 0`) while preserving genuine `H/D`-only rotamer splits (e.g., `Ser HG A/B` on single-conformer heavy atoms) and re-bisecting cross-letter amide `H/D` positions.
  2. **Hungarian + interval-cut matching:** Quota counts per residue are assigned by Hamilton largest-remainder rounding (`sum = M`), then paired sequentially and refined by single-site and sub-chain interval-cut Kuhn–Munkres sweeps minimizing peptide `C–N` bond/angle/$\omega$ strain, nucleic-acid `O3'–P` strain, disulfide `SG–SG` strain, metal-coordination / covalent link strain, and crystal-symmetry nonbonded clashes (`< 2.20 Å` hard + van der Waals ramp).

---

## 3. Layer 1 — Protonation-Aware 10-Class Solute–Water PMF, Explicit Lone Pairs & Optional Crystal E-Field (`field.py`)

### Directional Hydrogen Bonding from Explicit Protons and $sp^2 / sp^3$ Lone-Pair Virtual Sites
Heavy-atom-only solvation potentials must approximate angular hydrogen-bond geometry using multi-body coordinate frames because a heavy `N` or `O` atom does not encode where its proton or lone pairs point. `surfNturf` resolves donor and acceptor directionality directly through **explicit macromolecular hydrogens** and **explicit lone-pair virtual sites (`use_lp=True`)**:

* **Donor linearity (`N–H...O_w`):** An amide or amine group consists of a repulsive heavy `N_don` core ($r_0 \approx 2.9\text{ \AA}$) behind an attractive `H_pol` / `H_pos` proton displaced by `0.86–1.01 Å` along the covalent `N–H` bond ($r_{\min} \approx 1.85\text{ \AA}$). Summing $u_{\text{N\_don}} + u_{\text{H\_pol}}$ creates a directional minimum along the linear `N–H...O_w` vector while sterically blocking the back and sides of the nitrogen.
* **Acceptor lone-pair lobes (`LP_sp2`, `LP_neg`, `LP_sp3` at $d_{\text{LP}} = 0.65\text{ \AA}$):** Around a bare oxygen nucleus, a radial pair spline $u_O(r)$ is azimuthally symmetric around the `C=O` axis, smearing 1-tethered waters into a cone. Placing massless virtual sites (`LP_TYPES`, indices `10..12`) at $d_{\text{LP}} = 0.65\text{ \AA}$ breaks that rotational degeneracy using purely additive radial splines:
  * **`LP_sp2` (`10`):** Two in-plane $120^\circ$ lone pairs on neutral carbonyl/amide `C=O`, plus one external-bisector in-plane lone pair on unprotonated aromatic `N` (`HIS ND1/NE2`, nucleic ring `N`).
  * **`LP_neg` (`11`):** Two in-plane $120^\circ$ lone pairs on anionic carboxylate `COO-` (`ASP`, `GLU`, `OXT`) and outward oxyanion lobes (`PO4`, `SO4`).
  * **`LP_sp3` (`12`):** Two tetrahedral ($109.47^\circ$) out-of-plane lone pairs on hydroxyl `O` (`SER OG`, `THR OG1`, `TYR OH`, falling back to canonical staggered gauche lobes `CA–CB–OG` when explicit hydroxyl `H` is absent) and on unprotonated $sp^3$ ether / glycosidic / phosphodiester bridge oxygens (`C–O–C`, ribose `O4'`, nucleic `O3'/O5'`).

### Optional Screened Crystal Electric-Field Polarization (`use_epol=False` by default)
In deep electropositive or push–pull salt-bridge clefts (`Lys+/Arg+ ... HOH ... O=C`), non-additive electrostatic polarization further stabilizes bridging waters. When `use_epol=True` (`surfnturf train --epol`), `field.py` assigns tautomer-aware AMBER partial charges $q_a$ and evaluates the Thole-damped ($a_{\text{Thole}} = 0.65\text{ \AA}$), Yukawa-screened ($\lambda_D = 5.5\text{ \AA}$, $R_E = 7.5\text{ \AA}$) electric field $\tilde{\mathbf{E}}^{(m)}(\mathbf{x})$ gated smoothly outside covalent cores, contributing linear alignment $|\tilde{\mathbf{E}}|$, induction $|\tilde{\mathbf{E}}|^2$, and pure 3-body cross-atom cooperativity $|\sum_a \tilde{\mathbf{E}}_a|^2 - \sum_a |\tilde{\mathbf{E}}_a|^2$ (constructive in donor–acceptor push–pull clefts; destructive in like-charge frustrated pockets). Because its marginal crystallographic gain beyond `+LP` and tetrahedral closure is small (`< 0.02%` $R_{\text{free}}$) at higher setup cost, `+Epol` is kept **off by default**.

### Chemical Atom Typing (`ATOM_TYPES`)
Each non-water atom in conformer $m$ is assigned to one of 10 physical classes:

| Index | Class | Physical Species | Primary Role in $U_0^{(m)}(\mathbf{x})$ |
|---:|---|---|---|
| `0` | `O_carb` | Neutral carbonyl/amide `O` (`C=O`, `ASN OD1`, `GLN OE1`) | Lone-pair H-bond acceptor well around `2.7–3.0 Å` |
| `1` | `O_neg` | Carboxylate/oxyanion `O` (`ASP`, `GLU`, `OXT`, `PO4`, `SO4`), `F-`, compact divalent/alkali cations (`Mg2+`, `Zn2+`, `Ca2+`, `Na+`, `Fe`, `Cu`, `Mn`) | Deep electrostatic first-shell coordination well (`2.2–2.8 Å`) |
| `2` | `O_hyd` | Hydroxyl/ether `O` (`SER OG`, `THR OG1`, `TYR OH`, `C–O–C` / `P–O–C` bridges) & unprotonated `HIS ND1/NE2` | Amphoteric / lone-pair H-bond acceptor (`2.8–3.1 Å`) |
| `3` | `N_don` | Protonated/tertiary `N` core + large polar ions (`Cl-`, `Br-`, `I-`, `K+`) | Steric core `< 2.75 Å` + `3.1–3.4 Å` polar shell |
| `4` | `H_pol` | Neutral polar donor `H/D` (bonded to neutral `N` or `O`) | Directional H-bond donor attraction (`1.7–2.1 Å`) |
| `5` | `H_pos` | Cationic donor `H/D` (`LYS HZ*`, `ARG HE/HH*`, N-term `NH3+`, doubly protonated `HIP`) | Tighter salt-bridge / cationic H-bond attraction (`1.65–2.05 Å`) |
| `6` | `C_sp2` | Planar $sp^2$ carbon (backbone `C`, aromatic/guanidinium/carboxyl `C`) & tetrahedral `P` | Steric exclusion `< 3.3 Å`, weak $\pi$/dispersion trough at `3.6–3.8 Å` |
| `7` | `C_sp3` | Aliphatic $sp^3$ carbon (`CA`, `CB`, methyls) | Hydrophobic steric exclusion `< 3.4 Å`, shallow `3.8 Å` packing shell |
| `8` | `H_apol` | Nonpolar carbon-bonded `H/D` | Fine steric contour (`< 2.5 Å`) defining surface grooves |
| `9` | `S_sul` | Sulfur & selenium (`CYS SG`, `MET SD`, `MSE SE`) | Larger chalcogen steric core (`< 3.1 Å`), dispersion well at `3.5–4.0 Å` |

### $C^2$ Cardinal Cubic B-Spline Potentials
For each atom class $t \in \{0 \dots 9\}$, the dimensionless pair potential $u_t(r)$ (in units of $k_B T$) is expanded in uniform $C^2$-continuous cardinal cubic B-splines across $N_{\text{knots}} = 14$ knots at $r_k = 1.40 + 0.30\,k\text{ \AA}$ ($k = 0 \dots 13$, spanning `[1.40, 5.30 Å]`), with knots $k \ge 14$ ($r \ge 5.60\text{ \AA}$) pinned identically to zero so $u_t(r) \to 0$ in bulk water:
$$U_0^{(m)}(\mathbf{x}) = \sum_{a \in \mathcal{A}_m^{\text{sym}}} u_{t_a}\!\left(\|\mathbf{x} - \mathbf{x}_a\|_2\right),$$
where $\mathcal{A}_m^{\text{sym}}$ includes all crystal space-group and unit-cell lattice images within $R_{\text{cut}} = 5.60\text{ \AA}$ of the Asymmetric Unit (`ASUGrid`). Second-difference spline curvature penalties enforce smooth physical radial profiles.

---

## 4. Layer 2 — Grand-Canonical Langmuir Saturation (`field.py`)

In a pure dilute-gas Boltzmann approximation $g(\mathbf{x}) = e^{-U_0(\mathbf{x})}$, additive pair potentials in a buried cleft surrounded by three or four polar H-bond partners ($U_0 \approx -8$ to $-12$) produce unphysical water densities of $e^{+10} \approx 22{,}000\times$ bulk—as if thousands of water oxygens could overlap inside one `2.8 Å` cavity.

Modeling each microscopic hydration site as a grand-canonical lattice-gas cell of hard-core molecular exclusion volume $v_0$ with bulk occupancy fraction $\alpha = n_{\text{bulk}} v_0 \in (0, 1)$ gives the exact Fermi–Dirac / Langmuir site probability $p^{(m)}(\mathbf{x}) = \sigma(\ln\frac{\alpha}{1-\alpha} - U_0^{(m)}(\mathbf{x}))$. Dividing by the bulk occupancy $\alpha$ (where $U_0 = 0 \implies p_{\text{bulk}} = \alpha$) yields the relative oxygen pair distribution:
$$g_1^{(m)}(\mathbf{x}) = \frac{1}{\alpha}\,\sigma\!\left(\ln\frac{\alpha}{1-\alpha} - U_0^{(m)}(\mathbf{x})\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

This single learned parameter (`logit_alpha` $= \ln\frac{\alpha}{1-\alpha}$) satisfies all three physical limits smoothly:
* **Bulk water ($U_0 = 0$):** $g_1^{(m)} = 1$ identically.
* **Steric protein core ($U_0 \gg 1$):** $g_1^{(m)} \sim \frac{1}{1-\alpha} e^{-U_0} \to 0$.
* **Multi-tethered cleft ($U_0 \ll -1$):** $g_1^{(m)} \to \frac{1}{\alpha}$ (saturating smoothly at `100%` single-water site occupancy rather than diverging exponentially).

---

## 5. Layer 3 — Multi-Step Tetrahedral ($109.47^\circ$) Ornstein–Zernike Closure & Rosenfeld 0D Cavity Cap (`field.py`)

### Multi-Step Radial + Spherical-Harmonic ($l = 0, 1, 2$) Tetrahedral H-Bond Closure (`use_tet_oz=True`)
Solute–water pair potentials $U_0(\mathbf{x})$—even with explicit carbonyl lone pairs—still leave a `1-HB` surface water (e.g. `HOH 113` on `Gly34 O`) split across two equivalent $sp^2$ lone-pair lobes until **water–water hydrogen-bond network closure** breaks the tie. Solving the full 6D molecular Ornstein–Zernike equation is prohibitively slow, whereas a purely isotropic radial shell cannot distinguish four neighbors arranged at the vertices of a regular $109.47^\circ$ tetrahedron from four neighbors bunched on a one-sided hemisphere.

`surfNturf` decomposes the Stillinger–Weber / mW 3-body tetrahedral frustration $\sum_{j \neq k} (\hat{\mathbf{u}}_{ij}\cdot\hat{\mathbf{u}}_{ik} + \tfrac{1}{3})^2$ **analytically into $l = 0, 1, 2$ spherical-harmonic moments** on the `2.80 Å` H-bond shell ($\sigma_{\text{tet}} = 0.30\text{ \AA}$):
$$N_c(\mathbf{x}) = (w_{\text{hb}} * \rho_{\text{hb}})(\mathbf{x}), \qquad \mathbf{J}_1(\mathbf{x}) = \big((w_{\text{hb}}\hat{\mathbf{u}}) * \rho_{\text{hb}}\big)(\mathbf{x}), \qquad Q_{2,p}(\mathbf{x}) = \big((w_{\text{hb}}\phi_p(\hat{\mathbf{u}})) * \rho_{\text{hb}}\big)(\mathbf{x}),$$
where $\{\phi_p\}_{p=1}^5$ are the five real orthonormal traceless $l = 2$ harmonics ($\sum_{p=1}^5 \phi_p(\hat{\mathbf{u}})\phi_p(\hat{\mathbf{v}}) \equiv (\hat{\mathbf{u}}\cdot\hat{\mathbf{v}})^2 - \tfrac{1}{3}$) and $\rho_{\text{hb}} = \beta_{\text{sol}}\,\rho_{\text{hb,sol}} + v_{\text{vox}} n_{\text{bulk}}\,\xi$ combines polar solute tethers (`O/N` heavy atoms) and excess neighboring waters $\xi = \text{softplus}(\bar{g} - 1;\,\beta=4)$:
$$A_{\text{frust}}(\mathbf{x}) = \frac{\|\mathbf{Q}_2(\mathbf{x})\|_2^2 + \tfrac{2}{3}\|\mathbf{J}_1(\mathbf{x})\|_2^2}{1 + N_c(\mathbf{x})^2}, \qquad T_{\text{coop}}(\mathbf{x}) = \frac{N_c(\mathbf{x})^2}{1 + \tfrac{1}{4}N_c(\mathbf{x})^2}\,e^{-2 A_{\text{frust}}(\mathbf{x})}.$$
At the center of a regular $109.47^\circ$ tetrahedron, $\mathbf{J}_1 = \mathbf{0}$ and $\mathbf{Q}_2 = \mathbf{0} \implies A_{\text{frust}} \equiv 0$, maximizing $T_{\text{coop}}$; on a frustrated one-sided arc at identical $N_c$, $A_{\text{frust}} > 0.6$ suppresses unphysical smeared arcs. Combined with the 5 radial Ornstein–Zernike shells $R_q \in \{1.25,\,1.90,\,2.80,\,3.60,\,4.50\}\text{ \AA}$ across $K = 4$ damped ($\eta = 0.65$) self-consistent steps:
$$\Delta U_{\text{step}}(\mathbf{x}) = \sum_{q=1}^{5} v_q\,(\psi_{R_q} * \xi)(\mathbf{x}) + w_{\text{frust}} A_{\text{frust}}(\mathbf{x}) + w_{\text{tet}} T_{\text{coop}}(\mathbf{x}) + w_{\text{nc}}\frac{N_c(\mathbf{x})^2}{1 + \tfrac{1}{4}N_c(\mathbf{x})^2} - c_{\text{bulk}}(\boldsymbol{\theta}),$$
where subtracting the exact closed-form empty-cell constant $c_{\text{bulk}}(\boldsymbol{\theta})$ (`zero_bulk_offset=True`) guarantees $\Delta U_{\text{step}}(\infty) \equiv 0$ and $\bar{g}_O(\infty) \equiv 1.000000$ identically in bulk solvent.

### Parameter-Free Rosenfeld 0D Hard-Sphere Cavity Cap (`use_pauli_cap=True`)
Pointwise Langmuir saturation bounds the peak height $g_O(\mathbf{x})$ at a single grid voxel (`~0.35 Å`), not the integrated number of waters inside an oxygen van der Waals exclusion volume. Because two liquid-water oxygens cannot coexist closer than $2.60\text{ \AA}$, any sphere of radius $r_{\text{cav}} = 1.30\text{ \AA}$ (diameter $2.60\text{ \AA}$) contains **at most 1 water molecule** in every instantaneous microstate (Rosenfeld's 0D Fundamental Measure Theory bound). Convolving $\bar{g}_{O,\text{raw}}$ with the smooth cavity indicator $w_{\text{cav}}(r) = \sigma((1.30 - r)/0.10)$ via one 3D FFT gives the local cavity occupancy $n_{\text{cav}}(\mathbf{x})$ and enforces:
$$g_O^{(m)}(\mathbf{x}) = \frac{g_{O,\text{raw}}^{(m)}(\mathbf{x})}{1 + \text{softplus}(n_{\text{cav}}(\mathbf{x}) - 1.05;\,\beta=10) - \text{softplus}(n_{\text{cav}}^{\text{bulk}} - 1.05;\,\beta=10)},$$
preserving `1-HB`/`2-HB` waters ($n_{\text{cav}} \approx 0.45$) and bulk water ($n_{\text{cav}}^{\text{bulk}} \approx 0.33$, denominator $\equiv 1$) untouched while preventing deep 3- and 4-tethered clefts (`HOH 108`) from exceeding physical 1-water (`~10 e`) stoichiometry.

---

## 6. Layer 4 — Dipolar Water-H Orientation, `IT92` Rendering & Ordered-Water Displacement

### Molecular `IT92` Water Scattering Kernel
Rather than treating water as a Gaussian blob, `surfNturf` converts molecular oxygen number density $n_{\text{bulk}}\,\bar{g}_O(\mathbf{x})$ into X-ray electron density (`e/Å³`) using International Tables (`IT92`) 4-Gaussian + constant scattering factors for `O` and `H`, rotationally averaged for two protons at bond length $r_{\text{OH}} = 0.9572\text{ \AA}$:
$$f_{\text{wat},0}(s) = f_{\text{O}}^{\text{IT92}}(s) + 2\,f_{\text{H}}^{\text{IT92}}(s)\,j_0(4\pi s r_{\text{OH}}), \qquad j_0(x) = \frac{\sin x}{x}, \qquad s = \frac{1}{2d}.$$
At $s = 0$, $f_{\text{wat},0}(0) = f_{\text{O}}^{\text{IT92}}(0) + 2 f_{\text{H}}^{\text{IT92}}(0) \approx 10.008\text{ e}$, and $n_{\text{bulk}} = \rho_{\text{bulk}} / f_{\text{wat},0}(0)$ guarantees that uniform $\bar{g}_O = 1$ renders identically to $\rho_{\text{bulk}} = 0.3344\text{ e/\AA}^3$.

### Zero-Net-Charge Acceptor Water-H Polarization
Around macromolecular H-bond acceptors (`O_carb`, `O_neg`, `O_hyd`), water molecules orient an `O–H` bond toward the acceptor lone pair (`d(O_acc...H_w) ≈ 1.75–2.05 Å`) while sterically excluding protons from all atoms (`< 1.25 Å` from `H`, `< 1.55 Å` from heavy atoms):
$$g_H^{(m)}(\mathbf{x}) = S_{\text{steric}}^{(m)}(\mathbf{x})\,\tanh\!\left(\sum_{a \in \mathcal{A}_{m,\text{acc}}^{\text{sym}}} u_{H,t_a}\!\left(\|\mathbf{x} - \mathbf{x}_a\|_2\right)\right).$$
In reciprocal space, $\bar{g}_H(\mathbf{x})$ is multiplied by $f_{H,0}^{\text{zero}}(s)$, which equals $f_{\text{H}}^{\text{IT92}}(s)$ for all $\mathbf{h} \neq \mathbf{0}$ and is **strictly zero at $\mathbf{h} = \mathbf{0}$**:
$$\widehat{\rho}_{\text{solv}}(\mathbf{h}) = e^{-B_{\text{map}} s_{\mathbf{h}}^2}\,n_{\text{bulk}}\Big[f_{\text{wat},0}(s_{\mathbf{h}})\,\mathcal{F}\{\bar{g}_O\}(\mathbf{h}) + f_{H,0}^{\text{zero}}(s_{\mathbf{h}})\,\mathcal{F}\{\bar{g}_H\}(\mathbf{h})\Big].$$
Consequently, water-H orientation redistributes charge dipolarly at finite resolution while adding **identically zero net electrons** (`ΔF000 = 0`).

### Electron-Conserving Ordered-Water Displacement (`io.py`)
When explicit crystallographic `HOH` oxygen atoms are already modeled in the PDB (`WET` refinement), they contribute $8\,\text{occ}_w$ oxygen electrons per site inside $F_{\text{calc}}$ (or `10` if explicit `DOD`/`HOH` hydrogens are present). To prevent double-counting without creating an arbitrary step-mask seam, `displace_ordered_waters` carves smooth Fermi holes $1 - \text{occ}_w\,\sigma((r_{\text{ex}} - d)/0.15)$ around modeled waters and solves for $r_{\text{ex}}$ by bisection on the ASU so that:
$$\int_{\text{cell}} \rho_{\text{bulk\_displaced}}(\mathbf{x})\,d^3x + 8\,N_{\text{ops}}\sum_{w} \text{occ}_w = \int_{\text{cell}} \rho_{\text{total\_solv}}(\mathbf{x})\,d^3x.$$

---

## 7. Layer 5 — Spatially Varying 3D Water Mobility $B(\mathbf{x})$ (`bmob.py`)

### Physical Anchoring to Local Protein $B_{\text{iso}}$ and H-Bond Tethering
A first-shell water hydrogen-bonded to a surface residue cannot remain sharper than the thermal motion of the protein atoms tethering it, nor is an untethered water over a hydrophobic patch as immobile as a doubly-bridged cleft water. `bmob.py` computes three geometric fields on the ASU from heavy atoms:
1. $d_{\min}(\mathbf{x})$: distance to nearest macromolecular heavy atom,
2. $B_{\text{prot}}(\mathbf{x})$: contact-weighted local protein $B_{\text{iso}}$ across the `2.4–3.8 Å` first-shell neighborhood (plus strict `3.2 Å` through-space shell `B_prot_tb32`),
3. $N_{\text{tether}}(\mathbf{x})$: smooth count of polar `N/O`/ion heavy atoms in hydrogen-bonding range `[2.25, 3.25 Å]`.

The 6-parameter transferable physical mobility law (`dry_a4` / `wet_b5b` presets, or refined via `fit-bmob --mode phys`) sets:
$$B_{\text{near}}(\mathbf{x}) = \alpha_{\text{prot}}\,B_{\text{prot}}(\mathbf{x}) + B_0 + \Delta B_{\text{unteth}}\,e^{-\gamma_{\text{teth}} N_{\text{tether}}(\mathbf{x})},$$
$$B_{\text{phys}}(\mathbf{x}) = p_{\text{near}}(\mathbf{x})\,B_{\text{near}}(\mathbf{x}) + \big(1 - p_{\text{near}}(\mathbf{x})\big)\,B_{\text{far}}, \qquad p_{\text{near}}(\mathbf{x}) = \sigma\!\left(\frac{d_{\text{far}} - d_{\min}(\mathbf{x})}{w_{\text{far}}}\right).$$
When diffraction data are available (`fit-bmob --mode tb32` or `joint`), $\ln B(\mathbf{x})$ is further nudged by a regularized spatial field $\delta z(\mathbf{x}) = (G_{\text{FWHM}=3.2\text{ \AA}} * \eta)(\mathbf{x})$ coupled across neighboring hydration shells (`3.2 Å` FWHM = continuous analogue of neighbor-coupled $B$-factor restraints).

### `F000`-Exact Catmull–Rom Partition-of-Unity Basis
Evaluating an exact spatially varying Gaussian blur $\int \rho(\mathbf{y})\,K(\mathbf{x}-\mathbf{y}; B(\mathbf{y}))\,d^3y$ directly in real space would scale poorly on `10^6` voxels, yet naive interpolation can leak `F000`. `MultiBRenderer` expands $z(\mathbf{x}) = \ln B(\mathbf{x})$ on `13` geometric basis knots $B_k = 5.0 \times 2^{k-3}\text{ \AA}^2 \in [0.625,\,2560]\text{ \AA}^2$ ($k = 0 \dots 12$, with $k = 3$ sitting strictly at $B = 5.0\text{ \AA}^2$) using cubic Catmull–Rom weights $c_k(z)$:
$$\sum_{k=0}^{12} c_k(z) \equiv 1 \quad \forall\,z \in \mathbb{R}, \qquad \widehat{\rho}(\mathbf{h}) = \sum_{k=0}^{12} e^{-B_k s_{\mathbf{h}}^2}\,n_{\text{bulk}}\Big[f_{\text{wat},0}(s_{\mathbf{h}})\,\mathcal{F}\{c_k(z)\bar{g}_O\} + f_{H,0}^{\text{zero}}(s_{\mathbf{h}})\,\mathcal{F}\{c_k(z)\bar{g}_H\}\Big].$$
Because $e^{-B_k \cdot 0} = 1$ for every level $k$ and $\sum_k c_k(z(\mathbf{x})) \equiv 1$ at every voxel $\mathbf{x}$, **any spatial mobility distribution $B(\mathbf{x})$ conserves total unit-cell electrons `F000` to machine precision (`< 1e-15`)**, reproduces uniform $B = 5.0\text{ \AA}^2$ identically at $z = \ln 5$, and renders $M = 32$ conformers in at most `13` 3D FFTs.

---

## 8. Layer 6 — Crystal-Specific Difference-Map Basins, Peaks & Mass-Conserving Transport (`divot.py`)

Even an ideal pure-water PMF cannot predict crystal-specific buffer ions, partially ordered cryoprotectants, unmodeled ordered waters, or packing-induced cavities (**divots**). Yet naive real-space coordinate warping $\rho_0(\mathbf{x} - \mathbf{u}(\mathbf{x}))$ fails on solvent for three mathematical reasons:
1. In flat bulk solvent ($\nabla\rho_0 \approx \mathbf{0}$), $\rho_0(\mathbf{x} - \mathbf{u}) \approx \rho_0(\mathbf{x})$ has zero sensitivity to $\mathbf{u}$, so shifts cannot create missing peaks or cavities.
2. Without the continuity Jacobian $\det(\mathbf{I} - \nabla\mathbf{u})$, compressing or dilating a hydration peak destroys total electron count (`F000`).
3. Unclamped shifts across the protein boundary drag solvent density inside the macromolecular core, absorbing protein coordinate errors.

`divot.py` (`surfnturf fit-divot`) addresses all three cleanly:

1. **Occupancy-weighted open-solvent envelope $f_{\text{open}}(\mathbf{x}) \in [0, 1]$:** Blocks the obligate macromolecular core ($d < 2.15\text{ \AA}$ in `100%` of conformers $\implies f_{\text{open}} \to 0$) while reducing accessibility at partial-occupancy altlocs strictly in proportion to their occupancy.
2. **Exact `e/Å³` difference maps across all 230 space groups:** Each reciprocal-ASU work reflection $\mathbf{h}$ is weighted by $N_{\text{ops}} / m_{\text{stab}}(\mathbf{h})$, where $m_{\text{stab}}(\mathbf{h}) = \sum_{g=1}^{N_{\text{ops}}} (\mathbb{I}[\mathbf{R}_g^T\mathbf{h} = \mathbf{h}] + \mathbb{I}[-\mathbf{R}_g^T\mathbf{h} = \mathbf{h}])$, removing centring, centric `×2`, and special-axis $\varepsilon$ distortions identically upon space-group symmetrization.
3. **`WatershedDivotDict` (Bader steepest-ascent basins):** Partitions significant positive and negative open-solvent `mFo - DFc` troughs/blobs (`r <= 2.2 Å`) into fixed symmetry-exact basin templates and refines one smooth-$L_1$-regularized scalar amplitude per basin.
4. **`ContinuousPeakDict` (sub-voxel `IT92` peak dictionary):** Parameterizes localized open-solvent peaks by continuous Cartesian coordinates `(x_j, y_j, z_j)`, signed water equivalents $\Delta q_j$, and site $B_j \in [5.5, 97]\text{ \AA}^2$ using normalized partition-of-unity discrete Gaussians convolved in **one 3D FFT** with the exact `IT92` `(O + 2H)` kernel (zero voxel-edge ripple and exact sub-voxel `F000` invariance).
5. **Wall-clamped mass-conserving transport (`sw6_mc`):** Applies continuity-equation density transport $\rho_{\text{mc}}(\mathbf{x}) = \text{sym}_s\!\big[\det(\mathbf{I} - \nabla\mathbf{u}_{\text{eff}}(\mathbf{x}))\,\rho_0(\mathbf{x} - \mathbf{u}_{\text{eff}}(\mathbf{x}))\big]$ with $\mathbf{u}_{\text{eff}} = f_{\text{open}}\,\mathbf{u}$, exact skew-Hermitian Cartesian spectral derivatives ($\langle \partial_b f, g \rangle = -\langle f, \partial_b g \rangle$ with Nyquist modes zeroed), exact $3\times 3$ cofactor adjoints, and a smooth barrier keeping $\det(\mathbf{I} - \nabla\mathbf{u}_{\text{eff}}) > 0.30$.

---

## 9. Variable Projection & Zero-`FreeR`-Leakage Validation

When fitting reciprocal-space amplitudes $|F_{\text{obs}}(\mathbf{h})|$:

* **Envelope Theorem for inner scales (`fit_kmask_per_bin`, `kmask_mode='fixed_one'`, & `solve_inner_scales`):** When estimating pure water mobility $B(\mathbf{x})$ with a frozen PMF (`fit-bmob` in `auto`, `uniform`, `phys`, or `tb32` mode), floating per-bin $k_{\text{mask}}(s)$ simultaneously creates a gauge collinearity with high-angle Debye–Waller attenuation; `fit-bmob` therefore defaults to **`kmask_mode='fixed_one'`** ($k_{\text{mask}}(\mathbf{h}) \equiv 1.000$ across all resolution shells, solving only the single overall scale $k_0$ by exact variable projection $\partial\mathcal{L}/\partial k_0 = 0$) for `auto / uniform / phys / tb32`, and to **`kmask_mode='per_bin'`** (solving resolution-shell scales $k_b \ge 0$ via `fit_kmask_per_bin`) for `--mode joint`. In `divot.py`, `solve_inner_scales` solves bounded `(k_prot, k_solv >= 0.05)` (the floor keeps the solvent gradient, which is proportional to `k_solv`, from vanishing). Because $\partial\mathcal{L}/\partial k = 0$ at interior optima, outer analytical gradients require zero approximate scale Jacobians.
* **Strict `90% / 5% / 5%` blind split:** All free-set reflections (`free_mask_from_flags`, supporting `0..19` CCP4 flags, `0/1` flags, SHELX `-1/1` flags, or explicit `--free-flag-value`) are masked out of every difference map, scale solve, and gradient. Across both `fit-bmob` and `fit-divot`, work reflections are split into `work_train` (`~90%`) and `work_val` (`~5%`, drawn from an independent 20-bin `FreeR_flag == 1` bin when available, or by a deterministic hash of the space-group-canonical / Friedel-canonical Miller index `canonical_hkl_keys`). Stage complexity (`0shot`, `uniform`, `phys`, `spatial` in `fit-bmob`; `base`, `scale_only`, `divot_peak` plus dictionary `(z_thresh, lam_l1)` in `fit-divot`) is gated strictly by minimum amplitude `R` on `work_val` before refitting on `work` in `fit-divot`, keeping `R_free` completely uncompromised.
