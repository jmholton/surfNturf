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

## 3. Layer 1 — Protonation-Aware 10-Class Solute–Water PMF (`field.py`)

### Directional Hydrogen Bonding from Isotropic Explicit-Atom Potentials
Heavy-atom-only solvation potentials must approximate angular hydrogen-bond geometry using multi-body coordinate frames because a heavy `N` or `O` atom does not encode where its proton points. Keeping **explicit macromolecular hydrogens** solves angular directionality automatically with purely radial pair splines:

* An amide group consists of a repulsive heavy `N_don` core ($r_0 \approx 2.9\text{ \AA}$) behind an attractive `H_pol` proton displaced by `0.86–1.01 Å` along the `N–H` bond ($r_{\min} \approx 1.85\text{ \AA}$).
* Summing $u_{\text{N\_don}}(|\mathbf{x} - \mathbf{x}_N|) + u_{\text{H\_pol}}(|\mathbf{x} - \mathbf{x}_H|)$ creates a deep directional minimum along the linear `N–H...O_w` vector (`d(N...O_w) ≈ 2.85 Å`, `d(H...O_w) ≈ 1.85 Å`) while sterically blocking the back and sides of the nitrogen.

### Chemical Atom Typing (`ATOM_TYPES`)
Each non-water atom in conformer $m$ is assigned to one of 10 physical classes:

| Index | Class | Physical Species | Primary Role in $U_0^{(m)}(\mathbf{x})$ |
|---:|---|---|---|
| `0` | `O_carb` | Neutral carbonyl/amide `O` (`C=O`, `ASN OD1`, `GLN OE1`) | Lone-pair H-bond acceptor well around `2.7–3.0 Å` |
| `1` | `O_neg` | Carboxylate/oxyanion `O` (`ASP`, `GLU`, `OXT`, `PO4`, `SO4`), `F-`, compact divalent/alkali cations (`Mg2+`, `Zn2+`, `Ca2+`, `Na+`, `Fe`, `Cu`, `Mn`) | Deep electrostatic first-shell coordination well (`2.2–2.8 Å`) |
| `2` | `O_hyd` | Hydroxyl `O` (`SER OG`, `THR OG1`, `TYR OH`) & unprotonated `HIS ND1/NE2` | Amphoteric / lone-pair H-bond acceptor (`2.8–3.1 Å`) |
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

## 5. Layer 3 — 1-FFT Ornstein–Zernike Water–Water Packing (`field.py`)

Solute–water pair potentials $U_0(\mathbf{x})$ alone know nothing about **water–water steric exclusion** or **water-bridged second shells**. Solving the full 3D-RISM Ornstein–Zernike integral equation iteratively on a multi-conformer crystal ensemble requires minutes to hours of compute per structure and often over-oscillates under the Kovalenko–Hirata closure.

`surfNturf` captures the leading physical Ornstein–Zernike feedback in **one 3D FFT pair** on the ensemble-averaged first-shell excess oxygen peak field $\xi(\mathbf{x}) = \text{softplus}(\bar{g}_1(\mathbf{x}) - 1;\,\beta=4)$, where $\bar{g}_1(\mathbf{x}) = \sum_m w_m g_1^{(m)}(\mathbf{x})$:
$$\Delta U_{ww}(\mathbf{x}) = \sum_{q=1}^{5} v_q\,\big(\psi_{R_q} * \xi\big)(\mathbf{x}), \qquad \widehat{\psi}_{R_q}(k) = \frac{\sin(k R_q)}{k R_q}\,\exp\!\left(-\frac{1}{2}\sigma_{\text{OZ}}^2 k^2\right),$$
at spherical shell radii $R_q \in \{1.25,\,1.90,\,2.80,\,3.60,\,4.50\}\text{ \AA}$ ($\sigma_{\text{OZ}} = 0.32\text{ \AA}$, $k = 4\pi s = 2\pi / d$).

In the MD-trained weights (`pmf_full_l2.npz`):
* $v_{1.25} = +0.40,\; v_{1.90} = +0.55 > 0$ carve the **annular exclusion moat** around pinned first-shell waters (suppressing unphysical shoulder density `1.2–2.2 Å` from a high-occupancy water),
* $v_{2.80} = -0.66,\; v_{4.50} = -0.17 < 0$ stabilize **hydrogen-bonded second and third shells** (`2.8 Å` and `4.5 Å` water bridges across grooves),

producing the total potential $U_{\text{tot}}^{(m)}(\mathbf{x}) = U_0^{(m)}(\mathbf{x}) + \Delta U_{ww}(\mathbf{x})$ and ensemble oxygen density $\bar{g}_O(\mathbf{x}) = \sum_m w_m g_O^{(m)}(\mathbf{x})$.

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

* **Envelope Theorem for inner scales (`fit_kmask_per_bin` & `solve_inner_scales`):** Resolution-shell solvent scale factors $k_b \ge 0$ (in `field.py` / `bmob.py`) and bounded `(k_prot, k_solv >= 0)` (in `divot.py`) are solved to machine precision on the active work reflections at every outer L-BFGS-B evaluation. Because $\partial\mathcal{L}/\partial k_b = 0$ at interior optima (and $\partial k_b/\partial\boldsymbol{\theta} = 0$ on active non-negativity bounds), outer analytical gradients require zero approximate scale Jacobians.
* **Strict `90% / 5% / 5%` blind split:** All `FreeR_flag == 0` reflections are masked out of every difference map, scale solve, and gradient. In `fit-divot`, work reflections are further split into `work_train` (`~90%`) and `work_val` (`~5%`, drawn from an independent 20-bin `FreeR_flag` bin when available or a resolution-stratified `1/19` subset of work). Dictionary thresholds (`z_thresh`, `lam_l1`) are selected strictly by minimum amplitude `R` on `work_val` before a single warm-started refit on `work`, keeping `R_free` completely uncompromised.
