# Xe RICH pre-demonstrator: summary of the options explored

Design options for a lab **pre-demonstrator** of pressurised xenon as the gas radiator of ARC (the demonstrator proper is the later set-up with all the ARC baseline specifications). The drivers in `studies/` produce every simulated number and plot quoted here; the [README](../README.md) gives their options, outputs, model assumptions and the terms used below (FT, FT×QE, FT×PDE, board, pad, ring hits, β = 1 / real β).

**Status of the numbers.** Numbers given with a source (paper, datasheet, k4geo) are from that source; other untagged numbers are model estimates of this package (β = 1 toys or the 3D tracer, as stated). Tags on inputs: *[datasheet]* vendor specification, *[literature]* published measurement or formula, *[assumed]* placeholder until measured, *[measured]* measured on this set-up (none yet).

**Summary.** Five configurations (§7): (1) gas-ring imaging with cosmic muons: a stock 3" spherical mirror with f ≈ 203 mm, a 2" × 2" SiPM plane at its focus and ~20 cm of Xe at the reference pressure of 2 bar (1–3 bar scanned); (2, 3) the aerogel ring imaged by a short-focal-length mirror, in air and in Xe, which measures how Xe in the pores shifts the aerogel index; (4) the same comparison by proximity focusing, without a mirror; (5) configuration 1 with the 1 cm aerogel tile 1 mm in front of the SiPM: its aerogel ring is not imaged (for near-vertical muons the light misses the mirror, §6.1), and the tile Rayleigh-scatters about 13 % of the gas-ring photons (§4). A full-size, ARC-like cell needs a custom mirror and is left for later.

**Reference pressure.** Xe 2 bar (`constants.P_REF`, the default of the drivers). Numbers at 3 bar say so, with the option that reproduces them.

---

## 1. Physics inputs

### 1.1 Cherenkov emission in Xe
- Frank–Tamm: d²N/dx dE ≈ 370 z² sin²θc eV⁻¹ cm⁻¹; in wavelength ∝ 1/λ², steepened toward the UV by the Xe dispersion.
- The emitted spectrum has **no peak** in the transparent region: it rises down to the Xe absorption edge at ~150–160 nm, near the 147 nm resonance. About 53 % of the photons emitted in 200–600 nm lie below 300 nm.
- The **detected** peak is set by mirror reflectivity × window transmission × SiPM PDE: ~300–350 nm for standard SiPMs.

### 1.2 Xe optical constants (`xerich.media`)
Börzsönyi et al. (2008) Sellmeier formula *[literature]* (273.15 K, 1000 mbar; fitted to an 800 nm measurement and literature UV–visible indices), scaled with Lorentz–Lorenz and the virial molar density (B = −130 cm³/mol at 293 K). Values at 20 °C (`scripts/investigations/xe_index/doc_values.py`):

| | 2 bar | 3 bar |
|---|---|---|
| (n−1) at 300 nm | 1.42 × 10⁻³ | 2.15 × 10⁻³ |
| θc (β = 1) at 400 nm | 51.9 mrad | 63.7 mrad |
| π / K / p threshold (300 nm) | 2.6 / 9.2 / 17.6 GeV | 2.1 / 7.5 / 14.3 GeV |
| N_γ/cm saturated, 200–600 nm | 4.4 | 6.7 |
| N_γ/cm saturated, 320–450 nm | 1.1 | 1.7 |

Model accuracy:
- Density: Xe is ~1.6 % non-ideal at 3 bar; the code's density agrees with NIST *[literature]* to 0.03 % at 1–3 bar and 20 °C (NIST effective B = −133 cm³/mol at 293 K).
- Index vs Koch's measurements *[literature]* (273.16 K, 1 atm; Koch's air wavelengths converted to vacuum): Börzsönyi (n−1) within 0.08 % above 290 nm, lower by 0.13 % at 264 nm and 0.46 % at 230 nm (0.04 mrad in θc at 264 nm, 3 bar).
- Bideau-Mehu (1981) (`--xe-index bm1981`): (n−1) lower by 0.7 / 1.7 / 2.2 % at 264 / 400 / 954 nm, θc 0.9 % lower at 400 nm, chromatic rms 9–12 % larger.

Consequences:
- 2 bar Xe is similar to C₄F₁₀ in index; 3 bar gives about 50 % more photons and lowers the K threshold.
- π/K angle separation at 300 nm, 2 vs 3 bar: 5.6 vs 4.5 mrad at 20 GeV, 1.0 vs 0.9 mrad at 45 GeV (Δθ ≈ Δm²/(2p² tanθc), so it shrinks slightly with pressure at high p).

### 1.3 Choice of wavelength band
Emitted yield and chromatic rms per photon, β ≈ 1:

| Band | N/cm (2 / 3 bar) | Chromatic rms per photon (2 / 3 bar) | σ/√N per cm, 2 bar |
|---|---|---|---|
| 320–450 nm | 1.1 / 1.7 | 0.41 / 0.50 mrad | ~0.38 mrad |
| 300–600 nm | 2.1 / 3.1 | 0.70 / 0.86 mrad | ~0.49 mrad |
| 250–600 nm | 3.0 / 4.5 | 1.2 / 1.4 mrad | — |
| 200–600 nm | 4.4 / 6.7 | 2.2 / 2.7 mrad | ~1.1 mrad |

320–450 nm gives the best chromatic term per track but keeps only ~25 % of the 200–600 nm photons; in practice the SiPM PDE sets the band, and 320–450 nm roughly matches standard SiPMs.

### 1.4 FCC-ee π/K momentum spectra (`gen_pk.py`, Pythia 8.317)
Signal processes only, |cosθ| < 0.9:

| √s | K median p | K above threshold (2 / 3 bar) | K between π and K thresholds (2 bar) | K with p > 20 GeV |
|---|---|---|---|---|
| Z, 91 GeV | 2.7 GeV | 14 / 18 % | ~38 % | 2.2 % |
| WW, 160 GeV | 2.7 GeV | 14 / 19 % | ~36 % | 2.2 % |
| ZH, 240 GeV | 3.1 GeV | 15 / 20 % | ~40 % | 2.6 % |
| tt̄, 365 GeV | 3.7 GeV | 20 / 26 % | ~42 % | 4.1 % |

In Xe most of the gas K-ID power comes from threshold mode; kaons below ~2–2.6 GeV need the aerogel or time-of-flight; the dependence on √s is weak.

---

## 2. Mirrors

### 2.1 Coating
| Coating | Verdict |
|---|---|
| **Al + MgF₂** | Standard RICH coating, R > 80–85 % down to ~160 nm. Needed only with VUV-sensitive photodetectors. |
| **Protected / UV-enhanced Al (Al + SiO₂)** | Fine above ~250 nm, cheap, broadband, insensitive to the angle of incidence. **The choice for the stock mirrors (configurations 1, 2, 3, 5).** |
| Broadband dielectric | Sharp band edges act as a filter, but the band shifts with angle of incidence (Edmund "320–450 nm" holds at 45°, 340–488 nm at 0°) and reflectance depends on polarisation. Usable, not needed. |
| Silver, gold | Unsuitable: reflectivity collapses below ~350–400 nm. |
| Bare Al | Unsuitable: the native oxide degrades reflectivity below ~200 nm and over time. |

### 2.2 Focal length and geometry
- With the SiPM at the focal plane, a straight track gives a sharp ring of radius f·tanθc: 10.5 mm at 2 bar and 13.0 mm at 3 bar for f ≈ 20 cm.
- The ring centre sits at f·tanθ_track, independent of where the track crosses: ~3.5 mm per degree of track tilt.
- A track illuminates a mirror patch of radius L·tanθc (L = radiator length): ≈ 10 mm at 2 bar with L = 20 cm.

### 2.3 Spherical aberration (`raytrace.py --pressure 3`, f = 203 mm, 3 bar)
| Track offset from the mirror axis | Outermost mirror radius hit | Ring smearing, rms per photon |
|---|---|---|
| 0 | 13 mm | 0.03 mrad |
| 25 mm | 38 mm | 0.4 mrad |
| 50 mm | 62 mm | 1.8 mrad |
| 75 mm | 87 mm | 5.4 mrad |

At f ≈ 20 cm only the central ~8 cm of a large (f/1) mirror is good for precision ring imaging; a track's tilt can partly compensate an offset.

### 2.4 Commercial mirrors considered
Diameter, f, coating and price are *[datasheet]* or vendor quotes; the comments are model estimates or inferences.

| Part | Diameter | f | Coating | Price | Comment |
|---|---|---|---|---|---|
| **Edmund #43-584** | **76.2 mm (3")** | **203.2 mm** (f/2.7) | Enhanced Al, Ravg > 89 % at 250–450 nm | **€316** | **Recommended for configurations 1 and 5.** Largest stock mirror at f ≈ 20 cm. |
| Thorlabs CM750-200-F01 | 75 mm | 200 mm | UV-enhanced Al, R > 90 % at 250–450 nm, N-BK7 | quote | Equivalent alternative. |
| Edmund #49-608 | 50.8 mm (2") | 203.2 mm (f/4) | Enhanced Al, Ravg > 89 % at 250–450 nm | $199 | Centred beam tracks only. |
| Edmund #49-606 | 25.4 mm (1") | 203.2 mm (f/8) | Enhanced Al | $135 | Too small: the ring fills the mirror at 3 bar. |
| Newport 20DC500AL.2 | 50.8 mm | ~250 mm (model inference from the part name) | UV-enhanced Al | — | Specs not checked. |
| Edmund #72-975 | 76.2 mm | 203.2 mm | Dielectric, 320–450 nm | €464 | Built-in band filter; angle of incidence matters. |
| Edmund #49-609 | 76.2 mm | 304.8 mm (f/4) | Enhanced Al | €316 | If the vessel allows ~35 cm. |
| Edmund #43-970 | 152.4 mm (6") | **609.6 mm only** | Enhanced Al | €745 | 3× the photons with a ~60 cm radiator (model estimate); ring radius 3.9 cm (model estimate), poor for cosmics (needs a ~16 cm sensor). Not ARC-like. |
| Edmund #72-979 | 152.4 mm | 609.6 mm | Dielectric, 320–450 nm | €1310 | As above. |
| Edmund #72-960 | 25.4 mm | 25.4 mm (f/1) | Dielectric, 320–450 nm | $186 | Ring only ~1.6 mm: not useful. |

- Stock spherical mirrors stop at 75–76 mm diameter for f ≈ 200 mm (Thorlabs, Edmund, OptoSigma, Newport, Knight Optical). Edmund has no 4" option; its 6" mirrors come only at f = 610 mm. Telescope primaries (200 mm f/4–f/6 paraboloids, f = 0.8–1.2 m) are not suitable.
- Edmund specs that matter: **EFL tolerance ±2 %**, a 2 % scale on θc (measure f in situ or calibrate on β ≈ 1 tracks); λ/4 surface accuracy and 60-40 surface quality, irrelevant at the mrad level; Borofloat substrate, fine in Xe (equal pressure on both sides); vacuum/pressure compatibility of mounts and adhesives not specified (ask).

### 2.5 Large or custom mirrors
- **Custom spherical mirror**, ~200–260 mm diameter, R ≈ 400 mm (sag ≈ 13 mm), Al + SiO₂ or Al + MgF₂; lead time weeks to months; λ/2 or even λ is sufficient for mrad-level imaging. Possible suppliers: **Avantier** (5–500 mm, λ/4 standard, UV-enhanced Al), **asphericon** (Jena; up to 420 mm, coatings from 190 nm), the suppliers or spare segments of the LHCb RICH and ARC groups.
- **Mosaic** of 2–4 × 3" mirrors around a common centre of curvature: each segment works near its own axis; needs alignment.

---

## 3. ARC reference geometry (k4geo `CLD_o3_v01` / `ARC_o1_v01`)
- Hexagonal prism cells with a **14.815 cm side**: 25.7 cm flat-to-flat, 29.6 cm corner-to-corner.
- **Radial depth 20 cm**: the barrel spans r = 190–210 cm, ~18 cm of gas inside 1 cm Rohacell walls.
- Barrel: 27 cells in φ (17–18 distinct designs, mirrored in z). Endcaps: 6 sectors × 21 distinct cells.
- **Mirror:** spherical, 2 mm thick, per-cell optimised R, sphere-centre position and sensor tilt.
  - Barrel R = 25.5–36.9 cm (median 33.6); central cell R = 36.9 cm (f = 18.5 cm), sensor tilt −1.2°.
  - Toward |z| ≈ 1–2 m, R falls to ~31–33 cm, the sphere centre moves 12–14 cm toward the IP (steering the ring onto the sensor) and the sensor tilts by 8–15°.
  - Endcap R = 22.1–36.0 cm, falling with radius.
- **Sensor:** 8 × 8 cm SiPM, 0.8 mm pitch in the simulation (0.5 mm baseline in the JINST paper), 2 mm cooling plate.
- **Aerogel:** 1 cm, n = 1.03 at 400 nm.
- Optimised for C₄F₁₀ at 1 bar (θc ≈ 53 mrad); Xe at 2 bar is similar, at 3 bar the ring is ~20 % larger and needs re-optimisation. For the pre-demonstrator, f ≈ 20 cm with ~18–20 cm of gas reproduces the central barrel cell.

---

## 4. Photodetector
**Hardware.** The ARC prototype board *[literature]*: 2" × 2", four Hamamatsu SiPM arrays of 8 × 8 SiPMs (pads of 3 × 3 mm² at 3.2 mm pitch, 256 channels).
- Geometry *[datasheet outline; abutting packages assumed]*: packages of 25.8 mm, footprint 51.6 mm; pads 0.2 mm from the package edge, spanning 51.2 mm (= 16 × 3.2 mm, the scripts' grid); pad pitch 3.4 mm across the package seams; active fraction 0.879 inside ±25.6 mm, 0.865 of the footprint.
- Part number: S13361-3050NE-08, given for the first baseboard version (Cardinale, slide 10); to be confirmed for the ceramic board's arrays.
- Window *[datasheet, to be confirmed for the board's arrays]*: epoxy, n = 1.55, resin thickness 0.1 ± 0.05 mm.
- Readout: ALCOR SiPM-readout ASIC on INFN Bologna front-end boards.
- Cooling: ceramic baseboard with internal coolant flow, designed for −60 to −80 °C (Pezzulo et al.; Cardinale, slide 11); the pre-demonstrator runs it at −20 °C, the low end of the arrays' datasheet range (−20 to +60 °C without dew condensation).

**Input card.** The defaults of `cosmic_rates.py` (`cli.BASELINE`; options in the README), at Xe 2 bar with `--nphot 7200`. `--preset published` selects the simplified model (README).

| Input (option) | Value | Status |
|---|---|---|
| PDE (`--response s13361`) | S13361-3050 typical curve at 25 °C and 3 V overvoltage, through the window: 320–900 nm, peak 40.2 % at 460 nm, 39.95 % at 450 nm (KAPD1054E07, read from its figure) | datasheet; not at the operating temperature and overvoltage |
| Board (`--sensor-model board`), ring hits | geometry above; binary pad hits of the unscattered ring photons, without the muon's pad (model in the README); no crosstalk, dark counts or afterpulses | assumed |
| Mirror reflectivity (`--mirror-r nominal`) | 0.89 at 250–450 nm, 0.87 at 700 nm, 0.82 at 800–850 nm, 0.85 at 950 nm, linear in between (vendor: Ravg > 89 % at 250–450 nm, > 85 % at 250–700 nm) | assumed from the datasheet; placeholder for the measured curve |
| Aerogel dispersion (`--aerogel-index single-pole`) | single pole, λ₀ = 83.22 nm (Bellunato et al.), equal to the run's tile index at 400 nm | literature shape, assumed for this tile |
| Porosity (`--porosity-vac`) | 0.935, from the tile density (§6.3) | assumed until the vendor data |
| Pore-gas temperature (`--pore-t`) | 293.15 K | assumed until the box thermal state is known |
| Board reflectance (`--board-r-spec`, `--board-r-diff`) | specular 0.11 (range 0.08–0.14), diffuse (Lambertian) 0.10, flat in λ | literature (below), to be confirmed on the board |
| Rayleigh, configuration 4 (`--prox-rayleigh per-photon`) | per-photon path transmission, β = 1 FT×PDE spectrum for every muon: 0.933 for a vertical muon (single-pole index), 0.934 (constant index); photons of the ≥ 80 % rings at Xe 2 bar keep 0.929 / 0.932 / 0.933 at gaps 30 / 50 / 70 mm | model |
| Ring definitions | geometric: a ring counts from its light in the ARC QE band whatever the response | convention |

- Board reflectance source: Congedo et al., arXiv:2602.20139, Sec. 3.1, Table 1: S13361-3050 and -2050 8 × 8 arrays (50 µm cells, epoxy, no window), integrating sphere at 8°: total 18–24 % above 380 nm, diffuse 6–10 % (about 10 % for the 50 µm epoxy arrays), specular = total − diffuse.
- Board reflectance, not modelled: the drop below about 380 nm (15–18 % of the reflected detected light lies there, so the flat model overestimates it by up to that share); reflection beyond the footprint; vessel walls; configuration 4, where the light returning to the board is 0.23 / 0.06 / 0.01 % of the ring photons at gaps 30 / 50 / 70 mm for vertical muons (`scripts/investigations/board_reflection/config4_size.py`).
- Configuration 4, known approximation (not corrected): containment and pads use one Cherenkov angle (n = 1.03), the yield the dispersive index. With the dispersion in the containment (Xe 2 bar, plain tile) the ≥ 80 % ring rate changes by +0.56 ± 0.06 / +0.90 ± 0.07 / +1.9 ± 0.4 % at gaps 30 / 50 / 70 mm (`scripts/investigations/config4_single_angle/c4_dispersion_selection.py`).

**Results with the input card at Xe 2 bar** (board table of `cosmic_rates_plots.py`, `cosmic_rates_board.md`, which also holds air and 1–3 bar): ≥ 80 % rings through the telescope, real β, mean ± seed std. Photons are FT×PDE; on the footprint: all; on active pads and ring hits: without the muon's pad; reflected light from the board counts as other hits, never ring hits.

| Configuration | rings /h | rings /day | photons on the footprint | photons on active pads | fired pads (ring hits) | P(≥ 3 hits) | P(≥ 10 hits) | rings /day ≥ 3 hits | rings /day ≥ 10 hits | Rayleigh-scattered gas-ring photons | reflected photons on active pads | other hits from board reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10.2 ± 0.3 | 246 | 6.77 | 5.85 | 4.80 | 0.749 | 0.056 | 184 ± 4 | 14 ± 0 | | 0.447 | 0.426 |
| 5 (gas ring) | 10.3 ± 0.3 | 248 | 5.57 | 4.82 | 4.08 | 0.697 | 0.022 | 173 ± 4 | 5 ± 0 | 0.127 | 0.356 | 0.340 |
| 4, gap 30 mm (aerogel ring) | 235.2 ± 2.1 | 5645 | 11.24 | 9.74 | 7.91 | 0.945 | 0.305 | 5337 ± 43 | 1721 ± 21 | | | |
| 4, gap 50 mm (aerogel ring) | 95.3 ± 1.8 | 2288 | 10.29 | 8.92 | 7.74 | 0.922 | 0.307 | 2109 ± 41 | 703 ± 23 | | | |
| 4, gap 70 mm (aerogel ring) | 34.5 ± 1.5 | 827 | 9.16 | 7.98 | 7.18 | 0.878 | 0.268 | 726 ± 36 | 222 ± 11 | | | |

**Pixel pitch.** Pixel term per photon, (pitch/√12)/f at f = 203 mm: 3.2 mm (the board) 4.5 mrad, dominant; 0.8 mm (as in the ARC simulation) 1.1 mrad; 0.5 mm 0.7 mrad.

---

## 5. Cosmic-ray rates, configurations 1 and 5 (gas ring)
Assumptions (`cosmic_spectrum.py`, `cosmic_rates.py`):
- Muons from the Guan et al. momentum–zenith parametrisation *[literature]*, each at its own β: 60 m⁻²s⁻¹sr⁻¹ above 1 GeV/c (Reyna 70, Shukla–Sankrith 58; the PDG 2022 review quotes ≈ 70, with recent measurements 10–15 % lower). Against cos²θ, its zenith law changes the ring rates by < 0.3 %.
- Zenith angles below 75° only (`constants.MUON_THMAX`; README, model assumptions).
- Sea level, no roof, no altitude or geomagnetic correction (CERN is at ≈ 450 m). The BESS spectrum at Lynn Lake *[literature]* (360 m, cutoff 0.4 GV) lies 14–20 % above the Guan model below 5 GeV/c and about 6 % below it above; this affects the trigger and aerogel-ring rates more than the gas-ring rates (p > 1.6–2.8 GeV/c).
- Trigger telescope, idealised: counters of 5 × 5 cm at the SiPM plane and 8 × 8 cm at the mirror vertex plane, 105 muons per hour (88 for β = 1 muons with cos²θ and I0 = 70 above 1 GeV/c). Real counters sit above the SiPM and below the mirror, which lowers the rates (§8). Xe threshold at the UV edge of the ARC QE band (264 nm): 2.75 / 1.94 / 1.58 GeV/c at 1 / 2 / 3 bar, above which lie 41 / 53 / 59 % of the telescope muons.
- Ring definitions: a **≥ 80 % ring** has ≥ 80 % of its sampled photons on the mirror and ≥ 80 % of the ring on the SiPM; a **full ring** ≥ 99 % on both.
- Photon counts: FT×PDE (mirror reflectivity and S13361 PDE of the §4 input card), on the active pads (the muon's pad included), without Rayleigh-scattered photons; estimated per muon from its sampled photons, so cuts such as "≥ 7 photons" are approximate near the cut. The 3D-tracer tables of §5 and §6 use the §4 input card (the larger layouts of §6.1 keep a square sensor).

3D tracer, 3" mirror (f = 203 mm), 2" SiPM, through the telescope (5 seeds, seed spread 0.2–0.3 per hour):

| | 1 bar | 2 bar | 3 bar |
|---|---|---|---|
| ≥ 80 % rings per hour | 9.7 | 10.2 | 9.4 |
| full rings per hour | 7.7 | 7.0 | 5.8 |
| ≥ 80 % rings per hour with ≥ 7 FT×PDE photons | 0.0 | 4.8 | 6.4 |
| FT×PDE photons per ≥ 80 % ring (on active pads, the muon's pad included) | 2.9 | 5.9 | 8.9 |

- Reyna and Shukla–Sankrith change the ≥ 80 % rates by +14 to +17 % and −5 to −10 %; β = 1 muons (cos²θ, I0 = 70) give 17.5 / 13.5 / 10.6 per hour.
- **The SiPM limits the angular acceptance:** full rings need |θ_μ| ≲ 3.6° for a 5 cm sensor (atan((half sensor − ring radius)/f) at 400 nm, 3 bar).
- **The mirror limits the position acceptance:** about ±2.1 cm for a 3" mirror without the tile, ±2.2 cm with it (clear radius − L·tanθc, 3 bar).

Mirror and SiPM size (`cosmic_spectrum.py`; muons up to 25° from vertical over a square at the mirror plane containing every track able to give a ring, radiator 190 mm; not comparable with the telescope table), ≥ 80 % rings per hour:

| Mirror | SiPM | 2 bar, β = 1 | 2 bar, Guan | 3 bar, β = 1 | 3 bar, Guan | Full rings, 2 bar, Guan |
|---|---|---|---|---|---|---|
| 2" | 5.1 cm | 7.0 | 6.1 | 5.0 | 5.4 | 3.6 |
| **3"** | **5.1 cm** | **19.3** | **16.1** | **13.3** | **13.5** | **10.9** |
| 3" | 4 cm | 8.3 | 8.0 | 5.0 | 6.2 | 5.2 |
| 3" | 8 cm | 65.4 | 47.4 | 53.7 | 46.0 | 33.5 |

β = 1 mirror scan (`cosmic_acc.py --pressure 3`: 3 bar, 8 cm sensor, L = 200 mm, at 63.8 mrad; rings entirely on the sensor with all / ≥ 80 % of the photons on the mirror, per hour, statistical errors up to about 12): 2" 6/17, 3" 27/46, 4" 59/90, 6" 172/215, 20 cm 331/394, ARC 26 cm 774/891. The rate grows roughly with the mirror area until the sensor angle cut takes over; much of the extra rate of large f/1 mirrors is only good for photon counting.

---

## 6. Aerogel
### 6.1 With the f = 203 mm mirror (configuration 5): not imaged (`cosmic_rates.py --groups s61`, `toy_aerogel.py`, `toy_scan.py`)
- θ_aero = 242 mrad inside the tile for n = 1.03 (249 mrad in the gas, used for the distances below); with the Xe pore shift at 2 bar (`toy_aerogel.py`, `toy_scan.py`): n ≈ 1.031, 247 mrad in the tile.
- After ~19 cm of gas the photons are ~48 mm from the track and **miss the 3" mirror**; the imaged ring would have **r ≈ 52 mm** and **miss the 5 cm SiPM**.
- β = 1 muons, 3" mirror, 5.1 cm SiPM: only muons at 6–22° put aerogel light on the sensor, ≤ ~3.5 emitted photons per muon (~1 detected), on an arc of ≤ 20 % of the ring (median 12 %). Gas rings need muons within ~8° of vertical, so gas and aerogel light never appear in the same event.

3D tracer, Xe 2 bar (`cosmic_rates.py --groups s61`), n = 1.03 tile, muons uniform over the 5 × 5 cm tile (985 per hour for β = 1, 1152 with the spectrum). Rates per hour, β = 1 → real β:

| SiPM | Mirror Ø | Aerogel arcs ≥ 50 % | Full aerogel rings | Gas rings (≥ 80 % of the emitted photons on the SiPM) |
|---|---|---|---|---|
| 5.1 cm | 76 mm | 0 → 0.2 | 0 → 0.2 | 12.4 → 9.4 |
| 8 cm | 152 mm | 0 → 6.6 | 0 → 2.2 | 57.2 → 41.4 |
| 10.2 cm | 152 mm | 30 → 46 | 0.4 → 5.4 | 102 → 69 |
| 12.8 cm | 152 mm | 73 → 93 | 8.3 → 18.5 | 143 → 96 |
| 12.8 cm | 200 mm | 114 → 138 | 17 → 30 | 172 → 115 |

- Aerogel ring definitions: an **arc ≥ 50 %** has at least half of the 36 azimuth bins holding photons above threshold with half of their emitted photons on the SiPM; a **full ring** has ≥ 90 % of those bins reached; the tile side faces are black.
- With the spectrum the arcs rise (larger muon flux, smaller rings of slow muons) and the gas rings fall by 24–33 %. Xe in the pores (§6.3) lowers the arcs by 2–10 % and the full rings by 2–12 %.
- Imaging the aerogel with β = 1 muons needs **both** a SiPM of ≥ 10 cm (≥ 16 arrays, ≥ 1024 channels at 3 mm) **and** a mirror of ≥ 15 cm at f ≈ 20 cm (custom). The arcs and full rings of the 8 cm row come from slow muons with small aerogel angles, which do not measure the index without the muon momentum.
- Even then the rings are mostly partial; a full ring carries only ~25 emitted photons in 320–450 nm (~6 detected). A statistical measurement from the arcs would need external tracking of the muon direction at the mrad level.

### 6.2 Proximity focusing: configuration 4 (`cosmic_rates.py --groups c4`, `toy_prox.py`)
Tile → gas gap (air or Xe) → 2" SiPM, no mirror; plain n = 1.03 tile, each muon at its own β. Per hour: ≥ 80 % rings through the telescope (β = 1 in brackets) and arcs ≥ 50 % of the muons through the tile (half of the azimuths lit from at least one of six emission depths); radii (inner–outer at the median gas angle) from `toy_prox.py`, Xe 2 bar.

| Gap | Ring radius | Air | Xe 2 bar | Arcs ≥ 50 % per hour (air) |
|---|---|---|---|---|
| 30 mm | 7.5–9.9 mm | 235 (202) | 235 (202) | 424 |
| 50 mm | 12.4–14.8 mm | 95 (75) | 95 (76) | 239 |
| 70 mm | 17.1–19.5 mm | 34 (23) | 34 (23) | 129 |

- Xe in the pores (§6.3) lowers the Xe 2 bar ≥ 80 % rates by 1–7 %.
- Aerogel threshold for muons ~0.43 GeV/c: 91 % of the §5 telescope muons radiate.
- Median ring angle in the gas: 245 mrad (16–84 %: 210–249 mrad at 30 mm); the velocity shifts the aerogel angle in the tile by −21 mrad on average (rms 38 mrad). Estimators of the Xe − air shift: §7.
- β = 1 toy (`toy_prox.py --legacy --pressure 3`: Xe 3 bar, pore-vac φ = 0.97, radii from the fixed 0.249 rad): radii 8–10 / 13–15 / 18–20 mm, 193 / 69 / 18 rings ≥ 80 % and 376 / 208 / 104 arcs ≥ 50 % per hour on the 5.1 cm SiPM at 30 / 50 / 70 mm.

### 6.3 Aerogel index with Xe in the pores
- Expected shift Δn ≈ φ × (n_Xe − n_air), porosity φ ≈ 0.935 for n = 1.03 (ρ ≈ 0.14 g/cm³; φ ≈ 1 − 0.14/2.2 with 2.2 g/cm³ for silica, `AEROGEL_POROSITY_AIRREF`). At 400 nm, relative to the in-air specification (pore-air):
  - 2 bar: **+0.0010**, Δθc ≈ +3.9 mrad;
  - 3 bar: **+0.0016**, Δθc ≈ +6.3 mrad.

  Relative to empty pores: +0.0013 and +0.0019. The pore-vac variant with φ = 0.97 (`media.aerogel_index` default, used by `toy_*.py` and `proximity_config4.py`; `cosmic_rates.py` uses `--porosity-vac`, default 0.935) gives +0.0013 and +0.0020. The porosity of the actual tile is open (§8).
- This is a **lower bound**: Xe adsorbed on the large internal surface (500–1000 m²/g) could add to it, and pore filling may cause a transient of minutes to hours. No published measurement in pressurised Xe was found.
- **Precedent** *[literature]*: the LHCb RICH1 aerogel in contact with C₄F₁₀ had ~1.8× worse Cherenkov-angle resolution than simulated, attributed at least partly to absorbed C₄F₁₀; it was isolated from the gas for 2012 running (arXiv:1211.6759). Relevant to ARC, where the aerogel sits in the gas.
- Proposed measurement (configurations 2–4): the aerogel θc in air and in Xe at 1 / 2 / 3 bar (vacuum as an optional extra point), and its time dependence after each pressure change.
- Configuration 5: the 1 cm tile 1 mm in front of the SiPM Rayleigh-scatters 28 % of the gas photons at 350 nm, 18 % at 400 nm and 21 % of the FT×QE photons (ARC QE band), for the ARC simulation clarity C = 0.005 µm⁴/cm (`doc_values.py`, `hitmap_onaxis.py`); it depends on the tile clarity. With the §4 input card (FT×PDE): 13 % (§4 table).

---

## 7. Plan
Five configurations, in an order still to be decided (§8). 1 and 5 share the f = 203 mm mirror (`Phase1Layout` without and with the tile), 2 and 3 share a short-focal-length mirror, 4 needs no mirror.
1. **Xe gas ring:** Edmund #43-584 (or Thorlabs CM750-200-F01), plus a spare; the 2" × 2" SiPM plane at the focal plane; ~20 cm of Xe at 2 bar (reference) and 3 bar, no aerogel; trigger: the §5 telescope. Rates and photon counts: §4 (per day) and §5 (per hour, 1–3 bar).
2. **Aerogel ring in air:** the 1 cm tile 1 mm in front of the SiPM, air at atmospheric pressure, a short-focal-length mirror (Edmund #43-543, 2", f = 50.8 mm, or #43-553, 3", f = 76.2 mm). Measures the aerogel index, dispersion and clarity with air in the pores, the condition in which the vendor quotes n. With the spectrum: 48 (#43-543) and 25 (#43-553) ≥ 80 % rings (of the aerogel photons on the SiPM) and 150 / 106 arcs ≥ 50 % per hour. Xe in the gas changes this by < 2 %; Xe in the pores (configuration 3) lowers the ≥ 80 % rate by 1–6 % / 2–11 % at 1–3 bar. The tile side faces are black in the tracer; polished faces would add light, mostly off the ring.
3. **Aerogel ring in Xe:** configuration 2 with Xe at 1–3 bar and the same mirror; relative to 2, measures the aerogel-angle shift from Xe in the pores and its time dependence after each pressure change.
4. **Proximity focusing in air and in Xe:** the tile with a 3–5 cm gap to the same SiPM, no mirror (§6.2). The same air–Xe comparison without mirror systematics; needs a mechanically stable gap.
5. **Full set-up:** configuration 1 with the tile 1 mm in front of the SiPM. Its light misses the mirror for near-vertical muons (§6.1); it Rayleigh-scatters about 13 % of the gas-ring photons (§4). A ≥ 80 % ring carries 2.4 / 4.8 / 7.3 FT×PDE photons on average at 1 / 2 / 3 bar; about 58 / 139 rings per day carry ≥ 7 at 2 / 3 bar.
6. **Later:** an ARC-like cell with an ~26 cm hexagonal mirror (R ≈ 37 cm, custom or from ARC), an 8 × 8 cm SiPM tile and beam tests; it images the gas and aerogel rings together at ARC scale, the aerogel ring still partial for most tracks. For pure Xe characterisation with maximum photon statistics: the 6" f = 610 mm mirror (#43-970) with a ~65 cm vessel, ~3× the Xe and a larger sensor.

Aerogel-angle measurement (configurations 2–4):
- The muon velocity smears the aerogel angle (rms 38–70 mrad in the tile). On true emission angles (`cosmic_rates_plots.py`, no ring resolution), the β → 1 peak of each distribution carries the 3.86 mrad signal of Xe at 2 bar (3.75–3.86 mrad in a window around the known β = 1 angle), as do the rings of muons above 2 GeV/c; a peak finder on measured angles is still to be built (§8). The plain mean is unstable and biased low by 2–85 %, mainly from muons just below the air threshold that radiate only with Xe in the pores; a window fixed at the air angle keeps about 80 %.
- Momentum handles: 0.5–1 m of lead (cut at 0.8–1.5 GeV/c) needs the bottom counter enlarged behind it (the 8 × 8 cm counter keeps 1–8 % of the configuration-3 rings with #43-543, 5–25 % with #43-553; geometry only); the Xe gas-ring tag of configuration 3 keeps 17–43 % of the rings (residual shift ≈ −1 mrad at 2 bar) but cannot tag the air reference.

## 8. Open items
- SiPM: window and part number of the ceramic board's arrays; PDE at the operating overvoltage and temperature (§4).
- Porosity of the actual tile and the vendor data (§6.3).
- Order of the five configurations (§7); mechanical design of configurations 2–4 (tile holder, gap, trigger).
- Measured mirror reflectivity, PDE at the operating point and aerogel transmission, to replace the assumed inputs (§4); crosstalk and dark counts are not modelled.
- Cosmic rates: altitude and geomagnetic cutoff (§5), the trigger-counter geometry, the momentum handle and a data-level estimator of the aerogel angle for configurations 2–4, the tile side faces (§7).
- Comparison with the ARC full simulation (`ARC_Simulation_Reconstruction`), with Xe properties tuned.

## Sources
- Xe dispersion:
  - [Börzsönyi et al., Appl. Opt. 47, 4856 (2008)](https://doi.org/10.1364/AO.47.004856); coefficients as on [refractiveindex.info, Börzsönyi](https://refractiveindex.info/?shelf=main&book=Xe&page=Borzsonyi), which corrects a typo of the paper (C₁ = 12.75 × 10⁻³ µm²)
  - [refractiveindex.info, Bideau-Mehu et al. 1981](https://refractiveindex.info/?shelf=main&book=Xe&page=Bideau-Mehu)
  - [Leonard, At. Data Nucl. Data Tables 14, 21 (1974)](https://doi.org/10.1016/S0092-640X%2874%2980028-8), Table II(e): Koch's data (Kgl. Fysiogr. Sällsk. Lund Förh. 19, 173 (1949)) at 273.16 K and 1 atm, wavelengths in air (converted to vacuum for the comparison)
- Cosmic muons:
  - [PDG Review of Particle Physics (2022), Cosmic rays](https://pdg.lbl.gov/2022/reviews/rpp2022-rev-cosmic-rays.pdf)
  - [Guan et al., arXiv:1509.06176](https://arxiv.org/abs/1509.06176) (default spectrum)
  - [Reyna, arXiv:hep-ph/0604145](https://arxiv.org/abs/hep-ph/0604145)
  - [Shukla and Sankrith, arXiv:1606.06907](https://arxiv.org/abs/1606.06907)
  - BESS muon spectra at Tsukuba and Lynn Lake: [Motoki et al., Astropart. Phys. 19, 113 (2003), arXiv:astro-ph/0205344](https://arxiv.org/abs/astro-ph/0205344)
  - Muon range: Groom, Mokhov and Striganov, At. Data Nucl. Data Tables 78, 183 (2001); [PDG atomic and nuclear properties](https://pdg.lbl.gov/2024/AtomicNuclearProperties/)
- Xe density: [NIST Chemistry WebBook, fluid properties of xenon](https://webbook.nist.gov/cgi/fluid.cgi?ID=C7440633&Action=Page)
- Mirrors:
  - [Edmund #43-584](https://www.edmundoptics.com/p/3quot-dia-8quot-fl-uv-enhanced-alum-spherical-mirror/5147/)
  - [Edmund #43-543](https://www.edmundoptics.eu/p/2quot-dia-2quot-fl-uv-enhanced-alum-spherical-mirror/5106/) and [#43-553](https://www.edmundoptics.eu/p/3quot-dia-3quot-fl-uv-enhanced-alum-spherical-mirror/5116/) (short focal length)
  - [Edmund #49-608](https://www.edmundoptics.com/p/2quot-dia-8quot-fl-uv-enhanced-alum-spherical-mirror/10121/)
  - [Edmund #49-606](https://www.edmundoptics.com/p/1quot-dia-8quot-fl-uv-enhanced-alum-spherical-mirror/10119/)
  - [Edmund #72-960](https://www.edmundoptics.com/p/1-dia-1-fl-320-450nm-spherical-mirror/55204/)
  - [Edmund precision spherical mirror family](https://www.edmundoptics.eu/f/PdfExport/12013)
  - [Edmund broadband dielectric spherical mirror family](https://www.edmundoptics.eu/f/PdfExport/40178)
  - [Thorlabs CM750-200-F01](https://www.gophotonics.com/products/mirrors/thorlabs-inc/40-46-cm750-200-f01)
  - [Newport 20DC500AL.2](https://www.newport.com/p/20DC500AL.2)
  - [Avantier custom concave mirrors](https://avantierinc.com/solutions/custom-optics/concave-spherical-mirrors/)
  - [asphericon spherical mirrors](https://www.asphericon.com/en/solutions/products/optical-mirrors/spherical-mirrors/)
- ARC:
  - [k4geo ARC geometry](https://github.com/key4hep/k4geo/tree/main/FCCee/CLD/compact/CLD_o3_v01)
  - [ARC_geo_o1_v01.cpp](https://github.com/key4hep/k4geo/blob/main/detector/PID/ARC_geo_o1_v01.cpp)
  - [Pezzulo et al., JINST 21 C06015 (2026)](https://ora.ox.ac.uk/objects/uuid:db2688d0-abe4-47e5-b1b2-c48f67c455b8/files/rqr46r3422)
  - [ARC at RICH2025](https://indico.gsi.de/event/20387/contributions/90364/attachments/51775/78884/RICH2025-ARC-Pezzulo.pdf)
  - [R. Cardinale, SiPM housing with active cooling, RICH2025](https://indico.gsi.de/event/20387/contributions/90422/attachments/51826/79107/RICH2025_RCardinale.pdf) (slides 10–12, 18: baseboards v1 and v2)
- SiPM:
  - [ePIC dRICH photodetector unit, NIM A (2024)](https://iris.unito.it/retrieve/handle/2318/2012671/1373309/1-s2.0-S0168900224005953-main.pdf)
  - [Hamamatsu S13361-3050 series datasheet, KAPD1054E07](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/s13361-3050_series_kapd1054e.pdf)
  - [Congedo et al., arXiv:2602.20139](https://arxiv.org/abs/2602.20139) (SiPM array reflectance)
- Aerogel:
  - Bellunato et al., NIM A 595 (2008) 183 (single-pole dispersion)
  - [LHCb RICH performance, arXiv:1211.6759](https://arxiv.org/pdf/1211.6759)
  - [LHCb RICH aerogel with LHC data, NIM A](https://www.sciencedirect.com/science/article/abs/pii/S0168900210021522)
