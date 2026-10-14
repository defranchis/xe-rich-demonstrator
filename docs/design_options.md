# Xe RICH pre-demonstrator: summary of the options explored

This summarises the design exploration for a lab pre-demonstrator of pressurised xenon as the ARC gas radiator (the demonstrator proper is the later set-up with all the ARC baseline specifications). The study drivers in `studies/` produce every number quoted here and the corresponding plots; the [README](../README.md) lists their outputs and the assumptions and simplifications.

**Status in one paragraph.** The plan has five configurations (§7). (1) Gas-ring imaging with cosmic muons: a stock 3" spherical mirror with f ≈ 203 mm, a 2" × 2" SiPM plane at its focus and ~20 cm of Xe at the reference pressure of 2 bar (1–3 bar scanned). (2, 3) The aerogel ring imaged with a short-focal-length mirror, in air and in Xe, which measures how Xe in the pores shifts the aerogel index. (4) The same comparison by proximity focusing, without a mirror, in air and in Xe. (5) The full set-up: configuration 1 with the 1 cm aerogel tile 1 mm in front of the SiPM; the aerogel ring is not imaged in this geometry (for near-vertical muons its light misses the mirror, §6.1), and the tile Rayleigh-scatters about a fifth of the gas-ring photons (§6.3). A full-size, ARC-like cell needs a custom mirror and is left for later.

**Reference pressure.** Xe 2 bar (`constants.P_REF`, the default of the drivers). Numbers still quoted at 3 bar say so, with the driver option that reproduces them.

---

## 1. Physics inputs

### 1.1 Cherenkov emission in Xe
- The Frank–Tamm spectrum is d²N/dx dE ≈ 370 z² sin²θc eV⁻¹ cm⁻¹. In wavelength it goes as ∝ 1/λ², and the Xe dispersion steepens it further toward the UV.
- The emitted spectrum has **no peak** in the transparent region. It rises monotonically down to the Xe absorption edge at ~150–160 nm, close to the 147 nm resonance.
- The **detected** peak is set by the product of mirror reflectivity, window transmission and SiPM PDE. With standard SiPMs it falls at ~300–350 nm.
- About 53 % of the photons emitted in 200–600 nm lie below 300 nm.

### 1.2 Xe optical constants (`xerich.media`)
These use the Börzsönyi et al. (2008) Sellmeier formula (273 K, taken as 273.15 K, and 1000 mbar; fitted to an 800 nm measurement and literature UV–visible indices), scaled with Lorentz–Lorenz and the virial molar density. Xe is ~1.6 % non-ideal at 3 bar; the code's density agrees with NIST to 0.03 % at 1–3 bar and 20 °C (NIST effective B = −133 cm³/mol at 293 K vs the −130 used). All values are at 20 °C (`scripts/investigations/xe_index/doc_values.py`). The Bideau-Mehu (1981) formula gives an (n−1) lower by 0.7 % at 264 nm, 1.7 % at 400 nm and 2.2 % at 954 nm, a θc 0.9 % lower at 400 nm, and a chromatic rms 9–12 % larger. Against Koch's measurements (as tabulated by Leonard, 1974; 273.16 K, 1 atm, air wavelengths converted to vacuum), the Börzsönyi (n−1) agrees to ≤ 0.08 % above 290 nm and is lower by 0.13 % at 264 nm and 0.46 % at 230 nm; at 264 nm and 3 bar this is 0.04 mrad in θc.

| | 2 bar | 3 bar |
|---|---|---|
| (n−1) at 300 nm | 1.42 × 10⁻³ | 2.15 × 10⁻³ |
| θc (β = 1) at 400 nm | 51.9 mrad | 63.7 mrad |
| π / K / p threshold (300 nm) | 2.6 / 9.2 / 17.6 GeV | 2.1 / 7.5 / 14.3 GeV |
| N_γ/cm saturated, 200–600 nm | 4.4 | 6.7 |
| N_γ/cm saturated, 320–450 nm | 1.1 | 1.7 |

- 2 bar Xe is similar to C₄F₁₀ in index. 3 bar gives about 50 % more photons and lowers the K threshold.
- π/K angle separation at 300 nm, for 2 bar vs 3 bar:
  - 20 GeV: 5.6 vs 4.5 mrad.
  - 45 GeV: 1.0 vs 0.9 mrad.
  - The separation shrinks slightly with pressure at high p, because it scales as Δm²/(2p² tanθc).

### 1.3 Choice of wavelength band
Emitted yield and chromatic rms per photon, for β ≈ 1:

| Band | N/cm (2 / 3 bar) | Chromatic rms per photon (2 / 3 bar) | σ/√N per cm, 3 bar |
|---|---|---|---|
| 320–450 nm | 1.1 / 1.7 | 0.41 / 0.50 mrad | ~0.38 mrad |
| 300–600 nm | 2.1 / 3.1 | 0.70 / 0.86 mrad | ~0.49 mrad |
| 250–600 nm | 3.0 / 4.5 | 1.2 / 1.4 mrad | — |
| 200–600 nm | 4.4 / 6.7 | 2.2 / 2.7 mrad | ~1.1 mrad |

- The 320–450 nm band gives the best chromatic term per track but keeps only ~25 % of the 200–600 nm photons.
- In practice the SiPM PDE sets the band. 320–450 nm roughly matches standard SiPMs.

### 1.4 FCC-ee π/K momentum spectra (`gen_pk.py`, Pythia 8.317)
Signal processes only, with |cosθ| < 0.9:

| √s | K median p | K above threshold (2 / 3 bar) | K between π and K thresholds | K with p > 20 GeV |
|---|---|---|---|---|
| Z, 91 GeV | 2.7 GeV | 14 / 18 % | ~38 % | 2.2 % |
| WW, 160 GeV | 2.7 GeV | 14 / 19 % | ~37 % | 2.2 % |
| ZH, 240 GeV | 3.1 GeV | 15 / 20 % | ~41 % | 2.6 % |
| tt̄, 365 GeV | 3.7 GeV | 20 / 26 % | ~42 % | 4.1 % |

- In Xe, most of the gas K-ID power comes from threshold mode.
- Kaons below ~2–2.6 GeV need the aerogel or time-of-flight.
- The dependence on √s is weak.

---

## 2. Mirrors

### 2.1 Coating
| Coating | Verdict |
|---|---|
| **Al + MgF₂** | The standard RICH coating. R > 80–85 % down to ~160 nm. Needed only if VUV-sensitive photodetectors are used. |
| **Protected / UV-enhanced Al (Al + SiO₂)** | Fine above ~250 nm, cheap, broadband, insensitive to angle of incidence. **The choice for the stock mirrors (configurations 1, 2, 3 and 5).** |
| Broadband dielectric | Sharp band edges act as a wavelength filter. But the band shifts with angle of incidence: the Edmund "320–450 nm" spec holds at 45°, and at 0° it becomes 340–488 nm. Reflectance is also polarisation-dependent. Usable, but not needed. |
| Silver, gold | Unsuitable: reflectivity collapses below ~350–400 nm. |
| Bare Al | Unsuitable: the native oxide degrades reflectivity below ~200 nm and over time. |

### 2.2 Focal length and geometry
- The SiPM sits at the focal plane. All photons with the same direction converge on one point, so a straight track gives a sharp ring of radius f·tanθc.
- With f ≈ 20 cm, the ring radius is 10.5 mm at 2 bar and 12.9 mm at 3 bar.
- The ring centre on the sensor sits at f·tanθ_track, independent of where the track crosses. It moves by ~3.5 mm per degree of track tilt.
- A track illuminates a patch of the mirror of radius L·tanθc, where L is the radiator length. That is ≈ 13 mm at 3 bar with L = 20 cm.

### 2.3 Spherical aberration (`raytrace.py --pressure 3`, f = 203 mm, 3 bar)
| Track offset from the mirror axis | Outermost mirror radius hit | Ring smearing, rms per photon |
|---|---|---|
| 0 | 13 mm | 0.03 mrad |
| 25 mm | 38 mm | 0.4 mrad |
| 50 mm | 62 mm | 1.8 mrad |
| 75 mm | 87 mm | 5.4 mrad |

At f ≈ 20 cm, only the central ~8 cm of a large (f/1) mirror is good for precision ring imaging. A track's tilt can partly compensate an offset.

### 2.4 Commercial mirrors considered (f ≈ 20 cm unless stated)
| Part | Diameter | f | Coating | Price | Comment |
|---|---|---|---|---|---|
| **Edmund #43-584** | **76.2 mm (3")** | **203.2 mm** (f/2.7) | Enhanced Al, Ravg > 89 % at 250–450 nm | **€316** | **Recommended for configurations 1 and 5.** Largest stock mirror at f ≈ 20 cm. |
| Thorlabs CM750-200-F01 | 75 mm | 200 mm | UV-enhanced Al, R > 90 % at 250–450 nm, N-BK7 | quote | Equivalent alternative; f is exactly 200 mm. |
| Edmund #49-608 | 50.8 mm (2") | 203.2 mm (f/4) | Enhanced Al, Ravg > 89 % at 250–450 nm | $199 | OK for centred beam tracks only. |
| Edmund #49-606 | 25.4 mm (1") | 203.2 mm (f/8) | Enhanced Al | $135 | Too small: the ring fills the mirror at 3 bar. |
| Newport 20DC500AL.2 | 50.8 mm | ~250 mm (from the naming) | UV-enhanced Al | — | Specs not verified. |
| Edmund #72-975 | 76.2 mm | 203.2 mm | Dielectric, 320–450 nm | €464 | Built-in band filter; angle of incidence matters. |
| Edmund #49-609 | 76.2 mm | 304.8 mm (f/4) | Enhanced Al | €316 | Intermediate option if the vessel allows ~35 cm. |
| Edmund #43-970 | 152.4 mm (6") | **609.6 mm only** | Enhanced Al | €745 | 3× the photons with a ~60 cm radiator. Ring radius 3.9 cm, and it is poor for cosmics (needs a ~16 cm sensor). Not ARC-like. |
| Edmund #72-979 | 152.4 mm | 609.6 mm | Dielectric, 320–450 nm | €1310 | As above. |
| Edmund #72-960 | 25.4 mm | 25.4 mm (f/1) | Dielectric, 320–450 nm | $186 | First candidate looked at. Ring only ~1.6 mm. Not useful. |

Notes:
- Stock spherical mirrors stop at 75–76 mm diameter for f ≈ 200 mm, at Thorlabs, Edmund, OptoSigma, Newport and Knight Optical.
- Edmund has no 4" option, and its 6" mirrors come only at f = 610 mm.
- Telescope primaries are 200 mm f/4–f/6 paraboloids (f = 0.8–1.2 m), so they are not suitable.
- Specs that matter for the Edmund mirrors:
  - **EFL tolerance ±2 %:** this gives a 2 % scale on θc. Measure f in situ or calibrate on β ≈ 1 tracks.
  - λ/4 surface accuracy and 60-40 surface quality: irrelevant at the mrad level.
  - Borofloat substrate: fine in Xe, since the pressure is equal on both sides.
  - Vacuum/pressure compatibility of mounts and adhesives: not specified, so ask.

### 2.5 Large or custom mirrors
- **Custom spherical mirror**, ~200–260 mm diameter, R ≈ 400 mm (sag ≈ 13 mm), Al + SiO₂ or Al + MgF₂. Lead time is weeks to months. Possible suppliers:
  - **Avantier:** 5–500 mm, λ/4 standard, UV-enhanced Al offered.
  - **asphericon (Jena):** up to 420 mm, coatings from 190 nm.
  - Also ask the LHCb RICH and ARC groups for their suppliers and any spare segments.
- **Mosaic** of 2–4 × 3" mirrors around a common centre of curvature. Each segment works near its own axis, but alignment is needed.
- A specification of λ/2 or even λ is sufficient for mrad-level imaging.

---

## 3. ARC reference geometry (k4geo `CLD_o3_v01` / `ARC_o1_v01`)
- Hexagonal prism cells with a **14.815 cm side**: 25.7 cm flat-to-flat, 29.6 cm corner-to-corner.
- **Radial depth 20 cm.** The barrel spans r = 190–210 cm, giving ~18 cm of gas inside 1 cm Rohacell walls.
- The barrel has 27 cells in φ (17–18 distinct designs, mirrored in z). The endcaps have 6 sectors × 21 distinct cells.
- **Mirror:** spherical, 2 mm thick, with a per-cell optimised R, sphere-centre position and sensor tilt.
  - Barrel R = 25.5–36.9 cm (median 33.6). The central cell has R = 36.9 cm (f = 18.5 cm) and a sensor tilt of −1.2°.
  - Toward |z| ≈ 1–2 m, R falls to ~31–33 cm. The sphere centre moves 12–14 cm toward the IP, which steers the ring onto the sensor, and the sensor tilts by 8–15°.
  - Endcap R = 22.1–36.0 cm, falling with radius.
- **Sensor:** 8 × 8 cm SiPM, 0.8 mm pitch in the simulation (the JINST paper quotes 0.5 mm as baseline), with a 2 mm cooling plate.
- **Aerogel:** 1 cm, n = 1.03 at 400 nm.
- These parameters were optimised for C₄F₁₀ at 1 bar (θc ≈ 53 mrad). Xe at 2 bar is similar. At 3 bar the ring is ~20 % larger, so the parameters need re-optimisation.
- Implication for the pre-demonstrator: f ≈ 20 cm with ~18–20 cm of gas reproduces the central barrel cell.

---

## 4. Photodetector
- Baseline: the published ARC prototype board, 2" × 2" with four Hamamatsu SiPM arrays of 8 × 8 SiPMs (3 × 3 mm² pads at 3.2 mm pitch, 256 channels). With the packages abutting, the pads span 51.2 mm (= 16 × 3.2 mm, the scripts' grid) within a 51.6 mm outline; across the package seams the pad pitch is 3.4 mm. The front end is ALCOR (INFN-BO front-end boards). The cooled baseboard is a ceramic PCB with internal coolant flow, designed for −60 to −80 °C (Pezzulo et al.; Cardinale, slide 11); the pre-demonstrator runs it at −20 °C, the lower end of the arrays' datasheet range.
- Part number: Hamamatsu S13361-3050NE-08 is the one given for the first baseboard version (Cardinale, slide 10), and is to be confirmed for the ceramic board's arrays, like the window values below.
- Window: the S13361 datasheet gives an epoxy window of refractive index 1.55 and resin thickness 0.1 ± 0.05 mm. These are datasheet values, still to be confirmed for the arrays on the board.
- Response: 320–900 nm; typical PDE curve at 25 °C and 3 V overvoltage, peak 40.2 % at 460 nm, 39.95 % at 450 nm (datasheet KAPD1054E07, read from its figure). The datasheet operating range is −20 to +60 °C without dew condensation.
- **Input card: the defaults of `cosmic_rates.py` are the validated baseline** (`cli.BASELINE`, values from the named constants; README, model assumptions), at Xe 2 bar (`constants.P_REF`) with ring hits that leave out the muon's pad and `--nphot 7200`. The tables published before it used the ARC QE, a continuous 2" square, reflectivity 1, a constant tile index, porosity 0.97, no board reflection and a vertical-muon Rayleigh factor in configuration 4; `--preset published` reproduces that model. The values and their status:
  - PDE (`--response s13361`): the typical curve above, through the window. Datasheet; not at the operating temperature and overvoltage.
  - Board (`--sensor-model board`): 2 × 2 abutting packages of 25.8 mm (footprint 51.6 mm), 8 × 8 pads of 3.0 mm at 3.2 mm pitch, 0.2 mm from the package edge; active fraction 0.879 inside ±25.6 mm, 0.865 of the footprint. Datasheet outline; abutting packages assumed. Containment and arcs use the footprint, photon counts the active pads.
  - Ring hits: binary pad hits, P = 1 − exp(−μ_pad) per pad, of the unscattered ring photons, without the pad crossed by the muon; no crosstalk, dark counts or afterpulses. Assumed. Sampled with 7200 tile photons per muon (pad statistics biased low with fewer).
  - Mirror reflectivity (`--mirror-r nominal`): 0.89 at 250–450 nm, 0.87 at 700 nm, 0.82 at 800–850 nm, 0.85 at 950 nm, linear in between. Assumed from the vendor spec (Ravg > 89 % at 250–450 nm, > 85 % at 250–700 nm); placeholder for the measured curve.
  - Aerogel dispersion: single pole, λ₀ = 83.22 nm (Bellunato et al.), equal to the run's tile index at 400 nm. Literature shape, assumed for this tile.
  - Porosity 0.935 (`--porosity-vac`, from the tile density, §6.3) and pore-gas temperature 293.15 K (`--pore-t`, `constants.T_PORE`). Placeholders until the vendor data and the box thermal state are known.
  - Ring definitions stay geometric (a ring counts from its light in the ARC QE band whatever the response), as in the published tables.
  - Configuration 4, Rayleigh scattering in the tile (`--prox-rayleigh per-photon`): each photon direction is attenuated along its own path from the emission depth to the tile exit, averaged over the FT × PDE spectrum of the run's tile index, for the photon counts and the pad hits alike, with the β = 1 FT × PDE spectrum for every muon (a vertical muon's photons keep 0.933 with the single-pole index, 0.934 with a constant one; at Xe 2 bar the photons of the ≥ 80 % rings keep 0.929 / 0.932 / 0.933 at gaps 30 / 50 / 70 mm). Known approximation, not corrected: the configuration-4 containment and pads use one Cherenkov angle, at n = 1.03, while the yield uses the dispersive index; this underestimates the ≥ 80 % ring rate by up to about 2 %, growing with the gap.
  - Board reflectance (`--board-r-spec`, `--board-r-diff`; defaults 0.11 and 0.10, `constants.BOARD_R_SPEC`, `BOARD_R_DIFF`): specular 0.11 (range 0.08–0.14) and diffuse (Lambertian) 0.10, flat in wavelength. Source: Congedo et al., arXiv:2602.20139, Sec. 3.1, Table 1: S13361-3050 and -2050 8 × 8 arrays (50 µm cells, epoxy, no window) in an integrating sphere at 8°: total reflectance 18–24 % above 380 nm, diffuse 6–10 % (about 10 % for the 50 µm epoxy arrays), specular = total − diffuse (the caption of the paper's Fig. 6 names the S13360 series, an inconsistency of the paper). Literature measurement of the same part; to be confirmed on our board. The light on the footprint is shared into detected (PDE, per incident photon), reflected and absorbed, so the ring hits are unchanged; the reflected light (two board reflections, through the same mirror and tile) is other light, never ring hits. Not modelled: the drop of the reflectance below about 380 nm (15–18 % of the reflected detected light lies there, so the flat model overestimates it by up to that share), reflection beyond the 51.6 mm footprint, the vessel walls, and configuration 4 (no mirror: the light returning to the board is 0.23 / 0.06 / 0.01 % of the ring photons at gaps 30 / 50 / 70 mm for vertical muons, `scripts/investigations/board_reflection/config4_size.py`).
  - Results with the defaults at Xe 2 bar: **[PLACEHOLDER: board table from the full-statistics reruns at `--nphot 7200`, to be filled in]**
- Alternative: a 0.8 mm pitch, as in the ARC simulation.

| Pixel pitch | Pixel term per photon at f = 203 mm |
|---|---|
| 3.2 mm | 4.5 mrad (dominant) |
| 0.8 mm | 1.1 mrad |
| 0.5 mm | 0.7 mrad |

---

## 5. Cosmic-ray rates, configurations 1 and 5 (gas ring)
Assumptions (`cosmic_spectrum.py`, `cosmic_rates.py`):
- Muons from the momentum–zenith parametrisation of Guan et al. (arXiv:1509.06176), each at its own β: 60 m⁻²s⁻¹sr⁻¹ above 1 GeV/c (Reyna 70, Shukla–Sankrith 58; the PDG 2022 review quotes ≈ 70, with recent measurements 10–15 % lower). Against cos²θ, the zenith law changes the ring rates by < 0.3 %.
- Sea level, no roof, no altitude or geomagnetic correction (CERN is at ≈ 450 m). The BESS spectrum at Lynn Lake (360 m, cutoff 0.4 GV) lies 14–20 % above the Guan model below 5 GeV/c and about 6 % below it above; this affects the trigger and aerogel-ring rates more than the gas-ring rates (p > 1.6–2.8 GeV/c).
- Trigger telescope, idealised: 5 × 5 cm at the SiPM plane and 8 × 8 cm at the mirror vertex plane, 105 muons per hour (88 for β = 1 muons with cos²θ and I0 = 70 above 1 GeV/c). Real counters sit above the SiPM and below the mirror, which lowers the rates (§8). The threshold at the UV edge of the QE band (264 nm) is 2.75 / 1.94 / 1.58 GeV/c at 1 / 2 / 3 bar; 41 / 53 / 59 % of the telescope muons are above it.
- A ≥ 80 % ring has ≥ 80 % of its sampled photons on the mirror and ≥ 80 % of the ring on the SiPM; a full ring has ≥ 99 % on both. FT×QE photons are Frank–Tamm photons weighted by the SiPM quantum efficiency (the QE table of the ARC simulation; no window), without Rayleigh-scattered photons. The tables of §5 and §6 use this published model (`--preset published`), not the §4 input card. The tracer estimates them per muon from its sampled photons, so cuts such as "≥ 10 photons" are approximate for rings whose count is near the cut.

3D tracer, 3" mirror (f = 203 mm), 2" SiPM, through the telescope (5 seeds, seed spread 0.2–0.3 per hour):

| | 1 bar | 2 bar | 3 bar |
|---|---|---|---|
| ≥ 80 % rings per hour | 9.4 | 9.9 | 9.1 |
| full rings per hour | 7.6 | 6.8 | 5.6 |
| ≥ 80 % rings per hour with ≥ 10 FT×QE photons | 3.7 | 7.3 | 7.6 |
| FT×QE photons per ≥ 80 % ring (with the board: on active pads, the muon's pad included) | 7.9 | 15.9 | 24.1 |

- Reyna and Shukla–Sankrith change the ≥ 80 % rates by +14 to +17 % and −5 to −10 %; β = 1 muons (cos²θ, I0 = 70) give 17.0 / 13.0 / 10.1 per hour.
- **The SiPM limits the angular acceptance.** Full rings need |θ_μ| ≲ 3.6° for a 5 cm sensor (atan((half sensor − ring radius)/f) at 400 nm, 3 bar).
- **The mirror limits the position acceptance:** about ±2.1 cm for a 3" mirror without the tile and ±2.2 cm with it (clear radius − L·tanθc, 3 bar).

Mirror and SiPM size (`cosmic_spectrum.py`; muons up to 25° from vertical over a square at the mirror plane that contains every track able to give a ring, radiator 190 mm; not comparable with the telescope table), ≥ 80 % rings per hour:

| Mirror | SiPM | 2 bar, β = 1 | 2 bar, Guan | 3 bar, β = 1 | 3 bar, Guan | Full rings, 3 bar, Guan |
|---|---|---|---|---|---|---|
| 2" | 5.1 cm | 7.0 | 6.1 | 5.0 | 5.4 | 2.8 |
| **3"** | **5.1 cm** | **19.3** | **16.1** | **13.3** | **13.5** | **8.4** |
| 3" | 4 cm | 8.3 | 8.0 | 5.0 | 6.2 | 3.7 |
| 3" | 8 cm | 65.4 | 47.4 | 53.7 | 46.0 | 28.9 |

- The β = 1 mirror scan of `cosmic_acc.py --pressure 3` (3 bar, 8 cm sensor, L = 200 mm; rings entirely on the sensor with all / ≥ 80 % of the photons on the mirror, per hour): 2" 6/17, 3" 27/46, 4" 59/91, 6" 173/216, 20 cm 333/396, ARC 26 cm 776/893 (at 63.6 mrad; statistical errors up to about 12 per hour). A larger mirror increases the rate roughly with its area until the sensor angle cut takes over. Much of the extra rate from large f/1 mirrors is only good for photon counting.

---

## 6. Aerogel
### 6.1 With the f = 203 mm mirror (configuration 5): not imaged (`cosmic_rates.py --groups s61`, `toy_aerogel.py`, `toy_scan.py`)
- θ_aero = 242 mrad inside the tile for n = 1.03 (249 mrad in the gas, the value used for the distances below); `toy_aerogel.py` and `toy_scan.py` add the Xe pore shift (with `--pressure 3`: n ≈ 1.032, 250 mrad in the tile).
- After ~19 cm of gas the photons are ~48 mm from the track, which **misses the 3" mirror**. The imaged ring would have **r ≈ 52 mm**, which **misses the 5 cm SiPM**.
- For β = 1 muons with a 3" mirror and a 5.1 cm SiPM, only muons at 6–22° put aerogel light on the sensor: ≤ ~3.5 emitted photons per muon (~1 detected), on an arc of ≤ 20 % of the ring (median 12 %).
- Gas rings need muons within ~8° of vertical, so gas and aerogel light never appear in the same event.

3D tracer, Xe 3 bar (`cosmic_rates.py --groups s61` at 435dd80; it now runs these layouts at `P_REF`), n = 1.03 tile, muons uniform over the 5 × 5 cm tile (985 per hour for β = 1, 1152 with the spectrum). Rates per hour, β = 1 → spectrum (each muon at its own β):

| SiPM | Mirror Ø | Aerogel arcs ≥ 50 % | Full aerogel rings | Gas rings (≥ 80 % of the emitted photons on the SiPM) |
|---|---|---|---|---|
| 5.1 cm | 76 mm | 0 → < 0.5 | 0 → 0 | 9.6 → 9.4 |
| 8 cm | 152 mm | 0 → 5.9 | 0 → 1.3 | 49.8 → 40.9 |
| 10.2 cm | 152 mm | 30 → 45 | 0.2 → 3.7 | 94 → 73 |
| 12.8 cm | 152 mm | 73 → 92 | 7.1 → 14.8 | 136 → 103 |
| 12.8 cm | 200 mm | 114 → 137 | 15 → 25 | 161 → 123 |

- An arc ≥ 50 % has at least half of the 36 azimuth bins holding photons above threshold with half of their emitted photons on the SiPM; a full ring has ≥ 90 % of those bins reached; the tile side faces are black.
- With the spectrum the arcs rise (larger muon flux, smaller rings of slow muons) and the gas rings fall by 2–24 %. Xe in the pores (§6.3) lowers the arcs by 4–17 % and the full rings by 10–24 %.

- For β = 1 muons, imaging the aerogel needs **both** a SiPM of ≥ 10 cm (≥ 16 matrices, ≥ 1024 channels at 3 mm) **and** a mirror of ≥ 15 cm at f ≈ 20 cm (custom). The arcs and full rings of the 8 cm row come from slow muons with small aerogel angles, so they do not measure the index without the muon momentum.
- Even then the rings are mostly partial, and a full ring carries only ~25 emitted photons in 320–450 nm (~6 detected).
- A statistical measurement from the arcs would need external tracking of the muon direction at the mrad level.

### 6.2 Proximity focusing: configuration 4 (`cosmic_rates.py --groups c4`, `toy_prox.py`)
Aerogel tile → gas gap (air or Xe) → 2" SiPM, no mirror; plain n = 1.03 tile, each muon at its own β. Per hour: ≥ 80 % rings through the telescope (β = 1 in brackets) and arcs ≥ 50 % of the muons through the tile (half of the azimuths lit from at least one of six emission depths); radii (inner–outer at the median gas angle) from `toy_prox.py --pressure 3`, Xe 3 bar.

| Gap | Ring radius | Air | Xe 3 bar | Arcs ≥ 50 % per hour (air) |
|---|---|---|---|---|
| 30 mm | 7.5–9.9 mm | 226 (194) | 227 (195) | 414 |
| 50 mm | 12.4–14.8 mm | 90 (71) | 91 (72) | 231 |
| 70 mm | 17.1–19.5 mm | 32 (20) | 32 (21) | 124 |

- Xe in the pores (§6.3) lowers the Xe 3 bar ≥ 80 % rates by 2–12 %.
- The aerogel threshold for muons is ~0.43 GeV/c, so nearly every triggered muon radiates (91 % of the §5 telescope muons are above it).
- Median ring angle in the gas: 245 mrad (16–84 %: 210–249 mrad at 30 mm); the velocity shifts the aerogel angle in the tile by −21 mrad on average (rms 38 mrad). Estimators for the Xe-minus-air comparison: §7.
- The β = 1 toy (`toy_prox.py --legacy --pressure 3`: Xe 3 bar, pore shift 0.97 (n_gas − 1), radii from the fixed 0.249 rad) gives radii of 8–10 / 13–15 / 18–20 mm, 193 / 69 / 18 rings ≥ 80 % and 376 / 208 / 104 arcs ≥ 50 % per hour on the 5.1 cm SiPM at 30 / 50 / 70 mm.

### 6.3 Aerogel index with Xe in the pores
- Expected shift: Δn ≈ porosity × (n_Xe − n_air), with a porosity of ≈ 0.935 for n = 1.03 (ρ ≈ 0.14 g/cm³). At 400 nm:
  - 2 bar: **+0.0010**, Δθc ≈ +3.9 mrad;
  - 3 bar: **+0.0016**, Δθc ≈ +6.3 mrad.
  
  These are relative to the in-air specification (≈ 1 − 0.14/2.2 with 2.2 g/cm³ for silica, `AEROGEL_POROSITY_AIRREF`); relative to empty pores they are +0.0013 and +0.0019. The code's vacuum-referenced variant (`pore-vac`) uses porosity 0.97 (+0.0013 and +0.0020); the porosity of the actual tile is open (§8).
- This is a **lower bound**: Xe adsorbed on the large internal surface (500–1000 m²/g) could add to it. Pore filling may also cause a transient of minutes to hours.
- No published measurement in pressurised Xe was found.
- **Precedent:** the LHCb RICH1 aerogel in contact with C₄F₁₀ had ~1.8× worse Cherenkov-angle resolution than simulated, attributed at least partly to absorbed C₄F₁₀. The aerogel was isolated from the gas for 2012 running (arXiv:1211.6759). This is relevant to ARC, where the aerogel sits in the gas.
- Proposed measurement (configurations 2–4): the aerogel θc in air and in Xe at 1/2/3 bar (vacuum as an optional extra point), plus its time dependence after each pressure change.
- In configuration 5, the 1 cm tile 1 mm in front of the SiPM Rayleigh-scatters 28 % of the gas photons at 350 nm and 18 % at 400 nm, 21 % of the FT×QE photons over the SiPM band, for the clarity of the ARC simulation (C = 0.005 µm⁴/cm; `doc_values.py`, `hitmap_onaxis.py`). That depends on the tile clarity.

---

## 7. Plan
Five configurations, in an order still to be decided (§8). Configurations 1 and 5 share the f = 203 mm mirror (`Phase1Layout` without and with the tile), 2 and 3 share a short-focal-length mirror, and 4 needs no mirror.
1. **Xe gas ring:**
   - Mirror: Edmund #43-584 (or Thorlabs CM750-200-F01). Buy a spare.
   - SiPM: the 2" × 2" plane at the focal plane.
   - Radiator: ~20 cm of Xe at 2 bar (reference) and 3 bar, no aerogel.
   - Trigger: the matched scintillator telescope from §5.
   - Expect about 230 ≥ 80 % rings per day through the telescope, of which about 180 carry ≥ 10 FT×QE photons at 2–3 bar (90 at 1 bar; §5).
2. **Aerogel ring in air:** the 1 cm aerogel tile 1 mm in front of the SiPM, air at atmospheric pressure, and a short-focal-length mirror (Edmund #43-543, 2", f = 50.8 mm, or #43-553, 3", f = 76.2 mm). It measures the aerogel index, dispersion and clarity with air in the pores, the condition in which the vendor quotes n. With the spectrum, 47 (#43-543) and 23 (#43-553) ≥ 80 % rings (of the aerogel photons on the SiPM) and 149 / 104 arcs ≥ 50 % per hour. Xe in the gas changes this by < 2 %; Xe in the pores (configuration 3) lowers the ≥ 80 % rate by 1–6 % / 1–11 % at 1–3 bar. The tile side faces are black in the tracer; polished faces would add light, mostly off the ring.
3. **Aerogel ring in Xe:** configuration 2 with Xe at 1–3 bar and the same mirror. Relative to 2, it measures the shift of the aerogel angle from Xe in the pores and its time dependence after each pressure change.
4. **Proximity focusing in air and in Xe:** the tile with a 3–5 cm gap to the same SiPM and no mirror (§6.2). It makes the same air–Xe comparison without mirror systematics and needs a mechanically stable gap.
   - The muon velocity smears the aerogel angle (rms 38–60 mrad in the tile). With true emission angles (`cosmic_rates_plots.py`, no ring resolution), the β → 1 peak of each distribution carries the 6.32 mrad signal of Xe at 3 bar (6.18–6.36 mrad in a window around the known β = 1 angle), as do the rings of muons above 2 GeV/c; a peak finder on measured angles is still to be built (§8). The plain mean is unstable and biased low by 3–51 %, mainly from muons just below the air threshold that radiate only with Xe in the pores. A window fixed at the air angle keeps about 80 %.
   - Momentum handles: 0.5–1 m of lead (cut at 0.8–1.5 GeV/c) needs the bottom counter enlarged behind it (the 8 × 8 cm counter keeps 1–8 % of the configuration 3 rings with #43-543, 5–27 % with #43-553; geometry only); the Xe gas-ring tag of configuration 3 keeps 26–51 % of the rings (residual shift ≈ −2 mrad at 3 bar) but cannot tag the air reference.
5. **Full set-up:** configuration 1 with the 1 cm aerogel tile 1 mm in front of the SiPM. Its light misses the mirror for near-vertical muons (§6.1), and it Rayleigh-scatters about a fifth of the gas-ring photons (§6.3). With the tile, a ≥ 80 % ring carries 5.8 FT×QE photons on average at 1 bar and 17.9 at 3 bar, where about 175 rings per day carry ≥ 10.
6. **Later:** an ARC-like cell with an ~26 cm hexagonal mirror (R ≈ 37 cm, custom or from ARC), an 8 × 8 cm SiPM tile and beam tests.
   - It images the gas and aerogel rings together at ARC scale; even there the aerogel ring is partial for most tracks.
   - Alternative if the goal is purely Xe characterisation with maximum photon statistics: the 6" f = 610 mm mirror (#43-970) with a ~65 cm vessel. That needs ~3× the Xe and a larger sensor.

## 8. Open items
- Window of the board's arrays (§4: datasheet values, to be confirmed), the part number of the ceramic board's arrays (§4) and the PDE at the operating overvoltage and temperature.
- Porosity of the actual tile (0.935 from its density, 0.97 in the code's vacuum-referenced variant; §6.3) and the vendor data.
- Order of the five configurations (§7).
- Mechanical design of configurations 2–4: tile holder, gap, trigger.
- Replace the assumed mirror reflectivity and the 25 °C PDE curve (options in §4) by measured values at the operating point; measure the aerogel transmission. Ring hits on the board exist as an option (§4); crosstalk and dark counts are not modelled.
- Xe virial coefficient and UV dispersion: checked (§1.2).
- Cosmic rates (Guan et al. parametrisation, sea level): altitude and geomagnetic cutoff (§5), the trigger counter geometry, the momentum handle and a data-level estimator of the aerogel angle for configurations 2–4, and the tile side faces (§7) are open.
- Compare against the ARC full simulation (`ARC_Simulation_Reconstruction`), with Xe properties tuned.

## Sources
- Xe dispersion:
  - [Börzsönyi et al., Appl. Opt. 47, 4856 (2008)](https://doi.org/10.1364/AO.47.004856); coefficients as on [refractiveindex.info, Börzsönyi](https://refractiveindex.info/?shelf=main&book=Xe&page=Borzsonyi), which corrects a typo of the paper (C₁ = 12.75 × 10⁻³ µm²)
  - [refractiveindex.info, Bideau-Mehu et al. 1981](https://refractiveindex.info/?shelf=main&book=Xe&page=Bideau-Mehu)
  - [Leonard, At. Data Nucl. Data Tables 14, 21 (1974)](https://doi.org/10.1016/S0092-640X%2874%2980028-8), Table II(e): Koch's data (Kgl. Fysiogr. Sällsk. Lund Förh. 19, 173 (1949)) at 273.16 K and 1 atm, wavelengths in air
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
- Aerogel in gas:
  - [LHCb RICH performance, arXiv:1211.6759](https://arxiv.org/pdf/1211.6759)
  - [LHCb RICH aerogel with LHC data, NIM A](https://www.sciencedirect.com/science/article/abs/pii/S0168900210021522)
