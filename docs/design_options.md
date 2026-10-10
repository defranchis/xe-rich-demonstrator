# Xe RICH demonstrator: summary of the options explored

This summarises the design exploration for a lab demonstrator of pressurised xenon as the ARC gas radiator. The study drivers in `studies/` produce every number quoted here, and the plots are on the [design-figures page](https://mdefranc.web.cern.ch/ARC/xe_demonstrator/design_figures/). Assumptions and simplifications are listed in the [README](../README.md).

**Status in one paragraph.** Phase 1 is gas-only ring imaging with cosmic muons. It uses a stock 3" spherical mirror with f ≈ 203 mm, a 2" × 2" SiPM plane at its focus and ~20 cm of Xe at 2–3 bar. The aerogel cannot be imaged in this geometry. It moves to a separate proximity-focusing configuration (phase 2), which also measures how Xe in the pores shifts the aerogel index. A full-size, ARC-like cell needs a custom mirror and is left for later.

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
| **Protected / UV-enhanced Al (Al + SiO₂)** | Fine above ~250 nm, cheap, broadband, insensitive to angle of incidence. **The choice for phase 1.** |
| Broadband dielectric | Sharp band edges act as a wavelength filter. But the band shifts with angle of incidence: the Edmund "320–450 nm" spec holds at 45°, and at 0° it becomes 340–488 nm. Reflectance is also polarisation-dependent. Usable, but not needed. |
| Silver, gold | Unsuitable: reflectivity collapses below ~350–400 nm. |
| Bare Al | Unsuitable: the native oxide degrades reflectivity below ~200 nm and over time. |

### 2.2 Focal length and geometry
- The SiPM sits at the focal plane. All photons with the same direction converge on one point, so a straight track gives a sharp ring of radius f·tanθc.
- With f ≈ 20 cm, the ring radius is 10.5 mm at 2 bar and 12.9 mm at 3 bar.
- The ring centre on the sensor sits at f·tanθ_track, independent of where the track crosses. It moves by ~3.5 mm per degree of track tilt.
- A track illuminates a patch of the mirror of radius L·tanθc, where L is the radiator length. That is ≈ 13 mm at 3 bar with L = 20 cm.

### 2.3 Spherical aberration (`raytrace.py`, f = 203 mm, 3 bar)
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
| **Edmund #43-584** | **76.2 mm (3")** | **203.2 mm** (f/2.7) | Enhanced Al, Ravg > 89 % at 250–450 nm | **€316** | **Recommended for phase 1.** Largest stock mirror at f ≈ 20 cm. |
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
- Implication for the demonstrator: f ≈ 20 cm with ~18–20 cm of gas reproduces the central barrel cell.

---

## 4. Photodetector
- **Confirmed:** a 2" × 2" active area (50.8 × 50.8 mm).
- **Still to be confirmed:** the pixel size and part number. It may be a prototype-specific device.
- The published ARC prototype board has the same footprint: four 8×8 matrices of 3 × 3 mm² SiPMs (dRICH-style, Hamamatsu S13360-3050VS family), 256 channels, ALCOR/dRICH front-end, operating target −60 to −80 °C.

| Pixel pitch | Pixel term per photon at f = 203 mm |
|---|---|
| 3.2 mm | 4.5 mrad (dominant) |
| 0.8 mm | 1.1 mrad |
| 0.5 mm | 0.7 mrad |

---

## 5. Cosmic-ray acceptance, phase 1 (gas ring)
Assumptions:
- Vertical intensity I0 ≈ 70 m⁻²s⁻¹sr⁻¹ with a cos²θ distribution.
- About 1000 muons per hour cross the 5 × 5 cm plane.
- Most cosmic muons are above the μ threshold (~1.6 GeV at 3 bar), so their rings are nearly saturated.

| Mirror | SiPM | ≥ 80 % rings/h (3 bar) | Full rings/h (3 bar) | ≥ 80 % rings/h (2 bar) |
|---|---|---|---|---|
| 2" | 5.1 cm | 4.6 | 1.8 | 7.3 |
| **3"** | **5.1 cm** | **13** | **6** | **18** |
| 3" | 4 cm | 5 | 2.5 | 7 |
| 3" | 8 cm | 47 | 23 | 58 |

- **The SiPM limits the angular acceptance.** Full rings need |θ_μ| ≲ 3–4° for a 5 cm sensor.
- **The mirror limits the position acceptance:** about ±2 cm for a 3" mirror.
- Suggested trigger: a ~5 × 5 cm scintillator above the SiPM and a ~8 × 8 cm one below the mirror.
- A larger mirror increases the rate roughly with its area until the sensor angle cut takes over. That earlier scan used an 8 cm sensor: 2" 4/14 h⁻¹, 3" 24/41, 4" 54/87, 6" 160/200, 20 cm 315/390, ARC 26 cm 750/870 (full / ≥ 80 % rings). Much of the extra rate from large f/1 mirrors is only good for photon counting.

---

## 6. Aerogel
### 6.1 In the mirror geometry: not imaged (`toy_aerogel.py`, `toy_scan.py`)
- In the gas, θ_aero ≈ 249 mrad (n ≈ 1.032 with Xe in the pores).
- After ~19 cm of gas the photons are ~48 mm from the track, which **misses the 3" mirror**. The imaged ring would have **r ≈ 52 mm**, which **misses the 5 cm SiPM**.
- With a 3" mirror and a 5.1 cm SiPM, only muons at 6–22° put aerogel light on the sensor:
  - ≤ ~3.5 emitted photons per muon (~1 detected);
  - on an arc covering ≤ 20 % of the ring (median 12 %).
- Gas rings need muons within ~8° of vertical, so gas and aerogel light never appear in the same event.

Scan with muons uniform over the 5 × 5 cm tile (~985 /h):

| SiPM | Mirror Ø | Aerogel arcs ≥ 50 % /h | Full aerogel rings /h | Gas rings (≥ 80 %) /h |
|---|---|---|---|---|
| 5.1 cm | 76 mm | 0 | 0 | 9 |
| 8 cm | 152 mm | 0 | 0 | 53 |
| 10.2 cm | 152 mm | 22 | 0 | 90 |
| 12.8 cm | 152 mm | 62 | 5 | 134 |
| 12.8 cm | 200 mm | 104 | 12 | 158 |

- Imaging the aerogel needs **both** a SiPM of ≥ 10 cm (≥ 16 matrices, ≥ 1024 channels at 3 mm) **and** a mirror of ≥ 15 cm at f ≈ 20 cm (custom).
- Even then the rings are mostly partial, and a full ring carries only ~25 emitted photons in 320–450 nm (~6 detected).
- A statistical measurement from the arcs would need external tracking of the muon direction at the mrad level.

### 6.2 Proximity focusing: recommended for phase 2 (`toy_prox.py`)
The layout is aerogel tile → Xe gap → SiPM, with no mirror.

| Xe gap | Ring radius | ≥ 80 % of the ring on the 5.1 cm SiPM /h | ≥ 50 % arc /h |
|---|---|---|---|
| 30 mm | 8–10 mm | ~190 | ~380 |
| 50 mm | 13–15 mm | ~70 | ~210 |
| 70 mm | 18–20 mm | ~20 | ~100 |

The aerogel threshold for muons is ~0.43 GeV, so nearly every triggered muon radiates.

### 6.3 Aerogel index with Xe in the pores
- Expected shift: Δn ≈ porosity × (n_Xe − n_air), with a porosity of ≈ 0.935 for n = 1.03 (ρ ≈ 0.14 g/cm³). At 400 nm:
  - 2 bar: **+0.0010**, Δθc ≈ +3.9 mrad;
  - 3 bar: **+0.0016**, Δθc ≈ +6.3 mrad.
  
  These are relative to the in-air specification; relative to vacuum the shifts are +0.0013 and +0.0019.
- This is a **lower bound**: Xe adsorbed on the large internal surface (500–1000 m²/g) could add to it. Pore filling may also cause a transient of minutes to hours.
- No published measurement in pressurised Xe was found.
- **Precedent:** the LHCb RICH1 aerogel in contact with C₄F₁₀ had ~1.8× worse Cherenkov-angle resolution than simulated, attributed at least partly to absorbed C₄F₁₀. The aerogel was isolated from the gas for 2012 running (arXiv:1211.6759). This is relevant to ARC, where the aerogel sits in the gas.
- Proposed measurement: the aerogel θc in vacuum, air, and Xe at 1/2/3 bar, plus its time dependence after each pressure change.
- In the gas-imaging configuration, 1 cm of aerogel in front of the SiPM Rayleigh-scatters 28 % of the gas photons at 350 nm and 18 % at 400 nm, 21 % of the FT×QE photons over the SiPM band, for the clarity of the ARC simulation (C = 0.005 µm⁴/cm; `doc_values.py`, `hitmap_onaxis.py`). That depends on the tile clarity, and it is one reason phase 1 runs without aerogel.

---

## 7. Recommendations
1. **Phase 1 (now):**
   - Mirror: Edmund #43-584 (or Thorlabs CM750-200-F01). Buy a spare.
   - SiPM: the 2" × 2" plane at the focal plane.
   - Radiator: ~20 cm of Xe at 2 and 3 bar, no aerogel.
   - Trigger: the matched scintillator telescope from §5.
   - Expect ~300–450 usable rings per day.
2. **Phase 2:** an aerogel-in-Xe proximity-focusing run with a 3–5 cm gap, using the same SiPM.
3. **Later:** an ARC-like cell with an ~26 cm hexagonal mirror (R ≈ 37 cm, custom or from ARC), an 8 × 8 cm SiPM tile and beam tests.
   - Only this configuration images gas and aerogel together, and even there the aerogel ring is partial for most tracks.
   - Alternative if the goal is purely Xe characterisation with maximum photon statistics: the 6" f = 610 mm mirror (#43-970) with a ~65 cm vessel. That needs ~3× the Xe and a larger sensor.

## 8. Open items
- SiPM pixel size and part number. This drives the resolution and acceptance numbers.
- Phase-2 mechanical design: gap, tile holder, trigger.
- Fold in the measured mirror reflectivity, SiPM PDE and aerogel transmission. Add photon statistics to the toys.
- Xe virial coefficient and UV dispersion: checked (§1.2).
- Compare against the ARC full simulation (`ARC_Simulation_Reconstruction`), with Xe properties tuned.

## Sources
- Xe dispersion:
  - [Börzsönyi et al., Appl. Opt. 47, 4856 (2008)](https://doi.org/10.1364/AO.47.004856); coefficients as on [refractiveindex.info, Börzsönyi](https://refractiveindex.info/?shelf=main&book=Xe&page=Borzsonyi), which corrects a typo of the paper (C₁ = 12.75 × 10⁻³ µm²)
  - [refractiveindex.info, Bideau-Mehu et al. 1981](https://refractiveindex.info/?shelf=main&book=Xe&page=Bideau-Mehu)
  - [Leonard, At. Data Nucl. Data Tables 14, 21 (1974)](https://doi.org/10.1016/S0092-640X%2874%2980028-8), Table II(e): Koch's data (Kgl. Fysiogr. Sällsk. Lund Förh. 19, 173 (1949)) at 273.16 K and 1 atm, wavelengths in air
- Xe density: [NIST Chemistry WebBook, fluid properties of xenon](https://webbook.nist.gov/cgi/fluid.cgi?ID=C7440633&Action=Page)
- Mirrors:
  - [Edmund #43-584](https://www.edmundoptics.com/p/3quot-dia-8quot-fl-uv-enhanced-alum-spherical-mirror/5147/)
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
- SiPM: [ePIC dRICH photodetector unit, NIM A (2024)](https://iris.unito.it/retrieve/handle/2318/2012671/1373309/1-s2.0-S0168900224005953-main.pdf)
- Aerogel in gas:
  - [LHCb RICH performance, arXiv:1211.6759](https://arxiv.org/pdf/1211.6759)
  - [LHCb RICH aerogel with LHC data, NIM A](https://www.sciencedirect.com/science/article/abs/pii/S0168900210021522)
