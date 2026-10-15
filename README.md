# xe-rich-demonstrator

Fast optical simulation of the pressurised-xenon RICH pre-demonstrator (ARC / FCC-ee): a lab set-up that tests Xe as the ARC gas radiator in place of C₄F₁₀, which has a high global-warming potential. Three model types:
- **fast simulation**: the `xerich` 3D tracer (Snell and polarised Fresnel optics, Rayleigh scattering, cosmic muons at their own β);
- **β = 1 toys**: purely geometric drivers (`cosmic_acc*.py`, `toy_*.py`);
- **full simulation**: ARC's Geant4 framework (not in this repository).

Reference pressure: Xe 2 bar (`constants.P_REF`), the default of the single-pressure drivers, the headline numbers and the reference figures; scans cover 1–3 bar. The design record [`docs/design_options.md`](docs/design_options.md) (below: "design record") holds the numbers, hardware, ARC geometry, configurations and open items.

## Terms
| Term | Meaning |
|---|---|
| configuration 1–5 | the five set-ups (design record §7): **1** gas ring imaged by the f = 203 mm mirror; **2 / 3** aerogel ring imaged by a short-focal-length mirror, in air / in Xe; **4** aerogel ring by proximity focusing (no mirror), in air and in Xe; **5** = 1 plus the aerogel tile 1 mm in front of the SiPM |
| tile | the 5 × 5 × 1 cm aerogel tile, n = 1.03 at 400 nm |
| board | the ARC prototype SiPM board: four Hamamatsu S13361 arrays of 8 × 8 SiPMs (design record §4) |
| pad, pixel | pad: one 3 × 3 mm SiPM of the board; pixel: a generic readout cell of a given pitch |
| FT | Frank–Tamm: emitted photon count or spectrum, no detection efficiency |
| FT×QE, FT×PDE, FT×response | FT weighted with the ARC simulation SiPM QE (`arc_qe`), with the S13361 photon-detection efficiency (`s13361`), or with whichever `--response` selects |
| ring hits | fired board pads (binary) from the ring photons without Rayleigh scatter, leaving out the pad crossed by the muon |
| β = 1, real β | muons traced at β = 1, or at the velocity of a momentum drawn from the cosmic spectrum |
| ≥ 80 % ring, full ring, arc ≥ 50 % | ring definitions: design record §5 (gas ring) and §6.1 (aerogel ring) |
| telescope | the two trigger counters of design record §5 |
| pore-vac, pore-air | tile index with gas in the pores: 1.03 + φ (n_gas − 1) (`media.aerogel_index`) or 1.03 + φ (n_gas − n_air) (`media.aerogel_index_air`), φ = porosity (design record §6.3) |

## Model assumptions
### All models
- Pressure: single-pressure drivers take `--pressure` [bar abs] (default `P_REF`). The regression check against `legacy/` passes 3 bar, the pressure of the legacy scripts.
- Xe refractive index (`media.n_xe`, option `model` / `--xe-index`): `borzsonyi` (default; Börzsönyi et al. 2008, 273.15 K, 1000 mbar) or `bm1981` (Bideau-Mehu et al. 1981, 0 °C, 1 atm), scaled with Lorentz–Lorenz and the virial molar density (B = −130 cm³/mol at 293 K). Model accuracy: design record §1.2.
- Photon yields are FT: no detection efficiency, mirror reflectivity or window transmission unless a script states otherwise.

### β = 1 toys
`cosmic_acc*.py`, `toy_aerogel.py`, `toy_scan.py`, `toy_prox.py --legacy`: β = 1 muons with a cos²θ intensity, I0 = 70 m⁻²s⁻¹sr⁻¹, mirror clear aperture 90 % of the diameter. Not modelled: QE, Rayleigh scattering, vessel walls, SiPM fill factor. Aerogel photons leaving through the tile sides are lost.

### Momentum-aware cosmic muons
`muons.MuonSpectrum`, `cosmic`, `cosmic_spectrum.py`, `cosmic_rates.py`, `toy_prox.py`.
- Joint momentum–zenith spectrum d²I/dp dΩ at sea level, `--spectrum`: `guan` (default; Guan et al., modified Gaisser formula with Chirkin's cos θ*), `reyna` (ζ = p cos θ scaling, fitted above 1 GeV/c), `shukla` (Shukla–Sankrith energy-loss form, their Eqs. (8)–(9)), `shukla-eq10` (their factorised Eq. (10)). Each has its own normalisation; `norm='I0'` rescales to 70 m⁻²s⁻¹sr⁻¹ above 1 GeV/c. 0.2 < p < 1000 GeV/c, azimuth uniform; no altitude correction (`altitude_factor` raises NotImplementedError), no geomagnetic effect.
- Zenith angles are generated below 75° (`constants.MUON_THMAX`; muon samplers and flux normalisation).
- Each muon radiates at its own β: cos θ = 1/(nβ), FT yield ∝ 1 − 1/(n²β²) per wavelength.
- Overburden (optional, information only) and absorber cuts: PDG CSDA muon range (`muon_data.RANGE_TABLES`, 10 MeV–100 GeV kinetic energy; power law through the last two rows above).
- Tile side faces are black: a photon reaching one ends there, also on its way back from the mirror.

### SiPM and detector inputs (`cosmic_rates.py`)
The scripts use a continuous square SiPM of 50.8 mm (2") or 51.2 mm (2 × 2 arrays at 3.2 mm pitch); `cosmic_rates.py --sensor-model board` uses the board pad map (`layout.Board`; geometry in design record §4).

Detector-input options (`cli.detector_inputs`; recorded in every output, see Provenance). The defaults (`cli.BASELINE`, from the named constants) equal `--preset baseline-board`; `--preset published` selects the simplified model: ARC QE, continuous 2" square, reflectivity 1, constant tile index, porosity 0.97, no board reflection, vertical-muon Rayleigh factor in configuration 4, `--nphot 180` (an option of `cosmic_rates.py`, below). Explicit options override a preset. Values, sources and status: design record §4.

| Option | Default | Values and meaning |
|---|---|---|
| `--response` | `s13361` | `s13361`: S13361-3050 PDE through the epoxy window (`cherenkov.pde_s13361`); `arc_qe`: ARC simulation QE |
| `--mirror-r` | `nominal` | `nominal`: curve from the vendor spec (`constants.MIRROR_REFL_NOMINAL`); `1`; or a flat value in (0, 1]. The absorbed weight ends as `mirror absorption` |
| `--sensor-model` | `board` | `board`: pad map; `square`: continuous, fully active. Applies to the 2" sensor only; the larger layouts (`--groups s61`) stay square |
| `--aerogel-index` | `single-pole` | tile dispersion (`media.aerogel_medium`): `constant`, `single-pole` (n² − 1 = a₀λ²/(λ² − λ₀²), λ₀ = 83.22 nm) or `k4geo` (ARC table, n − 1 scaled); equal to the run's tile index (pore shift included) at `--aerogel-lambda-ref` |
| `--aerogel-lambda-ref` | 400 | reference wavelength [nm] of the dispersion |
| `--porosity-vac` | 0.935 | φ of the pore-vac variant, also used for its pore-gas signal (pore-air always uses `constants.AEROGEL_POROSITY_AIRREF` = 0.935) |
| `--pore-t` | 293.15 | temperature [K] of the gas in the pores (`constants.T_PORE`) |
| `--board-r-spec`, `--board-r-diff` | 0.11, 0.10 | specular and diffuse (Lambertian) board reflectance in [0, 1), flat in wavelength (`constants.BOARD_R_SPEC`, `BOARD_R_DIFF`); board only; their sum with the PDE is not checked against 1 |
| `--prox-rayleigh` | `per-photon` | configuration-4 Rayleigh transmission in the tile: `per-photon` along each photon path from its emission depth to the tile exit (`sim.rayleigh_transmission`), averaged over the β = 1 FT×response spectrum of the run's tile index, for photon counts and pad hits alike; `vertical`: the factor of a vertical muon for every muon (`sim.proximity_rayleigh`) |

- Board: containment fractions and arcs use the 51.6 mm footprint, photon counts the active pads.
- Ring hits (`analysis.pad_hits`): P = 1 − exp(−μ_pad) per pad, μ_pad = expected ring photons on the pad; no crosstalk, dark counts or afterpulses. Biased low at small `--nphot`.
- Board reflection (`cosmic.board_reflections`): the light reaching the footprint is split into detected, reflected and absorbed, so ring hits and all other outputs are unchanged. The reflected light is re-traced through the same scene (own random stream, two board reflections, no partial Fresnel reflections) in configuration 1 and the configuration-5 gas-ring runs; on active pads it counts as other light, never as ring hits (`refl_qe`, `pads_refl`). Not modelled: wavelength dependence, reflection beyond the footprint, vessel walls, a window slab (refused), configuration 4. Size of the neglected parts: design record §4.
- Configuration 4 uses one Cherenkov angle (n = 1.03) for containment and pads but the dispersive index for the yield, which underestimates the ≥ 80 % ring rate by up to about 2 % (design record §4).
- Ring definitions are geometric: a ring counts from its light in the ARC QE band whatever `--response`.
- Muons are traced in chunks of at most `CHUNK` muons and `CHUNK_PHOTONS` photons (`cosmic_rates.py`), which bounds the memory of a worker; the random streams depend on the chunk size.

### On-axis hit map (`hitmap_onaxis.py`)
- Layout: mirror R = 406.4 mm, Ø 76.2 mm, clear aperture 90 %, reflectivity 1. The 50.8 × 50.8 mm SiPM plane at z = f on the axis; below it a 1 mm gas gap, the tile, then gas down to the mirror; one gas fills the vessel.
- Media: Xe 2 and 3 bar (20 °C) from `n_xe` (`--xe-index`); C₄F₁₀ 1 bar from the ARC table RINDEX__C4F10_PFRICH, held at its 660 nm value above 660 nm (16.6 % of the FT and 6.8 % of the FT×QE photons); tile n = 1.03, no pore shift. `--aerogel-index`: `constant` (default), `single-pole` or `k4geo` (ARC table RINDEX__Aerogel_PFRICH, n − 1 scaled), equal to 1.03 at `--aerogel-lambda-ref` (default 400 nm).
- Tile Rayleigh scattering (default on): mean free path Λ = λ⁴/C, `--rayleigh-c` C = 0.005 µm⁴/cm (ARC table RAYLEIGH__Aerogel_BelleII; Λ = 1.62 / 5.12 / 25.9 cm at 300 / 400 / 600 nm; 0 = off), polarised dipole redirection, weight kept. `--surface-a` A (default 1; ignored when C = 0): straight-pass transmission T = A exp(−C t/λ⁴), √A per face crossing, the rest lost (`surface loss`). Paths with a scatter are class `scattered`; scatters count towards the interaction cap. Applies to gas photons crossing the tile and to aerogel photons.
- Muon's own hit (`--[no-]muon-cluster`, default on): the muon crosses the silicon, then the protective window (`--window-n` 1.55, `--window-t` 0.1 mm; design record §4). Its Cherenkov light is totally reflected at the window bottom face onto the silicon, in an annulus of radii t·tanθc to 2t·tanθc (origin `window`); the direct hit adds one fired cell (not in the maps). `--window-external` also puts the window in the path of all other photons (Fresnel at its bottom face, gas gap reduced by t).
- Photons: vertical β = 1 muon on the axis, from the sensor plane to the mirror vertex, emitting in the gap, tile and gas. Energies sampled from FT over 1.3–4.7 eV (264–954 nm), where the ARC QE is non-zero; FT×QE weights by QE(λ). Each photon uses n(λ) for its angle and every refraction.
- Optics: Snell, Fresnel and total internal reflection at both tile faces, with the Cherenkov polarisation carried as a complex field through every interface and the mirror (perfect conductor). Rays split into reflected and transmitted branches; paths with more than `--max-reflections` (default 2) partial reflections are dropped (reported as the interaction cap).
- Not modelled: absorption, Rayleigh scattering in the gas, mirror reflectivity below 1, SiPM reflection (perfect absorber), fill factor, QE angle dependence, pixelisation. Photons reaching a tile side face are absorbed.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                             # the xerich package: numpy >= 2, matplotlib
pip install pytest                           # tests
pip install --only-binary :all: pythia8mc   # only for studies/gen_pk.py; prebuilt wheels exist for Python 3.11 and 3.12
```

## Layout
| Path | Content |
|---|---|
| `src/xerich/` | the library (below) |
| `studies/` | one driver per study; each writes into `--outdir` |
| `tests/` | unit tests and the regression check against `legacy/` |
| `legacy/` | the original standalone scripts, for reference and the regression check; they write to the current directory |
| `scripts/investigations/` | one-off scripts behind numbers in the design record |

| Module | Content |
|---|---|
| `constants.py` | physical constants, particle masses, cosmic-muon intensity, hardware dimensions, fixed reference values |
| `media.py` | Xe n(λ, P, T) (`n_xe`); C₄F₁₀, air and aerogel indices from the ARC tables; tile index with gas in the pores (`aerogel_index`: φ = 0.97, shift relative to vacuum; `aerogel_index_air`: φ = 0.935, relative to air) and its dispersion (`aerogel_medium`); `Medium`, `ConstantIndex`, `Rayleigh` (scattering length λ⁴/C, surface factor) |
| `cherenkov.py` | Cherenkov angle, threshold, FT spectra and yields; FT×QE angle, yield and rms (`ftqe_angle_yield`, `theta_xe`, `chromatic_rms`); pore-gas angle shift (`pore_signal`); ARC QE (`qe_arc`, `qe_band`), S13361 PDE (`pde_s13361`), `response`; `PhotonSpectrum`: FT photon-energy sampler |
| `layout.py` | mirror, sensor (`square` or `Board`), tile, SiPM window, gas; `Phase1Layout` (configurations 1, 2, 3, 5 and the larger mirror layouts) and its `scene()` for the tracer |
| `tracer.py` | vectorised ray geometry; polarised Fresnel (`fresnel_amplitudes`, `interface_split`, ...) and Rayleigh redirection (`rayleigh_scatter`); `trace`: weight-splitting tracer through a `Scene` of slabs, a spherical mirror and a sensor plane (`record=True` keeps the branch vertices; a mirror reflectivity adds the end reason `mirror absorption`, in `REASONS_ALL`, and `wr`, the product of the reflectivities along the branch) |
| `muons.py`, `muon_data.py` | muon sampling and flux through a horizontal square, momentum–zenith spectra (`intensity`, `MuonSpectrum`), zenith exponent, CSDA range and its inverse; PDG range tables (concrete, polystyrene, aluminium, iron, lead), BESS vertical spectra (Tsukuba, Lynn Lake) |
| `cosmic.py` | gas-ring footprint model (`gas_ring_fractions`; `slope_term=True`: the legacy slope term); batch tracing of muons at their own β (`emit_batch`, `trace_batch`, `fractions`); board reflection (`board_reflections`) |
| `analysis.py` | ring containment on a square sensor, Monte Carlo rates, pixel resolution, binning, scattered fraction vs λ (`scattered_fraction`), binary pad hits (`pad_hits`) |
| `sim.py` | setup-level toys (spherical aberration, ring rates, `MirrorToy`, `gas_yield`, proximity focusing); photons of one muon in a `Phase1Layout` (`emit_photons`, `muon_photons`, `onaxis_muon`); the muon's own hit (`window_cluster`, `cluster_stats`); configuration 4 (`proximity_fractions`, `proximity_photons`, `proximity_rayleigh`, `rayleigh_transmission`) |
| `cli.py` | shared options (`--outdir`, `--thc-mrad`, `--slope-term`, `--porosity`, `--pressure`, `--gas-yield`); detector inputs and presets (`detector_inputs`, `inputs`, `BASELINE`, `PRESETS`, `NPHOT_DEFAULT`, `PRESET_NPHOT`) |
| `plotting.py`, `sketch.py`, `provenance.py` | plot style and helpers; setup sketch; provenance record of the outputs |

## Studies
Run from any directory, e.g. `python studies/toy_aerogel.py --outdir out/`. Common options:
- `--outdir`: default `$XERICH_OUT` if set, else `xerich-output/` next to the repository checkout.
- `--xe-index {borzsonyi,bm1981}` (default `borzsonyi`) in `plot_spectra`, `plot_lambda`, `plot_fcc`, `raytrace`, `cosmic_acc`, `cosmic_acc2`, `toy_aerogel`, `toy_plots`, `toy_scan`, `toy_prox`, `hitmap_onaxis`; the other drivers use `borzsonyi`. The regression check passes `bm1981`, the model of the legacy scripts.
- `--pressure P`: Xe pressure [bar abs], default `P_REF`.
- `--nproc N`: worker processes; `--seed S`: master seed, default 20261010 unless stated.

| Driver | What it does; specific options (defaults) | Output |
|---|---|---|
| `plot_spectra.py` | Xe dispersion, dN/dxdE for π/K, yield vs p, chromatic range | stdout, `xe_cherenkov_spectra.png` |
| `gen_pk.py <91\|160\|240\|365> <Nevt>` | Pythia π/K/p momenta at FCC-ee √s | stdout, `mom_<key>.npz` |
| `plot_lambda.py`, `plot_fcc.py` | dN/dxdλ at fixed p and averaged over FCC-ee tracks; momentum spectra vs the Xe thresholds. Read `mom_*.npz` from `--momdir` (default `--outdir`) | stdout, `xe_cherenkov_lambda.png`, `xe_fccee_pik.png` |
| `raytrace.py` | spherical-aberration smearing vs track offset and tilt. `--thc-mrad A` fixes the angle; default: FT×QE mean over the ARC QE band (63.8 mrad at 3 bar). The regression check passes the legacy 63.6 and 3 bar | stdout |
| `cosmic_acc.py`, `cosmic_acc2.py` | β = 1 gas-ring rates vs mirror and sensor size (`cosmic_acc.py` at `--pressure`, `--thc-mrad` as `raytrace.py`; `cosmic_acc2.py` at 2 and 3 bar). `--slope-term`: the legacy footprint, which counts the track displacement twice (the regression check passes it and 3 bar) | stdout |
| `sketch.py`, `sketch_phase1.py` | to-scale sketches of configurations 5 and 1 | `arc_xe_demonstrator_*.png` |
| `toy_aerogel.py` | 3D geometric toy: gas and aerogel photons, 3" and 6" mirrors. `--porosity X` (also `toy_scan.py`, `toy_plots.py`): φ in 1.03 + φ (n_gas − 1), default 0.97. `--gas-yield N` (also `toy_scan.py`): emitted Xe photons over 193 mm in 320–450 nm, default `sim.gas_yield(--pressure)` (`constants.GAS_YIELD_REF` at `P_REF` scaled by the FT integral of the index model); the regression check passes the legacy 33.5 at 3 bar | stdout, `toy_76.npz`, `toy_152.npz` |
| `toy_plots.py` | acceptance vs muon angle and hit maps from `--input` (default `<outdir>/toy_76.npz`); pressure from its provenance record, `--pressure` only for inputs without one | stdout, `toy_aerogel_acceptance.png` |
| `toy_scan.py` | SiPM and mirror size scan, aerogel and gas, muons uniform over the tile | stdout |
| `toy_prox.py` | configuration-4 ring rates and radii vs gap (30 / 50 / 70 mm), air and Xe: muons from the spectrum, per-muon aerogel angle, plain tile with pore-vac (φ = 0.97) and pore-air (φ = 0.935); radii (median and 16–84 %; outer radius gap·tan θ_gas + T·tan θ_tile) from the same angles; β = 1 rates of the same muons alongside (cos²θ zenith law, I0 = 70 above 1 GeV/c, no momentum weighting). `--spectrum` (`guan`), `--nmu` (40000), `--seed` (7), `--nproc` (8); `--legacy`: the β = 1 toy of design record §6.2 | stdout |
| `cosmic_spectrum.py` | spectrum parametrisations vs their papers and BESS; zenith exponent n(p); fractions of the telescope muons above the Xe and aerogel thresholds vs pressure (all muons of the spectrum, p > 0.2 GeV/c); vertical intensities above 0.2, 0.5 and 1 GeV/c; overburden sensitivity; aerogel angle vs p; absorber momentum cuts; the mirror and SiPM size table of design record §5 (slope term on and off, β = 1 and spectrum); fast footprint model vs the 3D tracer. `--nmu` (2e6, threshold fractions), `--nseed` (5), `--nproc` (20) | stdout, `cosmic_spectrum.json`, `cosmic_spectrum_{vertical,zenith,thresholds,angle_vs_p}.png` |
| `cosmic_rates.py` | configurations 1–5 and the larger mirror layouts (3D tracer; configuration 4 geometric); see below | `cosmic_rates_<group>.npz` (large), `cosmic_rates_meta.json` |
| `cosmic_rates_plots.py --input DIR` | tables and figures from `cosmic_rates.py`; see below. `--short-f F`, `--nproc` (16) | stdout, `cosmic_rates_summary.json`, `cosmic_rates_table.json` (`{provenance, rows}`), `cosmic_rates_tables.md`, `cosmic_rates_board.md` (board runs), `cosmic_rates_*.png` |
| `hitmap_onaxis.py` | mean SiPM hit density of configuration 5 for a vertical muon on the axis, Xe 2 / 3 bar and C₄F₁₀, FT×QE and FT, by origin (gas, aerogel, window) and path class; options and assumptions in the on-axis section above. `--nphot` (4e6 per origin and gas), `--chunk` (50000 photons), `--nproc` (8); `--no-unpolarised` skips the unpolarised-Fresnel rerun (which doubles the runtime). The npz also holds per-chunk class sums and ring moments (`<gas>_chunk_cls/_ring/_spot`, for Monte Carlo uncertainties) and the weights vs λ by origin, end reason and scattering (`<gas>_<weighting>_lam_end`) | stdout, `hitmap_onaxis_<gas>_<ftqe\|ft>.png`, `hitmap_onaxis_radial_gases.png`, `hitmap_onaxis_halo.png` (Rayleigh halo vs ring), `hitmap_onaxis_scatter_lambda.png` (scattered fraction vs λ), `hitmap_onaxis_summary.csv/.md`, `hitmap_onaxis.npz` |
| `muon_window.py` | the muon's own SiPM hit vs window thickness 0.1–0.5 mm and index 1.45 / 1.5 / 1.55: photons on the silicon (FT×QE, FT, plus the direct hit) and spot radii. `--nphot` (2e5 per configuration), `--nproc` (8) | stdout, `muon_window_cluster.png/.csv` |
| `sketch_effects.py` | ten traced sketches of configuration 5 at `P_REF` unless stated: 1 overview, 2 tile refraction, 3 aerogel reflection, 4 gas ghosts, 5 chromatic dispersion, 6 spherical aberration, 7 total internal reflection, 8 polarisation (1–8 without Rayleigh scattering), 9 Rayleigh halo, 10 the muon's own hit. `--nphot` (2e6), `--chunk` (50000), `--nproc` (24), `--only K ...` (sketch numbers, default all) | `sketch_effects_<n>_<name>.png`, `sketch_effects_numbers.txt` |
| `short_f_mirrors.py` | configurations 2 and 3: aerogel ring imaged by an f/1 mirror (Edmund #43-543 or #43-553), tile 1 mm in front of the 2" SiPM at z = f, air or Xe 1–3 bar, β = 1, reflectivity 1, FT×QE and FT. `--parts` (comma-separated, default all): `onaxis` (vertical muon on the axis: rings, path classes, losses, window cluster, pixel maps at 3.2 and 0.8 mm), `terms` (ring-width terms one at a time; deterministic meridional trace), `zscan` (sensor position), `angles` (containment vs zenith, azimuth, impact offset), `tol` (focal length, dr/dn, clear aperture); on-axis figures, `zscan` and `tol` at `P_REF` and in air. `--nphot` (2e6, on-axis), `--nseeds` (4), `--nphot-terms` (1e6), `--nphot-zscan` (2e5, also `tol`), `--nphot-angle` (10000), `--chunk` (50000), `--nproc` (8); `--replot` redraws from the npz files | `short_f_<part>.npz`, `short_f_angles.npz` (with its axes; read by `cosmic_rates_plots.py --short-f`), `short_f_summary.json`, `short_f_*.png`, `short_f_*.csv` |
| `proximity_config4.py` | configuration 4 with the 3D tracer: tile, gap 30 / 50 / 70 mm, SiPM with its window, air or Xe 1–3 bar, pore shift off, pore-vac or pore-air, β = 1. `--parts` (default `onaxis,terms,angles`): `onaxis`, `points` (off-centre impact points in air and at `P_REF`, run with `onaxis`; alone: `--replot --parts onaxis` redraws), `terms` (width terms, r(depth), dr/dθ, dr/dn), `angles` (containment vs zenith, azimuth, offset); maps and profiles at `P_REF`. The scene is the mirror image (z → −z) of the set-up: the tracer does not handle a window below a downward-facing sensor. `--nphot` (1e6, on-axis), `--nseeds` (4), `--nphot-points` (2e5), `--nphot-terms` (1e6), `--nphot-angle` (10000), `--chunk` (50000), `--nproc` (8); `--replot` as above | `prox_<part>.npz`, `prox_angles.npz` (with its axes), `prox_summary.json`, `prox_*.png`, `prox_*.csv` |

To regenerate the momentum files (read from `--outdir` unless `--momdir` is given): `for k in 91 160 240 365; do python studies/gen_pk.py $k 50000 --outdir out/; done`

### `cosmic_rates.py`
Options: `--nphot N --nmu-scale X --nseed K --seed S --nproc N --groups G --spectrum M`, plus the detector inputs above. Defaults `--nphot 7200` (sampled tile photons per muon, gas 0.4×, `cli.NPHOT_DEFAULT`; 180 with `--preset published`), `--nmu-scale 1` (scale of the muon numbers per run), `--nseed 5` independent seeds per run, `--nproc 20`, `--spectrum guan`.
- Runs: configurations 1, 2, 3, 5 and the larger mirror layouts with the 3D tracer (`cosmic.trace_batch`; Rayleigh scattering off except in the `_ray` runs), configuration 4 with `sim.proximity_fractions`; Xe 1–3 bar, both short-f mirrors (Edmund #43-543 and #43-553, clear aperture 90 % assumed), gaps 30 / 50 / 70 mm, tile plain, pore-vac and pore-air.
- Muons: zenith from cos³ through the top counter, momentum from the spectrum at that zenith; each sample is traced at β = 1 and at its real β; per-muon weights give the spectrum and the other parametrisations.
- Counters: top 5 × 5 cm at the SiPM plane (tile top face in configuration 4); bottom 8 × 8 cm at the mirror vertex plane (5 × 5 cm at the SiPM plane in 4). The "plane" normalisation counts every muon through the 2" SiPM square (half side 25.4 mm), or through the tile in 4.
- `--groups` (default all): `c1` (configuration 1, Xe 1–3 bar), `c5` (configuration 5: aerogel light, gas-ring geometry `_geo` and with Rayleigh scattering `_ray`), `c23` (configurations 2 and 3, both mirrors), `s61` (larger mirror layouts, design record §6.1), `c4` (configuration 4).

npz keys `<run>/<field>` (`qe` in a key = weighted with the run's response):

| Key | Meaning |
|---|---|
| `c`, `phi`, `x0`, `y0`, `p`, `seed` | per muon: cos zenith, azimuth, entry point at the top-counter plane, momentum, seed index (0 to `--nseed` − 1) |
| `top`, `bot` | counter flags |
| `<b1\|br>_<gas\|tile>_` + field | per β tag (`b1` β = 1, `br` real β) and origin: |
| `fm`, `fs`, `fd` | FT fraction of the emitted photons on the mirror; of the ring on the sensor; of the emitted photons detected |
| `arc`, `arc_half` | fraction of the 36 azimuth bins (those holding emitted photons) with a photon on the sensor; with at least half of their emitted photons on it |
| `emit_qe`, `det_qe`, `det_qe_clean` | emitted photons; photons on the sensor (board: active pads); the same without Rayleigh-scattered photons |
| board: `det_qe_plane`, `det_qe_ring`, `pads`, `pads_all`, `pads_pge` | unscattered photons on the footprint; unscattered photons on active pads without the muon's pad; mean ring hits; the same with scattered photons; P(≥ k ring hits), k = 0–40 |
| board reflectance: `refl_qe`, `pads_refl` | reflected photons on active pads, without the muon's pad; mean fired pads from ring photons and reflected light together |
| configuration 4: `<b1\|br>_tile_fd`, `_arc`, `_th`, `_tr` | fraction of the aerogel photons on the sensor; fraction of azimuths hit; aerogel angle in the tile [rad]; mean Rayleigh transmission of the photons on the sensor (`--prox-rayleigh per-photon`) |
| configuration 4, board: `_act`, `_act_ring`, `_pads`, `_pads_pge` | fraction of the photons on the sensor that land on active pads; the same without the muon's pad; ring hits; P(≥ k) (transmission-weighted with `per-photon`) |

### `cosmic_rates_plots.py`
Reads `cosmic_rates_<group>.npz` and `cosmic_rates_meta.json` (response, aerogel and pore-gas inputs) from `--input`; figures need every run group. Contents:
- rates per configuration and ring definition, per hour and per day, β = 1 and real β, seed spread, parametrisation band, both normalisations; containment maps; the larger-layout table;
- FT×response photons per ring (Monte Carlo estimate per muon; board: active pads, the muon's pad included) and rates vs a minimum photon count;
- precision per hour on the mean ring angle (pixel term, Xe dispersion, velocity spread);
- Xe − air aerogel-angle shift on the true emission angles, four estimators: plain mean; mean in a −10 to +20 mrad window around the plain-tile β = 1 angle, or around each run's own β = 1 angle; 95 % quantile;
- momentum handles: absorber with the bottom counter at its plane or moved behind it (geometry only, no multiple scattering), the counter side keeping 90 % of the range-cut rings, the Xe gas-ring tag;
- board runs (`cosmic_rates_board.md`; configurations 1, 4 and the gas ring of 5): per ≥ 80 % ring the photons on the footprint, on active pads and the ring hits (the latter two without the muon's pad), rates with ≥ 3 and ≥ 10 hits; with a board reflectance the reflected photons and the other hits they add;
- `rayleigh_c4` in `cosmic_rates_summary.json`: configuration-4 Rayleigh mode and mean `tile_tr` of the ≥ 80 % rings per gap.

`--short-f`: `short_f_angles.npz` of `short_f_mirrors.py` (with `short_f_summary.json` in the same directory) adds the impact-offset term of configurations 2 and 3.

## Tests
- `pytest`: unit tests (optics, Xe and aerogel index models, Fresnel and polarisation, Rayleigh scattering, tracer closure and weight conservation, the muon's window light), a quick legacy-vs-driver comparison and the golden outputs. The check of the embedded ARC tables parses `materials_arc_o1_v01.xml` under `$XERICH_K4GEO` (a key4hep/k4geo checkout or an installed `share/k4geo`); skipped when unset.
- `XERICH_BASELINE=A XERICH_NEW=B pytest tests/test_regression.py`: compares two existing output directories A and B with the regression comparison (skipped unless both variables are set).
- `XERICH_FULL_REGRESSION=1 pytest`: reruns every legacy script and driver with the legacy settings (3 bar, `--gas-yield 33.5`, the legacy angle) and requires identical printed output and npz arrays (provenance key ignored); `python tests/regression.py --help` gives the steps by hand.
- Golden outputs (`tests/test_golden.py`, about 4 min with 4 workers): small fixed-seed runs of the defaults and of `--preset published` (`cosmic_rates.py --groups c1 c5 c4` and `cosmic_rates_plots.py`; configuration 1 with more muons per task than one chunk; stdout of `raytrace.py`, `cosmic_acc.py`, `toy_prox.py`) must reproduce `tests/golden_reference_<cpu>.json` bit for bit (sha256 of every npz array per run and field; summary, rates and board tables and stdout verbatim); a failure names what moved. `<cpu>` (`avx512`, `avx2`, ...) is the numpy CPU-feature type, since float results differ in the last bit between types; without a reference for the current type the digests are skipped and every number is compared at relative tolerance 1e-10. `python tests/golden.py regen` rewrites the reference of the current type; regenerate only with a change of the input card (design record §4), never to absorb a code change.

## Provenance
Every npz and JSON (key `provenance`), markdown table (first line, HTML comment), text log (first line) and PNG (metadata `Description`) a driver writes carries the git commit and a dirty-tree flag (`unknown` and null outside a checkout), the python and numpy versions, the command line, every argument, the resolved detector inputs and the reference constants (`xerich.provenance`). The csv outputs (`hitmap_onaxis_summary.csv`, `prox_onaxis.csv`, `short_f_onaxis.csv`, `muon_window_cluster.csv`) carry none.
