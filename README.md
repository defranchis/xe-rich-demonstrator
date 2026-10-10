# xe-rich-demonstrator

Optics toys for a pressurised-xenon RICH demonstrator (ARC / FCC-ee). Xe at 2–3 bar is studied as the gas radiator in place of the C₄F₁₀ baseline, which has a high global-warming potential. [`docs/design_options.md`](docs/design_options.md) summarises every option explored, including commercial mirrors, SiPM, aerogel and cosmic acceptance.

## Model assumptions and simplifications
- Xe refractive index: Bideau-Mehu et al. (1981) Sellmeier formula (0 °C, 1 atm), scaled to pressure and temperature with Lorentz–Lorenz and a second virial coefficient B(T) ≈ −130 cm³/mol at 293 K. That B is approximate and not yet checked against a reference.
- Photon yields are emitted Frank–Tamm yields: no photodetector QE, mirror reflectivity or window transmission is applied unless a script states otherwise.
- The cosmic toys are pure geometry: β = 1 muons with a cos²θ intensity, I0 = 70 m⁻²s⁻¹sr⁻¹, and a mirror clear aperture of 90 % of the diameter. QE, Rayleigh scattering, vessel walls and the SiPM fill factor are not modelled. Aerogel photons leaving through the tile sides are counted as lost.
- SiPM: 2" × 2" active area. The scripts use a 50.8–51.2 mm sensor and a 3.2 mm pixel pitch as placeholders until the part is confirmed.

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
| `studies/` | one driver per study; each writes its files into `--outdir` |
| `tests/` | unit tests and the regression check against `legacy/` |
| `legacy/` | the original standalone scripts, kept for reference and for the regression check; they write to the current directory |

| Module | Content |
|---|---|
| `constants.py` | physical constants, particle masses, cosmic-muon intensity, hardware dimensions, and the fixed values the studies use in place of computed ones |
| `media.py` | Xe n(λ, P, T); aerogel index with gas-filled pores |
| `cherenkov.py` | Cherenkov angle and threshold, Frank–Tamm spectra and integrated yields |
| `layout.py` | dataclasses for the mirror, sensor, aerogel tile and gas radiator |
| `tracer.py` | vectorised 3D ray geometry: Cherenkov cones, Snell refraction with a total-internal-reflection flag, reflection on a sphere, propagation to planes |
| `muons.py` | cosmic-muon sampling and the muon flux through a horizontal square |
| `analysis.py` | ring containment on a square sensor, Monte Carlo rates, pixel resolution, binning |
| `sim.py` | setup-level toys: spherical aberration, cosmic ring rates, the 3D mirror toy with aerogel, proximity focusing |
| `plotting.py`, `sketch.py`, `cli.py` | shared plot style and helpers, the setup sketch, the `--outdir` option |

## Studies
Run from any directory, e.g. `python studies/toy_aerogel.py --outdir out/`. The default output directory is `$XERICH_OUT` if set, else `xerich-output/` next to the repository checkout.

| Driver | What it does | Output |
|---|---|---|
| `plot_spectra.py` | dispersion, dN/dxdE for π/K, yield vs p, chromatic range | stdout, `xe_cherenkov_spectra.png` |
| `gen_pk.py <91\|160\|240\|365> <Nevt>` | Pythia π/K/p momenta at FCC-ee √s | stdout, `mom_<key>.npz` |
| `plot_lambda.py` | dN/dxdλ at fixed p and averaged over FCC-ee tracks (reads `mom_*.npz`) | stdout, `xe_cherenkov_lambda.png` |
| `plot_fcc.py` | momentum spectra vs Xe thresholds (reads `mom_*.npz`) | stdout, `xe_fccee_pik.png` |
| `raytrace.py` | spherical-aberration smearing vs track offset/tilt | stdout |
| `cosmic_acc.py`, `cosmic_acc2.py` | cosmic gas-ring rates vs mirror and sensor size | stdout |
| `sketch.py`, `sketch_phase1.py` | to-scale setup sketches | `arc_xe_demonstrator_*.png` |
| `toy_aerogel.py` | 3D geometric toy: gas and aerogel photons, mirror geometry | stdout, `toy_76.npz`, `toy_152.npz` |
| `toy_plots.py` | acceptance vs muon angle and hit maps (reads `toy_76.npz`) | stdout, `toy_aerogel_acceptance.png` |
| `toy_scan.py` | scan of SiPM and mirror size for aerogel and gas | stdout |
| `toy_prox.py` | proximity-focusing aerogel ring rates | stdout |

`toy_plots.py`, `plot_lambda.py` and `plot_fcc.py` read their inputs from `--outdir` unless `--input` (toy) or `--momdir` (momenta) is given. To regenerate the momentum files: `for k in 91 160 240 365; do python studies/gen_pk.py $k 50000 --outdir out/; done`

## Tests
`pytest` runs the unit tests and a quick legacy-vs-driver comparison. `XERICH_FULL_REGRESSION=1 pytest` reruns every legacy script and every driver and requires identical printed output and npz arrays; `python tests/regression.py --help` runs the same steps by hand.
