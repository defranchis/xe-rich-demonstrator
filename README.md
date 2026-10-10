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
pip install numpy scipy matplotlib
pip install --only-binary :all: pythia8mc   # only for gen_pk.py; prebuilt wheels exist for Python 3.12
```

## Scripts (run from `scripts/`)
| Script | What it does | Output |
|---|---|---|
| `xe_optics.py` | Xe n(λ, P, T), Frank–Tamm yield (imported by the others) | – |
| `plot_spectra.py` | dispersion, dN/dxdE for π/K, yield vs p | `xe_cherenkov_spectra.png` |
| `plot_lambda.py` | dN/dxdλ at fixed p and averaged over FCC-ee tracks | `xe_cherenkov_lambda.png` (needs `mom_*.npz`) |
| `gen_pk.py <91\|160\|240\|365> <Nevt>` | Pythia π/K/p momenta at FCC-ee √s | `mom_<key>.npz` |
| `plot_fcc.py` | momentum spectra vs Xe thresholds | `xe_fccee_pik.png` |
| `raytrace.py` | spherical-aberration smearing vs track offset/tilt | stdout |
| `cosmic_acc.py`, `cosmic_acc2.py` | cosmic gas-ring rates vs mirror and sensor size | stdout |
| `sketch.py`, `sketch_phase1.py` | to-scale setup sketches | `arc_xe_demonstrator_*.png` |
| `toy_aerogel.py` | 3D geometric toy: gas and aerogel photons, mirror geometry | stdout, `toy_76.npz`, `toy_152.npz` |
| `toy_plots.py` | acceptance vs muon angle and hit maps (needs `toy_76.npz`) | `toy_aerogel_acceptance.png` |
| `toy_scan.py` | scan of SiPM and mirror size for aerogel and gas | stdout |
| `toy_prox.py` | proximity-focusing aerogel ring rates | stdout |

To regenerate the momentum files: `for k in 91 160 240 365; do python gen_pk.py $k 50000; done`
