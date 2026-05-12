# Experiment Command Registry

Run from `src/` (`cd src` first). See `README.md` for setup (Julia, datasets, env vars).

## Main classification benchmark (paper Tables IV–V)

| ID | Purpose | Command |
|---|---|---|
| `tm.all` | GLADE + FPTM on all datasets. | `python -m paper_2.run_tm` |
| `ml.all` | Library-default ML baselines, all datasets. | `python -m paper_2.run_ml` |
| `ml_tmmatched.all` | Capacity-matched tree baselines (`n_estimators = C`). | `python -m paper_2.run_ml_tmmatched` |
| `tinyml.all` | Small-MLP baselines, all datasets. | `python -m paper_2.run_tinyml` |
| `run_all` | All model families (long). | `python -m paper_2.run_all` (`--skip-tm` / `--skip-ml` / `--skip-tinyml` / `--skip-ml-tmmatched`) |
| `tm.one` | GLADE + FPTM on one dataset. | `python -m paper_2.run_any --task tm --dataset nslkdd` |
| `ml.one` | ML baselines on one dataset / subset. | `python -m paper_2.run_any --task ml --dataset medsec --models XGBoost,RandomForest` |
| `tinyml.one` | TinyML on one dataset / subset. | `python -m paper_2.run_any --task tinyml --dataset wustl --models MLP_tiny` |

## Binariser comparison — FPTM with 3 binarisers (paper Table III)

| ID | Purpose | Command |
|---|---|---|
| `binariser.all` | FPTM trained on GLADE / StandardBinarizer / Thermometer, all datasets. | `python -m paper_2.run_binarizer_comparison` |
| `binariser.one` | One dataset, all 3 binarisers. | `python -m paper_2.run_binarizer_comparison --dataset wustl` |
| `binariser.glade` | One binariser, all datasets. | `python -m paper_2.run_binarizer_comparison --binariser glade` |
| `binariser.latex` | Print Table III from cached results. | `python -m paper_2.run_binarizer_comparison --render-latex` |

## GLADE component ablation

| ID | Purpose | Command |
|---|---|---|
| `ablation.cell` | One (dataset, variant) cell. `variant ∈ {full, no_budget, no_gap, no_perturb}`. | `python -m paper_2.glade_ablation --dataset nslkdd --variant no_gap --out ../results/ablation/nslkdd_no_gap.json` |

## Serialization, compression, inference

| ID | Purpose | Command |
|---|---|---|
| `bundle` | Build `results/<ds>/models/model.pkl` deployment bundles. | `python -m paper_2.save_uncompressed_bundle` |
| `compress` | Pickle vs FBZ (zstd-1) sizes. | `python -m paper_2.bench_compression` |
| `codec.sweep` | Full codec/level sweep. | `python -m paper_2.bench_codecs_sweep` |
| `verify.fbz` | Byte-exact `.fbz` round-trip. | `python -m paper_2.verify_fbz` |
| `infer.all` | Per-sample inference latency, all models. | `python -m paper_2.bench_inference_all` |
| `verify.infer` | FPTM predict path == Julia reference. | `python -m paper_2.verify_tm_inference` |

## Raspberry Pi 5 (edge-latency figure)

| ID | Purpose | Command |
|---|---|---|
| `pi.bench` | Binarisation/inference latency on the Pi (run on the device). | `python pi_bench/bench_pi.py` |

## Figures & tools

| ID | Purpose | Command |
|---|---|---|
| `plot.storage` | Storage-footprint bar chart. | `python -m paper_2.plot_storage_bar` |
| `plot.codec` | Codec sweep figures. | `python -m paper_2.plot_codec_sweep` |
| `plot.infer` | Inference-latency figure. | `python -m paper_2.plot_inference` |
| `atlas.export` | Structured JSON/HTML export of a trained model. | `python -m paper_2.atlas ../results/medsec/models/model.pkl --dataset medsec --out a.json --html a.html` |
| `atlas.serve` | Serve the model viewer. | `python -m paper_2.atlas serve` |
