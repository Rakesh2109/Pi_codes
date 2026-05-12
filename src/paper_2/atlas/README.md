# `paper_2.atlas` — TMAtlas-style export for GLADE + FPTM

`sonnets-project/TMAtlas` exports a trained TMU model into a serialisable
`{model, metadata, features, clauses}` document (plus a per-sample CSV of
predictions and clause activations). That package only understands the **TMU**
binarisers and the TMU clause bank, so this sub-package provides the equivalent
for Paper 2's stack: the **GLADE** booleaniser + the **Fuzzy Pattern Tsetlin
Machine** (`tm_rules.json` / `model.pkl`).

## Layout (mirrors TMAtlas)

| TMAtlas | here | role |
|---|---|---|
| `booleanisation.Thermometer` | `atlas.GLADEAdapter` | binariser exposing `unique_values` + `get_feature_names_out` (+ `transform`) |
| `inspectors.FeatureInspector` | `atlas.FeatureInspector` | per-feature definitions (`name`, `type`, `range`, `thresholds`) |
| `inspectors.TMUInspector` | `atlas.FPTMInspector` | model info / metadata / clauses from a `tm_rules` dict |
| `exporters.JsonExporter` | `atlas.JsonExporter` | assembles the JSON document |
| `exporters.format_clause_activations` / `export_data_to_csv` | `atlas.format_clause_activations` / `atlas.export_predictions_csv` | per-sample CSV |
| — | `atlas.clause_output_matrix` | graded per-clause output `max(clamp − violations, 0)` for every row |
| — | `atlas.build_atlas` / `atlas.load_bundle` | one-call wrapper / bundle loader |

Only dependency: NumPy (`tm_inference` / numba is used **optionally**, just for
the `predicted` column of the CSV).

## CLI

```bash
cd src

# JSON to stdout (generic feature names, threshold-derived feature ranges)
python -m paper_2.atlas ../results/medsec/models/model.pkl

# JSON with real names / classes / value ranges, plus a predictions+activations CSV
python -m paper_2.atlas ../results/medsec/models/model.pkl \
    --dataset medsec --out medsec_atlas.json --csv medsec_predictions.csv

# straight from the Julia clause dump (sibling glade.json is picked up automatically)
python -m paper_2.atlas ../results/nslkdd/models/tm_rules.json --dataset nslkdd
```

`--dataset` is one of `wustl|nslkdd|ton_iot|medsec`; it is required for `--csv`
(the CSV needs the raw input rows). `--feature-names a,b,c` / `--class-names ...`
let you supply names without loading a dataset. `--csv-split {train,test}`
(default `test`) and `--limit N` cap the CSV.

## Library

```python
from paper_2.atlas import build_atlas
import json

doc = build_atlas("../results/medsec/models/model.pkl")          # dict, .pkl, or tm_rules.json
json.dump(doc, open("medsec_atlas.json", "w"), indent=2, ensure_ascii=False)
```

```python
from paper_2.atlas import (
    GLADEAdapter, FeatureInspector, FPTMInspector, JsonExporter,
    clause_output_matrix, format_clause_activations, export_predictions_csv, load_bundle,
)
tm_rules, glade = load_bundle("../results/wustl/models/model.pkl")
adapter   = GLADEAdapter(glade)
feat_ins  = FeatureInspector(adapter, X=X_train, feature_names=names)   # X optional
inspector = FPTMInspector(tm_rules, feat_ins, class_names=class_names)  # class_names optional
doc       = JsonExporter(inspector, feat_ins).export()

acts = format_clause_activations(clause_output_matrix(tm_rules, adapter, X_test))
export_predictions_csv("preds.csv", names, X_test, y_test, y_pred, acts)
```

## Browser viewer (GUI)

`viewer.html` is a single-file, dependency-free web UI for an atlas JSON:
**Overview** (model/task/classes + metadata), **Features** (sortable, searchable
table with each feature's value range and threshold list), **Clauses** (filter by
class / polarity / literal text / minimum literal count; each clause rendered as a
readable rule, e.g. `＋ class normal · clamp 15 · SrcBytes ≥ 436 ∧ DstBytes < 252 ∧ …`),
and **Predictions** (load a `*_predictions.csv`; filter, "mis-classified only",
click a row → see its activated clauses rendered inline).

Three ways to use it:

```bash
cd src

# 1. self-contained: embeds the atlas in the page, just open the .html
python -m paper_2.atlas ../results/medsec/models/model.pkl --dataset medsec \
    --html medsec_atlas.html        # then open medsec_atlas.html in a browser

# 2. open viewer.html directly (file://) and pick a *_atlas.json via the
#    "Load atlas JSON" button (or drag-and-drop it onto the page)

# 3. serve results/atlas/ over HTTP (works with VS Code port forwarding)
python -m paper_2.atlas serve        # prints http://localhost:8000/viewer.html
                                     #   …/viewer.html?src=wustl_atlas.json  auto-loads it
python -m paper_2.atlas serve --dir . --port 8123
```

(There is no native GUI toolkit in this environment — Tk/Streamlit/Flask aren't
installed — hence a static HTML app you open in your own browser.)

## Output schema

```jsonc
{
  "model":    { "type": "classification", "task": "binary|multiclass", "classes": ["…"] },
  "metadata": { "numClausesPerClass": 80, "numClauses": 400, "numClasses": 5,
                "numLiterals": 351, "T": 12, "s": 75, "L": 30, "LF": 8, "epochs": 300,
                "weightedClauses": false, "binariser": "GLADEv2",
                "created": "2026-…T…+00:00" },
  "features": [ { "name": "SrcBytes", "type": "binary|continuous|unknown",
                  "range": [310.0, 2298.0], "thresholds": [430.0, 436.0, …] }, … ],
  "clauses":  [ { "id": 0, "class": "normal", "polarity": "positive|negative",
                  "clamp": 15, "nLiterals": 53,
                  "literals": [ { "feature": "SrcBytes", "operator": "≥|<", "threshold": 436.0 }, … ],
                  "weights":  { "normal": { "value": 15.0, "polarity": "positive" } } }, … ]
}
```

Differences from TMAtlas, all driven by FPTM semantics:

- An FPTM clause belongs to **one** class and **one** polarity (positive
  clauses vote *for* the class, negative clauses *against*), so each clause
  record carries `class` and `polarity`, and `weights` holds only that class.
- `clamp` is the fuzzy literal-sum cap: a clause emits `max(clamp − violations, 0)`
  (violations = unsatisfied `≥` literals + satisfied-but-forbidden `<` literals),
  i.e. the clause vote lies in `[0, clamp]`. `clause_output_matrix(...)` returns
  these graded outputs; `> 0` is the binary "activated?" view.
- Class prediction is `argmax_k (Σ positive-clause outputs − Σ negative-clause outputs)`
  over classes — `clause_output_matrix` reconstructs `tm_inference.TMModel.predict`
  exactly.
- `binariser`/`metadata` carry the GLADE-specific fields (`L`, `LF`, `epochs`);
  `feature.type` is `"unknown"` when the export is built without the raw data.
