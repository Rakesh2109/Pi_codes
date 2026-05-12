#!/usr/bin/env python3
"""Verify a serialised .fbz model by running Python inference on the
test split and comparing macro-F1 to the baseline metrics produced
during Julia training.

Inference logic mirrors `vote()` in FuzzyPatternTM/src/Tsetlin.jl:

    For each clause k of class c, polarity p:
        violations = | { i in include_k : x_i == 0 } |
                   + | { i in exclude_k : x_i == 1 } |
        clause_out = max(clamp_k - violations, 0)

    pos_vote_c = sum_k clause_out  (positive-polarity clauses)
    neg_vote_c = sum_k clause_out  (negative-polarity clauses)
    score_c    = pos_vote_c - neg_vote_c

    y_hat = argmax_c score_c
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

from . import fbz
from .config import DATASETS, RESULTS_DIR
from .data_loader import load_and_preprocess


def _build_clause_matrices(clauses: list, n_bits: int):
    """Stack a polarity's clauses into dense int8 matrices for batch eval."""
    K = len(clauses)
    if K == 0:
        return (np.zeros((0, n_bits), dtype=np.int8),
                np.zeros((0, n_bits), dtype=np.int8),
                np.zeros(0, dtype=np.int32))
    pos_mask = np.zeros((K, n_bits), dtype=np.int8)
    neg_mask = np.zeros((K, n_bits), dtype=np.int8)
    clamps = np.zeros(K, dtype=np.int32)
    for k, c in enumerate(clauses):
        if c["include"]:
            pos_mask[k, c["include"]] = 1
        if c["exclude"]:
            neg_mask[k, c["exclude"]] = 1
        clamps[k] = c["clamp"]
    return pos_mask, neg_mask, clamps


def _class_vote(X: np.ndarray, pos_mask, neg_mask, clamps):
    """Sum of clause outputs for one polarity, vectorised over all samples.

    X         : (n_samples, n_bits)        int8/uint8 0/1
    pos_mask  : (n_clauses, n_bits)        1 where literal x_j is required
    neg_mask  : (n_clauses, n_bits)        1 where literal NOT x_j is required
    clamps    : (n_clauses,)
    returns   : (n_samples,)               int votes
    """
    if pos_mask.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=np.int64)
    inc_total = pos_mask.sum(axis=1)               # (K,)
    inc_hits = X @ pos_mask.T                       # (S, K) — x_j==1 AND included
    pos_violations = inc_total[None, :] - inc_hits  # x_j==0 AND included
    neg_violations = X @ neg_mask.T                 # x_j==1 AND inverted-included
    violations = pos_violations + neg_violations
    clause_out = np.maximum(clamps[None, :] - violations, 0)
    return clause_out.sum(axis=1)


def class_scores(model: dict, X_bin: np.ndarray) -> np.ndarray:
    """Per-sample, per-class vote scores (pos_vote - neg_vote)."""
    n_bits = model["n_bits"]
    labels = model["labels"]
    rules = model["rules"]
    assert X_bin.shape[1] == n_bits, \
        f"input has {X_bin.shape[1]} bits, model expects {n_bits}"

    X = X_bin.astype(np.int8, copy=False)
    scores = np.zeros((X.shape[0], len(labels)), dtype=np.int64)
    for c, lbl in enumerate(labels):
        spec = rules[lbl]
        pm, nm, cl = _build_clause_matrices(spec["positive"], n_bits)
        pos_vote = _class_vote(X, pm, nm, cl)
        pm, nm, cl = _build_clause_matrices(spec["negative"], n_bits)
        neg_vote = _class_vote(X, pm, nm, cl)
        scores[:, c] = pos_vote - neg_vote
    return scores


def predict(model: dict, X_bin: np.ndarray) -> np.ndarray:
    """Predict class labels from a normalised FBZ dict + binarised X."""
    labels = model["labels"]
    pred_idx = class_scores(model, X_bin).argmax(axis=1)
    return np.asarray([labels[i] for i in pred_idx])


def verify_dataset(loader_name: str, short_id: str, human_name: str):
    results_root = Path(RESULTS_DIR)
    fbz_path = results_root / short_id / "models" / "model.fbz"
    report_path = results_root / short_id / "reports" / "GLADE_FPTM.json"

    print(f"\n{'=' * 68}")
    print(f"  {human_name}  —  {fbz_path.name}")
    print(f"{'=' * 68}")

    model = fbz.read(fbz_path)
    has_glade = "glade" in model
    print(f"  codec        : {model['codec']} (level {model['codec_level']})")
    print(f"  sections     : "
          f"{'GLADE + TM' if has_glade else 'TM only'}")
    print(f"  n_bits       : {model['n_bits']}")
    print(f"  n_classes    : {len(model['labels'])}  labels={model['labels']}")
    print(f"  clamp_max    : {model['clamp_max']}")
    if has_glade:
        g = model["glade"]
        print(f"  GLADE        : {g['n_features_in']} raw features → "
              f"{model['n_bits']} bits  "
              f"(thresholds f32, indices i32)")
    n_pos = sum(len(model['rules'][l]['positive']) for l in model['labels'])
    n_neg = sum(len(model['rules'][l]['negative']) for l in model['labels'])
    print(f"  clauses      : {n_pos} positive + {n_neg} negative "
          f"= {n_pos + n_neg} total")
    print(f"  file size    : {fbz_path.stat().st_size} B")

    data = load_and_preprocess(loader_name)
    Xte, yte = data["X_test"], data["y_test"]
    # Binarise straight from the .fbz — no external glade.json needed
    Xte_b = fbz.transform(model, Xte)

    scores = class_scores(model, Xte_b)
    n_total = int(scores.shape[0])
    y_true = np.asarray(yte)
    labels_arr = np.asarray(model["labels"])

    # ── Two deterministic tie-break rules ─────────────────────────
    # (a) numpy argmax — picks the smallest tied index
    pred_first = labels_arr[scores.argmax(axis=1)]
    # (b) reverse argmax — picks the largest tied index (predict-from-end)
    pred_last = labels_arr[(scores.shape[1] - 1)
                           - scores[:, ::-1].argmax(axis=1)]

    acc_first = accuracy_score(y_true, pred_first)
    f1_first = f1_score(y_true, pred_first, average="macro", zero_division=0)
    wf1_first = f1_score(y_true, pred_first, average="weighted", zero_division=0)

    acc_last = accuracy_score(y_true, pred_last)
    f1_last = f1_score(y_true, pred_last, average="macro", zero_division=0)

    # ── Lossless round-trip: .fbz preds == tm_rules.json preds ───
    json_path = results_root / short_id / "models" / "tm_rules.json"
    rules_json = json.loads(json_path.read_text())
    model_from_json = fbz.unpack(fbz.pack(rules_json, level=0))
    yhat_json = predict(model_from_json, Xte_b)
    identical = bool(np.array_equal(pred_first, yhat_json))

    # ── Tie statistics ───────────────────────────────────────────
    max_scores = scores.max(axis=1, keepdims=True)
    tied_classes = (scores == max_scores).sum(axis=1)
    n_ties = int(np.sum(tied_classes > 1))
    # On non-tied samples the prediction is independent of tie-break;
    # the two rules MUST agree there.
    untied = tied_classes == 1
    untied_match = bool(np.array_equal(pred_first[untied], pred_last[untied]))
    # Disagreements between the two rules can only occur on tied samples
    n_disagree = int(np.sum(pred_first != pred_last))

    base = json.loads(report_path.read_text())["metrics"]
    print()
    print(f"  {'metric':<14}{'Julia':>11}"
          f"{'Python (smallest)':>19}{'Python (largest)':>19}")
    print(f"  {'-' * 62}")
    print(f"  {'accuracy':<14}{base['accuracy']:>11.6f}"
          f"{acc_first:>19.6f}{acc_last:>19.6f}")
    print(f"  {'macro F1':<14}{base['macro_f1']:>11.6f}"
          f"{f1_first:>19.6f}{f1_last:>19.6f}")
    print(f"  {'weighted F1':<14}{base['weighted_f1']:>11.6f}"
          f"{wf1_first:>19.6f}{'—':>19}")

    py_lo, py_hi = sorted([f1_first, f1_last])
    julia_in = py_lo - 1e-9 <= base["macro_f1"] <= py_hi + 1e-9
    print()
    print(f"  Tie analysis:")
    print(f"    test samples            : {n_total}")
    print(f"    samples with tied scores: {n_ties}  ({n_ties/n_total:.2%})")
    print(f"    Python tie-break disagreements (smallest vs largest): "
          f"{n_disagree}  (matches n_ties: "
          f"{'YES ✓' if n_disagree <= n_ties else 'NO ✗'})")
    print(f"    Non-tied predictions identical between rules: "
          f"{'YES ✓' if untied_match else 'NO ✗'}")
    print(f"    Python F1 range across both tie-breaks: "
          f"[{py_lo:.6f}, {py_hi:.6f}]")
    julia_mf1 = base["macro_f1"]
    julia_gap = (julia_mf1 - py_hi) if julia_mf1 > py_hi else (py_lo - julia_mf1)
    print(f"    Julia F1 within Python's tie-break range: "
          f"{'YES ✓' if julia_in else f'gap = {julia_gap:+.4f}'}")
    print(f"  Lossless round-trip (.fbz preds == tm_rules.json preds): "
          f"{'YES ✓' if identical else 'NO ✗'}")

    # Substitute names in the legacy summary
    acc, mf1 = acc_first, f1_first
    ok = identical
    return {"dataset": short_id, "acc": acc, "macro_f1": mf1,
            "baseline_acc": base["accuracy"], "baseline_mf1": base["macro_f1"],
            "match": ok}


def main():
    rows = []
    for loader_name, short_id, human_name in DATASETS:
        try:
            rows.append(verify_dataset(loader_name, short_id, human_name))
        except Exception as e:
            print(f"ERROR {short_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 68)
    print("  SUMMARY  (Python inference from .fbz vs Julia training metrics)")
    print("=" * 68)
    print(f"  {'dataset':<10} {'acc(fbz)':>10} {'acc(base)':>10} "
          f"{'mf1(fbz)':>10} {'mf1(base)':>10}  match")
    for r in rows:
        print(f"  {r['dataset']:<10} {r['acc']:>10.6f} "
              f"{r['baseline_acc']:>10.6f} {r['macro_f1']:>10.6f} "
              f"{r['baseline_mf1']:>10.6f}  "
              f"{'✓' if r['match'] else '✗'}")


if __name__ == "__main__":
    main()
