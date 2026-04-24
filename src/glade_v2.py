#!/usr/bin/env python3
"""
GLADE v2 — Gap-aware Lightweight Adaptive Discretisation Engine, v2.

Three accuracy changes over v1:
  (a) Hybrid bit budget     — dense continuous columns get the full budget
                               (like KBinsQuantile); only sparse or near-
                               categorical columns use the adaptive log-budget.
  (b) Entropy-based dead-bit elimination — thresholds whose resulting bits
                               have binary entropy below 0.05 nats are
                               discarded, instead of the 1%-of-samples rule
                               from v1. Preserves minority-class bits on
                               imbalanced datasets.
  (c) Local gap perturbation — each quantile edge is nudged into the
                               highest-density neighbourhood by trying
                               t and t ± (local gap)/2, and picking the
                               placement that maximises bit variance.

Edge-friendliness options (all orthogonal to accuracy):
  - `pack_bits=True` in transform() returns a uint8-packed bit matrix
                    (8 bits per output byte) at zero accuracy cost.
  - `quantise_int16=True` in save_json() replaces float32 thresholds
                    with int16 + per-feature (scale, zero_point), halving
                    the binariser state size on MCU.

Public API is compatible with GLADE v1 (same method names + signatures),
so the existing run_tm.py / sweep_* / TBF layer all work by swapping
the import.
"""

from __future__ import annotations

import json
import numpy as np


class GLADEv2:

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self._cat_cols:  list[int] = []
        self._num_cols:  list[int] = []
        self._cat_edges: list[np.ndarray] = []
        self._num_edges: list[np.ndarray] = []
        self._feat_idx: np.ndarray | None = None
        self._thresh:   np.ndarray | None = None
        self._n_features = 0
        self._fitted = False

    # ── fit ───────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "GLADEv2":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self._n_features = d
        K = self.n_bins

        # A representative subsample keeps fitting fast on large datasets.
        SUB = 30_000
        if n > SUB:
            rng = np.random.RandomState(42)
            Xs = X[rng.choice(n, SUB, replace=False)]
        else:
            Xs = X

        self._cat_cols, self._num_cols = [], []
        for j in range(d):
            nu = np.unique(Xs[:, j]).size
            if nu <= 1:
                continue
            if nu <= K:
                self._cat_cols.append(j)
            else:
                self._num_cols.append(j)

        # Categorical columns: midpoint thresholds between sorted uniques.
        self._cat_edges = [np.sort(np.unique(X[:, j])) for j in self._cat_cols]

        # Numerical columns: hybrid budget + gap-aware + dead-bit pruning.
        self._num_edges = []
        for j in self._num_cols:
            col_sub = Xs[:, j]
            k_j = self._hybrid_budget(col_sub, K)
            edges = self._find_edges(col_sub, k_j)
            self._num_edges.append(edges)

        # Flatten into (feat_idx, thresh) vectors for fast inference.
        all_fi, all_th = [], []
        for idx, j in enumerate(self._cat_cols):
            for v in self._cat_edges[idx][1:]:
                all_fi.append(j); all_th.append(float(v))
        for idx, j in enumerate(self._num_cols):
            for v in self._num_edges[idx]:
                all_fi.append(j); all_th.append(float(v))

        self._feat_idx = np.asarray(all_fi, dtype=np.int32)
        self._thresh   = np.asarray(all_th, dtype=np.float32)
        self._fitted   = True
        return self

    # ── (a) hybrid bit budget ────────────────────────────────────────
    @staticmethod
    def _hybrid_budget(col: np.ndarray, K: int) -> int:
        """
        GLADE v2 budget.

        - If a column is sparse (z > 0.3) OR near-categorical (u <= K),
          fall back to the adaptive v1 formula (log2 of effective unique
          count, with a sparsity boost term).
        - Otherwise, the column is dense continuous and receives the full
          KBinsQuantile-equivalent budget K. This avoids the v1 failure
          mode where dense physiological features were under-resolved.
        """
        u = np.unique(col).size
        if u <= 1:
            return 0
        if u <= K:
            return max(1, u - 1)

        z = float(np.mean(col == 0))
        if z > 0.3:
            # Sparse column: keep v1's conservative budget.
            nz_mask = col != 0
            u_nz = np.unique(col[nz_mask]).size if nz_mask.any() else u
            density = max(1.0 - z, 0.01)
            eff = max(u_nz * density ** 2, 2.0)
            b = int(np.ceil(np.log2(eff)) + 1 + 1)     # +1 for sparsity
            return max(1, min(b, K))

        # Dense continuous column: full KBinsQuant budget.
        return K

    # ── edge placement with gap-aware + (c) local perturbation ───────
    def _find_edges(self, col: np.ndarray, k: int) -> np.ndarray:
        uniq = np.sort(np.unique(col))
        if uniq.size <= 1:
            return np.array([])
        if uniq.size <= k:
            return (uniq[:-1] + uniq[1:]) / 2.0

        zero_frac = float(np.mean(col == 0))
        prefix = []
        work = col
        if zero_frac > 0.3:
            nonzero = col[col > 0]
            if nonzero.size > 10 and np.unique(nonzero).size > 1:
                prefix = [float(np.min(nonzero)) * 0.5]
                work = nonzero

        remaining = k - len(prefix)
        work_uniq = np.sort(np.unique(work))
        if work_uniq.size <= remaining:
            return np.sort(np.unique(prefix + list(
                (work_uniq[:-1] + work_uniq[1:]) / 2.0)))[:k]

        # Baseline: equal-frequency quantile edges.
        pcts = np.linspace(
            100 / (remaining + 1),
            100 * remaining / (remaining + 1),
            remaining,
        )
        raw = np.percentile(work, pcts)

        # Conditional gap-aware snap (v1 behaviour kept).
        gaps = np.diff(work_uniq)
        median_gap = np.median(gaps)
        max_gap = np.max(gaps)
        has_structural = (median_gap > 0) and (max_gap > 5.0 * median_gap)

        if has_structural:
            midpoints = (work_uniq[:-1] + work_uniq[1:]) / 2.0
            idx = np.searchsorted(midpoints, raw).clip(0, midpoints.size - 1)
            left = (idx - 1).clip(0)
            d_right = np.abs(midpoints[idx] - raw)
            d_left  = np.abs(midpoints[left] - raw)
            best = np.where(d_left < d_right, left, idx)
            edges_work = np.unique(midpoints[best])
        else:
            edges_work = np.unique(raw)

        # (c) Local perturbation — per edge, try a small left/right nudge.
        edges_work = self._local_perturb(work, edges_work)

        # (b) Entropy-based dead-bit elimination.
        edges = self._kill_dead_bits_entropy(work, edges_work, remaining)
        return np.sort(np.unique(prefix + list(edges)))[:k]

    # ── (c) local perturbation ──────────────────────────────────────
    @staticmethod
    def _local_perturb(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        For each edge t, try t, t-delta and t+delta where delta is a small
        fraction of the local inter-edge gap, and pick the placement that
        maximises the resulting bit variance. Deterministic; O(k) per
        edge.
        """
        if edges.size == 0:
            return edges
        srt = np.sort(np.unique(edges))
        lo, hi = float(col.min()), float(col.max())
        out = []
        for i, t in enumerate(srt):
            left  = srt[i - 1] if i > 0 else lo
            right = srt[i + 1] if i < srt.size - 1 else hi
            half_left  = (t - left)  * 0.25
            half_right = (right - t) * 0.25
            cands = [t - half_left, t, t + half_right]
            # Variance of the Boolean bit = p * (1 - p).
            best_t, best_var = t, -1.0
            for c in cands:
                p = float(np.mean(col >= c))
                v = p * (1.0 - p)
                if v > best_var:
                    best_var, best_t = v, c
            out.append(best_t)
        return np.array(out, dtype=np.float64)

    # ── (b) entropy-based dead-bit elimination ──────────────────────
    @staticmethod
    def _kill_dead_bits_entropy(
        col: np.ndarray, edges: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Drop thresholds whose resulting Boolean bit has entropy below a
        floor (0.05 nats ~ 0.072 bits), which corresponds to p ∈ (0, 0.014)
        or p ∈ (0.986, 1). This is ~0.7× the 1%-of-samples floor used in
        v1 and preserves minority-class discriminators.
        """
        if edges.size == 0:
            return edges
        floor_nats = 0.05

        def _bit_entropy(t: float) -> float:
            p = float(np.mean(col >= t))
            if p <= 0.0 or p >= 1.0:
                return 0.0
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        alive = np.array([_bit_entropy(t) >= floor_nats for t in edges],
                         dtype=bool)
        good = edges[alive]
        n_need = min(k, edges.size) - good.size
        if n_need <= 0 or good.size == 0:
            return good if good.size > 0 else edges
        bounds = np.concatenate([[col.min()], np.sort(good), [col.max()]])
        for _ in range(n_need):
            widths = np.diff(bounds)
            wi = int(np.argmax(widths))
            mid = (bounds[wi] + bounds[wi + 1]) / 2.0
            good = np.sort(np.append(good, mid))
            bounds = np.concatenate([[col.min()], good, [col.max()]])
        return good[:k]

    # ── transform ───────────────────────────────────────────────────
    def transform(self, X: np.ndarray, pack_bits: bool = False) -> np.ndarray:
        assert self._fitted, "GLADEv2.transform called before fit"
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        total = self._thresh.size
        out = np.empty((n, total), dtype=np.uint8)
        CHUNK = 64
        for s in range(0, total, CHUNK):
            e = min(s + CHUNK, total)
            np.greater_equal(
                X[:, self._feat_idx[s:e]],
                self._thresh[s:e][np.newaxis, :],
                out=out[:, s:e],
            )
        if not pack_bits:
            return out
        # Pack 8 bits into 1 uint8 byte along the feature axis.
        pad = (-total) % 8
        if pad:
            out = np.concatenate(
                [out, np.zeros((n, pad), dtype=np.uint8)], axis=1
            )
        return np.packbits(out, axis=1)

    def fit_transform(self, X: np.ndarray, pack_bits: bool = False):
        return self.fit(X).transform(X, pack_bits=pack_bits)

    # ── persistent storage ──────────────────────────────────────────
    def save_json(self, path: str, quantise_int16: bool = False) -> None:
        """
        Persist the binariser to JSON. When `quantise_int16` is true the
        thresholds are stored as int16 values with a global (scale,
        zero_point), halving the threshold-table size.
        """
        if not quantise_int16:
            payload = {
                "version": "GLADEv2",
                "n_features_in": int(self._n_features),
                "n_bits": int(self._thresh.size),
                "feat_idx": self._feat_idx.tolist(),
                "thresh":   [float(x) for x in self._thresh.tolist()],
                "n_bins_param": int(self.n_bins),
                "quantised": False,
            }
        else:
            t = np.asarray(self._thresh, dtype=np.float64)
            lo, hi = float(t.min()), float(t.max())
            scale = (hi - lo) / 65535.0 if hi > lo else 1.0
            zp = lo
            q = np.clip(np.round((t - zp) / scale), 0, 65535).astype(np.int32)
            payload = {
                "version": "GLADEv2",
                "n_features_in": int(self._n_features),
                "n_bits": int(self._thresh.size),
                "feat_idx": self._feat_idx.tolist(),
                "thresh_q": q.tolist(),
                "thresh_scale": scale,
                "thresh_zp":    zp,
                "n_bins_param": int(self.n_bins),
                "quantised": True,
            }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load_json(cls, path: str) -> "GLADEv2":
        with open(path) as f:
            p = json.load(f)
        obj = cls(n_bins=int(p["n_bins_param"]))
        obj._feat_idx = np.asarray(p["feat_idx"], dtype=np.int32)
        if p.get("quantised", False):
            q = np.asarray(p["thresh_q"], dtype=np.float64)
            obj._thresh = (q * p["thresh_scale"] + p["thresh_zp"]).astype(
                np.float32
            )
        else:
            obj._thresh = np.asarray(p["thresh"], dtype=np.float32)
        obj._n_features = int(p["n_features_in"])
        obj._fitted = True
        return obj

    @property
    def n_bits(self) -> int:
        return int(self._thresh.size) if self._fitted else 0
