from __future__ import annotations

from .benchmarks import run_full_comparison, run_real_assets, run_tm_vs_dt

__all__ = ["run_full_comparison", "run_real_assets", "run_tm_vs_dt"]


if __name__ == "__main__":
    run_real_assets()
