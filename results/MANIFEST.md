# Frozen results manifest

This tree holds **read-only, git-tracked** paper numbers. Eval drivers must not
overwrite these files; new runs go under `runs/<script>/<utc-timestamp>/`.

| Folder | Backs (paper) | Key files |
|--------|---------------|-----------|
| `static/` | Core static GE-F1 tables (§5.1) | `static_macro.json`, `static_pooled_micro.json` |
| `dynamic/` | Core dynamic / transfer tables (§5.1) | `dynamic_*.json`, `processed/final_all_methods.*`, `processed/refinement_ablation.*`, `scene_level/` |
| `evaluation_targets/` | Verified transition target list for core protocol | `operation_consistent_dynamic_verified.json` |
| `expanded_transfer/` | Expanded transfer suite (§5.2) | `results.json`, `targets.json`, `predeclared_manifest.json` |
| `component_analysis/` | Component analysis (§5.3; paper term) | `results.json` |
| `bootstrap/` | Bootstrap CIs (§5.4) | `bootstrap_*.json`, `bootstrap_summary.md` |
| `multiseed/` | Multi-seed aggregates | `results.json`, `*_per_seed.csv` |
| `diagnostics/` | Scorer / dropout diagnostics | `scorer_ablations_seed0.json`, `candidate_dropout_results.json` |

Bytes of these JSON/CSV files are frozen for the workshop release. Re-running
evaluation writes elsewhere under `runs/`; compare against this tree with
`scripts/verify_release.py`.
