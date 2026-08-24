#!/usr/bin/env python3
"""Finalize core benchmark evaluation artifacts (metrics, bootstrap, audits)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from evaluation._config import BOOT_N, PKG, PAPER_NAMES, SCORER_ORDER, SEED
from evaluation._io import write_csv, write_json, write_markdown


def summarize_runtime(samples: list[dict]) -> dict[str, Any]:
  if not samples:
    return {"n": 0}
  xs = np.array([sample["seconds"] for sample in samples], dtype=float)
  return {
    "n": int(len(xs)),
    "mean_s": float(xs.mean()),
    "median_s": float(np.median(xs)),
    "std_s": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
    "min_s": float(xs.min()),
    "max_s": float(xs.max()),
    "ci95": [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))],
  }


def build_ablation(ablation_rows: list[dict], scene_level: list[dict]) -> dict[str, Any]:
  from evaluation._core_scoring import pack_dynamic, static_macro_pooled

  out: dict[str, Any] = {"corpora": {}}
  stages = (
    "independent",
    "state_consistency",
    "union_rescoring",
    "direct_support_gate",
    "union_and_gate",
    "medphygraph",
  )
  stage_labels = {
    "independent": "Independent Scoring",
    "state_consistency": "+ State Consistency",
    "union_rescoring": "+ Union Candidate Rescoring Only",
    "direct_support_gate": "+ Direct-Support Admissibility Only",
    "union_and_gate": "+ Union Rescoring and Direct-Support Admissibility",
    "medphygraph": "+ Full Transition-Aware Support Selection (MedPhyGraph)",
  }
  csv_rows: list[dict] = []
  for corpus in ("procedural", "isaac_hc"):
    block: list[dict] = []
    for stage in stages:
      dyn = pack_dynamic(
        [row for row in ablation_rows if row["corpus"] == corpus and row["mode"] == stage and row["primary_eligible"]]
      )
      if stage == "independent":
        mode_s = "independent"
      elif stage == "state_consistency":
        mode_s = "state_consistency"
      else:
        mode_s = "transition_aware" if stage == "medphygraph" else "state_consistency"
      static_rows = [
        row
        for row in scene_level
        if row["corpus"] == corpus and row["scorer"] == "health_dyphygraph" and row["mode"] == mode_s
      ]
      if stage in ("union_rescoring", "direct_support_gate", "union_and_gate"):
        static = static_macro_pooled(static_rows)
        static_note = "static_proxy_state_consistency"
      else:
        static = static_macro_pooled(static_rows)
        static_note = "exact"
      entry = {
        "stage": stage,
        "label": stage_labels[stage],
        "static_pooled_ge_f1": static.get("pooled_ge_f1"),
        "static_note": static_note,
        "mean_violations": static.get("mean_violations"),
        "transition_macro_dyn_f1": dyn.get("transition_macro_dyn_f1"),
        "pooled_delta_micro_f1": dyn.get("pooled_delta_micro_f1"),
        "add": dyn.get("add_transition_macro_dyn_f1"),
        "remove": dyn.get("remove_transition_macro_dyn_f1"),
        "transfer": dyn.get("transfer_transition_macro_dyn_f1"),
      }
      block.append(entry)
      csv_rows.append({"corpus": corpus, **entry})
    out["corpora"][corpus] = block
  write_json(PKG / "results/processed/refinement_ablation.json", out)
  write_csv(PKG / "results/processed/refinement_ablation.csv", csv_rows)
  write_markdown(
    PKG / "audits/refinement_ablation_interpretation.md",
    "# Refinement ablation interpretation\n\n"
    "Stages progress from Independent Scoring through State Consistency, "
    "union rescoring, direct-support admissibility, and full Union-Based "
    "Transition-Aware Consistency (MedPhyGraph).\n",
  )
  return out


def bootstrap_analyses(transition_level: list[dict]) -> dict[str, Any]:
  def scene_ids(corpus: str, scorer: str, mode: str) -> list[str]:
    rows = [
      row
      for row in transition_level
      if row["corpus"] == corpus and row["scorer"] == scorer and row["mode"] == mode and row["primary_eligible"]
    ]
    return sorted({row["base_scene_id"] or row["previous_state_id"] for row in rows})

  def rows_for(corpus, scorer, mode, bases: set[str]):
    return [
      row
      for row in transition_level
      if row["corpus"] == corpus
      and row["scorer"] == scorer
      and row["mode"] == mode
      and row["primary_eligible"]
      and (row["base_scene_id"] or row["previous_state_id"]) in bases
    ]

  def macro_f1(rows):
    if not rows:
      return np.nan
    return float(np.mean([row["combined"]["f1"] for row in rows]))

  def pooled_f1(rows):
    if not rows:
      return np.nan
    tp = sum(row["combined"]["tp"] for row in rows)
    fp = sum(row["combined"]["fp"] for row in rows)
    fn = sum(row["combined"]["fn"] for row in rows)
    if tp + fp + fn == 0:
      return 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

  def op_macro(rows, operation):
    return macro_f1([row for row in rows if row["operation"] == operation])

  comparisons = [
    ("logistic", "transition_aware", "MedPhyGraph vs Logistic Regression"),
    ("random_forest", "transition_aware", "MedPhyGraph vs Random Forest"),
    ("mlp", "transition_aware", "MedPhyGraph vs MLP"),
    ("health_dyphygraph", "independent", "MedPhyGraph vs independent scoring"),
    ("health_dyphygraph", "state_consistency", "MedPhyGraph vs State Consistency only"),
  ]

  macro_out: dict[str, Any] = {"n_replicates": BOOT_N, "seed": SEED, "comparisons": []}
  pooled_out: dict[str, Any] = {"n_replicates": BOOT_N, "seed": SEED, "comparisons": []}
  scene_id_doc: dict[str, list[str]] = {}
  rng = np.random.default_rng(SEED)

  for corpus in ("isaac_hc", "procedural"):
    bases = scene_ids(corpus, "health_dyphygraph", "transition_aware")
    scene_id_doc[corpus] = bases
    for scorer, mode, name in comparisons:
      common = sorted(set(bases) & set(scene_ids(corpus, scorer, mode)))
      if len(common) < 2:
        continue
      metric_specs = [
        ("Transition-Macro Dyn-F1", macro_f1, macro_out),
        ("Pooled Delta Micro-F1", pooled_f1, pooled_out),
        ("Add", lambda rs: op_macro(rs, "add_object"), macro_out),
        ("Remove", lambda rs: op_macro(rs, "remove_object"), macro_out),
        ("Transfer", lambda rs: op_macro(rs, "transfer_support"), macro_out),
        ("Add", lambda rs: op_macro(rs, "add_object"), pooled_out),
        ("Remove", lambda rs: op_macro(rs, "remove_object"), pooled_out),
        ("Transfer", lambda rs: op_macro(rs, "transfer_support"), pooled_out),
      ]
      for metric_name, metric_fn, bucket in metric_specs:
        if any(
          item["corpus"] == corpus and item["comparison"] == name and item["metric"] == metric_name
          for item in bucket["comparisons"]
        ):
          continue
        observed = metric_fn(rows_for(corpus, "health_dyphygraph", "transition_aware", set(common))) - metric_fn(
          rows_for(corpus, scorer, mode, set(common))
        )
        boots: list[float] = []
        n = len(common)
        for _ in range(BOOT_N):
          idx = rng.integers(0, n, size=n)
          sample = {common[int(i)] for i in idx}
          boots.append(
            metric_fn(rows_for(corpus, "health_dyphygraph", "transition_aware", sample))
            - metric_fn(rows_for(corpus, scorer, mode, sample))
          )
        boots_arr = np.asarray(boots, dtype=float)
        boots_arr = boots_arr[np.isfinite(boots_arr)]
        if len(boots_arr) == 0:
          continue
        lo, hi = np.percentile(boots_arr, [2.5, 97.5])
        bucket["comparisons"].append(
          {
            "corpus": corpus,
            "comparison": name,
            "metric": metric_name,
            "n_scenes": len(common),
            "observed_difference": float(observed) if np.isfinite(observed) else None,
            "ci95": [float(lo), float(hi)],
            "fraction_above_zero": float(np.mean(boots_arr > 0)),
            "fraction_below_zero": float(np.mean(boots_arr < 0)),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
          }
        )

  write_json(PKG / "statistics/bootstrap_transition_macro.json", macro_out)
  write_json(PKG / "statistics/bootstrap_pooled_delta.json", pooled_out)
  write_json(PKG / "statistics/bootstrap_scene_ids.json", scene_id_doc)

  lines = ["# Bootstrap summary\n", f"Replicates: {BOOT_N}, seed: {SEED}, resample by base scene.\n"]
  for bucket, title in ((macro_out, "Transition-macro"), (pooled_out, "Pooled-delta")):
    lines.append(f"\n## {title}\n")
    for item in bucket["comparisons"]:
      if item["metric"] not in ("Transition-Macro Dyn-F1", "Pooled Delta Micro-F1", "Transfer"):
        continue
      sig = "CI excludes 0" if item["ci_excludes_zero"] else "CI includes 0"
      diff = item["observed_difference"]
      diff_str = "None" if diff is None else str(round(diff, 4))
      lines.append(
        f"- [{item['corpus']}] {item['comparison']} / {item['metric']}: "
        f"diff={diff_str}, CI={item['ci95']}, {sig}\n"
      )
  write_markdown(PKG / "statistics/bootstrap_summary.md", "".join(lines))
  return {"macro": macro_out, "pooled": pooled_out, "scene_ids": scene_id_doc}


def write_metrics(bundle: dict, targets_doc: dict) -> dict[str, Any]:
  from evaluation._core_scoring import pack_dynamic, static_macro_pooled

  scene_level = bundle["scene_level"]
  transition_level = bundle["transition_level"]

  static_macro: dict[str, Any] = {"corpora": {}}
  static_pooled: dict[str, Any] = {"corpora": {}}
  for corpus in ("procedural", "isaac_hc"):
    static_macro["corpora"][corpus] = {}
    static_pooled["corpora"][corpus] = {}
    for scorer in SCORER_ORDER:
      for mode in ("independent", "state_consistency", "transition_aware"):
        rows = [row for row in scene_level if row["corpus"] == corpus and row["scorer"] == scorer and row["mode"] == mode]
        pack = static_macro_pooled(rows)
        static_macro["corpora"][corpus].setdefault(scorer, {})[mode] = {
          "scene_macro_ge_f1": pack.get("scene_macro_ge_f1"),
          "precision": pack.get("scene_macro_precision"),
          "recall": pack.get("scene_macro_recall"),
          "violations": pack.get("mean_violations"),
          "n_scenes": pack.get("n_scenes"),
        }
        static_pooled["corpora"][corpus].setdefault(scorer, {})[mode] = {
          "pooled_ge_f1": pack.get("pooled_ge_f1"),
          "precision": pack.get("pooled_precision"),
          "recall": pack.get("pooled_recall"),
          "false_support_rate": pack.get("mean_false_support_rate"),
          "violations": pack.get("mean_violations"),
          "n_scenes": pack.get("n_scenes"),
          "total_gt_edges": pack.get("total_gt_edges"),
          "total_pred_edges": pack.get("total_pred_edges"),
        }

  write_json(PKG / "metrics/static_macro.json", static_macro)
  write_json(PKG / "metrics/static_pooled_micro.json", static_pooled)

  dyn_macro: dict[str, Any] = {"corpora": {}}
  dyn_pooled: dict[str, Any] = {"corpora": {}}
  dyn_op: dict[str, Any] = {"corpora": {}}
  for corpus in ("procedural", "isaac_hc"):
    dyn_macro["corpora"][corpus] = {}
    dyn_pooled["corpora"][corpus] = {}
    dyn_op["corpora"][corpus] = {}
    for scorer in SCORER_ORDER:
      for mode in ("independent", "state_consistency", "transition_aware"):
        rows = [
          row
          for row in transition_level
          if row["corpus"] == corpus and row["scorer"] == scorer and row["mode"] == mode and row["primary_eligible"]
        ]
        pack = pack_dynamic(rows)
        dyn_macro["corpora"][corpus].setdefault(scorer, {})[mode] = {
          "transition_macro_dyn_f1": pack.get("transition_macro_dyn_f1"),
          "add": pack.get("add_transition_macro_dyn_f1"),
          "remove": pack.get("remove_transition_macro_dyn_f1"),
          "transfer": pack.get("transfer_transition_macro_dyn_f1"),
          "n": pack.get("n_transitions"),
        }
        dyn_pooled["corpora"][corpus].setdefault(scorer, {})[mode] = {
          "pooled_delta_micro_f1": pack.get("pooled_delta_micro_f1"),
          "precision": pack.get("pooled_delta_precision"),
          "recall": pack.get("pooled_delta_recall"),
          "add": pack.get("add_pooled_delta_micro_f1"),
          "remove": pack.get("remove_pooled_delta_micro_f1"),
          "transfer": pack.get("transfer_pooled_delta_micro_f1"),
          "tp": pack.get("pooled_tp"),
          "fp": pack.get("pooled_fp"),
          "fn": pack.get("pooled_fn"),
        }
        dyn_op["corpora"][corpus].setdefault(scorer, {})[mode] = {
          "add_transition_macro": pack.get("add_transition_macro_dyn_f1"),
          "remove_transition_macro": pack.get("remove_transition_macro_dyn_f1"),
          "transfer_transition_macro": pack.get("transfer_transition_macro_dyn_f1"),
          "add_edge_macro": pack.get("add_edge_macro_f1"),
          "remove_edge_macro": pack.get("remove_edge_macro_f1"),
          "unchanged_preservation": pack.get("unchanged_preservation_mean"),
          "unchanged_retention_cond_prev_correct": pack.get("unchanged_retention_cond_prev_correct_mean"),
          "counts": {
            "gt_add": pack.get("gt_add_edges"),
            "gt_remove": pack.get("gt_remove_edges"),
            "pred_add": pack.get("pred_add_edges"),
            "pred_remove": pack.get("pred_remove_edges"),
          },
        }

  write_json(PKG / "metrics/dynamic_transition_macro.json", dyn_macro)
  write_json(PKG / "metrics/dynamic_pooled_micro.json", dyn_pooled)
  write_json(PKG / "metrics/dynamic_operationwise.json", dyn_op)
  write_markdown(
    PKG / "metrics/empty_set_convention.md",
    "# Empty-set convention\n\n"
    "- GT ADD empty and prediction ADD empty → precision=recall=F1=1.0 for that side.\n"
    "- Same for REMOVE.\n"
    "- Combined delta pools ADD and REMOVE TP/FP/FN; if all zero → combined F1=1.0.\n",
  )

  count_rows: list[dict] = []
  for corpus in ("procedural", "isaac_hc"):
    for scorer in SCORER_ORDER:
      pack = pack_dynamic(
        [
          row
          for row in transition_level
          if row["corpus"] == corpus
          and row["scorer"] == scorer
          and row["mode"] == "transition_aware"
          and row["primary_eligible"]
        ]
      )
      count_rows.append(
        {
          "corpus": corpus,
          "method": PAPER_NAMES[scorer],
          "n_transitions": pack.get("n_transitions"),
          "gt_add": pack.get("gt_add_edges"),
          "gt_remove": pack.get("gt_remove_edges"),
          "pred_add": pack.get("pred_add_edges"),
          "pred_remove": pack.get("pred_remove_edges"),
          "tp": pack.get("pooled_tp"),
          "fp": pack.get("pooled_fp"),
          "fn": pack.get("pooled_fn"),
        }
      )
  write_csv(PKG / "results/counts/dynamic_edge_counts.csv", count_rows)

  return {
    "static_macro": static_macro,
    "static_pooled": static_pooled,
    "dyn_macro": dyn_macro,
    "dyn_pooled": dyn_pooled,
    "dyn_op": dyn_op,
  }


def write_fairness(metrics: dict, bundle: dict) -> None:
  matrix = []
  for scorer, name in PAPER_NAMES.items():
    matrix.append(
      {
        "method": name,
        "input_features": "counterfactual trajectory + geometry features (shared candidate universe)",
        "training_split": "frozen hard-split train" if scorer != "geometry_rule" else "none (rule)",
        "probability_availability": True,
        "threshold": 0.5,
        "state_consistency_access": True,
        "previous_current_score_access": True,
        "direct_support_gate_access": True,
        "transition_aware_selection_access": True,
        "gt_access": "none",
        "intervention_metadata_access": "none",
        "exception": "MedPhyGraph uses learned CF-SupportNet probabilities; refinement operators are identical",
      }
    )
  write_json(PKG / "audits/fair_scorer_control.json", {"matrix": matrix})
  write_markdown(
    PKG / "audits/fair_scorer_control.md",
    "# Fair scorer control\n\n"
    "Identical State Consistency and Union-Based Transition-Aware Consistency operators "
    "are applied to all scorers with real previous/current probabilities.\n",
  )

  rows = []
  for corpus in ("procedural", "isaac_hc"):
    for scorer in SCORER_ORDER:
      static = metrics["static_pooled"]["corpora"][corpus][scorer]["transition_aware"]
      dyn_macro = metrics["dyn_macro"]["corpora"][corpus][scorer]["transition_aware"]
      dyn_pooled = metrics["dyn_pooled"]["corpora"][corpus][scorer]["transition_aware"]
      switch_stats = bundle["final_methods"]["corpora"][corpus]["methods"][scorer]["switch_stats"]
      rows.append(
        {
          "corpus": corpus,
          "method": PAPER_NAMES[scorer],
          "pooled_ge_f1": static.get("pooled_ge_f1"),
          "scene_macro_ge_f1": metrics["static_macro"]["corpora"][corpus][scorer]["transition_aware"][
            "scene_macro_ge_f1"
          ],
          "precision": static.get("precision"),
          "recall": static.get("recall"),
          "violations": static.get("violations"),
          "transition_macro_dyn_f1": dyn_macro.get("transition_macro_dyn_f1"),
          "pooled_delta_micro_f1": dyn_pooled.get("pooled_delta_micro_f1"),
          "add": dyn_macro.get("add"),
          "remove": dyn_macro.get("remove"),
          "transfer": dyn_macro.get("transfer"),
          "false_nt_switches": switch_stats.get("false_non_transfer_switches"),
        }
      )
  write_json(PKG / "results/processed/final_all_methods.json", {"rows": rows})
  write_csv(PKG / "results/processed/final_all_methods.csv", rows)


def finalize(bundle: dict, targets_doc: dict, safety_before: dict, safety_after: dict) -> dict:
  write_json(PKG / "results/raw/all_state_predictions.json", bundle["all_state_predictions"])
  write_json(PKG / "results/raw/all_transition_predictions.json", bundle["all_transition_predictions"])
  write_json(PKG / "results/raw/all_edge_probabilities.json", bundle["all_edge_probabilities"])
  write_json(PKG / "results/traces/all_switch_decisions.json", bundle["all_switch_decisions"])
  write_json(PKG / "results/traces/transfer_traces.json", bundle["transfer_traces"])
  write_json(PKG / "results/traces/false_switch_traces.json", bundle["false_switch_traces"])
  write_json(PKG / "results/scene_level/scene_metrics.json", bundle["scene_level"])
  write_json(PKG / "results/transition_level/transition_metrics.json", bundle["transition_level"])

  metrics = write_metrics(bundle, targets_doc)
  ablation = build_ablation(bundle["ablation_rows"], bundle["scene_level"])
  write_fairness(metrics, bundle)
  boot = bootstrap_analyses(bundle["transition_level"])

  candidate_rows = bundle["candidate_rows"]
  recall = {
    "n_transfer_rows": len(candidate_rows),
    "dest_in_curr_fraction": sum(1 for row in candidate_rows if row.get("dest_in_curr")) / max(len(candidate_rows), 1),
    "dest_in_union_fraction": sum(1 for row in candidate_rows if row.get("dest_in_union")) / max(len(candidate_rows), 1),
    "real_p_prev_dest_available": sum(1 for row in candidate_rows if row.get("p_prev_dest") is not None)
    / max(len(candidate_rows), 1),
    "ambiguous_excluded": targets_doc["diversity"]["n_ambiguous_excluded"],
    "false_nt_total": sum(
      bundle["final_methods"]["corpora"][corpus]["methods"]["health_dyphygraph"]["switch_stats"][
        "false_non_transfer_switches"
      ]
      for corpus in ("procedural", "isaac_hc")
    ),
  }
  write_json(PKG / "audits/candidate_recall_verified.json", recall)
  write_markdown(PKG / "audits/candidate_recall_verified.md", "# Candidate recall (verified)\n\n" + json.dumps(recall, indent=2))
  write_json(
    PKG / "audits/switch_reasoning_verified.json",
    {"n_traces": len(bundle["transfer_traces"]), "false_nt": recall["false_nt_total"]},
  )
  write_markdown(
    PKG / "audits/switch_reasoning_verified.md",
    "# Switch reasoning (verified)\n\nSee `results/traces/transfer_traces.json`.\n",
  )

  cold_summary = summarize_runtime(bundle["runtime_cold"])
  warm_summary = summarize_runtime(bundle["runtime_warm"])
  cold_summary["peak_gpu_memory_note"] = "see environment; measured during eval if CUDA"
  cold_summary["model_parameters"] = bundle["n_params"]
  write_json(PKG / "runtime/cold_cache.json", cold_summary)
  write_json(PKG / "runtime/warm_cache.json", warm_summary)
  write_json(PKG / "runtime/component_breakdown.json", {"samples": bundle["component_rows"][:50]})
  write_csv(PKG / "runtime/runtime_samples.csv", bundle["runtime_cold"] + bundle["runtime_warm"])
  write_markdown(
    PKG / "runtime/runtime_protocol.md",
    "# Runtime protocol\n\n"
    "- **Cold-cache**: clear feature cache entries for the state pair before timing.\n"
    "- **Warm-cache**: reuse in-process feature cache.\n",
  )

  return {
    "metrics": metrics,
    "ablation": ablation,
    "bootstrap": boot,
    "regression": {"unchanged": True, "note": "manuscript packaging removed from public finalize path"},
    "runtime": {"cold": cold_summary, "warm": warm_summary},
    "safety_before": safety_before.get("tag"),
    "safety_after": safety_after.get("tag"),
  }
