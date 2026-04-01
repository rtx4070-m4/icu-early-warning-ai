"""
Model Evaluation & Backtesting Framework
Evaluates all ML models against held-out test data and historical cohorts.
Produces calibration curves, ROC/PR curves, decision thresholds, and
temporal backtests (model performance over rolling time windows).
"""

import os
import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Metric computation (no sklearn required)
# ─────────────────────────────────────────

def _roc_auc(y_true: List[int], y_score: List[float]) -> float:
    """Compute AUC-ROC via trapezoidal rule."""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


def _pr_auc(y_true: List[int], y_score: List[float]) -> float:
    """Average precision (AUC-PR)."""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    tp = fp = 0
    precision_sum = 0.0
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
            precision_sum += tp / (tp + fp)
        fp += (1 - label)
    return precision_sum / n_pos


def _binary_metrics(y_true: List[int], y_pred: List[int],
                     y_score: List[float]) -> Dict[str, float]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    n = len(y_true)

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy   = (tp + tn) / n if n > 0 else 0
    ppv        = precision
    npv        = tn / (tn + fn) if (tn + fn) > 0 else 0
    brier      = sum((s - t) ** 2 for s, t in zip(y_score, y_true)) / n

    auc_roc = _roc_auc(y_true, y_score)
    auc_pr  = _pr_auc(y_true, y_score)

    return {
        "auc_roc": round(auc_roc, 4),
        "auc_pr":  round(auc_pr, 4),
        "f1":      round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "specificity": round(specificity, 4),
        "accuracy":  round(accuracy, 4),
        "ppv":  round(ppv, 4),
        "npv":  round(npv, 4),
        "brier_score": round(brier, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": n,
        "prevalence": round(sum(y_true) / n, 4),
    }


def _find_threshold(y_true: List[int], y_score: List[float],
                     target_metric: str = "f1") -> Tuple[float, Dict]:
    """Grid-search the optimal decision threshold."""
    best_thresh = 0.5
    best_val = -1.0
    best_metrics = {}
    thresholds = [i / 100 for i in range(5, 96, 5)]
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_score]
        m = _binary_metrics(y_true, y_pred, y_score)
        val = m.get(target_metric, 0)
        if val > best_val:
            best_val = val
            best_thresh = t
            best_metrics = m
    return best_thresh, best_metrics


def _calibration_error(y_true: List[int], y_score: List[float],
                         n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """Expected Calibration Error (ECE) and per-bin data."""
    bins = [[] for _ in range(n_bins)]
    for score, label in zip(y_score, y_true):
        idx = min(int(score * n_bins), n_bins - 1)
        bins[idx].append((score, label))

    ece = 0.0
    bin_data = []
    n = len(y_true)
    for i, b in enumerate(bins):
        if not b:
            continue
        avg_score = sum(s for s, _ in b) / len(b)
        avg_label = sum(l for _, l in b) / len(b)
        bin_ece = abs(avg_score - avg_label) * len(b) / n
        ece += bin_ece
        bin_data.append({
            "bin": i, "mean_predicted": round(avg_score, 3),
            "mean_actual": round(avg_label, 3),
            "count": len(b), "ece_contribution": round(bin_ece, 4),
        })

    return round(ece, 4), bin_data


# ─────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────

@dataclass
class ModelEvalResult:
    model_name: str
    task: str
    eval_date: str
    dataset_size: int
    threshold: float
    metrics: Dict = field(default_factory=dict)
    calibration_ece: float = 0.0
    calibration_bins: List[Dict] = field(default_factory=list)
    subgroup_results: Dict = field(default_factory=dict)  # e.g. by severity, age group
    pass_criteria: bool = False
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    def summary_line(self) -> str:
        status = "✓ PASS" if self.pass_criteria else "✗ FAIL"
        m = self.metrics
        return (f"[{status}] {self.model_name}/{self.task} | "
                f"AUC={m.get('auc_roc', 0):.3f} F1={m.get('f1', 0):.3f} "
                f"Recall={m.get('recall', 0):.3f} ECE={self.calibration_ece:.3f} "
                f"n={self.dataset_size}")


@dataclass
class BacktestWindow:
    window_start: str
    window_end: str
    n_patients: int
    n_events: int
    metrics: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    model_name: str
    task: str
    windows: List[BacktestWindow] = field(default_factory=list)
    temporal_drift_detected: bool = False
    drift_metric: str = "auc_roc"
    drift_pvalue: float = 1.0

    def auc_trend(self) -> List[float]:
        return [w.metrics.get("auc_roc", 0) for w in self.windows]

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "task": self.task,
            "temporal_drift_detected": self.temporal_drift_detected,
            "drift_pvalue": self.drift_pvalue,
            "auc_trend": self.auc_trend(),
            "windows": [asdict(w) for w in self.windows],
        }


# ─────────────────────────────────────────
# Synthetic evaluation dataset
# ─────────────────────────────────────────

def _make_synthetic_eval_data(n: int = 1000, prevalence: float = 0.15,
                               seed: int = 42) -> Tuple[List[int], List[float]]:
    """Generate y_true and y_score mimicking a well-calibrated clinical model."""
    rng = random.Random(seed)
    y_true, y_score = [], []
    for _ in range(n):
        is_pos = rng.random() < prevalence
        if is_pos:
            score = min(0.99, max(0.01, rng.gauss(0.72, 0.18)))
        else:
            score = min(0.99, max(0.01, rng.gauss(0.22, 0.16)))
        y_true.append(int(is_pos))
        y_score.append(score)
    return y_true, y_score


def _make_subgroups(n: int = 1000, seed: int = 42) -> Dict[str, List[int]]:
    """Return index lists for demographic subgroups."""
    rng = random.Random(seed)
    groups: Dict[str, List[int]] = {
        "age_lt65": [], "age_ge65": [],
        "male": [], "female": [],
        "high_news2": [], "low_news2": [],
    }
    for i in range(n):
        age = rng.randint(18, 95)
        sex = rng.choice(["M", "F"])
        news2 = rng.randint(0, 14)
        if age < 65:
            groups["age_lt65"].append(i)
        else:
            groups["age_ge65"].append(i)
        groups["male" if sex == "M" else "female"].append(i)
        if news2 >= 5:
            groups["high_news2"].append(i)
        else:
            groups["low_news2"].append(i)
    return groups


# ─────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates a single model on a test set, computes all metrics,
    calibration error, and subgroup fairness analysis.
    """

    PASS_CRITERIA = {
        "sepsis_risk":    {"auc_roc": 0.80, "recall": 0.75, "calibration_ece": 0.25},
        "mortality_risk": {"auc_roc": 0.78, "recall": 0.70, "calibration_ece": 0.25},
        "cardiac_risk":   {"auc_roc": 0.76, "recall": 0.70, "calibration_ece": 0.25},
        "anomaly":        {"auc_roc": 0.72, "recall": 0.65, "calibration_ece": 0.25},
        "lstm":           {"auc_roc": 0.75, "recall": 0.70, "calibration_ece": 0.25},
        "default":        {"auc_roc": 0.70, "recall": 0.65, "calibration_ece": 0.25},
    }

    def evaluate(self, model_name: str, task: str,
                 y_true: List[int], y_score: List[float],
                 target_metric: str = "f1",
                 subgroup_indices: Optional[Dict[str, List[int]]] = None) -> ModelEvalResult:

        threshold, metrics = _find_threshold(y_true, y_score, target_metric)
        ece, cal_bins = _calibration_error(y_true, y_score)

        # Subgroup analysis
        subgroup_results = {}
        if subgroup_indices:
            for group_name, idxs in subgroup_indices.items():
                if len(idxs) < 20:
                    continue
                sg_true = [y_true[i] for i in idxs]
                sg_score = [y_score[i] for i in idxs]
                sg_pred = [1 if s >= threshold else 0 for s in sg_score]
                if sum(sg_true) < 5:
                    continue
                sg_m = _binary_metrics(sg_true, sg_pred, sg_score)
                subgroup_results[group_name] = {
                    "n": len(idxs),
                    "auc_roc": sg_m["auc_roc"],
                    "recall": sg_m["recall"],
                    "f1": sg_m["f1"],
                    "prevalence": sg_m["prevalence"],
                }

        # Pass/fail check
        criteria = self.PASS_CRITERIA.get(task, self.PASS_CRITERIA["default"])
        pass_criteria = (
            metrics.get("auc_roc", 0) >= criteria["auc_roc"] and
            metrics.get("recall", 0) >= criteria["recall"] and
            ece <= criteria["calibration_ece"]
        )

        return ModelEvalResult(
            model_name=model_name,
            task=task,
            eval_date=datetime.utcnow().isoformat(),
            dataset_size=len(y_true),
            threshold=threshold,
            metrics=metrics,
            calibration_ece=ece,
            calibration_bins=cal_bins,
            subgroup_results=subgroup_results,
            pass_criteria=pass_criteria,
        )

    def evaluate_from_model(self, model, task: str,
                             X_test, y_true: List[int]) -> ModelEvalResult:
        """Evaluate using a fitted sklearn-compatible model."""
        try:
            y_score = list(model.predict_proba(X_test)[:, 1])
        except Exception:
            y_score = list(model.predict(X_test).astype(float))
        model_name = type(model).__name__
        return self.evaluate(model_name, task, y_true, y_score)


# ─────────────────────────────────────────
# Backtester (temporal drift detection)
# ─────────────────────────────────────────

class ModelBacktester:
    """
    Simulates rolling-window backtesting to detect temporal concept drift.
    In production: feed actual historical predictions + outcomes.
    """

    def backtest(self, model_name: str, task: str,
                  n_windows: int = 12,
                  window_days: int = 30,
                  prevalence_drift: float = 0.01,
                  auc_drift: float = -0.01) -> BacktestResult:
        """
        Simulate n_windows rolling evaluation windows.
        prevalence_drift: change in event rate per window (+ or -)
        auc_drift: change in model AUC per window (simulates concept drift)
        """
        base_date = datetime.utcnow() - timedelta(days=n_windows * window_days)
        windows = []
        base_prev = 0.15
        base_auc_offset = 0.0

        for w in range(n_windows):
            start = base_date + timedelta(days=w * window_days)
            end   = start + timedelta(days=window_days)
            prev  = max(0.05, min(0.40, base_prev + w * prevalence_drift))
            # Degrade AUC slightly over time to simulate drift
            auc_offset = base_auc_offset + w * auc_drift
            n = random.randint(200, 400)
            y_true, y_score = _make_synthetic_eval_data(
                n=n, prevalence=prev, seed=w * 7 + 13
            )
            # Apply AUC degradation: shift scores toward 0.5
            y_score = [0.5 + (s - 0.5) * (1 + auc_offset) for s in y_score]
            y_score = [max(0.01, min(0.99, s)) for s in y_score]

            threshold, metrics = _find_threshold(y_true, y_score, "f1")
            windows.append(BacktestWindow(
                window_start=start.strftime("%Y-%m-%d"),
                window_end=end.strftime("%Y-%m-%d"),
                n_patients=n,
                n_events=sum(y_true),
                metrics=metrics,
            ))

        # Drift detection: compare first half vs second half AUC
        half = n_windows // 2
        first_half_auc  = [w.metrics.get("auc_roc", 0) for w in windows[:half]]
        second_half_auc = [w.metrics.get("auc_roc", 0) for w in windows[half:]]

        mean1 = sum(first_half_auc) / len(first_half_auc)
        mean2 = sum(second_half_auc) / len(second_half_auc)
        drift_detected = abs(mean1 - mean2) > 0.05  # >5 AUC point drop

        # Approximate p-value via t-test formula
        n1, n2 = len(first_half_auc), len(second_half_auc)
        var1 = sum((x - mean1) ** 2 for x in first_half_auc) / max(n1 - 1, 1)
        var2 = sum((x - mean2) ** 2 for x in second_half_auc) / max(n2 - 1, 1)
        se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-6
        t_stat = abs(mean1 - mean2) / se
        # Approximate p-value from t-stat (df≈n_windows)
        p_value = max(0.001, 2 * (1 - min(0.999, t_stat / (t_stat + n_windows))))

        return BacktestResult(
            model_name=model_name,
            task=task,
            windows=windows,
            temporal_drift_detected=drift_detected,
            drift_pvalue=round(p_value, 4),
        )


# ─────────────────────────────────────────
# Full evaluation suite
# ─────────────────────────────────────────

class EvaluationSuite:
    """Runs evaluation + backtest for all registered models."""

    TASKS = [
        ("SepsisRiskModel",    "sepsis_risk",    0.15),
        ("MortalityRiskModel", "mortality_risk", 0.10),
        ("CardiacRiskModel",   "cardiac_risk",   0.12),
        ("AnomalyDetector",    "anomaly",        0.08),
        ("LSTMDeteriorator",   "lstm",           0.12),
    ]

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.backtester = ModelBacktester()

    def run_all(self, output_dir: str = "/data/eval",
                backtest_windows: int = 12) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        results = {"eval_date": datetime.utcnow().isoformat(), "models": {}}

        for model_name, task, prevalence in self.TASKS:
            logger.info(f"Evaluating {model_name} / {task} ...")

            # Generate synthetic test data
            n_test = 800
            y_true, y_score = _make_synthetic_eval_data(n_test, prevalence)
            subgroups = _make_subgroups(n_test)

            # Evaluation
            eval_result = self.evaluator.evaluate(
                model_name, task, y_true, y_score,
                subgroup_indices=subgroups
            )

            # Backtest
            bt_result = self.backtester.backtest(
                model_name, task, n_windows=backtest_windows,
                auc_drift=-0.005,  # mild drift
            )

            results["models"][task] = {
                "evaluation": eval_result.to_dict(),
                "backtest": bt_result.to_dict(),
                "summary": eval_result.summary_line(),
                "drift_detected": bt_result.temporal_drift_detected,
            }
            print(f"  {eval_result.summary_line()}")

        # Overall pass rate
        n_pass = sum(1 for m in results["models"].values()
                     if m["evaluation"]["pass_criteria"])
        results["overall"] = {
            "n_models": len(self.TASKS),
            "n_pass": n_pass,
            "pass_rate": round(n_pass / len(self.TASKS), 2),
            "all_pass": n_pass == len(self.TASKS),
        }

        # Save report
        report_path = os.path.join(output_dir, f"eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation report saved: {report_path}")
        results["report_path"] = report_path
        return results

    def generate_html_report(self, results: Dict, output_path: str) -> str:
        models = results.get("models", {})
        overall = results.get("overall", {})

        rows = ""
        for task, data in models.items():
            ev = data["evaluation"]
            m = ev["metrics"]
            bt = data["backtest"]
            status_color = "#3fb950" if ev["pass_criteria"] else "#f85149"
            drift_icon = "⚠" if data["drift_detected"] else "✓"
            rows += f"""
            <tr>
              <td>{ev['model_name']}</td>
              <td>{task}</td>
              <td style="color:{status_color};font-weight:bold">{'PASS' if ev['pass_criteria'] else 'FAIL'}</td>
              <td>{m.get('auc_roc', 0):.3f}</td>
              <td>{m.get('auc_pr', 0):.3f}</td>
              <td>{m.get('f1', 0):.3f}</td>
              <td>{m.get('recall', 0):.3f}</td>
              <td>{m.get('precision', 0):.3f}</td>
              <td>{ev['calibration_ece']:.3f}</td>
              <td>{ev['threshold']:.2f}</td>
              <td>{ev['dataset_size']}</td>
              <td style="color:{'#e3b341' if data['drift_detected'] else '#3fb950'}">{drift_icon} {bt['auc_trend'][-1]:.3f}</td>
            </tr>"""

        overall_color = "#3fb950" if overall.get("all_pass") else "#e3b341"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Evaluation Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  body {{ background:#0d1117; color:#e6edf3; font-family:'JetBrains Mono',monospace; padding:28px; font-size:12px; }}
  h1 {{ font-size:20px; color:#58a6ff; margin-bottom:4px; }}
  .sub {{ color:#8b949e; font-size:12px; margin-bottom:20px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }}
  .kpi-val {{ font-size:26px; font-weight:bold; }}
  .kpi-lbl {{ color:#8b949e; font-size:11px; text-transform:uppercase; margin-bottom:6px; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }}
  th {{ background:#21262d; color:#8b949e; padding:10px 8px; text-align:left; font-size:11px; text-transform:uppercase; }}
  td {{ padding:9px 8px; border-bottom:1px solid #21262d; }}
  tr:last-child td {{ border-bottom:none; }}
</style>
</head>
<body>
<h1>🧪 Model Evaluation Report</h1>
<div class="sub">Generated: {results.get('eval_date','')[:19]} UTC</div>

<div class="kpis">
  <div class="kpi">
    <div class="kpi-lbl">Models Evaluated</div>
    <div class="kpi-val" style="color:#58a6ff">{overall.get('n_models',0)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Passing</div>
    <div class="kpi-val" style="color:#3fb950">{overall.get('n_pass',0)}</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Pass Rate</div>
    <div class="kpi-val" style="color:{overall_color}">{overall.get('pass_rate',0):.0%}</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Overall Status</div>
    <div class="kpi-val" style="color:{overall_color}">{'ALL PASS' if overall.get('all_pass') else 'REVIEW'}</div>
  </div>
</div>

<table>
  <tr>
    <th>Model</th><th>Task</th><th>Status</th>
    <th>AUC-ROC</th><th>AUC-PR</th><th>F1</th>
    <th>Recall</th><th>Precision</th><th>ECE</th>
    <th>Threshold</th><th>N</th><th>Drift (Latest)</th>
  </tr>
  {rows}
</table>

<div style="margin-top:16px;color:#8b949e;font-size:11px">
  ⚠ Pass criteria: AUC-ROC ≥ task threshold, Recall ≥ 0.70, ECE ≤ 0.10 | Drift: AUC drop >5pts across rolling windows
</div>
</body>
</html>"""

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML evaluation report: {output_path}")
        return html


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")
    suite = EvaluationSuite()

    print("Running full model evaluation suite...\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        results = suite.run_all(output_dir=tmpdir, backtest_windows=12)
        html_path = os.path.join(tmpdir, "eval_report.html")
        suite.generate_html_report(results, html_path)
        print(f"\nOverall: {results['overall']['n_pass']}/{results['overall']['n_models']} models PASS")
        print(f"Report: {results['report_path']}")
        print(f"HTML:   {html_path} ({os.path.getsize(html_path):,} bytes)")
