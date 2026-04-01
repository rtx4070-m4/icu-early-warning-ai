"""
Survival Analysis Module
ICU patient outcome modelling using:
  - Kaplan-Meier survival curves (non-parametric)
  - Cox Proportional Hazards (parametric, covariate-adjusted)
  - Competing risks (death vs discharge vs transfer)
  - Time-to-event prediction for ICU length-of-stay

All implemented from scratch — no lifelines/statsmodels required.
Optional: enhanced curves if matplotlib is available.
"""

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────

@dataclass
class SurvivalObservation:
    patient_id: str
    time_days: float        # Follow-up duration
    event: int              # 1 = event (death/discharge), 0 = censored
    event_type: str         # "died" | "discharged" | "transferred" | "censored"
    covariates: Dict[str, float] = field(default_factory=dict)


@dataclass
class KMPoint:
    time: float
    survival: float
    n_at_risk: int
    n_events: int
    n_censored: int
    ci_lower: float = 0.0
    ci_upper: float = 1.0


@dataclass
class SurvivalResult:
    method: str
    group_label: str
    km_curve: List[KMPoint]
    median_survival_days: Optional[float]
    survival_at_7d: float
    survival_at_14d: float
    survival_at_30d: float
    log_rank_p: Optional[float] = None   # vs reference group
    n_total: int = 0
    n_events: int = 0

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "group": self.group_label,
            "n_total": self.n_total,
            "n_events": self.n_events,
            "median_survival_days": self.median_survival_days,
            "survival_at_7d": round(self.survival_at_7d, 4),
            "survival_at_14d": round(self.survival_at_14d, 4),
            "survival_at_30d": round(self.survival_at_30d, 4),
            "log_rank_p": self.log_rank_p,
            "km_points": len(self.km_curve),
        }

    def summary(self) -> str:
        med = f"{self.median_survival_days:.1f}d" if self.median_survival_days else "NR"
        return (f"{self.group_label}: n={self.n_total} events={self.n_events} "
                f"median={med} 7d={self.survival_at_7d:.0%} "
                f"30d={self.survival_at_30d:.0%}")


# ─────────────────────────────────────────
# Kaplan-Meier estimator
# ─────────────────────────────────────────

class KaplanMeier:
    """
    Non-parametric Kaplan-Meier survival estimator.
    Computes S(t) = Π(1 - d_i/n_i) with Greenwood variance for CIs.
    """

    def __init__(self, observations: List[SurvivalObservation]):
        self.obs = observations
        self._curve: Optional[List[KMPoint]] = None

    def fit(self) -> "KaplanMeier":
        times = sorted(set(o.time_days for o in self.obs if o.event == 1))
        n_total = len(self.obs)

        points: List[KMPoint] = [
            KMPoint(time=0.0, survival=1.0, n_at_risk=n_total,
                     n_events=0, n_censored=0, ci_lower=1.0, ci_upper=1.0)
        ]

        survival = 1.0
        greenwood_sum = 0.0
        n_at_risk = n_total

        for t in times:
            # Events at this time
            d_t = sum(1 for o in self.obs if o.event == 1 and abs(o.time_days - t) < 1e-9)
            # Censorings strictly before this time (already removed from risk set)
            c_prev = sum(1 for o in self.obs if o.event == 0 and o.time_days < t)
            # Actual at-risk: those with time >= t
            n_at_risk = sum(1 for o in self.obs if o.time_days >= t)

            if n_at_risk > 0 and d_t > 0:
                survival *= (1 - d_t / n_at_risk)
                greenwood_sum += d_t / (n_at_risk * (n_at_risk - d_t + 1e-9))

            # Greenwood 95% CI (log-log scale)
            if survival > 0 and greenwood_sum > 0:
                log_log_s = math.log(-math.log(survival + 1e-9))
                se_log_log = math.sqrt(greenwood_sum) / abs(math.log(survival + 1e-9))
                z = 1.96
                ci_lo = math.exp(-math.exp(log_log_s + z * se_log_log))
                ci_hi = math.exp(-math.exp(log_log_s - z * se_log_log))
            else:
                ci_lo = max(0, survival - 0.1)
                ci_hi = min(1, survival + 0.1)

            # Censorings at this exact time
            c_t = sum(1 for o in self.obs if o.event == 0 and abs(o.time_days - t) < 1e-9)
            points.append(KMPoint(
                time=t, survival=max(0, survival), n_at_risk=n_at_risk,
                n_events=d_t, n_censored=c_t,
                ci_lower=max(0, ci_lo), ci_upper=min(1, ci_hi),
            ))

        self._curve = points
        return self

    def survival_at(self, t: float) -> float:
        """Interpolate S(t) at an arbitrary time."""
        if not self._curve:
            self.fit()
        last = 1.0
        for p in self._curve:
            if p.time > t:
                break
            last = p.survival
        return last

    def median_survival(self) -> Optional[float]:
        """Return the time at which S(t) first crosses 0.5."""
        if not self._curve:
            self.fit()
        for p in self._curve:
            if p.survival <= 0.5:
                return p.time
        return None  # Median not reached

    @property
    def curve(self) -> List[KMPoint]:
        if not self._curve:
            self.fit()
        return self._curve

    def to_result(self, group_label: str = "All") -> SurvivalResult:
        self.fit()
        return SurvivalResult(
            method="Kaplan-Meier",
            group_label=group_label,
            km_curve=self.curve,
            median_survival_days=self.median_survival(),
            survival_at_7d=self.survival_at(7),
            survival_at_14d=self.survival_at(14),
            survival_at_30d=self.survival_at(30),
            n_total=len(self.obs),
            n_events=sum(o.event for o in self.obs),
        )


# ─────────────────────────────────────────
# Log-rank test
# ─────────────────────────────────────────

def log_rank_test(group1: List[SurvivalObservation],
                   group2: List[SurvivalObservation]) -> Tuple[float, float]:
    """
    Log-rank test comparing two survival curves.
    Returns (test_statistic, p_value).
    """
    all_obs = group1 + group2
    event_times = sorted(set(o.time_days for o in all_obs if o.event == 1))

    O1_total = E1_total = 0.0
    O2_total = E2_total = 0.0
    V_total = 0.0

    for t in event_times:
        n1 = sum(1 for o in group1 if o.time_days >= t)
        n2 = sum(1 for o in group2 if o.time_days >= t)
        d1 = sum(1 for o in group1 if o.event == 1 and abs(o.time_days - t) < 1e-9)
        d2 = sum(1 for o in group2 if o.event == 1 and abs(o.time_days - t) < 1e-9)
        N  = n1 + n2
        d  = d1 + d2

        if N < 2:
            continue

        E1 = d * n1 / N
        E2 = d * n2 / N
        O1_total += d1
        E1_total += E1
        O2_total += d2
        E2_total += E2

        V = d * n1 * n2 * (N - d) / (N ** 2 * (N - 1) + 1e-9)
        V_total += V

    if V_total < 1e-9:
        return 0.0, 1.0

    chi2 = (O1_total - E1_total) ** 2 / V_total
    # Chi-squared with 1 df → p-value approximation
    p_value = math.exp(-0.5 * chi2) * (1 + chi2 * 0.5 + chi2**2 * 0.125)
    p_value = max(0.0001, min(1.0, p_value))
    return round(chi2, 4), round(p_value, 4)


# ─────────────────────────────────────────
# Cox Proportional Hazards (simplified)
# ─────────────────────────────────────────

class CoxPH:
    """
    Simplified Cox PH model using Newton-Raphson partial likelihood.
    Handles right-censored data; estimates hazard ratios for covariates.
    """

    def __init__(self, covariates: List[str], lr: float = 0.01, max_iter: int = 100):
        self.covariates = covariates
        self.lr = lr
        self.max_iter = max_iter
        self.betas: Optional[np.ndarray] = None
        self.hazard_ratios: Optional[Dict[str, float]] = None
        self.baseline_hazard: Dict[float, float] = {}
        self.fitted = False

    def _get_X(self, obs: List[SurvivalObservation]) -> np.ndarray:
        X = []
        for o in obs:
            row = [o.covariates.get(c, 0.0) for c in self.covariates]
            X.append(row)
        X = np.array(X, dtype=float)
        # Standardise
        self._mean = X.mean(0)
        self._std = X.std(0) + 1e-9
        return (X - self._mean) / self._std

    def fit(self, obs: List[SurvivalObservation]) -> "CoxPH":
        """Fit model via gradient ascent of partial log-likelihood."""
        self.obs = obs
        X = self._get_X(obs)
        n, p = X.shape
        betas = np.zeros(p)
        times = np.array([o.time_days for o in obs])
        events = np.array([o.event for o in obs])

        for iteration in range(self.max_iter):
            grad = np.zeros(p)
            hess = np.zeros((p, p))

            lin_pred = X @ betas
            exp_lp = np.exp(lin_pred - lin_pred.max())  # numerical stability

            for i in range(n):
                if events[i] == 0:
                    continue
                t_i = times[i]
                # Risk set: all j with time >= t_i
                risk_mask = times >= t_i
                risk_exp = exp_lp[risk_mask]
                risk_X = X[risk_mask]

                denom = risk_exp.sum() + 1e-9
                w = risk_exp / denom                         # weights
                mean_X = (w[:, None] * risk_X).sum(0)       # weighted mean of X
                mean_X2 = (w[:, None] * risk_X ** 2).sum(0) # weighted mean of X^2

                grad += X[i] - mean_X
                hess -= np.diag(mean_X2 - mean_X ** 2)

            # Newton step (using only diagonal of Hessian for stability)
            diag = np.diag(hess)
            safe_diag = np.where(np.abs(diag) > 1e-9, diag, -1.0)
            step = -grad / safe_diag
            betas += self.lr * np.clip(step, -1.0, 1.0)

            if np.linalg.norm(grad) < 1e-4:
                logger.debug(f"Cox PH converged at iteration {iteration}")
                break

        self.betas = betas
        self.hazard_ratios = {
            c: round(float(math.exp(b)), 4)
            for c, b in zip(self.covariates, betas)
        }
        self._compute_baseline_hazard(obs, X, betas)
        self.fitted = True
        return self

    def _compute_baseline_hazard(self, obs, X, betas):
        """Nelson-Aalen baseline cumulative hazard."""
        times = np.array([o.time_days for o in obs])
        events = np.array([o.event for o in obs])
        exp_lp = np.exp(X @ betas)
        event_times = sorted(set(times[events == 1]))
        H0 = 0.0
        for t in event_times:
            risk = exp_lp[times >= t].sum()
            d = (events[times == t]).sum()
            if risk > 0:
                H0 += d / risk
            self.baseline_hazard[t] = H0

    def predict_survival(self, covariates: Dict[str, float],
                          time_points: List[float]) -> Dict[float, float]:
        """Predict S(t) for a new patient."""
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        x = np.array([covariates.get(c, 0.0) for c in self.covariates])
        x_std = (x - self._mean) / self._std
        lp = float(x_std @ self.betas)
        exp_lp = math.exp(lp)

        bt_times = sorted(self.baseline_hazard.keys())
        result = {}
        for t in time_points:
            # Cumulative baseline hazard at t
            H0 = 0.0
            for bt in bt_times:
                if bt <= t:
                    H0 = self.baseline_hazard[bt]
                else:
                    break
            S_t = math.exp(-H0 * exp_lp)
            result[t] = round(max(0, min(1, S_t)), 4)
        return result

    def summary(self) -> Dict:
        if not self.fitted:
            return {}
        return {
            "covariates": self.covariates,
            "hazard_ratios": self.hazard_ratios,
            "interpretation": {
                c: f"HR={hr:.2f} ({'increases' if hr > 1 else 'decreases'} hazard "
                   f"by {abs(hr-1)*100:.0f}%)"
                for c, hr in (self.hazard_ratios or {}).items()
            }
        }


# ─────────────────────────────────────────
# Competing risks
# ─────────────────────────────────────────

def cause_specific_curves(obs: List[SurvivalObservation],
                            causes: List[str] = None) -> Dict[str, SurvivalResult]:
    """
    Compute cause-specific KM curves for competing events.
    Each cause is treated as the event of interest; others are censored.
    """
    causes = causes or list({o.event_type for o in obs if o.event == 1})
    results = {}
    for cause in causes:
        cause_obs = [
            SurvivalObservation(
                patient_id=o.patient_id,
                time_days=o.time_days,
                event=1 if (o.event == 1 and o.event_type == cause) else 0,
                event_type=o.event_type,
                covariates=o.covariates,
            )
            for o in obs
        ]
        km = KaplanMeier(cause_obs)
        results[cause] = km.to_result(group_label=cause)
    return results


# ─────────────────────────────────────────
# Synthetic cohort generator
# ─────────────────────────────────────────

def generate_synthetic_cohort(n: int = 200, seed: int = 42) -> List[SurvivalObservation]:
    """Generate a realistic synthetic ICU survival cohort."""
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        severity = rng.random()  # 0 = mild, 1 = critical
        # Higher severity → shorter time, higher mortality
        shape = 1.5
        scale = max(1.0, 15 - severity * 12)
        time = min(90, max(0.5, rng.weibullvariate(scale, shape)))

        # Mortality probability scales with severity
        mort_prob = 0.05 + severity * 0.55
        if rng.random() < mort_prob:
            event, event_type = 1, "died"
        elif rng.random() < 0.7:
            event, event_type = 1, "discharged"
        else:
            event, event_type = 1, "transferred"

        # Censored if time > 30 days and not died
        if time > 30 and event_type != "died":
            time = 30
            event, event_type = 0, "censored"

        covariates = {
            "age":           rng.gauss(65, 15),
            "news2":         severity * 12,
            "lactate":       1.0 + severity * 5,
            "creatinine":    0.9 + severity * 3,
            "sofa_score":    severity * 10,
            "sepsis_risk":   severity,
            "ventilated":    1.0 if severity > 0.6 else 0.0,
            "vasopressors":  1.0 if severity > 0.5 else 0.0,
        }
        obs.append(SurvivalObservation(
            patient_id=f"SIM_{i:04d}",
            time_days=round(time, 2),
            event=event,
            event_type=event_type,
            covariates=covariates,
        ))
    return obs


# ─────────────────────────────────────────
# Analysis runner
# ─────────────────────────────────────────

class SurvivalAnalyzer:
    """High-level survival analysis interface."""

    def __init__(self, observations: List[SurvivalObservation]):
        self.obs = observations

    def overall_survival(self) -> SurvivalResult:
        """Overall KM survival curve for all patients."""
        km = KaplanMeier(self.obs)
        return km.to_result("All patients")

    def survival_by_group(self, grouper: callable) -> Dict[str, SurvivalResult]:
        """Stratified KM curves by a grouping function."""
        groups: Dict[str, List] = {}
        for o in self.obs:
            key = grouper(o)
            groups.setdefault(key, []).append(o)
        results = {}
        for label, group_obs in sorted(groups.items()):
            km = KaplanMeier(group_obs)
            result = km.to_result(label)
            results[label] = result
        # Add log-rank p-values between pairs
        group_list = list(results.items())
        if len(group_list) == 2:
            g1_obs = [o for o in self.obs if grouper(o) == group_list[0][0]]
            g2_obs = [o for o in self.obs if grouper(o) == group_list[1][0]]
            _, p = log_rank_test(g1_obs, g2_obs)
            group_list[0][1].log_rank_p = p
            group_list[1][1].log_rank_p = p
        return results

    def cox_analysis(self, covariates: List[str]) -> Dict:
        """Fit Cox PH model and return hazard ratios."""
        cox = CoxPH(covariates)
        cox.fit(self.obs)
        return cox.summary()

    def competing_risks(self) -> Dict[str, SurvivalResult]:
        """Cause-specific survival curves."""
        return cause_specific_curves(self.obs)

    def predict_patient(self, covariates: Dict[str, float],
                          time_points: Optional[List[float]] = None) -> Dict:
        """Predict survival probability for a new patient."""
        time_points = time_points or [1, 3, 7, 14, 30]
        feat_names = ["age", "news2", "lactate", "creatinine",
                       "sofa_score", "sepsis_risk", "ventilated"]
        cox = CoxPH(feat_names)
        cox.fit(self.obs)
        survival = cox.predict_survival(covariates, time_points)
        return {
            "predicted_survival": survival,
            "predicted_icu_los_days": next(
                (t for t, s in sorted(survival.items()) if s < 0.5), None
            ),
            "hazard_ratios": cox.hazard_ratios,
        }

    def full_report(self) -> Dict:
        """Run all analyses and return a structured report."""
        overall = self.overall_survival()
        by_severity = self.survival_by_group(
            lambda o: "HIGH" if o.covariates.get("sepsis_risk", 0) > 0.5 else "LOW"
        )
        competing = self.competing_risks()
        cox = self.cox_analysis(["age", "news2", "lactate", "sofa_score",
                                   "ventilated", "vasopressors"])
        return {
            "overall": overall.to_dict(),
            "by_sepsis_risk": {k: v.to_dict() for k, v in by_severity.items()},
            "competing_risks": {k: v.to_dict() for k, v in competing.items()},
            "cox_model": cox,
            "n_total": len(self.obs),
            "n_events": sum(o.event for o in self.obs),
            "event_rate": round(sum(o.event for o in self.obs) / len(self.obs), 3),
        }


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Survival Analysis Demo ===\n")

    obs = generate_synthetic_cohort(n=300, seed=42)
    analyzer = SurvivalAnalyzer(obs)

    # Overall survival
    overall = analyzer.overall_survival()
    print(f"Overall: {overall.summary()}")

    # Stratified by sepsis risk
    print("\nBy sepsis risk level:")
    by_group = analyzer.survival_by_group(
        lambda o: "High risk (>50%)" if o.covariates.get("sepsis_risk", 0) > 0.5
                   else "Low risk (≤50%)"
    )
    for label, result in by_group.items():
        p = f" log-rank p={result.log_rank_p}" if result.log_rank_p else ""
        print(f"  {result.summary()}{p}")

    # Competing risks
    print("\nCompeting risks:")
    competing = analyzer.competing_risks()
    for cause, result in competing.items():
        print(f"  {cause:<15} 30d cumulative incidence: {1-result.survival_at_30d:.0%}")

    # Cox PH
    print("\nCox PH hazard ratios:")
    cox = analyzer.cox_analysis(["age", "news2", "lactate", "sofa_score", "ventilated"])
    for cov, hr in cox["hazard_ratios"].items():
        arrow = "↑" if hr > 1 else "↓"
        print(f"  {cov:<15} HR={hr:.3f} {arrow}")

    # Individual patient prediction
    print("\nPatient prediction (NEWS2=9, lactate=3.9, sofa=7):")
    pred = analyzer.predict_patient({
        "age": 72, "news2": 9, "lactate": 3.9, "creatinine": 2.1,
        "sofa_score": 7.5, "sepsis_risk": 0.78,
        "ventilated": 0, "vasopressors": 1,
    })
    for t, s in pred["predicted_survival"].items():
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  Day {t:>2}: {bar} {s:.0%}")
