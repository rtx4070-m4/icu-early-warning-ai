"""
Patient Deterioration Simulation Engine
Generates medically realistic ICU patient trajectories for:
  - Model stress-testing and validation
  - Alert system load testing
  - Training data augmentation
  - Demo and education

Models: septic shock progression, respiratory failure, cardiogenic shock,
        AKI cascade, post-op complications, and recovery curves.
"""

import math
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Iterator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Trajectory building blocks
# ─────────────────────────────────────────

@dataclass
class PhysiologyState:
    """Complete physiological state at one time point."""
    time_hours: float
    heart_rate: float
    sbp: float
    dbp: float
    respiratory_rate: float
    temperature: float
    spo2: float
    gcs: int
    # Labs (updated less frequently)
    lactate: float = 1.0
    creatinine: float = 1.0
    wbc: float = 8.0
    potassium: float = 4.0
    glucose: float = 100.0
    procalcitonin: float = 0.1
    # Derived
    map: float = 0.0
    shock_index: float = 0.0
    news2: int = 0

    def __post_init__(self):
        self.map = round((self.sbp + 2 * self.dbp) / 3, 1)
        self.shock_index = round(self.heart_rate / max(self.sbp, 1), 3)
        self.news2 = _compute_news2(self)

    def to_dict(self) -> Dict:
        return {
            "time_hours": round(self.time_hours, 2),
            "heart_rate": round(self.heart_rate, 1),
            "sbp": round(self.sbp, 1),
            "dbp": round(self.dbp, 1),
            "map": self.map,
            "respiratory_rate": round(self.respiratory_rate, 1),
            "temperature": round(self.temperature, 2),
            "spo2": round(self.spo2, 1),
            "gcs": self.gcs,
            "lactate": round(self.lactate, 2),
            "creatinine": round(self.creatinine, 2),
            "wbc": round(self.wbc, 1),
            "potassium": round(self.potassium, 1),
            "glucose": round(self.glucose, 0),
            "procalcitonin": round(self.procalcitonin, 2),
            "shock_index": self.shock_index,
            "news2": self.news2,
        }


def _compute_news2(s: PhysiologyState) -> int:
    score = 0
    # RR
    rr = s.respiratory_rate
    if rr <= 8 or rr >= 25: score += 3
    elif 21 <= rr <= 24: score += 2
    elif 9 <= rr <= 11: score += 1
    # SpO2
    spo2 = s.spo2
    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1
    # SBP
    sbp = s.sbp
    if sbp <= 90 or sbp >= 220: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    # HR
    hr = s.heart_rate
    if hr <= 40 or hr >= 131: score += 3
    elif 111 <= hr <= 130: score += 2
    elif hr <= 50 or 91 <= hr <= 110: score += 1
    # Temp
    t = s.temperature
    if t <= 35.0: score += 3
    elif t <= 36.0 or t >= 39.1: score += 1
    elif 38.1 <= t <= 39.0: score += 1
    # GCS
    if s.gcs < 15: score += 3
    return score


# ─────────────────────────────────────────
# Noise model
# ─────────────────────────────────────────

class PhysiologicalNoise:
    """Adds realistic measurement noise and physiological variability."""

    NOISE = {
        "heart_rate": 4.0, "sbp": 6.0, "dbp": 4.0,
        "respiratory_rate": 1.5, "temperature": 0.15,
        "spo2": 0.8, "lactate": 0.15, "creatinine": 0.08,
        "wbc": 0.8, "potassium": 0.12, "glucose": 8.0,
        "procalcitonin": 0.05,
    }

    def apply(self, state: PhysiologyState, rng: random.Random) -> PhysiologyState:
        """Return a copy of state with noise applied."""
        import copy
        s = copy.copy(state)
        for field, std in self.NOISE.items():
            val = getattr(s, field)
            noisy = val + rng.gauss(0, std)
            # Apply clinical floor/ceiling
            noisy = max(self._floor(field), min(self._ceiling(field), noisy))
            setattr(s, field, round(noisy, 2))
        s.map = round((s.sbp + 2 * s.dbp) / 3, 1)
        s.shock_index = round(s.heart_rate / max(s.sbp, 1), 3)
        s.news2 = _compute_news2(s)
        return s

    def _floor(self, field: str) -> float:
        floors = {"heart_rate": 20, "sbp": 40, "dbp": 20, "respiratory_rate": 4,
                   "temperature": 33.0, "spo2": 60, "lactate": 0.3,
                   "creatinine": 0.3, "wbc": 0.5, "potassium": 1.5,
                   "glucose": 30, "procalcitonin": 0.0}
        return floors.get(field, 0)

    def _ceiling(self, field: str) -> float:
        ceilings = {"heart_rate": 300, "sbp": 300, "dbp": 180,
                     "respiratory_rate": 60, "temperature": 43.0,
                     "spo2": 100, "lactate": 30, "creatinine": 25,
                     "wbc": 100, "potassium": 9.0, "glucose": 1500,
                     "procalcitonin": 1000}
        return ceilings.get(field, 1e9)


# ─────────────────────────────────────────
# Sigmoid & curve helpers
# ─────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _linear(t: float, t0: float, t1: float,
             v0: float, v1: float) -> float:
    """Linearly interpolate between v0 at t0 and v1 at t1."""
    if t <= t0: return v0
    if t >= t1: return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

def _smooth_step(t: float, t0: float, t1: float,
                  v0: float, v1: float) -> float:
    """Smooth sigmoid interpolation."""
    if t <= t0: return v0
    if t >= t1: return v1
    progress = (t - t0) / (t1 - t0)
    smooth = 3 * progress ** 2 - 2 * progress ** 3
    return v0 + (v1 - v0) * smooth


# ─────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────

@dataclass
class ScenarioPhase:
    """One phase of a clinical scenario (e.g., 'pre-sepsis', 'fulminant shock')."""
    name: str
    duration_hours: float
    target: Dict[str, float]   # physiological targets at end of phase
    transition: str = "smooth"  # linear | smooth | instant


SCENARIOS: Dict[str, List[ScenarioPhase]] = {

    "septic_shock_untreated": [
        ScenarioPhase("early_infection", 6,
            {"heart_rate": 100, "sbp": 115, "dbp": 70, "respiratory_rate": 20,
             "temperature": 38.5, "spo2": 95, "gcs": 15,
             "lactate": 2.0, "wbc": 14, "procalcitonin": 2.0}),
        ScenarioPhase("sepsis_onset", 8,
            {"heart_rate": 120, "sbp": 95, "dbp": 58, "respiratory_rate": 25,
             "temperature": 39.2, "spo2": 91, "gcs": 14,
             "lactate": 3.5, "wbc": 20, "procalcitonin": 15}),
        ScenarioPhase("septic_shock", 10,
            {"heart_rate": 138, "sbp": 72, "dbp": 44, "respiratory_rate": 30,
             "temperature": 39.5, "spo2": 85, "gcs": 11,
             "lactate": 6.5, "wbc": 25, "procalcitonin": 45,
             "creatinine": 3.2, "potassium": 5.8}),
        ScenarioPhase("decompensation", 6,
            {"heart_rate": 155, "sbp": 58, "dbp": 35, "respiratory_rate": 35,
             "temperature": 40.1, "spo2": 76, "gcs": 7,
             "lactate": 12.0, "creatinine": 5.5, "potassium": 6.8}),
    ],

    "septic_shock_treated": [
        ScenarioPhase("early_infection", 4,
            {"heart_rate": 102, "sbp": 110, "dbp": 68, "respiratory_rate": 21,
             "temperature": 38.5, "spo2": 94, "gcs": 15,
             "lactate": 2.2, "wbc": 15, "procalcitonin": 3.0}),
        ScenarioPhase("initial_deterioration", 4,
            {"heart_rate": 122, "sbp": 88, "dbp": 55, "respiratory_rate": 26,
             "temperature": 39.0, "spo2": 90, "gcs": 14,
             "lactate": 4.0, "wbc": 22, "procalcitonin": 20}),
        ScenarioPhase("treatment_response", 12,
            {"heart_rate": 105, "sbp": 100, "dbp": 63, "respiratory_rate": 22,
             "temperature": 38.2, "spo2": 94, "gcs": 15,
             "lactate": 2.5, "creatinine": 1.8, "procalcitonin": 10}),
        ScenarioPhase("stabilization", 12,
            {"heart_rate": 92, "sbp": 112, "dbp": 70, "respiratory_rate": 18,
             "temperature": 37.8, "spo2": 96, "gcs": 15,
             "lactate": 1.5, "creatinine": 1.3, "procalcitonin": 4}),
        ScenarioPhase("recovery", 20,
            {"heart_rate": 82, "sbp": 122, "dbp": 76, "respiratory_rate": 16,
             "temperature": 37.2, "spo2": 97, "gcs": 15,
             "lactate": 1.0, "creatinine": 1.0, "wbc": 10, "procalcitonin": 0.5}),
    ],

    "respiratory_failure": [
        ScenarioPhase("mild_distress", 8,
            {"heart_rate": 95, "sbp": 130, "dbp": 80, "respiratory_rate": 22,
             "temperature": 37.5, "spo2": 94, "gcs": 15,
             "lactate": 1.5, "wbc": 12}),
        ScenarioPhase("moderate_distress", 6,
            {"heart_rate": 110, "sbp": 125, "dbp": 78, "respiratory_rate": 26,
             "temperature": 38.0, "spo2": 89, "gcs": 15,
             "lactate": 2.0}),
        ScenarioPhase("severe_distress", 4,
            {"heart_rate": 128, "sbp": 118, "dbp": 72, "respiratory_rate": 32,
             "temperature": 38.3, "spo2": 82, "gcs": 14,
             "lactate": 3.0}),
        ScenarioPhase("intubation_stabilized", 16,
            {"heart_rate": 88, "sbp": 115, "dbp": 70, "respiratory_rate": 14,
             "temperature": 37.8, "spo2": 96, "gcs": 12,
             "lactate": 1.8}),
    ],

    "cardiogenic_shock": [
        ScenarioPhase("early_mi", 3,
            {"heart_rate": 88, "sbp": 125, "dbp": 78, "respiratory_rate": 20,
             "temperature": 37.1, "spo2": 94, "gcs": 15,
             "lactate": 1.5}),
        ScenarioPhase("pump_failure", 6,
            {"heart_rate": 115, "sbp": 88, "dbp": 58, "respiratory_rate": 26,
             "temperature": 37.0, "spo2": 88, "gcs": 14,
             "lactate": 3.5, "creatinine": 2.0}),
        ScenarioPhase("cardiogenic_shock", 8,
            {"heart_rate": 130, "sbp": 70, "dbp": 45, "respiratory_rate": 30,
             "temperature": 36.5, "spo2": 82, "gcs": 12,
             "lactate": 6.0, "creatinine": 3.5, "potassium": 5.5}),
        ScenarioPhase("intervention_response", 16,
            {"heart_rate": 92, "sbp": 105, "dbp": 65, "respiratory_rate": 18,
             "temperature": 37.2, "spo2": 94, "gcs": 14,
             "lactate": 2.2, "creatinine": 2.0}),
    ],

    "aki_progression": [
        ScenarioPhase("baseline", 12,
            {"heart_rate": 85, "sbp": 128, "dbp": 78, "respiratory_rate": 16,
             "temperature": 37.0, "spo2": 97, "gcs": 15,
             "lactate": 1.0, "creatinine": 1.2}),
        ScenarioPhase("aki_stage1", 12,
            {"heart_rate": 92, "sbp": 120, "dbp": 74, "respiratory_rate": 18,
             "temperature": 37.3, "spo2": 96, "gcs": 15,
             "lactate": 1.4, "creatinine": 2.0, "potassium": 4.8}),
        ScenarioPhase("aki_stage2", 12,
            {"heart_rate": 100, "sbp": 112, "dbp": 70, "respiratory_rate": 20,
             "temperature": 37.5, "spo2": 95, "gcs": 14,
             "lactate": 1.8, "creatinine": 3.5, "potassium": 5.5}),
        ScenarioPhase("aki_stage3", 16,
            {"heart_rate": 108, "sbp": 105, "dbp": 65, "respiratory_rate": 22,
             "temperature": 37.8, "spo2": 93, "gcs": 13,
             "lactate": 2.5, "creatinine": 6.5, "potassium": 6.2}),
    ],

    "post_op_recovery": [
        ScenarioPhase("immediate_post_op", 4,
            {"heart_rate": 95, "sbp": 118, "dbp": 72, "respiratory_rate": 16,
             "temperature": 36.8, "spo2": 96, "gcs": 13,
             "lactate": 1.8, "glucose": 180}),
        ScenarioPhase("early_recovery", 12,
            {"heart_rate": 88, "sbp": 122, "dbp": 76, "respiratory_rate": 15,
             "temperature": 37.0, "spo2": 97, "gcs": 14,
             "lactate": 1.2, "glucose": 140}),
        ScenarioPhase("recovery", 20,
            {"heart_rate": 78, "sbp": 126, "dbp": 78, "respiratory_rate": 14,
             "temperature": 37.0, "spo2": 98, "gcs": 15,
             "lactate": 1.0, "glucose": 110}),
    ],

    "stable_icu": [
        ScenarioPhase("stable", 72,
            {"heart_rate": 78, "sbp": 124, "dbp": 76, "respiratory_rate": 15,
             "temperature": 37.1, "spo2": 97, "gcs": 15,
             "lactate": 1.0, "creatinine": 1.0, "wbc": 8.5}),
    ],
}

# Default starting states per scenario
SCENARIO_BASELINES: Dict[str, Dict] = {
    "septic_shock_untreated": dict(heart_rate=85, sbp=125, dbp=78, respiratory_rate=17,
                                    temperature=37.2, spo2=97, gcs=15,
                                    lactate=1.0, creatinine=0.9, wbc=9.0,
                                    potassium=4.0, glucose=110, procalcitonin=0.1),
    "septic_shock_treated":   dict(heart_rate=85, sbp=125, dbp=78, respiratory_rate=17,
                                    temperature=37.2, spo2=97, gcs=15,
                                    lactate=1.0, creatinine=0.9, wbc=9.0,
                                    potassium=4.0, glucose=110, procalcitonin=0.1),
    "respiratory_failure":    dict(heart_rate=82, sbp=130, dbp=82, respiratory_rate=18,
                                    temperature=37.0, spo2=97, gcs=15,
                                    lactate=1.0, creatinine=1.0, wbc=10.0,
                                    potassium=4.0, glucose=105, procalcitonin=0.15),
    "cardiogenic_shock":      dict(heart_rate=80, sbp=135, dbp=85, respiratory_rate=16,
                                    temperature=37.0, spo2=97, gcs=15,
                                    lactate=1.0, creatinine=1.1, wbc=8.0,
                                    potassium=4.2, glucose=115, procalcitonin=0.1),
    "aki_progression":        dict(heart_rate=82, sbp=130, dbp=80, respiratory_rate=16,
                                    temperature=37.0, spo2=97, gcs=15,
                                    lactate=1.0, creatinine=1.2, wbc=9.0,
                                    potassium=4.0, glucose=100, procalcitonin=0.1),
    "post_op_recovery":       dict(heart_rate=92, sbp=120, dbp=74, respiratory_rate=16,
                                    temperature=36.8, spo2=96, gcs=13,
                                    lactate=1.8, creatinine=0.9, wbc=11.0,
                                    potassium=4.0, glucose=175, procalcitonin=0.2),
    "stable_icu":             dict(heart_rate=76, sbp=124, dbp=76, respiratory_rate=14,
                                    temperature=37.0, spo2=98, gcs=15,
                                    lactate=0.9, creatinine=0.9, wbc=8.0,
                                    potassium=4.0, glucose=100, procalcitonin=0.08),
}


# ─────────────────────────────────────────
# Trajectory generator
# ─────────────────────────────────────────

class PatientTrajectory:
    """
    Generates a time-series of PhysiologyState objects following a clinical scenario.
    Supports 5-minute and hourly sampling.
    """

    VITAL_FIELDS = ["heart_rate", "sbp", "dbp", "respiratory_rate",
                     "temperature", "spo2", "gcs"]
    LAB_FIELDS = ["lactate", "creatinine", "wbc", "potassium",
                   "glucose", "procalcitonin"]

    def __init__(self, scenario: str, patient_id: str = "SIM001", seed: int = 42):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. "
                             f"Available: {list(SCENARIOS.keys())}")
        self.scenario = scenario
        self.patient_id = patient_id
        self.rng = random.Random(seed)
        self.noise = PhysiologicalNoise()
        self.phases = SCENARIOS[scenario]
        self.baseline = SCENARIO_BASELINES.get(scenario, SCENARIO_BASELINES["stable_icu"])
        self.total_hours = sum(p.duration_hours for p in self.phases)

    def _interpolate(self, t: float, phase: ScenarioPhase, t_start: float,
                      prev_state: PhysiologyState) -> PhysiologyState:
        """Interpolate physiology at time t within a phase."""
        t_end = t_start + phase.duration_hours
        fields = {**{f: getattr(prev_state, f) for f in self.VITAL_FIELDS + self.LAB_FIELDS}}

        for field, target_val in phase.target.items():
            if field not in fields:
                continue
            start_val = fields[field]
            if phase.transition == "instant":
                fields[field] = target_val
            elif phase.transition == "linear":
                fields[field] = _linear(t, t_start, t_end, start_val, target_val)
            else:  # smooth
                fields[field] = _smooth_step(t, t_start, t_end, start_val, target_val)

        return PhysiologyState(time_hours=t, **{k: float(v) if k != "gcs"
                                                 else int(round(v)) for k, v in fields.items()})

    def generate(self, interval_minutes: int = 5) -> List[PhysiologyState]:
        """Generate full trajectory with given sampling interval."""
        states: List[PhysiologyState] = []
        interval_h = interval_minutes / 60.0

        # Initial state
        current = PhysiologyState(time_hours=0.0, **{k: float(v) if k != "gcs"
                                                       else int(v) for k, v in self.baseline.items()})
        states.append(self.noise.apply(current, self.rng))

        t = interval_h
        phase_start = 0.0
        phase_idx = 0

        while t <= self.total_hours:
            # Advance phase if needed
            while phase_idx < len(self.phases) - 1:
                phase_end = phase_start + self.phases[phase_idx].duration_hours
                if t > phase_end:
                    phase_start = phase_end
                    phase_idx += 1
                else:
                    break

            phase = self.phases[phase_idx]
            prev = states[-1]
            interpolated = self._interpolate(t, phase, phase_start, prev)
            noisy = self.noise.apply(interpolated, self.rng)
            states.append(noisy)
            t = round(t + interval_h, 4)

        return states

    def generate_stream(self, interval_minutes: int = 5) -> Iterator[PhysiologyState]:
        """Yield states one at a time (memory-efficient streaming)."""
        yield from self.generate(interval_minutes)

    def to_records(self, interval_minutes: int = 5) -> List[Dict]:
        """Return list of dicts with patient_id and timestamp."""
        base_time = datetime.utcnow()
        records = []
        for state in self.generate(interval_minutes):
            d = state.to_dict()
            d["patient_id"] = self.patient_id
            d["timestamp"] = (base_time + timedelta(
                hours=state.time_hours)).isoformat()
            records.append(d)
        return records


# ─────────────────────────────────────────
# Multi-patient ICU simulator
# ─────────────────────────────────────────

@dataclass
class SimulationResult:
    scenario: str
    patient_id: str
    n_states: int
    duration_hours: float
    max_news2: int
    min_sbp: float
    max_lactate: float
    peak_phase: str
    states: List[PhysiologyState] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.patient_id} [{self.scenario}] "
                f"n={self.n_states} hrs={self.duration_hours:.0f} "
                f"max_NEWS2={self.max_news2} min_SBP={self.min_sbp:.0f} "
                f"max_lactate={self.max_lactate:.1f}")


class ICUSimulator:
    """
    Simulates an entire ICU ward with multiple patient trajectories.
    Used for: load testing, model validation, demo generation.
    """

    def simulate_patient(self, patient_id: str, scenario: str,
                          interval_minutes: int = 5,
                          seed: int = None) -> SimulationResult:
        seed = seed or hash(patient_id) % 10000
        traj = PatientTrajectory(scenario, patient_id, seed=seed)
        states = traj.generate(interval_minutes)

        max_news2 = max(s.news2 for s in states)
        min_sbp   = min(s.sbp for s in states)
        max_lac   = max(s.lactate for s in states)

        # Find peak phase (worst NEWS2)
        worst_idx = max(range(len(states)), key=lambda i: states[i].news2)
        cum_h = 0.0
        peak_phase = traj.phases[-1].name
        for ph in traj.phases:
            cum_h += ph.duration_hours
            if states[worst_idx].time_hours <= cum_h:
                peak_phase = ph.name
                break

        return SimulationResult(
            scenario=scenario, patient_id=patient_id,
            n_states=len(states), duration_hours=traj.total_hours,
            max_news2=max_news2, min_sbp=min_sbp, max_lactate=max_lac,
            peak_phase=peak_phase, states=states,
        )

    def simulate_ward(self, n_patients: int = 10,
                       scenario_mix: Optional[Dict[str, float]] = None,
                       interval_minutes: int = 5) -> List[SimulationResult]:
        """Simulate a full ICU ward with a mix of scenarios."""
        if scenario_mix is None:
            scenario_mix = {
                "stable_icu": 0.25,
                "septic_shock_treated": 0.20,
                "septic_shock_untreated": 0.10,
                "respiratory_failure": 0.20,
                "cardiogenic_shock": 0.10,
                "aki_progression": 0.10,
                "post_op_recovery": 0.05,
            }

        # Normalise weights
        total = sum(scenario_mix.values())
        weights = {k: v / total for k, v in scenario_mix.items()}

        results = []
        rng = random.Random(99)
        scenarios_list = list(weights.keys())
        scenario_weights = [weights[s] for s in scenarios_list]

        for i in range(n_patients):
            pid = f"SIM_{i+1:03d}"
            scenario = rng.choices(scenarios_list, weights=scenario_weights)[0]
            result = self.simulate_patient(pid, scenario,
                                            interval_minutes=interval_minutes,
                                            seed=i * 17 + 3)
            results.append(result)

        return sorted(results, key=lambda r: -r.max_news2)

    def generate_training_dataset(self, n_patients: int = 200,
                                    interval_minutes: int = 60,
                                    label_horizon_hours: int = 6) -> Tuple[List[Dict], List[int]]:
        """
        Generate a labelled dataset for ML training.
        Label = 1 if patient deteriorates (NEWS2 ≥ 7) within next `label_horizon_hours`.
        """
        X_records = []
        y_labels = []
        rng = random.Random(42)

        all_scenarios = list(SCENARIOS.keys())
        for i in range(n_patients):
            scenario = rng.choice(all_scenarios)
            pid = f"TRAIN_{i:04d}"
            result = self.simulate_patient(pid, scenario,
                                            interval_minutes=interval_minutes,
                                            seed=i)
            states = result.states
            horizon_steps = int(label_horizon_hours * 60 / interval_minutes)

            for j, state in enumerate(states[:-horizon_steps]):
                future = states[j+1: j+1+horizon_steps]
                label = 1 if any(s.news2 >= 7 for s in future) else 0
                record = state.to_dict()
                record["patient_id"] = pid
                record["scenario"] = scenario
                X_records.append(record)
                y_labels.append(label)

        logger.info(f"Training dataset: {len(X_records)} records, "
                    f"prevalence={sum(y_labels)/len(y_labels):.1%}")
        return X_records, y_labels


# ─────────────────────────────────────────
# Entry point / demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    sim = ICUSimulator()

    print("=== Single Patient Simulation ===")
    result = sim.simulate_patient("P_SIM_001", "septic_shock_treated", interval_minutes=30)
    print(result.summary())
    print(f"First state: NEWS2={result.states[0].news2}, "
          f"HR={result.states[0].heart_rate:.0f}, SBP={result.states[0].sbp:.0f}")
    worst = max(result.states, key=lambda s: s.news2)
    print(f"Worst state: NEWS2={worst.news2}, t={worst.time_hours:.1f}h, "
          f"HR={worst.heart_rate:.0f}, SBP={worst.sbp:.0f}, "
          f"lactate={worst.lactate:.1f}, SpO2={worst.spo2:.0f}%")

    print("\n=== ICU Ward Simulation (8 patients) ===")
    ward = sim.simulate_ward(n_patients=8, interval_minutes=60)
    print(f"{'Patient':<12} {'Scenario':<28} {'NEWS2':>6} {'SBP':>6} {'Lac':>6} {'Peak Phase'}")
    print("-" * 75)
    for r in ward:
        print(f"{r.patient_id:<12} {r.scenario:<28} {r.max_news2:>6} "
              f"{r.min_sbp:>6.0f} {r.max_lactate:>6.1f}  {r.peak_phase}")

    print("\n=== Training Dataset Generation ===")
    X, y = sim.generate_training_dataset(n_patients=50, interval_minutes=60, label_horizon_hours=6)
    prevalence = sum(y) / len(y)
    print(f"Records: {len(X)}, Positive labels: {sum(y)} ({prevalence:.1%})")

    print("\nAll scenarios available:")
    for name in SCENARIOS:
        phases = SCENARIOS[name]
        total_h = sum(p.duration_hours for p in phases)
        print(f"  {name:<30} {len(phases)} phases, {total_h:.0f}h total")
