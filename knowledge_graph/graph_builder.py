"""
Medical Knowledge Graph Builder
Constructs a graph of: diseases ↔ symptoms ↔ medications ↔ procedures ↔ labs
Supports path queries, drug interactions, differential diagnosis, and risk factor lookup
Uses networkx; optionally exports to Neo4j Cypher
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Node / Edge types
# ─────────────────────────────────────────

class NodeType:
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    PROCEDURE = "Procedure"
    LAB = "Lab"
    RISK_FACTOR = "RiskFactor"
    ORGANISM = "Organism"
    BODY_SYSTEM = "BodySystem"


class EdgeType:
    HAS_SYMPTOM = "HAS_SYMPTOM"
    TREATED_BY = "TREATED_BY"
    DIAGNOSED_BY = "DIAGNOSED_BY"
    CAUSES = "CAUSES"
    RISK_FACTOR_FOR = "RISK_FACTOR_FOR"
    INTERACTS_WITH = "INTERACTS_WITH"
    CONTRAINDICATES = "CONTRAINDICATES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    PART_OF = "PART_OF"
    CAUSED_BY = "CAUSED_BY"


@dataclass
class KGNode:
    id: str
    type: str
    name: str
    properties: Dict = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "type": self.type, "name": self.name, **self.properties}


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    properties: Dict = field(default_factory=dict)


# ─────────────────────────────────────────
# Knowledge Base (curated clinical facts)
# ─────────────────────────────────────────

KNOWLEDGE_BASE = {
    "diseases": {
        "sepsis": {
            "symptoms": ["fever", "tachycardia", "hypotension", "altered mental status",
                         "tachypnea", "diaphoresis", "oliguria"],
            "risk_factors": ["immunocompromised", "diabetes", "recent surgery",
                             "indwelling catheter", "elderly"],
            "labs": ["elevated WBC", "elevated lactate", "elevated procalcitonin",
                     "elevated CRP", "positive blood culture"],
            "treatments": ["broad-spectrum antibiotics", "fluid resuscitation",
                           "vasopressors", "source control"],
            "procedures": ["blood culture", "central venous catheter", "arterial line"],
            "caused_by": ["bacteremia", "pneumonia", "UTI", "abdominal infection"],
            "icd10": "A41.9",
            "severity": "critical",
            "system": "systemic",
        },
        "septic_shock": {
            "symptoms": ["hypotension", "fever", "altered mental status", "oliguria",
                         "tachycardia", "tachypnea"],
            "risk_factors": ["sepsis", "immunocompromised", "elderly"],
            "labs": ["elevated lactate >2 mmol/L", "elevated WBC", "metabolic acidosis"],
            "treatments": ["norepinephrine", "vasopressin", "broad-spectrum antibiotics",
                           "hydrocortisone", "fluid resuscitation"],
            "procedures": ["arterial line", "central venous catheter", "mechanical ventilation"],
            "caused_by": ["sepsis"],
            "icd10": "A41.9",
            "severity": "critical",
            "system": "systemic",
        },
        "pneumonia": {
            "symptoms": ["cough", "fever", "dyspnea", "chest pain", "tachypnea",
                         "productive cough", "chills"],
            "risk_factors": ["smoking", "COPD", "elderly", "immunocompromised",
                             "alcoholism", "aspiration"],
            "labs": ["elevated WBC", "elevated CRP", "elevated procalcitonin"],
            "treatments": ["ceftriaxone", "azithromycin", "levofloxacin",
                           "piperacillin-tazobactam", "vancomycin"],
            "procedures": ["chest X-ray", "CT scan", "sputum culture", "blood culture"],
            "caused_by": ["Streptococcus pneumoniae", "Haemophilus influenzae",
                          "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "MRSA"],
            "icd10": "J18.9",
            "severity": "moderate",
            "system": "respiratory",
        },
        "ards": {
            "symptoms": ["dyspnea", "tachypnea", "hypoxemia", "bilateral infiltrates"],
            "risk_factors": ["sepsis", "pneumonia", "trauma", "aspiration", "pancreatitis"],
            "labs": ["low PaO2", "low PaO2/FiO2 ratio"],
            "treatments": ["mechanical ventilation", "prone positioning", "neuromuscular blockade",
                           "corticosteroids"],
            "procedures": ["mechanical ventilation", "arterial line", "bronchoscopy"],
            "icd10": "J80",
            "severity": "critical",
            "system": "respiratory",
        },
        "acute_kidney_injury": {
            "symptoms": ["oliguria", "edema", "confusion", "fatigue"],
            "risk_factors": ["sepsis", "hypovolemia", "nephrotoxic drugs",
                             "contrast dye", "chronic kidney disease"],
            "labs": ["elevated creatinine", "elevated BUN", "hyperkalemia",
                     "metabolic acidosis", "elevated phosphorus"],
            "treatments": ["fluid resuscitation", "furosemide", "dialysis",
                           "CRRT", "sodium bicarbonate"],
            "procedures": ["dialysis", "CRRT", "foley catheter"],
            "icd10": "N17.9",
            "severity": "moderate",
            "system": "renal",
        },
        "congestive_heart_failure": {
            "symptoms": ["dyspnea", "edema", "orthopnea", "fatigue",
                         "tachycardia", "chest pain"],
            "risk_factors": ["hypertension", "coronary artery disease",
                             "atrial fibrillation", "diabetes", "obesity"],
            "labs": ["elevated BNP", "elevated proBNP", "elevated troponin"],
            "treatments": ["furosemide", "metoprolol", "lisinopril",
                           "spironolactone", "digoxin"],
            "procedures": ["echocardiogram", "ECG", "chest X-ray"],
            "icd10": "I50.9",
            "severity": "moderate",
            "system": "cardiovascular",
        },
        "myocardial_infarction": {
            "symptoms": ["chest pain", "dyspnea", "diaphoresis", "nausea",
                         "palpitations", "syncope"],
            "risk_factors": ["hypertension", "diabetes", "smoking",
                             "hyperlipidemia", "family history", "obesity"],
            "labs": ["elevated troponin", "elevated CK-MB", "elevated BNP"],
            "treatments": ["aspirin", "heparin", "clopidogrel", "nitroglycerin",
                           "metoprolol", "atorvastatin"],
            "procedures": ["ECG", "echocardiogram", "coronary angiography", "PCI"],
            "icd10": "I21.9",
            "severity": "critical",
            "system": "cardiovascular",
        },
        "atrial_fibrillation": {
            "symptoms": ["palpitations", "dyspnea", "fatigue", "chest pain",
                         "dizziness", "syncope"],
            "risk_factors": ["hypertension", "heart failure", "coronary artery disease",
                             "elderly", "hyperthyroidism", "COPD"],
            "labs": ["elevated TSH", "elevated BNP"],
            "treatments": ["metoprolol", "amiodarone", "digoxin",
                           "warfarin", "apixaban", "cardioversion"],
            "procedures": ["ECG", "echocardiogram", "Holter monitor"],
            "icd10": "I48.91",
            "severity": "moderate",
            "system": "cardiovascular",
        },
        "diabetic_ketoacidosis": {
            "symptoms": ["nausea", "vomiting", "abdominal pain", "polyuria",
                         "polydipsia", "weakness", "altered mental status"],
            "risk_factors": ["type 1 diabetes", "insulin non-compliance",
                             "infection", "stress"],
            "labs": ["elevated glucose", "elevated ketones", "low pH",
                     "low bicarbonate", "low potassium"],
            "treatments": ["insulin", "IV fluids", "potassium replacement",
                           "sodium bicarbonate"],
            "icd10": "E10.10",
            "severity": "moderate",
            "system": "endocrine",
        },
        "pulmonary_embolism": {
            "symptoms": ["dyspnea", "chest pain", "hemoptysis", "tachycardia",
                         "syncope", "tachypnea"],
            "risk_factors": ["DVT", "immobility", "malignancy", "oral contraceptives",
                             "hypercoagulable state", "recent surgery"],
            "labs": ["elevated D-dimer", "elevated troponin", "elevated BNP",
                     "low PaO2"],
            "treatments": ["heparin", "enoxaparin", "warfarin", "apixaban",
                           "thrombolytics"],
            "procedures": ["CT pulmonary angiography", "V/Q scan", "echocardiogram"],
            "icd10": "I26.99",
            "severity": "high",
            "system": "cardiovascular",
        },
        "copd_exacerbation": {
            "symptoms": ["dyspnea", "cough", "increased sputum", "wheezing",
                         "tachypnea", "cyanosis"],
            "risk_factors": ["smoking", "COPD", "infection", "air pollution"],
            "labs": ["elevated WBC", "elevated procalcitonin", "elevated CO2"],
            "treatments": ["albuterol", "ipratropium", "methylprednisolone",
                           "azithromycin", "levofloxacin"],
            "procedures": ["chest X-ray", "ABG", "ECG", "mechanical ventilation"],
            "icd10": "J44.1",
            "severity": "moderate",
            "system": "respiratory",
        },
    },

    "drug_interactions": [
        {"drug1": "warfarin", "drug2": "aspirin", "severity": "high",
         "effect": "increased bleeding risk"},
        {"drug1": "warfarin", "drug2": "metronidazole", "severity": "high",
         "effect": "warfarin potentiation"},
        {"drug1": "metoprolol", "drug2": "amiodarone", "severity": "moderate",
         "effect": "bradycardia risk"},
        {"drug1": "digoxin", "drug2": "amiodarone", "severity": "high",
         "effect": "digoxin toxicity"},
        {"drug1": "heparin", "drug2": "enoxaparin", "severity": "contraindicated",
         "effect": "severe bleeding risk"},
        {"drug1": "furosemide", "drug2": "gentamicin", "severity": "high",
         "effect": "ototoxicity and nephrotoxicity"},
        {"drug1": "vancomycin", "drug2": "piperacillin-tazobactam", "severity": "moderate",
         "effect": "increased nephrotoxicity risk"},
        {"drug1": "propofol", "drug2": "midazolam", "severity": "moderate",
         "effect": "additive CNS depression"},
        {"drug1": "fentanyl", "drug2": "midazolam", "severity": "moderate",
         "effect": "respiratory depression"},
        {"drug1": "ciprofloxacin", "drug2": "warfarin", "severity": "high",
         "effect": "warfarin potentiation"},
    ],

    "lab_critical_values": {
        "potassium": {"low": 3.0, "high": 6.0, "unit": "mEq/L"},
        "sodium": {"low": 120, "high": 160, "unit": "mEq/L"},
        "glucose": {"low": 50, "high": 500, "unit": "mg/dL"},
        "hemoglobin": {"low": 7.0, "high": None, "unit": "g/dL"},
        "platelet": {"low": 50, "high": None, "unit": "k/uL"},
        "INR": {"low": None, "high": 3.5, "unit": ""},
        "pH": {"low": 7.20, "high": 7.60, "unit": ""},
        "lactate": {"low": None, "high": 4.0, "unit": "mmol/L"},
        "troponin": {"low": None, "high": 0.04, "unit": "ng/mL"},
        "creatinine": {"low": None, "high": 4.0, "unit": "mg/dL"},
    },
}


# ─────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────

class MedicalKnowledgeGraph:
    """Medical knowledge graph with clinical query capabilities."""

    def __init__(self):
        if not HAS_NETWORKX:
            logger.warning("networkx not installed — using dict-based fallback")
        self.graph = nx.DiGraph() if HAS_NETWORKX else None
        self._adjacency: Dict[str, List[Dict]] = {}  # fallback
        self._nodes: Dict[str, KGNode] = {}
        self._build()

    # ── Build ──────────────────────────────

    def _add_node(self, node: KGNode):
        self._nodes[node.id] = node
        if self.graph is not None:
            self.graph.add_node(node.id, **node.to_dict())

    def _add_edge(self, edge: KGEdge):
        if self.graph is not None:
            self.graph.add_edge(edge.source, edge.target,
                                relation=edge.relation,
                                weight=edge.weight,
                                **edge.properties)
        else:
            self._adjacency.setdefault(edge.source, []).append(
                {"target": edge.target, "relation": edge.relation}
            )

    def _node_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

    def _ensure_node(self, name: str, node_type: str) -> str:
        nid = self._node_id(name)
        if nid not in self._nodes:
            self._add_node(KGNode(id=nid, type=node_type, name=name))
        return nid

    def _build(self):
        """Populate graph from curated knowledge base."""
        kb = KNOWLEDGE_BASE

        for disease_name, info in kb["diseases"].items():
            d_id = self._ensure_node(disease_name, NodeType.DISEASE)
            d_node = self._nodes[d_id]
            d_node.properties.update({
                "icd10": info.get("icd10", ""),
                "severity": info.get("severity", ""),
                "system": info.get("system", ""),
            })

            for sx in info.get("symptoms", []):
                sx_id = self._ensure_node(sx, NodeType.SYMPTOM)
                self._add_edge(KGEdge(d_id, sx_id, EdgeType.HAS_SYMPTOM, 0.9))

            for rf in info.get("risk_factors", []):
                rf_id = self._ensure_node(rf, NodeType.RISK_FACTOR)
                self._add_edge(KGEdge(rf_id, d_id, EdgeType.RISK_FACTOR_FOR, 0.8))

            for lab in info.get("labs", []):
                lab_id = self._ensure_node(lab, NodeType.LAB)
                self._add_edge(KGEdge(d_id, lab_id, EdgeType.DIAGNOSED_BY, 0.85))

            for tx in info.get("treatments", []):
                med_id = self._ensure_node(tx, NodeType.MEDICATION)
                self._add_edge(KGEdge(d_id, med_id, EdgeType.TREATED_BY, 0.9))

            for proc in info.get("procedures", []):
                proc_id = self._ensure_node(proc, NodeType.PROCEDURE)
                self._add_edge(KGEdge(d_id, proc_id, EdgeType.DIAGNOSED_BY, 0.8))

            for cause in info.get("caused_by", []):
                cause_id = self._ensure_node(cause, NodeType.ORGANISM)
                self._add_edge(KGEdge(disease_name, cause_id, EdgeType.CAUSED_BY, 0.85))

            body_system = info.get("system", "")
            if body_system:
                sys_id = self._ensure_node(body_system, NodeType.BODY_SYSTEM)
                self._add_edge(KGEdge(d_id, sys_id, EdgeType.PART_OF, 1.0))

        for interaction in kb["drug_interactions"]:
            d1 = self._ensure_node(interaction["drug1"], NodeType.MEDICATION)
            d2 = self._ensure_node(interaction["drug2"], NodeType.MEDICATION)
            severity_weight = {"low": 0.3, "moderate": 0.6, "high": 0.9,
                                "contraindicated": 1.0}.get(interaction["severity"], 0.5)
            self._add_edge(KGEdge(d1, d2, EdgeType.INTERACTS_WITH, severity_weight,
                                  {"severity": interaction["severity"],
                                   "effect": interaction["effect"]}))
            self._add_edge(KGEdge(d2, d1, EdgeType.INTERACTS_WITH, severity_weight,
                                  {"severity": interaction["severity"],
                                   "effect": interaction["effect"]}))

        logger.info(f"Knowledge graph built: {len(self._nodes)} nodes, "
                    f"{self.graph.number_of_edges() if self.graph else '?'} edges")

    # ── Query API ─────────────────────────

    def get_symptoms(self, disease: str) -> List[str]:
        """Return symptoms associated with a disease."""
        d_id = self._node_id(disease)
        if self.graph is None:
            return [e["target"] for e in self._adjacency.get(d_id, [])
                    if e["relation"] == EdgeType.HAS_SYMPTOM]
        return [self._nodes[n].name for n in self.graph.successors(d_id)
                if self.graph[d_id][n].get("relation") == EdgeType.HAS_SYMPTOM
                and n in self._nodes]

    def get_treatments(self, disease: str) -> List[str]:
        d_id = self._node_id(disease)
        if self.graph is None:
            return []
        return [self._nodes[n].name for n in self.graph.successors(d_id)
                if self.graph[d_id][n].get("relation") == EdgeType.TREATED_BY
                and n in self._nodes]

    def get_diagnostic_labs(self, disease: str) -> List[str]:
        d_id = self._node_id(disease)
        if self.graph is None:
            return []
        return [self._nodes[n].name for n in self.graph.successors(d_id)
                if self.graph[d_id][n].get("relation") == EdgeType.DIAGNOSED_BY
                and n in self._nodes]

    def get_risk_factors(self, disease: str) -> List[str]:
        d_id = self._node_id(disease)
        if self.graph is None:
            return []
        return [self._nodes[n].name for n in self.graph.predecessors(d_id)
                if self.graph[n][d_id].get("relation") == EdgeType.RISK_FACTOR_FOR
                and n in self._nodes]

    def differential_diagnosis(self, symptoms: List[str], top_n: int = 5) -> List[Dict]:
        """Score diseases by symptom overlap."""
        symptom_ids = {self._node_id(s) for s in symptoms}
        scores = {}

        for node_id, node in self._nodes.items():
            if node.type != NodeType.DISEASE:
                continue
            disease_symptoms = set(self.get_symptoms(node.name))
            disease_symptom_ids = {self._node_id(s) for s in disease_symptoms}
            if not disease_symptom_ids:
                continue
            overlap = symptom_ids & disease_symptom_ids
            precision = len(overlap) / len(symptom_ids) if symptom_ids else 0
            recall = len(overlap) / len(disease_symptom_ids)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            if overlap:
                scores[node_id] = {
                    "disease": node.name,
                    "score": f1,
                    "matched_symptoms": list(overlap),
                    "severity": node.properties.get("severity", ""),
                    "icd10": node.properties.get("icd10", ""),
                }

        ranked = sorted(scores.values(), key=lambda x: -x["score"])
        return ranked[:top_n]

    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Return all interactions among a medication list."""
        med_ids = [self._node_id(m) for m in medications]
        interactions = []
        seen = set()
        if self.graph is None:
            return []
        for i, m1 in enumerate(med_ids):
            for m2 in med_ids[i+1:]:
                if self.graph.has_edge(m1, m2):
                    edge = self.graph[m1][m2]
                    if edge.get("relation") == EdgeType.INTERACTS_WITH:
                        pair = tuple(sorted([m1, m2]))
                        if pair not in seen:
                            seen.add(pair)
                            interactions.append({
                                "drug1": self._nodes[m1].name if m1 in self._nodes else m1,
                                "drug2": self._nodes[m2].name if m2 in self._nodes else m2,
                                "severity": edge.get("severity", "unknown"),
                                "effect": edge.get("effect", "unknown"),
                            })
        return sorted(interactions, key=lambda x: {"contraindicated": 0, "high": 1,
                                                    "moderate": 2, "low": 3}.get(x["severity"], 4))

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two entities in the graph."""
        if self.graph is None:
            return None
        s_id = self._node_id(source)
        t_id = self._node_id(target)
        try:
            path = nx.shortest_path(self.graph.to_undirected(), s_id, t_id)
            return [self._nodes[n].name if n in self._nodes else n for n in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_disease_info(self, disease: str) -> Dict:
        d_id = self._node_id(disease)
        node = self._nodes.get(d_id)
        if not node:
            return {}
        return {
            "name": node.name,
            "icd10": node.properties.get("icd10", ""),
            "severity": node.properties.get("severity", ""),
            "system": node.properties.get("system", ""),
            "symptoms": self.get_symptoms(disease),
            "treatments": self.get_treatments(disease),
            "diagnostic_labs": self.get_diagnostic_labs(disease),
            "risk_factors": self.get_risk_factors(disease),
        }

    def graph_stats(self) -> Dict:
        if self.graph is None:
            return {"nodes": len(self._nodes)}
        type_counts = {}
        for n in self._nodes.values():
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
        }

    def export_cypher(self) -> str:
        """Export as Neo4j Cypher CREATE statements."""
        lines = ["// Medical Knowledge Graph — Neo4j Cypher Export\n"]
        for node in self._nodes.values():
            props = json.dumps({k: v for k, v in node.properties.items()
                                 if isinstance(v, (str, int, float, bool))})
            lines.append(f"CREATE (:{node.type} {{id: '{node.id}', name: '{node.name}', "
                         f"props: {props}}});")
        if self.graph:
            for s, t, data in self.graph.edges(data=True):
                rel = data.get("relation", "RELATED_TO")
                lines.append(f"MATCH (a {{id: '{s}'}}), (b {{id: '{t}'}}) "
                             f"CREATE (a)-[:{rel}]->(b);")
        return "\n".join(lines)


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────

if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    print("Graph stats:", kg.graph_stats())

    print("\n--- Sepsis info ---")
    info = kg.get_disease_info("sepsis")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n--- Differential diagnosis for: fever, hypotension, tachycardia, dyspnea ---")
    ddx = kg.differential_diagnosis(["fever", "hypotension", "tachycardia", "dyspnea"])
    for d in ddx:
        print(f"  {d['disease']} (score={d['score']:.2f}, severity={d['severity']})")

    print("\n--- Drug interactions for ICU polypharmacy ---")
    meds = ["vancomycin", "piperacillin-tazobactam", "heparin", "furosemide",
            "metoprolol", "amiodarone", "propofol", "midazolam"]
    interactions = kg.check_drug_interactions(meds)
    for ix in interactions:
        print(f"  {ix['drug1']} + {ix['drug2']}: [{ix['severity'].upper()}] {ix['effect']}")

    print("\n--- Path: diabetes → sepsis ---")
    path = kg.shortest_path("diabetes", "sepsis")
    print(f"  {' → '.join(path) if path else 'No path found'}")
