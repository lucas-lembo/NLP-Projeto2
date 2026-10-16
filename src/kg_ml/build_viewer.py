"""Visualizador comparador: grafo do Projeto 1 × grafo do Projeto 2, caso a caso.

Gera um HTML autocontido (Cytoscape.js via CDN) a partir de:
- grafo do P1: nodes.csv / edges.csv do repositório do Projeto 1 (P1_DIR);
- grafo do P2: data/processed/{nodes,edges}.csv (saída do build_graph);
- evidência: pairs_test_predictions.csv (a frase de cada par candidato);
- associação: concept_pmi.csv (PMI entre conceitos no corpus).

Os nós são os mesmos nos dois grafos (mesmo NER, mesmo node_id), então as
arestas são comparadas pela chave (source_id, target_id, relation). Arestas
repetidas do P1 (o mesmo par ligado em várias frases) viram uma só.

Uso: PYTHONPATH=src python3 -m kg_ml.build_viewer
Abre depois: assets/graph_viewer/index.html direto no navegador.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kg_ml.build_graph import ENTITY_TYPES, MODEL, TYPE_LABEL
from kg_ml.config import DATA_PROCESSED, P1_CASES, P1_EDGES, P1_NODES, ROOT

TEMPLATE = Path(__file__).with_name("viewer_template.html")
OUT_DIR = ROOT / "assets" / "graph_viewer"

# Mesmas cores e anéis do visualizador do Projeto 1.
COLORS = {
    "Patient": ("#fef3c7", "#d97706"),
    "History": ("#e0e7ff", "#4338ca"),
    "Symptom": ("#fee2e2", "#b91c1c"),
    "Finding": ("#fce7f3", "#be185d"),
    "Exam": ("#dbeafe", "#1d4ed8"),
    "ExamResult": ("#cffafe", "#0e7490"),
    "Diagnosis": ("#dcfce7", "#15803d"),
    "Treatment": ("#fef9c3", "#a16207"),
    "AnatomicalSite": ("#f3e8ff", "#7e22ce"),
    "Unit": ("#f1f5f9", "#64748b"),
}
RINGS = {
    "Patient": 100, "History": 90, "Symptom": 80, "AnatomicalSite": 70, "Diagnosis": 60,
    "Exam": 50, "ExamResult": 40, "Finding": 30, "Treatment": 20, "Unit": 10,
}


def parse_prob(attributes: str) -> float | None:
    for part in attributes.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "prob":
            return float(value)
    return None


def load_edges(path: Path) -> pd.DataFrame:
    edges = pd.read_csv(path, dtype=str).fillna("")
    return edges.drop_duplicates(["case_id", "source_id", "target_id", "relation"])


def pair_mentions(predictions: pd.DataFrame, node_ids: dict) -> tuple[dict, dict]:
    """Menções de cada par de nós nas frases do caso.

    Retorna {(case, source, target): [[sent_idx, e1_start, e1_end, e2_start, e2_end,
    p1_label, pred, score], ...]} e {(case, sent_idx): frase}.
    """
    mentions, sentences = {}, {}
    for row in predictions.itertuples():
        source = node_ids.get((row.case_id, TYPE_LABEL[row.e1_type], row.e1_label))
        target = node_ids.get((row.case_id, TYPE_LABEL[row.e2_type], row.e2_label))
        if source is None or target is None:
            continue
        mentions.setdefault((row.case_id, source, target), []).append([
            int(row.sent_idx), int(row.e1_start), int(row.e1_end), int(row.e2_start), int(row.e2_end),
            row.p1_label, getattr(row, f"pred_{MODEL}"), round(float(getattr(row, f"score_{MODEL}")), 3),
        ])
        sentences[(row.case_id, int(row.sent_idx))] = row.sentence
    return mentions, sentences


def pmi_lookup(path: Path) -> dict:
    """{'Tipo:label|Tipo:label' (ordem alfabética): [pmi, casos juntos]}."""
    if not path.exists():
        return {}
    pmi = pd.read_csv(path)
    lookup = {}
    for row in pmi.itertuples():
        a = f"{TYPE_LABEL[row.source_type]}:{row.source}"
        b = f"{TYPE_LABEL[row.target_type]}:{row.target}"
        lookup["|".join(sorted((a, b)))] = [round(float(row.pmi), 2), int(row.cases_together)]
    return lookup


def main() -> None:
    nodes = pd.read_csv(P1_NODES, dtype=str).fillna("")
    p2_nodes = pd.read_csv(DATA_PROCESSED / "nodes.csv", dtype=str).fillna("")
    if not nodes.equals(p2_nodes):
        print("aviso: nodes.csv do P1 e do P2 diferem; usando os nós do P1")
    p1_edges = load_edges(P1_EDGES)
    p2_edges = load_edges(DATA_PROCESSED / "edges.csv")
    predictions = pd.read_csv(DATA_PROCESSED / "pairs_test_predictions.csv")
    cases = pd.read_csv(P1_CASES)

    nodes_by_id = nodes.set_index("node_id").to_dict(orient="index")
    node_ids = {(r.case_id, r.type, r.label): r.node_id for r in nodes.itertuples()}
    mentions, sentences = pair_mentions(predictions, node_ids)
    pmi = pmi_lookup(DATA_PROCESSED / "concept_pmi.csv")
    case_text = dict(zip(cases["case_id"], cases["case_text"]))

    def is_entity(node_id: str) -> bool:
        return nodes_by_id.get(node_id, {}).get("type") in ENTITY_TYPES

    payload, used_pmi = {}, {}
    for case_id in sorted(nodes.loc[nodes["case_id"] != "", "case_id"].unique()):
        e1 = p1_edges[p1_edges["case_id"] == case_id]
        e2 = p2_edges[p2_edges["case_id"] == case_id]

        # Une as duas listas por par (source, target): relações do P1, do P2 e a prob do P2.
        pairs: dict[tuple[str, str], dict] = {}
        for graph, frame in (("r1", e1), ("r2", e2)):
            for r in frame.itertuples():
                rec = pairs.setdefault((r.source_id, r.target_id), {"r1": [], "r2": [], "p": None})
                rec[graph].append(r.relation)
                if graph == "r2":
                    prob = parse_prob(r.attributes)
                    if prob is not None:
                        rec["p"] = prob

        edges, case_mentions = [], {}
        for (s, t), rec in pairs.items():
            ent = is_entity(s) and is_entity(t)
            edges.append({"s": s, "t": t, "r1": rec["r1"], "r2": rec["r2"], "p": rec["p"], "ent": ent})
            if not ent:
                continue
            found = mentions.get((case_id, s, t))
            if found:
                case_mentions[f"{s}|{t}"] = found
            key = "|".join(sorted((f"{nodes_by_id[s]['type']}:{nodes_by_id[s]['label']}",
                                   f"{nodes_by_id[t]['type']}:{nodes_by_id[t]['label']}")))
            if key in pmi:
                used_pmi[key] = pmi[key]

        node_set = set(nodes.loc[nodes["case_id"] == case_id, "node_id"])
        node_set |= {s for s, _ in pairs} | {t for _, t in pairs}
        case_nodes = [
            {"id": nid, "label": nodes_by_id[nid]["label"], "type": nodes_by_id[nid]["type"],
             "attributes": nodes_by_id[nid]["attributes"]}
            for nid in sorted(node_set) if nid in nodes_by_id
        ]
        patient = nodes[(nodes["case_id"] == case_id) & (nodes["type"] == "Patient")]

        payload[case_id] = {
            "patient": patient.iloc[0]["attributes"] if len(patient) else "",
            "text": case_text.get(case_id, ""),
            "nodes": case_nodes,
            "edges": edges,
            "mentions": case_mentions,
            "sentences": {str(i): text for (c, i), text in sentences.items() if c == case_id},
        }

    def to_js(obj) -> str:
        # "</" escapado para um texto de caso nunca fechar a tag <script> por acidente
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    html = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__DATA__", to_js(payload))
        .replace("__PMI__", to_js(used_pmi))
        .replace("__COLORS__", json.dumps(COLORS))
        .replace("__RINGS__", json.dumps(RINGS))
        .replace("__MODEL__", MODEL)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(payload)} casos -> {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
