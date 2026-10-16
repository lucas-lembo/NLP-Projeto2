"""Passo 5: monta o grafo do Projeto 2 e compara com o do Projeto 1.

Os nós são os mesmos do Projeto 1 (mesmo NER). Mudam só as arestas entre
entidades (TREATED_BY, SUPPORTS, LOCATED_IN, TARGETS, PERFORMED_ON,
CONFIRMS/EXCLUDES/REVEALS): no P1 vêm das regras, aqui vêm do classificador,
cada uma com a probabilidade como atributo. As demais arestas (paciente,
histórico, resultado de exame, unidade, achado) são mantidas.

Uso: PYTHONPATH=src python3 -m kg_ml.build_graph
"""

from __future__ import annotations

import pandas as pd

from kg_ml.config import DATA_PROCESSED, NONE_LABEL, P1_EDGES, P1_NODES

MODEL = "logreg_ngram2"
TYPE_LABEL = {
    "symptom": "Symptom", "diagnosis": "Diagnosis", "exam": "Exam",
    "treatment": "Treatment", "anatomical_site": "AnatomicalSite",
}
ENTITY_TYPES = set(TYPE_LABEL.values())


def model_edges(predictions: pd.DataFrame, node_ids: dict) -> tuple[pd.DataFrame, int]:
    """Uma aresta por par de nós: fica a menção com maior probabilidade."""
    related = predictions[predictions[f"pred_{MODEL}"] != NONE_LABEL]
    best = {}
    missing = 0
    for row in related.itertuples():
        source = node_ids.get((row.case_id, TYPE_LABEL[row.e1_type], row.e1_label))
        target = node_ids.get((row.case_id, TYPE_LABEL[row.e2_type], row.e2_label))
        if source is None or target is None:
            missing += 1
            continue
        score = getattr(row, f"score_{MODEL}")
        key = (row.case_id, source, target)
        if key not in best or score > best[key][1]:
            best[key] = (getattr(row, f"pred_{MODEL}"), score)

    rows = [
        {"case_id": case_id, "source_id": s, "target_id": t, "relation": rel, "attributes": f"prob={score:.3f}"}
        for (case_id, s, t), (rel, score) in best.items()
    ]
    return pd.DataFrame(rows), missing


def main() -> None:
    nodes = pd.read_csv(P1_NODES, dtype=str).fillna("")
    p1_edges = pd.read_csv(P1_EDGES, dtype=str).fillna("")
    predictions = pd.read_csv(DATA_PROCESSED / "pairs_test_predictions.csv")

    node_type = dict(zip(nodes["node_id"], nodes["type"]))
    node_ids = {(r.case_id, r.type, r.label): r.node_id for r in nodes.itertuples()}

    # arestas entidade-entidade do P1 (as que o modelo substitui)
    is_entity_edge = p1_edges["source_id"].map(node_type).isin(ENTITY_TYPES) & p1_edges["target_id"].map(node_type).isin(ENTITY_TYPES)
    kept = p1_edges[~is_entity_edge]
    p1_entity_edges = p1_edges[is_entity_edge]

    new_edges, missing = model_edges(predictions, node_ids)
    p2_edges = pd.concat([kept.drop(columns=["edge_id"]), new_edges], ignore_index=True)
    p2_edges.insert(0, "edge_id", [f"e{i + 1}" for i in range(len(p2_edges))])

    nodes.to_csv(DATA_PROCESSED / "nodes.csv", index=False)
    p2_edges.to_csv(DATA_PROCESSED / "edges.csv", index=False)
    if missing:
        print(f"{missing} predições sem nó correspondente no P1 (ignoradas)")

    # comparação P1 x P2 nas arestas entre entidades
    p1_set = set(zip(p1_entity_edges["source_id"], p1_entity_edges["target_id"], p1_entity_edges["relation"]))
    p2_set = set(zip(new_edges["source_id"], new_edges["target_id"], new_edges["relation"]))
    rows = []
    for relation in sorted({r for _, _, r in p1_set | p2_set}):
        a = {e for e in p1_set if e[2] == relation}
        b = {e for e in p2_set if e[2] == relation}
        rows.append({"relation": relation, "p1": len(a), "p2": len(b), "both": len(a & b),
                     "only_p1": len(a - b), "only_p2": len(b - a)})
    comparison = pd.DataFrame(rows)
    comparison.loc[len(comparison)] = ["TOTAL", *comparison.drop(columns="relation").sum()]
    comparison.to_csv(DATA_PROCESSED / "graph_comparison.csv", index=False)

    print(f"P1: {len(nodes)} nós, {len(p1_edges)} arestas | P2: {len(nodes)} nós, {len(p2_edges)} arestas")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
