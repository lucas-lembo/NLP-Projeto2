"""Gera as figuras do relatório em assets/images/.

Uso: PYTHONPATH=src python3 -m kg_ml.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from kg_ml.config import DATA_PROCESSED, IMAGES

TYPE_COLORS = {
    "diagnosis": "#d62728", "symptom": "#ff7f0e", "exam": "#1f77b4",
    "treatment": "#2ca02c", "anatomical_site": "#9467bd",
}


def plot_metrics() -> None:
    metrics = pd.read_csv(DATA_PROCESSED / "metrics.csv").set_index("system")
    systems = ["p1_rules", "weak_rules", "nb_ngram1", "logreg_ngram1", "logreg_ngram2", "logreg_ngram3", "logreg_ngram2_noP1"]
    names = ["Regras P1", "Rótulos\nautomáticos", "NB\nuni", "LR\nuni", "LR\nuni+bi", "LR\nuni+bi+tri", "LR uni+bi\nsem P1"]
    metrics = metrics.loc[systems]
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (col, label) in enumerate([("precision", "Precisão"), ("recall", "Recall"), ("f1", "F1")]):
        ax.bar(x + (i - 1) * 0.27, metrics[col], width=0.27, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("valor")
    ax.set_title("Relações nos pares anotados: regras do Projeto 1 x modelos do Projeto 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "metricas.png", dpi=150)
    plt.close(fig)


def plot_graph_comparison() -> None:
    comparison = pd.read_csv(DATA_PROCESSED / "graph_comparison.csv")
    comparison = comparison[comparison["relation"] != "TOTAL"]
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, comparison["p1"], width=0.4, label="Projeto 1 (regras)")
    ax.bar(x + 0.2, comparison["p2"], width=0.4, label="Projeto 2 (modelo)")
    ax.set_xticks(x, comparison["relation"], rotation=30)
    ax.set_ylabel("nº de arestas")
    ax.set_title("Arestas entre entidades nos 56 casos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "comparacao-grafos.png", dpi=150)
    plt.close(fig)


def plot_concept_graph(min_cases: int = 15, top: int = 45) -> None:
    concepts = pd.read_csv(DATA_PROCESSED / "concept_pmi.csv")
    concepts = concepts[concepts["cases_together"] >= min_cases].head(top)
    graph = nx.Graph()
    for row in concepts.itertuples():
        graph.add_node(row.source, kind=row.source_type)
        graph.add_node(row.target, kind=row.target_type)
        graph.add_edge(row.source, row.target, weight=row.pmi)

    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(graph, k=1.2, iterations=200, seed=1)
    colors = [TYPE_COLORS[graph.nodes[n]["kind"]] for n in graph.nodes]
    widths = [graph.edges[e]["weight"] / 2 for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=300, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    for kind, color in TYPE_COLORS.items():
        ax.scatter([], [], color=color, label=kind)
    ax.legend(loc="lower left")
    ax.set_title(f"Associações por PMI entre conceitos (amostra MultiCaRe, ≥{min_cases} casos juntos)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMAGES / "grafo-associacao-pmi.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_metrics()
    plot_graph_comparison()
    plot_concept_graph()
    print(f"figuras salvas em {IMAGES}")


if __name__ == "__main__":
    main()
