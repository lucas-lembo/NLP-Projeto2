"""Atributos dos pares e predição restrita ao tipo do par.

Fica separado do resto porque não depende do código do Projeto 1: a
demonstração (notebook) importa só este arquivo para usar o modelo salvo.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NONE_LABEL = "NONE"
PAIR_RELATIONS = {
    ("diagnosis", "treatment"): ["TREATED_BY"],
    ("symptom", "diagnosis"): ["SUPPORTS"],
    ("diagnosis", "anatomical_site"): ["LOCATED_IN"],
    ("treatment", "anatomical_site"): ["TARGETS"],
    ("exam", "anatomical_site"): ["PERFORMED_ON"],
    ("exam", "diagnosis"): ["CONFIRMS", "EXCLUDES", "REVEALS"],
    ("exam", "symptom"): ["CONFIRMS", "EXCLUDES", "REVEALS"],
}


def clean_between(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(re.findall(r"[a-z][a-z\-]*|[;:,()]", text.lower()))


def distance_bin(n: int) -> str:
    if n <= 2:
        return "0-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def prepare(pairs: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=pairs.index)
    X["between"] = pairs["between"].map(clean_between)
    left = pairs["left"].fillna("").map(lambda s: " ".join("L_" + w for w in s.split()))
    right = pairs["right"].fillna("").map(lambda s: " ".join("R_" + w for w in s.split()))
    X["context"] = left + " " + right
    X["pair_type"] = pairs["pair_type"]
    X["order"] = np.where(pairs["e1_first"], "e1_first", "e2_first")
    X["distance"] = pairs["n_between"].map(distance_bin)
    return X


def predict_allowed(model, pairs: pd.DataFrame) -> tuple[list[str], list[float]]:
    """Escolhe a classe mais provável entre as permitidas para o tipo do par."""
    proba = model.predict_proba(prepare(pairs))
    classes = list(model.classes_)
    labels, scores = [], []
    for row_proba, e1_type, e2_type in zip(proba, pairs["e1_type"], pairs["e2_type"]):
        allowed = PAIR_RELATIONS[(e1_type, e2_type)] + [NONE_LABEL]
        best = max((c for c in allowed if c in classes), key=lambda c: row_proba[classes.index(c)])
        labels.append(best)
        # "confiança da relação" = 1 - P(NONE), útil como peso da aresta
        scores.append(1.0 - row_proba[classes.index(NONE_LABEL)])
    return labels, scores
