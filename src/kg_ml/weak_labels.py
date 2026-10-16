"""Passo 3: rótulos automáticos (weak supervision).

Várias funções de rotulagem simples votam em cada par de treino. Cada uma
devolve um rótulo (a relação ou NONE) ou None quando não tem opinião. O rótulo
final é o voto da maioria; pares sem voto ou com empate são descartados.

Fontes:
- gatilhos do Projeto 1 (triggers.csv)
- gatilhos aprendidos por PMI (passo 2)
- heurísticas simples de distância, separadores e negação

A versão "sem P1" refaz a votação sem a função dos gatilhos do P1, para testar
se o modelo depende deles.

Uso: PYTHONPATH=src python3 -m kg_ml.weak_labels
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from kg_ml.config import DATA_INTERIM, DATA_PROCESSED, NONE_LABEL, PAIR_RELATIONS
from kg_ml.ngram_patterns import between_tokens, ngrams

from kg_extraction.features.triggers import load_triggers

POLARITY_TO_RELATION = {"positive": "CONFIRMS", "negative": "EXCLUDES", "neutral": "REVEALS"}
NEGATION_WORDS = {"no", "without", "absence", "negative", "not"}
FAR = 15


def contains_phrase(text, phrase: str) -> bool:
    if not isinstance(text, str):
        return False
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE) is not None


def default_relation(row) -> str:
    return PAIR_RELATIONS[(row.e1_type, row.e2_type)][0]


def is_exam_finding(row) -> bool:
    return row.e1_type == "exam" and row.e2_type in ("diagnosis", "symptom")


class LabelingFunctions:
    def __init__(self, learned_patterns: pd.DataFrame, entity_spans: dict):
        self.triggers = load_triggers()
        self.learned = {t: set(g["ngram"]) for t, g in learned_patterns.groupby("pair_type")}
        self.entity_spans = entity_spans  # (case_id, sent_idx) -> [(start, end, type)]

    def lf_p1_trigger(self, row):
        if row.pair_type == "diagnosis-treatment":
            for t in self.triggers:
                if t.use == "administration" and contains_phrase(row.between, t.phrase):
                    return "TREATED_BY"
        if is_exam_finding(row):
            for t in self.triggers:
                if t.use == "exam_result" and contains_phrase(row.between, t.phrase):
                    return POLARITY_TO_RELATION[t.polarity]
        return None

    def lf_learned_pattern(self, row):
        patterns = self.learned.get(row.pair_type, set())
        if patterns & ngrams(between_tokens(row.between)):
            return "REVEALS" if is_exam_finding(row) else default_relation(row)
        return None

    def lf_adjacent(self, row):
        # "lung adenocarcinoma", "CT of the abdomen": entidades coladas costumam se relacionar
        if is_exam_finding(row) or row.n_between > 2:
            return None
        if isinstance(row.between, str) and re.search(r"[,;()]", row.between):
            return None
        return default_relation(row)

    def lf_far(self, row):
        return NONE_LABEL if row.n_between > FAR else None

    def lf_separator(self, row):
        if isinstance(row.between, str) and re.search(r";|\bbut\b|\bwhereas\b|\bhowever\b", row.between, re.IGNORECASE):
            return NONE_LABEL
        return None

    def lf_entity_between(self, row):
        # outra entidade do mesmo tipo no meio: a relação provavelmente é com ela
        start = min(row.e1_end, row.e2_end)
        end = max(row.e1_start, row.e2_start)
        for s, e, etype in self.entity_spans.get((row.case_id, row.sent_idx), []):
            if s >= start and e <= end and etype in (row.e1_type, row.e2_type):
                return NONE_LABEL
        return None

    def lf_negation(self, row):
        if not is_exam_finding(row) or not row.e1_first:
            return None
        last_words = between_tokens(row.between)[-3:]
        if NEGATION_WORDS & set(last_words):
            return "EXCLUDES"
        return None

    def all(self, use_p1: bool = True):
        lfs = [self.lf_learned_pattern, self.lf_adjacent, self.lf_far,
               self.lf_separator, self.lf_entity_between, self.lf_negation]
        if use_p1:
            lfs.insert(0, self.lf_p1_trigger)
        return lfs


def majority_vote(votes: list) -> str | None:
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    ranked = Counter(votes).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def entity_spans_by_sentence(pairs: pd.DataFrame) -> dict:
    spans = {}
    for row in pairs.itertuples():
        key = (row.case_id, row.sent_idx)
        spans.setdefault(key, set()).add((row.e1_start, row.e1_end, row.e1_type))
        spans[key].add((row.e2_start, row.e2_end, row.e2_type))
    return spans


def apply_labeling_functions(pairs: pd.DataFrame, learned_patterns: pd.DataFrame) -> pd.DataFrame:
    lf = LabelingFunctions(learned_patterns, entity_spans_by_sentence(pairs))
    pairs = pairs.copy()
    for func in lf.all(use_p1=True):
        pairs[func.__name__] = [func(row) for row in pairs.itertuples()]

    lf_cols = [f.__name__ for f in lf.all(use_p1=True)]
    lf_cols_no_p1 = [f.__name__ for f in lf.all(use_p1=False)]
    pairs["weak_label"] = [majority_vote(list(v)) for v in pairs[lf_cols].itertuples(index=False)]
    pairs["weak_label_no_p1"] = [majority_vote(list(v)) for v in pairs[lf_cols_no_p1].itertuples(index=False)]
    return pairs


def lf_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    """Cobertura e conflito de cada função (tabela clássica de weak supervision)."""
    lf_cols = [c for c in labeled.columns if c.startswith("lf_")]
    rows = []
    for col in lf_cols:
        voted = labeled[col].notna()
        others = labeled.loc[voted, [c for c in lf_cols if c != col]]
        conflict = [
            any(o is not None and o == o and o != v for o in other_votes)
            for v, other_votes in zip(labeled.loc[voted, col], others.itertuples(index=False))
        ]
        rows.append({
            "labeling_function": col,
            "coverage": round(voted.mean(), 4),
            "votes": int(voted.sum()),
            "conflict_rate": round(sum(conflict) / max(1, len(conflict)), 4),
            "most_common_vote": labeled.loc[voted, col].mode().iloc[0] if voted.any() else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    pairs = pd.read_csv(DATA_INTERIM / "pairs_train.csv")
    learned = pd.read_csv(DATA_PROCESSED / "learned_patterns.csv")
    labeled = apply_labeling_functions(pairs, learned)
    labeled.to_csv(DATA_INTERIM / "pairs_train_labeled.csv", index=False)

    summary = lf_summary(labeled)
    summary.to_csv(DATA_PROCESSED / "labeling_functions.csv", index=False)
    print(summary.to_string(index=False))
    print("\nrótulos finais (com P1):")
    print(labeled["weak_label"].value_counts(dropna=False).to_string())
    print("\nrótulos finais (sem P1):")
    print(labeled["weak_label_no_p1"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
