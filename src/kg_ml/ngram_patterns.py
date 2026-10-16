"""Passo 2: N-gramas + PMI.

(a) Padrões de relação: quais N-gramas aparecem entre duas entidades de um
    tipo de par (ex. diagnosis-treatment) muito mais do que entre pares em geral.

        PMI(g, T) = log2( P(g | T) / P(g) )

    Resultado: lista de gatilhos aprendidos por tipo de par, comparada com o
    triggers.csv escrito à mão no Projeto 1.

(b) Associação entre conceitos: PMI entre dois conceitos que aparecem no mesmo
    caso clínico (ex. "Tuberculosis" e "Isoniazid"), calculado na amostra inteira.
    Vira uma camada extra do grafo, de "conhecimento" do corpus.

Uso: PYTHONPATH=src python3 -m kg_ml.ngram_patterns
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import pandas as pd

from kg_ml.config import DATA_INTERIM, DATA_PROCESSED, PAIR_RELATIONS

from kg_extraction.features.preprocessing import tokenize_with_spans
from kg_extraction.features.triggers import load_triggers
from kg_extraction.features.value_unit_extraction import load_units

MAX_BETWEEN_FOR_PATTERNS = 8   # padrões só de pares próximos, onde a relação costuma estar
MIN_COUNT_PATTERN = 20
MIN_COUNT_CONCEPT = 5
TOP_K = 25

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "was", "were", "is", "are", "be", "been", "had", "has", "have", "that", "this", "which", "who",
    "his", "her", "their", "its", "it", "he", "she", "they", "also", "after", "before", "due", "both",
    "left", "right", "but", "not", "no", "into", "than", "then", "there", "these", "those", "while",
}


# pedaços de unidades ("g/dL" vira "g dl" na tokenização) não são padrão de relação
UNIT_WORDS = {piece.lower() for unit in load_units() for piece in unit.replace("^", "/").split("/")}


def ngrams(tokens: list[str], n_max: int = 3) -> set[str]:
    result = set()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            result.add(" ".join(tokens[i:i + n]))
    return result


def is_informative(gram: str) -> bool:
    tokens = gram.split()
    if any(any(ch.isdigit() for ch in t) or t in UNIT_WORDS or len(t) == 1 for t in tokens):
        return False
    return not all(t in STOPWORDS for t in tokens)


def between_tokens(text) -> list[str]:
    if not isinstance(text, str):
        return []
    return [w.lower() for w, _, _ in tokenize_with_spans(text)]


def relation_patterns(pairs: pd.DataFrame) -> pd.DataFrame:
    close = pairs[pairs["n_between"] <= MAX_BETWEEN_FOR_PATTERNS]
    count_gram_type = Counter()
    count_gram = Counter()
    count_type = Counter()
    for pair_type, between in zip(close["pair_type"], close["between"]):
        grams = ngrams(between_tokens(between))
        count_type[pair_type] += 1
        for g in grams:
            count_gram[g] += 1
            count_gram_type[(g, pair_type)] += 1

    total = sum(count_type.values())
    rows = []
    for (g, pair_type), c in count_gram_type.items():
        if c < MIN_COUNT_PATTERN or not is_informative(g):
            continue
        p_g_given_t = c / count_type[pair_type]
        p_g = count_gram[g] / total
        rows.append({
            "pair_type": pair_type,
            "ngram": g,
            "n": len(g.split()),
            "count": c,
            "p_given_type": round(p_g_given_t, 4),
            "pmi": round(math.log2(p_g_given_t / p_g), 3),
        })
    patterns = pd.DataFrame(rows).sort_values(["pair_type", "pmi"], ascending=[True, False])
    return patterns


def top_patterns(patterns: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    positive = patterns[patterns["pmi"] > 0]
    return positive.groupby("pair_type").head(k).reset_index(drop=True)


def compare_with_p1_triggers(patterns: pd.DataFrame) -> pd.DataFrame:
    """Onde cada gatilho manual do P1 aparece na lista aprendida (se aparece)."""
    use_to_types = {
        "administration": ["diagnosis-treatment"],
        "exam_result": ["exam-diagnosis", "exam-symptom"],
    }
    rows = []
    for trigger in load_triggers():
        for pair_type in use_to_types.get(trigger.use, []):
            ranked = patterns[(patterns["pair_type"] == pair_type) & (patterns["pmi"] > 0)].reset_index(drop=True)
            hit = ranked.index[ranked["ngram"] == trigger.phrase]
            rows.append({
                "trigger": trigger.phrase,
                "use": trigger.use,
                "pair_type": pair_type,
                "rank": int(hit[0]) + 1 if len(hit) else None,
                "pmi": ranked.loc[hit[0], "pmi"] if len(hit) else None,
                "count": int(ranked.loc[hit[0], "count"]) if len(hit) else None,
            })
    return pd.DataFrame(rows)


def concept_associations(pairs: pd.DataFrame) -> pd.DataFrame:
    """PMI entre conceitos que aparecem no mesmo caso (contagem por caso)."""
    valid_types = {f"{a}-{b}" for a, b in PAIR_RELATIONS}
    concepts_per_case = {}
    for row in pairs.itertuples():
        concepts = concepts_per_case.setdefault(row.case_id, set())
        concepts.add((row.e1_type, row.e1_label))
        concepts.add((row.e2_type, row.e2_label))

    n_cases = len(concepts_per_case)
    count_concept = Counter()
    count_pair = Counter()
    for concepts in concepts_per_case.values():
        count_concept.update(concepts)
        for a, b in combinations(sorted(concepts), 2):
            if f"{a[0]}-{b[0]}" in valid_types:
                count_pair[(a, b)] += 1
            elif f"{b[0]}-{a[0]}" in valid_types:
                count_pair[(b, a)] += 1

    rows = []
    for (a, b), c in count_pair.items():
        if c < MIN_COUNT_CONCEPT:
            continue
        pmi = math.log2(c * n_cases / (count_concept[a] * count_concept[b]))
        if pmi <= 0:
            continue
        rows.append({
            "source_type": a[0], "source": a[1],
            "target_type": b[0], "target": b[1],
            "cases_together": c, "pmi": round(pmi, 3),
        })
    return pd.DataFrame(rows).sort_values("pmi", ascending=False)


def main() -> None:
    pairs = pd.read_csv(DATA_INTERIM / "pairs_train.csv")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    patterns = relation_patterns(pairs)
    patterns.to_csv(DATA_INTERIM / "ngram_pmi_all.csv", index=False)
    learned = top_patterns(patterns)
    learned.to_csv(DATA_PROCESSED / "learned_patterns.csv", index=False)

    comparison = compare_with_p1_triggers(patterns)
    comparison.to_csv(DATA_PROCESSED / "p1_triggers_vs_pmi.csv", index=False)

    concepts = concept_associations(pairs)
    concepts.to_csv(DATA_PROCESSED / "concept_pmi.csv", index=False)

    for pair_type, group in learned.groupby("pair_type"):
        print(f"\n{pair_type}: " + ", ".join(group["ngram"].head(10)))
    found = comparison["rank"].notna().sum()
    print(f"\ngatilhos do P1 encontrados entre os padrões com PMI > 0: {found}/{len(comparison)}")
    print(f"{len(concepts)} associações entre conceitos com PMI > 0")


if __name__ == "__main__":
    main()
