"""Gera os pares candidatos de entidades (mesma sentença) usando o NER do Projeto 1.

Cada linha da saída é um par (entidade1, entidade2) de tipos que podem ter
relação (ver PAIR_RELATIONS), com o texto entre eles, o contexto ao redor e o
que o Projeto 1 teria dito sobre esse par (`p1_label`).

- Amostra de treino: só match exato (o fuzzy do P1 é lento demais para milhares de casos).
- 56 casos do Projeto 1 (teste): NER completo do P1, igual ao que gerou o grafo dele.

Uso: PYTHONPATH=src python3 -m kg_ml.candidates
"""

from __future__ import annotations

import pickle

import pandas as pd

from kg_ml.config import DATA_INTERIM, DATA_RAW, MAX_TOKENS_BETWEEN, P1_CASES, PAIR_RELATIONS

from kg_extraction.config import GAZETTEER_FILES
from kg_extraction.features.gazetteer_ner import EntityMatch, _find_exact_matches, find_entities_in_sentence, load_all_gazetteers
from kg_extraction.features.preprocessing import split_sentences, tokenize_with_spans
from kg_extraction.features.relations import assign_history, build_exam_result, find_finding, reclassify_treatment_matches
from kg_extraction.features.triggers import find_triggers_in_sentence, load_triggers
from kg_extraction.features.value_unit_extraction import build_value_unit_pattern, find_value_units, load_units

GAZETTEER_CACHE = DATA_INTERIM / "gazetteers.pkl"
CONTEXT_TOKENS = 3


def load_gazetteers():
    if GAZETTEER_CACHE.exists():
        return pickle.loads(GAZETTEER_CACHE.read_bytes())
    print("montando gazetteers do Projeto 1 (só na primeira vez, ~2-3 min)...")
    gazetteers = load_all_gazetteers(GAZETTEER_FILES)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    GAZETTEER_CACHE.write_bytes(pickle.dumps(gazetteers))
    return gazetteers


def find_entities_exact(sentence: str, gazetteers, sent_idx: int) -> list[EntityMatch]:
    """Mesma fase exata do NER do Projeto 1, sem a fase fuzzy."""
    occupied = [False] * len(sentence)
    matches = []
    for start, end, entity_type, synonym in _find_exact_matches(sentence, gazetteers):
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        entry = gazetteers[entity_type].entry_for(synonym)
        matches.append(EntityMatch(entity_type, entry.mesh_ui, entry.canonical, sentence[start:end], start, end, sent_idx, "exact", 0))
    matches.sort(key=lambda m: m.start)
    return matches


def words(text: str) -> list[str]:
    return [w.lower() for w, _, _ in tokenize_with_spans(text)]


def p1_exam_links(exams, matches, sentence, value_units, result_triggers) -> dict:
    """Para cada exame, a aresta CONFIRMS/EXCLUDES/REVEALS que o Projeto 1 criaria."""
    links = {}
    for exam in exams:
        if build_exam_result(exam, value_units):
            continue  # P1 liga o exame a um valor, não a um diagnóstico
        finding = find_finding(exam, sentence, result_triggers, matches)
        if finding and "linked_key" in finding:
            links[id(exam)] = (finding["linked_key"], finding["relation"])
    return links


def case_pairs(case_id: str, text: str, gazetteers, triggers, unit_pattern, fuzzy: bool) -> list[dict]:
    rows = []
    last_diagnosis = None  # para reproduzir o TREATED_BY do P1 ("último diagnóstico visto")
    seen_treatments = set()

    for sent_idx, sentence in enumerate(split_sentences(text)):
        if fuzzy:
            matches = find_entities_in_sentence(sentence, gazetteers, sent_idx)
        else:
            matches = find_entities_exact(sentence, gazetteers, sent_idx)
        if not matches:
            continue

        value_units = find_value_units(sentence, unit_pattern)
        admin_triggers = find_triggers_in_sentence(sentence, triggers, "administration")
        result_triggers = find_triggers_in_sentence(sentence, triggers, "exam_result")
        history_triggers = find_triggers_in_sentence(sentence, triggers, "medical_history")

        # Mesmas correções de tipo do Projeto 1: tratamento com valor vira exame,
        # e entidade depois de "history of" vira histórico (fica fora dos pares).
        reclassified = reclassify_treatment_matches(matches, value_units, admin_triggers)
        history = assign_history(matches, history_triggers) - reclassified
        entities = []
        for i, m in enumerate(matches):
            if i in history:
                continue
            etype = "exam" if i in reclassified else m.entity_type
            entities.append((etype, m))

        # TREATED_BY do P1: o tratamento (na 1a menção no caso) liga ao último diagnóstico visto.
        p1_treated_by = set()
        for etype, m in entities:
            if etype == "diagnosis":
                last_diagnosis = m.canonical_label
            elif etype == "treatment" and m.canonical_label not in seen_treatments:
                seen_treatments.add(m.canonical_label)
                if last_diagnosis:
                    p1_treated_by.add((last_diagnosis, m.canonical_label))

        exams = [m for etype, m in entities if etype == "exam"]
        exam_links = p1_exam_links(exams, matches, sentence, value_units, result_triggers)

        for type1, e1 in entities:
            for type2, e2 in entities:
                if (type1, type2) not in PAIR_RELATIONS:
                    continue
                first, second = (e1, e2) if e1.start < e2.start else (e2, e1)
                between = sentence[first.end:second.start]
                n_between = len(words(between))
                if n_between > MAX_TOKENS_BETWEEN:
                    continue

                relation = PAIR_RELATIONS[(type1, type2)][0]
                if relation == "TREATED_BY":
                    p1_label = relation if (e1.canonical_label, e2.canonical_label) in p1_treated_by else "NONE"
                elif type1 == "exam" and type2 in ("diagnosis", "symptom"):
                    link = exam_links.get(id(e1))
                    p1_label = link[1] if link and link[0] == (type2, e2.canonical_label) else "NONE"
                else:
                    p1_label = relation  # SUPPORTS / LOCATED_IN / TARGETS / PERFORMED_ON: co-ocorrência

                rows.append({
                    "case_id": case_id,
                    "sent_idx": sent_idx,
                    "sentence": sentence,
                    "pair_type": f"{type1}-{type2}",
                    "e1_type": type1,
                    "e1_label": e1.canonical_label,
                    "e1_text": e1.matched_text,
                    "e1_start": e1.start,
                    "e1_end": e1.end,
                    "e2_type": type2,
                    "e2_label": e2.canonical_label,
                    "e2_text": e2.matched_text,
                    "e2_start": e2.start,
                    "e2_end": e2.end,
                    "e1_first": e1.start < e2.start,
                    "between": between,
                    "n_between": n_between,
                    "left": " ".join(words(sentence[:first.start])[-CONTEXT_TOKENS:]),
                    "right": " ".join(words(sentence[second.end:])[:CONTEXT_TOKENS]),
                    "p1_label": p1_label,
                })
    return rows


def build_pairs(cases: pd.DataFrame, fuzzy: bool) -> pd.DataFrame:
    gazetteers = load_gazetteers()
    triggers = load_triggers()
    unit_pattern = build_value_unit_pattern(load_units())
    rows = []
    for n, case in enumerate(cases.itertuples(), 1):
        rows.extend(case_pairs(case.case_id, str(case.case_text), gazetteers, triggers, unit_pattern, fuzzy))
        if n % 500 == 0:
            print(f"  {n}/{len(cases)} casos, {len(rows)} pares")
    pairs = pd.DataFrame(rows)
    pairs.insert(0, "pair_id", [f"p{i}" for i in range(len(pairs))])
    return pairs


def main() -> None:
    print("pares de teste (56 casos do Projeto 1)...")
    test = build_pairs(pd.read_csv(P1_CASES), fuzzy=True)
    test.to_csv(DATA_INTERIM / "pairs_test.csv", index=False)
    print(f"{len(test)} pares de teste")

    print("pares de treino (amostra do MultiCaRe)...")
    train = build_pairs(pd.read_csv(DATA_RAW / "sample_cases.csv"), fuzzy=False)
    train.to_csv(DATA_INTERIM / "pairs_train.csv", index=False)
    print(f"{len(train)} pares de treino")


if __name__ == "__main__":
    main()
