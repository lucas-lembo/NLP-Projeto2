"""Sorteia os pares de teste para anotação manual (padrão-ouro).

Sorteio estratificado por tipo de par, só dos 56 casos do Projeto 1. A
planilha gerada tem a sentença com as entidades marcadas e as colunas:

- suggested_label: pré-anotação (feita pelo Claude, ver README)
- gold_label: rótulo final, revisado pela equipe
- reviewer / notes: quem revisou e observações

Uso: PYTHONPATH=src python3 -m kg_ml.annotation_sample
"""

from __future__ import annotations

import pandas as pd

from kg_ml.config import DATA_ANNOTATION, DATA_INTERIM, NONE_LABEL, PAIR_RELATIONS

N_PAIRS = 210
SEED = 7


def mark_sentence(row) -> str:
    spans = sorted(
        [(row.e1_start, row.e1_end, "E1"), (row.e2_start, row.e2_end, "E2")], reverse=True
    )
    text = row.sentence
    for start, end, tag in spans:
        text = text[:start] + f"[{tag} " + text[start:end] + "]" + text[end:]
    return text


def main() -> None:
    if (DATA_ANNOTATION / "test_pairs_to_annotate.csv").exists():
        print("planilha de anotação já existe; apague-a antes se quiser sortear de novo")
        return
    pairs = pd.read_csv(DATA_INTERIM / "pairs_test.csv")
    per_type = N_PAIRS // pairs["pair_type"].nunique()
    sample = pd.concat([
        group.sample(n=min(per_type, len(group)), random_state=SEED)
        for _, group in pairs.groupby("pair_type")
    ])
    # completa até N_PAIRS com pares aleatórios dos tipos que sobraram
    rest = pairs.drop(sample.index)
    missing = N_PAIRS - len(sample)
    if missing > 0:
        sample = pd.concat([sample, rest.sample(n=min(missing, len(rest)), random_state=SEED)])

    sample = sample.sample(frac=1, random_state=SEED)  # embaralha para não anotar em blocos
    sheet = pd.DataFrame({
        "pair_id": sample["pair_id"],
        "case_id": sample["case_id"],
        "pair_type": sample["pair_type"],
        "e1": sample["e1_label"],
        "e2": sample["e2_label"],
        "sentence": [mark_sentence(r) for r in sample.itertuples()],
        "allowed_labels": [
            " | ".join(PAIR_RELATIONS[(r.e1_type, r.e2_type)] + [NONE_LABEL]) for r in sample.itertuples()
        ],
        "suggested_label": "",
        "gold_label": "",
        "reviewer": "",
        "notes": "",
    })
    DATA_ANNOTATION.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(DATA_ANNOTATION / "test_pairs_to_annotate.csv", index=False)
    print(sheet["pair_type"].value_counts().to_string())
    print(f"{len(sheet)} pares salvos em {DATA_ANNOTATION / 'test_pairs_to_annotate.csv'}")


if __name__ == "__main__":
    main()
