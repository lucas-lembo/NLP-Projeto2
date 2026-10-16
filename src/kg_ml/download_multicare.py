"""Baixa o cases.parquet do MultiCaRe (Zenodo) e sorteia uma amostra de casos.

Os artigos que já estão nos 56 casos do Projeto 1 ficam de fora, porque esses
casos são o nosso conjunto de avaliação.

Uso: PYTHONPATH=src python3 -m kg_ml.download_multicare
"""

from __future__ import annotations

import pandas as pd
import requests

from kg_ml.config import DATA_RAW, P1_CASES, SAMPLE_SEED, SAMPLE_SIZE

ZENODO_CONCEPT_ID = "10079369"


def download(url: str, destination, expected_size: int) -> None:
    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"{destination.name} já baixado")
        return
    print(f"baixando {destination.name} ({expected_size / 1e6:.0f} MB)...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    record = requests.get(f"https://zenodo.org/api/records/{ZENODO_CONCEPT_ID}/versions/latest", timeout=60).json()
    print(f"MultiCaRe versão {record['metadata']['version']}")
    files = {f["key"]: f for f in record["files"]}
    parquet_path = DATA_RAW / "cases.parquet"
    download(files["cases.parquet"]["links"]["self"], parquet_path, files["cases.parquet"]["size"])

    # Cada linha do parquet é um artigo com uma lista de casos.
    articles = pd.read_parquet(parquet_path)
    cases = articles.explode("cases").dropna(subset=["cases"])
    cases = pd.concat([cases[["article_id"]].reset_index(drop=True), pd.json_normalize(cases["cases"].tolist())], axis=1)
    print(f"{len(cases)} casos no dataset completo")

    p1_articles = set(pd.read_csv(P1_CASES)["article_id"])
    cases = cases[~cases["article_id"].isin(p1_articles)]
    cases = cases[cases["case_text"].str.len() > 200]

    sample = cases.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
    sample = sample[["article_id", "case_id", "age", "gender", "case_text"]]
    sample.to_csv(DATA_RAW / "sample_cases.csv", index=False)
    print(f"amostra de {len(sample)} casos salva em {DATA_RAW / 'sample_cases.csv'}")


if __name__ == "__main__":
    main()
