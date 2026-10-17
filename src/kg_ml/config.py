"""Caminhos e constantes do Projeto 2.

O código do Projeto 1 é usado como biblioteca (NER, gatilhos, valor+unidade),
sem nenhuma alteração. Por padrão ele fica em ../NLP-Projeto1 em relação a
esta pasta; dá pra trocar com a variável de ambiente P1_DIR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Tipos de par considerados e relações possíveis ficam em features.py.
from kg_ml.features import NONE_LABEL, PAIR_RELATIONS  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]

P1_DIR = Path(os.environ.get("P1_DIR", ROOT.parent / "NLP-Projeto1")).resolve()
sys.path.insert(0, str(P1_DIR / "src"))

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ANNOTATION = ROOT / "data" / "annotation"
IMAGES = ROOT / "assets" / "images"

# Casos e grafo do Projeto 1 (conjunto de avaliação).
P1_CASES = P1_DIR / "data" / "raw" / "multicare" / "cases.csv"
P1_NODES = P1_DIR / "data" / "processed" / "nodes.csv"
P1_EDGES = P1_DIR / "data" / "processed" / "edges.csv"

# Amostra do MultiCaRe usada para treino (estatística de N-gramas + classificador).
SAMPLE_SIZE = 4000
SAMPLE_SEED = 42

# Só olhamos pares com no máximo esse número de tokens entre as entidades.
MAX_TOKENS_BETWEEN = 30
