#!/usr/bin/env bash
# Roda o pipeline inteiro do Projeto 2, na ordem dos passos.
# Uso (de dentro de project2/): bash pipelines/workflows/run_all.sh
# O código do Projeto 1 precisa estar em ../../NLP-Projeto1 ou em $P1_DIR.
set -e
export PYTHONPATH=src

python3 -m kg_ml.download_multicare     # passo 1: amostra do MultiCaRe
python3 -m kg_ml.candidates             # passo 1: NER do P1 + pares candidatos
python3 -m kg_ml.ngram_patterns         # passo 2: N-gramas + PMI
python3 -m kg_ml.weak_labels            # passo 3: rótulos automáticos
python3 -m kg_ml.train                  # passo 4: classificadores + avaliação
python3 -m kg_ml.build_graph            # passo 5: grafo do P2 e comparação com o P1
python3 -m kg_ml.figures                # figuras do relatório
python3 -m kg_ml.build_viewer           # visualizador P1 × P2
