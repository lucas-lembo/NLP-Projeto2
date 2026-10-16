# Código do Projeto 2

Pacote `kg_ml`. Ele usa o código do Projeto 1 (`kg_extraction`) como biblioteca, sem alterá-lo. Por padrão, o Projeto 1 é procurado em `../../NLP-Projeto1` (relativo a `project2/`). Para outro local, use a variável `P1_DIR`.

## Como rodar

```bash
cd project2
pip install -r requirements.txt
bash pipelines/workflows/run_all.sh        # ou os módulos um por um:
PYTHONPATH=src python3 -m kg_ml.<módulo>
```

Na primeira vez leva cerca de 6 minutos: o download tem 168 MB e o NER dos 4 mil casos leva ~3,5 min. Se o download pelo Python falhar com erro de SSL, baixe com `curl -L -o data/raw/cases.parquet https://zenodo.org/api/records/20416562/files/cases.parquet/content` e rode de novo.

## Módulos (na ordem do pipeline)

| Passo | Módulo | Entrada | Saída |
|---|---|---|---|
| 1 | `download_multicare.py` | Zenodo | `data/raw/sample_cases.csv` (4.000 casos, sem os artigos do P1) |
| 1 | `candidates.py` | casos + NER do P1 | `data/interim/pairs_{train,test}.csv` (pares de entidades na mesma sentença) |
| 2 | `ngram_patterns.py` | pares de treino | `learned_patterns.csv`, `p1_triggers_vs_pmi.csv`, `concept_pmi.csv` |
| 3 | `weak_labels.py` | pares + padrões | `data/interim/pairs_train_labeled.csv`, `labeling_functions.csv` |
| — | `annotation_sample.py` | pares de teste | `data/annotation/test_pairs_to_annotate.csv` (210 pares para revisão) |
| 4 | `train.py` | pares rotulados + anotação | `metrics*.csv`, `pairs_test_predictions.csv`, `relation_model.joblib` |
| 5 | `build_graph.py` | predições + grafo do P1 | `data/processed/{nodes,edges}.csv`, `graph_comparison.csv` |
| — | `figures.py` | resultados | `assets/images/*.png` |

`config.py` guarda caminhos e parâmetros. `features.py` tem os atributos do classificador e não depende do P1, por isso a demonstração consegue usar o modelo salvo.

## Ideia de cada passo

- **Pares candidatos:** todo par de entidades na mesma sentença cujos tipos podem se relacionar (ex.: diagnóstico–tratamento), com no máximo 30 tokens entre eles. Para cada par, guardamos também o que as regras do P1 diriam (`p1_label`).
- **N-gramas + PMI:** `PMI(g, T) = log2 P(g|T) / P(g)`, onde `g` é um N-grama (1 a 3 palavras) do texto entre as entidades e `T` é o tipo do par. Os N-gramas de maior PMI viram gatilhos aprendidos. Também calculamos o PMI entre conceitos que aparecem no mesmo caso, o que gera o grafo de associação.
- **Rótulos automáticos:** 7 funções simples votam em cada par (gatilhos do P1, gatilhos aprendidos, entidades coladas, distância grande, separadores, outra entidade no meio, negação). O rótulo é a maioria; empate ou nenhum voto descarta o par.
- **Classificador:** atributos são TF-IDF dos N-gramas entre as entidades, palavras de contexto, tipo do par, ordem e distância. Comparamos Regressão Logística e Naive Bayes, com unigramas, uni+bigramas e uni+bi+trigramas. No teste sem os gatilhos do P1, os rótulos são gerados sem a função `lf_p1_trigger`, para ver se o modelo depende deles.
- **Avaliação:** feita só nos 56 casos do P1, que não entram no treino. P/R/F1 das relações, onde acertar exige o rótulo exato e `NONE` não conta como acerto. "Detecção" mede só se existe relação ou não.
