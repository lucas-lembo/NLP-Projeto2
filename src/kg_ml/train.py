"""Passo 4 e avaliação: classificador de relações com N-gramas.

Atributos de cada par:
- N-gramas (TF-IDF) do texto entre as duas entidades
- palavras de contexto (3 antes e 3 depois do par)
- tipo do par, ordem das entidades e faixa de distância

Modelos: Regressão Logística e Multinomial Naive Bayes, treinados nos rótulos
automáticos do passo 3. Avaliação nos pares anotados dos 56 casos do Projeto 1,
comparando com as regras do Projeto 1.

Uso: PYTHONPATH=src python3 -m kg_ml.train
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kg_ml.config import DATA_ANNOTATION, DATA_INTERIM, DATA_PROCESSED, NONE_LABEL, PAIR_RELATIONS
from kg_ml.features import predict_allowed, prepare
from kg_ml.ngram_patterns import top_patterns
from kg_ml.weak_labels import LabelingFunctions, entity_spans_by_sentence, majority_vote

ANNOTATION_FILE = DATA_ANNOTATION / "test_pairs_to_annotate.csv"


def make_model(kind: str, ngram_max: int) -> Pipeline:
    features = ColumnTransformer([
        ("between", TfidfVectorizer(ngram_range=(1, ngram_max), token_pattern=r"\S+", min_df=3), "between"),
        ("context", CountVectorizer(token_pattern=r"\S+", min_df=3, binary=True), "context"),
        ("meta", OneHotEncoder(handle_unknown="ignore"), ["pair_type", "order", "distance"]),
    ])
    if kind == "logreg":
        classifier = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    else:
        classifier = MultinomialNB(alpha=0.5)
    return Pipeline([("features", features), ("clf", classifier)])


# ---------- métricas ----------

def relation_scores(gold: list[str], pred: list[str]) -> dict:
    """P/R/F1 sobre as relações (NONE não conta como acerto)."""
    tp = sum(g == p != NONE_LABEL for g, p in zip(gold, pred))
    fp = sum(p != NONE_LABEL and p != g for g, p in zip(gold, pred))
    fn = sum(g != NONE_LABEL and p != g for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # detecção: só "tem relação ou não", ignorando qual
    g_bin = [g != NONE_LABEL for g in gold]
    p_bin = [p != NONE_LABEL for p in pred]
    tp_b = sum(g and p for g, p in zip(g_bin, p_bin))
    prec_b = tp_b / sum(p_bin) if sum(p_bin) else 0.0
    rec_b = tp_b / sum(g_bin) if sum(g_bin) else 0.0
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if prec_b + rec_b else 0.0
    return {
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "detect_precision": round(prec_b, 3), "detect_recall": round(rec_b, 3), "detect_f1": round(f1_b, 3),
        "predicted_relations": sum(p_bin),
    }


def load_gold() -> pd.DataFrame | None:
    if not ANNOTATION_FILE.exists():
        return None
    sheet = pd.read_csv(ANNOTATION_FILE, dtype=str).fillna("")
    sheet["gold"] = np.where(sheet["gold_label"] != "", sheet["gold_label"], sheet["suggested_label"])
    not_reviewed = (sheet["gold_label"] == "").sum()
    if not_reviewed:
        print(f"AVISO: {not_reviewed} pares ainda sem revisão da equipe (usando a pré-anotação)")
    sheet = sheet[sheet["gold"] != ""]
    return sheet[["pair_id", "gold"]] if len(sheet) else None


def weak_vote_on(pairs: pd.DataFrame, train_pairs: pd.DataFrame) -> list[str]:
    """As próprias funções de rotulagem aplicadas no teste (sem voto = NONE)."""
    learned = top_patterns(pd.read_csv(DATA_INTERIM / "ngram_pmi_all.csv"))
    lf = LabelingFunctions(learned, entity_spans_by_sentence(pairs))
    labels = []
    for row in pairs.itertuples():
        vote = majority_vote([f(row) for f in lf.all(use_p1=True)])
        labels.append(vote or NONE_LABEL)
    return labels


def top_features(model: Pipeline, k: int = 15) -> pd.DataFrame:
    names = model.named_steps["features"].get_feature_names_out()
    clf = model.named_steps["clf"]
    rows = []
    for i, label in enumerate(clf.classes_):
        for j in np.argsort(clf.coef_[i])[::-1][:k]:
            rows.append({"label": label, "feature": names[j], "weight": round(clf.coef_[i][j], 3)})
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(DATA_INTERIM / "pairs_train_labeled.csv")
    test = pd.read_csv(DATA_INTERIM / "pairs_test.csv")

    experiments = [
        ("logreg", 1, "weak_label"), ("logreg", 2, "weak_label"), ("logreg", 3, "weak_label"),
        ("nb", 1, "weak_label"), ("nb", 2, "weak_label"), ("nb", 3, "weak_label"),
        ("logreg", 2, "weak_label_no_p1"),
    ]

    models = {}
    for kind, n, label_col in experiments:
        data = train[train[label_col].notna()]
        name = f"{kind}_ngram{n}" + ("_noP1" if label_col.endswith("no_p1") else "")
        print(f"treinando {name} em {len(data)} pares...")
        model = make_model(kind, n).fit(prepare(data), data[label_col])
        models[name] = model
        labels, scores = predict_allowed(model, test)
        test[f"pred_{name}"] = labels
        test[f"score_{name}"] = np.round(scores, 4)

    test["pred_weak_rules"] = weak_vote_on(test, train)
    test["pred_cooccurrence"] = [PAIR_RELATIONS[(a, b)][-1] if len(PAIR_RELATIONS[(a, b)]) == 1 else "REVEALS"
                                 for a, b in zip(test["e1_type"], test["e2_type"])]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    test.to_csv(DATA_PROCESSED / "pairs_test_predictions.csv", index=False)
    joblib.dump(models["logreg_ngram2"], DATA_PROCESSED / "relation_model.joblib")
    top_features(models["logreg_ngram2"]).to_csv(DATA_PROCESSED / "top_features.csv", index=False)

    gold = load_gold()
    if gold is None:
        print("sem anotações ainda: modelos treinados e predições salvas, avaliação pulada")
        return

    evaluated = test.merge(gold, on="pair_id")
    systems = {"p1_rules": "p1_label", "cooccurrence": "pred_cooccurrence", "weak_rules": "pred_weak_rules"}
    systems.update({name: f"pred_{name}" for name in models})

    results = []
    for system, col in systems.items():
        results.append({"system": system, **relation_scores(list(evaluated["gold"]), list(evaluated[col]))})
    results = pd.DataFrame(results)
    results.to_csv(DATA_PROCESSED / "metrics.csv", index=False)
    print(f"\navaliação em {len(evaluated)} pares anotados:")
    print(results.to_string(index=False))

    per_type = []
    for pair_type, group in evaluated.groupby("pair_type"):
        for system in ["p1_rules", "weak_rules", "logreg_ngram2"]:
            scores = relation_scores(list(group["gold"]), list(group[systems[system]]))
            per_type.append({"pair_type": pair_type, "system": system, "n": len(group),
                             "precision": scores["precision"], "recall": scores["recall"], "f1": scores["f1"]})
    pd.DataFrame(per_type).to_csv(DATA_PROCESSED / "metrics_by_pair_type.csv", index=False)

    labels = sorted(set(evaluated["gold"]) | set(evaluated["pred_logreg_ngram2"]))
    cm = confusion_matrix(evaluated["gold"], evaluated["pred_logreg_ngram2"], labels=labels)
    pd.DataFrame(cm, index=[f"gold_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]).to_csv(
        DATA_PROCESSED / "confusion_logreg_ngram2.csv")
    print("\n" + classification_report(evaluated["gold"], evaluated["pred_logreg_ngram2"], zero_division=0))


if __name__ == "__main__":
    main()
