from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import m2cgen as m2c
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


sys.setrecursionlimit(100000)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "data0120.csv"
FEATURES = [
    "temparature1",
    "humidity1",
    "valtage1",
    "usage1",
    "rotation1",
    "temparature2",
    "humidity2",
    "valtage2",
    "usage2",
    "rotation2",
    "temparature3",
    "humidity3",
    "valtage3",
    "usage3",
    "rotation3",
]
ROTATION_FEATURES = {"rotation1", "rotation2", "rotation3"}
THRESHOLD_RECALL_TARGET = 0.995


def js_global(filename: str, value: dict, global_name: str) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    source = (
        "(function(g){\n"
        f"  g.{global_name} = {payload};\n"
        "})(typeof window !== 'undefined' ? window : globalThis);\n"
    )
    (APP_DIR / filename).write_text(source, encoding="utf-8")


def rounded_row(values) -> list[float]:
    return [round(float(value), 1) for value in values]


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    X = frame[FEATURES].astype(float)
    y = (frame["fault_flag"] == "Yes").astype(int)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]
    positive_validation = sorted(p_val[y_val.to_numpy() == 1])
    threshold_index = math.floor((1 - THRESHOLD_RECALL_TARGET) * len(positive_validation))
    threshold = float(positive_validation[threshold_index])

    p_test = model.predict_proba(X_test)[:, 1]
    test_pred = p_test >= threshold
    rounded_test_values = X_test.round(1)
    p_rounded_test = model.predict_proba(rounded_test_values)[:, 1]
    rounded_test_pred = p_rounded_test >= threshold
    test_roc_auc = float(roc_auc_score(y_test, p_test))
    metrics = {
        "valRocAuc": float(roc_auc_score(y_val, p_val)),
        "testRocAuc": test_roc_auc,
        "testRecall": float(recall_score(y_test, test_pred, zero_division=0)),
        "testPrecision": float(precision_score(y_test, test_pred, zero_division=0)),
        "testVisitReductionRate": float((p_test < threshold).mean()),
        "testMissedCount": int(((y_test.to_numpy() == 1) & (~test_pred)).sum()),
        "roundedTestRecall": float(recall_score(y_test, rounded_test_pred, zero_division=0)),
        "roundedTestPrecision": float(precision_score(y_test, rounded_test_pred, zero_division=0)),
        "roundedTestVisitReductionRate": float((p_rounded_test < threshold).mean()),
        "roundedTestMissedCount": int(((y_test.to_numpy() == 1) & (~rounded_test_pred)).sum()),
    }

    feature_meta = {}
    for feature in FEATURES:
        train_values = X_train[feature].astype(float)
        train_min = float(train_values.min())
        train_max = float(train_values.max())
        feature_range = train_max - train_min
        hard_min = train_min - 0.2 * feature_range
        hard_max = train_max + 0.2 * feature_range
        item = {
            "p1": round(float(train_values.quantile(0.01)), 1),
            "p99": round(float(train_values.quantile(0.99)), 1),
            "hardMin": hard_min,
            "hardMax": hard_max,
            "isRotation": feature in ROTATION_FEATURES,
        }
        if feature in ROTATION_FEATURES:
            item["allowed"] = sorted(float(v) for v in train_values.unique())
        feature_meta[feature] = item

    def decision(values) -> bool:
        return float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1]) >= threshold

    test_indices = list(X_test.index)
    fault_demo = None
    safe_demo = None
    for index in test_indices:
        full_values = X_test.loc[index].to_numpy(dtype=float)
        expected_fault = bool(y_test.loc[index])
        if bool(decision(full_values)) != expected_fault:
            continue
        display_values = rounded_row(full_values)
        rounded_decision = decision(display_values)
        if rounded_decision != expected_fault:
            continue
        demo = {
            "features": display_values,
            "expectedDecision": "fault" if expected_fault else "safe",
        }
        if expected_fault and fault_demo is None:
            fault_demo = demo
        if not expected_fault and safe_demo is None:
            safe_demo = demo
        if fault_demo is not None and safe_demo is not None:
            break
    if fault_demo is None or safe_demo is None:
        raise RuntimeError("Could not find rounded test rows that preserve both demo decisions")

    generated_model = m2c.export_to_javascript(model)
    model_source = (
        "(function(g){\n"
        f"{generated_model}\n"
        "  g.ADKModel = {\n"
        "    score: score,\n"
        "    faultProbability: function(input) {\n"
        "      var output = score(input);\n"
        "      return Array.isArray(output) ? Number(output[1]) : Number(output);\n"
        "    }\n"
        "  };\n"
        "})(typeof window !== 'undefined' ? window : globalThis);\n"
    )
    (APP_DIR / "model.js").write_text(model_source, encoding="utf-8")

    meta = {
        "features": FEATURES,
        "threshold": threshold,
        "metrics": metrics,
        "featureMeta": feature_meta,
        "demoRows": {"fault": fault_demo, "safe": safe_demo},
    }
    js_global("model_meta.js", meta, "ADK_META")

    fixture_rows = []
    for index in X_test.index[:300]:
        values = X_test.loc[index].to_numpy(dtype=float)
        fixture_rows.append(
            {
                "features": [float(value) for value in values],
                "pythonProbability": float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1]),
            }
        )
    fixture = {
        "features": FEATURES,
        "threshold": threshold,
        "rows": fixture_rows,
    }
    (APP_DIR / "tests").mkdir(exist_ok=True)
    (APP_DIR / "tests" / "parity_fixture.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"validation ROC-AUC: {metrics['valRocAuc']:.6f}")
    print(f"test ROC-AUC: {metrics['testRocAuc']:.6f}")
    print("raw test metrics:")
    print(f"  recall: {metrics['testRecall']:.6f}")
    print(f"  precision: {metrics['testPrecision']:.6f}")
    print(f"  visit-reduction rate: {metrics['testVisitReductionRate']:.6f}")
    print(f"  missed count: {metrics['testMissedCount']}")
    print("rounded-input test metrics (X_test.round(1)):")
    print(f"  recall: {metrics['roundedTestRecall']:.6f}")
    print(f"  precision: {metrics['roundedTestPrecision']:.6f}")
    print(f"  visit-reduction rate: {metrics['roundedTestVisitReductionRate']:.6f}")
    print(f"  missed count: {metrics['roundedTestMissedCount']}")
    print(f"threshold: {threshold:.12f}")
    print(f"wrote model.js ({(APP_DIR / 'model.js').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
