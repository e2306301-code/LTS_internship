"""
data0120.xlsx 故障予測分析

VS Codeでは「# %%」ごとにセル実行できます。
このファイルは元のExcelを変更せず、analysis_outputsへ結果を保存します。
"""

# %% 1. ライブラリと入力ファイル
from pathlib import Path
import sys

# Codexの作業用Pythonパッケージを参照
sys.path.insert(
    0,
    "/Users/horinouchi/Documents/Codex/2026-09-26/adk-adk-adk-adk-1-2/work/pydeps",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT = Path(__file__).with_name("data0120.xlsx")
OUTPUT = Path(__file__).with_name("analysis_outputs")
OUTPUT.mkdir(exist_ok=True)

df = pd.read_excel(INPUT)
sensor_columns = [c for c in df.columns if c not in ["id", "fault_flag"]]
y = df["fault_flag"].map({"No": 0, "Yes": 1}).astype(int)
X = df[sensor_columns].astype(float)

print("読み込み完了:", INPUT)
print("データ形状:", df.shape)
print(df.head())


# %% 2. データ品質と故障割合
quality = pd.Series(
    {
        "行数": len(df),
        "列数": df.shape[1],
        "欠損セル数": int(df.isna().sum().sum()),
        "ID重複数": int(df["id"].duplicated().sum()),
        "センサー値の完全重複行": int(df[sensor_columns].duplicated().sum()),
        "故障あり件数": int(y.sum()),
        "故障なし件数": int((1 - y).sum()),
        "故障率": float(y.mean()),
    }
)
print(quality)
quality.to_csv(OUTPUT / "01_data_quality.csv", header=["value"])


# %% 3. 故障あり・なしのセンサー比較
group_means = df.groupby("fault_flag")[sensor_columns].mean().T
group_means["difference_Yes_minus_No"] = group_means["Yes"] - group_means["No"]

effect_rows = []
for column in sensor_columns:
    no_fault = X.loc[y == 0, column]
    fault = X.loc[y == 1, column]
    pooled_std = np.sqrt((no_fault.var() + fault.var()) / 2)
    effect_rows.append(
        {
            "feature": column,
            "mean_no_fault": no_fault.mean(),
            "mean_fault": fault.mean(),
            "standardized_difference": (fault.mean() - no_fault.mean()) / pooled_std,
        }
    )

effects = pd.DataFrame(effect_rows)
effects["absolute_difference"] = effects["standardized_difference"].abs()
effects = effects.sort_values("absolute_difference", ascending=False)
print(effects.head(10).to_string(index=False))
effects.to_csv(OUTPUT / "02_fault_no_fault_comparison.csv", index=False)


# %% 4. センサー間の相関
correlation = X.corr()
correlation.to_csv(OUTPUT / "03_sensor_correlation.csv")

plt.figure(figsize=(11, 9))
sns.heatmap(correlation, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Sensor correlation matrix")
plt.tight_layout()
plt.savefig(OUTPUT / "03_sensor_correlation.png", dpi=160)
plt.close()


# %% 5. センサー間の差・平均・ばらつきを追加
def add_engineered_features(source):
    result = source.copy()
    groups = {
        "temperature": ["temparature1", "temparature2", "temparature3"],
        "humidity": ["humidity1", "humidity2", "humidity3"],
        "voltage": ["valtage1", "valtage2", "valtage3"],
        "usage": ["usage1", "usage2", "usage3"],
        "rotation": ["rotation1", "rotation2", "rotation3"],
    }
    for name, columns in groups.items():
        values = result[columns]
        result[f"{name}_mean"] = values.mean(axis=1)
        result[f"{name}_range"] = values.max(axis=1) - values.min(axis=1)
        result[f"{name}_std"] = values.std(axis=1, ddof=0)
        result[f"{name}_diff_1_2"] = values.iloc[:, 0] - values.iloc[:, 1]
        result[f"{name}_diff_2_3"] = values.iloc[:, 1] - values.iloc[:, 2]
        result[f"{name}_diff_1_3"] = values.iloc[:, 0] - values.iloc[:, 2]
    return result


X_engineered = add_engineered_features(X)
print("元の説明変数数:", X.shape[1])
print("追加後の説明変数数:", X_engineered.shape[1])


# %% 6. 学習・検証・テストへの分割とモデル比較
all_indices = np.arange(len(df))
train_indices, remaining_indices = train_test_split(
    all_indices, test_size=0.4, stratify=y, random_state=42
)
validation_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.5,
    stratify=y.iloc[remaining_indices],
    random_state=42,
)

models = {
    "Logistic": Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
    "Extra trees": ExtraTreesClassifier(
        n_estimators=450,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "Gradient boosting": HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        random_state=42,
    ),
}

model_rows = []
trained_models = {}
for feature_name, features in {"Original": X, "Engineered": X_engineered}.items():
    for model_name, model_template in models.items():
        model = clone(model_template)
        model.fit(features.iloc[train_indices], y.iloc[train_indices])
        validation_probability = model.predict_proba(features.iloc[validation_indices])[:, 1]
        test_probability = model.predict_proba(features.iloc[test_indices])[:, 1]
        model_rows.append(
            {
                "features": feature_name,
                "model": model_name,
                "validation_ROC_AUC": roc_auc_score(y.iloc[validation_indices], validation_probability),
                "test_ROC_AUC": roc_auc_score(y.iloc[test_indices], test_probability),
                "test_PR_AUC": average_precision_score(y.iloc[test_indices], test_probability),
            }
        )
        trained_models[(feature_name, model_name)] = (
            model,
            features,
            validation_probability,
            test_probability,
        )

model_results = pd.DataFrame(model_rows).sort_values("test_ROC_AUC", ascending=False)
print(model_results.to_string(index=False))
model_results.to_csv(OUTPUT / "04_model_comparison.csv", index=False)


# %% 7. 故障見逃し率を抑えた訪問削減シミュレーション
best_key = (model_results.iloc[0]["features"], model_results.iloc[0]["model"])
best_model, best_features, validation_probability, test_probability = trained_models[best_key]


def select_threshold(actual, probability, maximum_miss_rate):
    best = None
    for threshold in np.unique(np.r_[0.0, probability, 1.0]):
        no_visit = probability < threshold
        missed = int(((actual == 1) & no_visit).sum())
        miss_rate = missed / int((actual == 1).sum())
        if miss_rate <= maximum_miss_rate:
            candidate = (int(no_visit.sum()), float(threshold))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1]


policy_rows = []
validation_actual = y.iloc[validation_indices].to_numpy()
test_actual = y.iloc[test_indices].to_numpy()

for miss_limit in [0.005, 0.01, 0.02, 0.05]:
    threshold = select_threshold(validation_actual, validation_probability, miss_limit)
    no_visit = test_probability < threshold
    missed_faults = int(((test_actual == 1) & no_visit).sum())
    policy_rows.append(
        {
            "validation_fault_miss_limit": miss_limit,
            "test_visit_reduction_rate": no_visit.mean(),
            "test_fault_miss_rate": missed_faults / int((test_actual == 1).sum()),
            "fault_rate_in_no_visit_group": missed_faults / max(1, int(no_visit.sum())),
            "test_no_visit_cases": int(no_visit.sum()),
            "test_missed_faults": missed_faults,
        }
    )

policies = pd.DataFrame(policy_rows)
print(policies.to_string(index=False))
policies.to_csv(OUTPUT / "05_visit_reduction_policy.csv", index=False)


# %% 8. 重要変数と最終グラフ
importance_result = permutation_importance(
    best_model,
    best_features.iloc[test_indices],
    y.iloc[test_indices],
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)
importance = pd.DataFrame(
    {
        "feature": best_features.columns,
        "ROC_AUC_drop": importance_result.importances_mean,
    }
).sort_values("ROC_AUC_drop", ascending=False)
print(importance.head(15).to_string(index=False))
importance.to_csv(OUTPUT / "06_feature_importance.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_plot = model_results.sort_values("test_ROC_AUC")
axes[0].barh(
    model_plot["model"] + " / " + model_plot["features"],
    model_plot["test_ROC_AUC"],
    color="#4C78A8",
)
axes[0].set_xlim(0.5, 1.0)
axes[0].set_title("Test ROC-AUC")

importance_plot = importance.head(12).sort_values("ROC_AUC_drop")
axes[1].barh(importance_plot["feature"], importance_plot["ROC_AUC_drop"], color="#72B7B2")
axes[1].set_title("Feature importance")

plt.tight_layout()
plt.savefig(OUTPUT / "06_model_and_importance.png", dpi=160)
plt.close()

print("\n分析完了。結果保存先:", OUTPUT)

