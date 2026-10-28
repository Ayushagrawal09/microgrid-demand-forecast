"""Trains the backup model on train_telemetry.csv and saves it as model.json.
Run once:  python train_model.py   (needs pandas + scikit-learn)"""
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from predict import FEATURES, build_features

df = pd.read_csv("train_telemetry.csv")
X = pd.DataFrame([build_features(r) for r in df.to_dict(orient="records")])[FEATURES]
y = df["target_demand_kw"]

model = GradientBoostingRegressor(random_state=42).fit(X, y)

trees = []
for est in model.estimators_[:, 0]:
    t = est.tree_
    trees.append({"feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                  "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                  "value": [v[0][0] for v in t.value]})

with open("model.json", "w") as fh:
    json.dump({"features": FEATURES, "init": float(model.init_.constant_[0][0]),
               "learning_rate": model.learning_rate, "trees": trees}, fh)
print("Saved model.json with", len(trees), "trees")
