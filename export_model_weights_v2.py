"""
Export v2 model (tuned GBM from ensemble) to portable JSON for React predictor.
Output: ncaa_data/model_weights.json + bracket-app/src/model_weights.json
"""
import pickle
import json
import numpy as np

with open("ncaa_data/model_v2.pkl", "rb") as f:
    bundle = pickle.load(f)

gbm = bundle["gbm"]
feat_names = bundle["feature_names"]

def extract_tree(tree):
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        nodes.append({
            "left": int(t.children_left[i]),
            "right": int(t.children_right[i]),
            "feature": int(t.feature[i]),
            "threshold": round(float(t.threshold[i]), 6),
            "value": round(float(t.value[i][0][0]), 8),
        })
    return nodes

trees = []
for stage in gbm.estimators_:
    for tree_est in stage:
        trees.append(extract_tree(tree_est))

weights = {
    "algorithm": "gradient_boosting_v2",
    "n_estimators": gbm.n_estimators,
    "learning_rate": gbm.learning_rate,
    "init_value": round(float(gbm.init_.class_prior_[1]), 8),
    "features": feat_names,
    "n_features": len(feat_names),
    "trees": trees,
    "best_params": bundle["best_params"],
}

for path in ["ncaa_data/model_weights.json", "bracket-app/src/model_weights.json"]:
    with open(path, "w") as f:
        json.dump(weights, f, separators=(",", ":"))

size_kb = len(json.dumps(weights, separators=(",", ":"))) / 1024
print(f"Exported {len(trees)} trees, {len(feat_names)} features -> {size_kb:.0f} KB")
