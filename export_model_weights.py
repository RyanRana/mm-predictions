"""
Export trained model weights to a portable JSON format.
Output: ncaa_data/model_weights.json
Can be loaded in JS to run predictions client-side.
"""
import pickle
import json
import numpy as np

with open("ncaa_data/model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_NAMES = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB", "SEED",
]

def extract_tree(tree):
    """Extract a single decision tree into a portable dict."""
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        node = {
            "left": int(t.children_left[i]),
            "right": int(t.children_right[i]),
            "feature": int(t.feature[i]),
            "threshold": round(float(t.threshold[i]), 6),
            "value": round(float(t.value[i][0][0]), 8),
        }
        nodes.append(node)
    return nodes

trees = []
for stage in model.estimators_:
    for tree_est in stage:
        trees.append(extract_tree(tree_est))

weights = {
    "algorithm": "gradient_boosting",
    "n_estimators": model.n_estimators,
    "learning_rate": model.learning_rate,
    "init_value": round(float(model.init_.class_prior_[1]), 8),
    "features": FEATURE_NAMES,
    "n_features": len(FEATURE_NAMES),
    "trees": trees,
}

with open("ncaa_data/model_weights.json", "w") as f:
    json.dump(weights, f, separators=(",", ":"))

size_kb = len(json.dumps(weights, separators=(",", ":"))) / 1024
print(f"Exported {len(trees)} trees, {len(FEATURE_NAMES)} features -> ncaa_data/model_weights.json ({size_kb:.0f} KB)")
