import modelWeights from "./model_weights.json";

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function predictTree(tree, features) {
  let nodeIdx = 0;
  while (true) {
    const node = tree[nodeIdx];
    if (node.left === -1) return node.value;
    if (features[node.feature] <= node.threshold) {
      nodeIdx = node.left;
    } else {
      nodeIdx = node.right;
    }
  }
}

/**
 * Predict P(team_a wins) given differential features (team_a - team_b).
 * @param {number[]} diff - 18-element array of feature differences
 * @returns {number} probability team_a wins
 */
export function predictProba(diff) {
  const { trees, learning_rate, init_value } = modelWeights;
  const initLogOdds = Math.log(init_value / (1 - init_value));
  let raw = initLogOdds;
  for (const tree of trees) {
    raw += learning_rate * predictTree(tree, diff);
  }
  return sigmoid(raw);
}

/**
 * Get the feature names in order.
 */
export function getFeatureNames() {
  return modelWeights.features;
}
