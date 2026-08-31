import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import mart

rng = np.random.RandomState(0)

# 1) brute-force check: delta_ndcg must equal the true NDCG change from
#    actually swapping the two docs in the model ranking
def brute_lambdas(labels, model_scores, k=10):
    size = len(labels)
    order = np.argsort(model_scores)[::-1]
    rank = np.empty(size, dtype=int)
    rank[order] = np.arange(size)
    idcg = mart.dcg(np.sort(labels)[::-1], k)
    lambdas = np.zeros(size)
    w = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if labels[i] > labels[j]:
                def dcg_of(assignment):
                    return sum((2 ** labels[d] - 1) / math.log2(assignment[d] + 2)
                               for d in range(size))
                cur = {d: rank[d] for d in range(size)}
                swp = dict(cur)
                swp[i], swp[j] = cur[j], cur[i]
                delta = abs(dcg_of(cur) - dcg_of(swp)) / idcg
                rho = 1 / (1 + math.exp(model_scores[i] - model_scores[j]))
                lambdas[i] += rho * delta
                lambdas[j] -= rho * delta
                w[i] += rho * (1 - rho) * delta
                w[j] += rho * (1 - rho) * delta
    return lambdas, w

for trial in range(20):
    n = rng.randint(2, 12)
    labels = rng.randint(0, 5, size=n).astype(float)
    scores = rng.randn(n)
    got_l, got_w = mart.query_lambdas(labels, scores)
    want_l, want_w = brute_lambdas(labels, scores)
    assert np.allclose(got_l, want_l), (trial, got_l, want_l)
    assert np.allclose(got_w, want_w), (trial, got_w, want_w)
print("lambda + weight brute-force check: OK")

# 2) training on synthetic data: NDCG must improve
n_queries, docs_per_q, n_feat = 40, 20, 5
rows = []
for q in range(n_queries):
    X = rng.randn(docs_per_q, n_feat)
    raw = 2 * X[:, 0] - X[:, 1] + 0.3 * rng.randn(docs_per_q)
    labels = np.digitize(raw, np.percentile(raw, [20, 40, 60, 80]))
    for d in range(docs_per_q):
        rows.append([labels[d], q, d] + list(X[d]))
data = np.array(rows)

labels, queries, features = data[:, 0], data[:, 1], data[:, 3:]
lr = 0.1
model_scores = np.zeros(len(features))
ndcg0 = mart.mean_ndcg(model_scores, labels, queries, 10)
for it in range(30):
    lambdas, weights = mart.compute_lambdas(model_scores, labels, queries)
    tree = DecisionTreeRegressor(max_depth=6)
    tree.fit(features, lambdas)
    mart.newton_leaf_values(tree, features, lambdas, weights)
    model_scores += lr * tree.predict(features)
ndcg30 = mart.mean_ndcg(model_scores, labels, queries, 10)
print(f"NDCG@10: start {ndcg0:.4f} -> after 30 trees {ndcg30:.4f}")
assert ndcg30 > ndcg0 + 0.1, "did not learn"
print("training check: OK")
