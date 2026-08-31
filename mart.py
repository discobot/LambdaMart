"""LambdaMART: gradient-boosted trees trained on LambdaRank gradients,
with Newton-step leaf values (sum of lambdas over sum of second derivatives).

Expects CSV input with a header row and columns:
    label, query_id, doc_id, feature_1, ..., feature_n
Rows must be grouped by query_id (all documents of a query contiguous).

Reference: Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" (2010).
"""

import argparse
import csv
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class Ensemble:
    def __init__(self, learning_rate: float) -> None:
        self.trees: list[DecisionTreeRegressor] = []
        self.learning_rate = learning_rate

    def __len__(self) -> int:
        return len(self.trees)

    def add(self, tree: DecisionTreeRegressor) -> None:
        self.trees.append(tree)

    def predict(self, features: np.ndarray) -> np.ndarray:
        result = np.zeros(len(features))
        for tree in self.trees:
            result += self.learning_rate * tree.predict(features)
        return result


def query_slices(queries: np.ndarray) -> list[slice]:
    """Slices of the contiguous query blocks, in order of appearance."""
    boundaries = np.flatnonzero(np.diff(queries)) + 1
    starts = [0, *boundaries, len(queries)]
    return [slice(a, b) for a, b in zip(starts, starts[1:])]


def dcg(labels: np.ndarray, k: int | None = None) -> float:
    """DCG of a result page given its labels in ranked order."""
    labels = labels[:k]
    gains = 2.0 ** labels - 1
    discounts = np.log2(np.arange(len(labels)) + 2)
    return float(np.sum(gains / discounts))


def ndcg(labels: np.ndarray, model_scores: np.ndarray, k: int = 10) -> float:
    """NDCG@k for one query: labels reordered by model score vs. ideal order."""
    ideal = dcg(np.sort(labels)[::-1], k)
    if ideal == 0:
        return 1.0
    ranked = labels[np.argsort(model_scores)[::-1]]
    return dcg(ranked, k) / ideal


def mean_ndcg(model_scores: np.ndarray, labels: np.ndarray,
              queries: np.ndarray, k: int = 10) -> float:
    pages = query_slices(queries)
    return sum(ndcg(labels[s], model_scores[s], k) for s in pages) / len(pages)


def query_lambdas(labels: np.ndarray, model_scores: np.ndarray,
                  k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """LambdaRank gradients and second derivatives for one query.

    For each pair (i, j) with labels[i] > labels[j]:
        lambda_ij = |dNDCG from swapping i and j| / (1 + exp(s_i - s_j))
    accumulated positively into i (push up) and negatively into j (push down).
    The second derivatives w_ij = rho * (1 - rho) * |dNDCG| accumulate
    positively into both documents; they are the Newton-step denominators.
    """
    idcg = dcg(np.sort(labels)[::-1], k)
    if idcg == 0:
        return np.zeros(len(labels)), np.zeros(len(labels))

    # position of each document in the model's current ranking (0 = best)
    order = np.argsort(model_scores)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))

    gain = 2.0 ** labels - 1
    discount = 1.0 / np.log2(rank + 2.0)

    # pairwise matrices; [i, j] refers to the document pair (i, j)
    delta_ndcg = np.abs(np.subtract.outer(gain, gain)
                        * np.subtract.outer(discount, discount)) / idcg
    rho = 1.0 / (1.0 + np.exp(np.subtract.outer(model_scores, model_scores)))
    pair_lambda = np.where(np.greater.outer(labels, labels), rho * delta_ndcg, 0.0)
    pair_w = pair_lambda * (1.0 - rho)

    lambdas = pair_lambda.sum(axis=1) - pair_lambda.sum(axis=0)
    weights = pair_w.sum(axis=1) + pair_w.sum(axis=0)
    return lambdas, weights


def compute_lambdas(model_scores: np.ndarray, labels: np.ndarray,
                    queries: np.ndarray, k: int = 10
                    ) -> tuple[np.ndarray, np.ndarray]:
    lambdas = np.zeros(len(labels))
    weights = np.zeros(len(labels))
    for s in query_slices(queries):
        lambdas[s], weights[s] = query_lambdas(labels[s], model_scores[s], k)
    return lambdas, weights


def newton_leaf_values(tree: DecisionTreeRegressor, features: np.ndarray,
                       lambdas: np.ndarray, weights: np.ndarray) -> None:
    """Replace each leaf's value with the Newton step: sum(lambda) / sum(w)."""
    leaves = tree.apply(features)
    num = np.bincount(leaves, weights=lambdas, minlength=tree.tree_.node_count)
    den = np.bincount(leaves, weights=weights, minlength=tree.tree_.node_count)
    values = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    tree.tree_.value[:, 0, 0] = values


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    labels, queries, doc_ids, features = (
        data[:, 0], data[:, 1], data[:, 2], data[:, 3:])
    return labels, queries, doc_ids, features


def learn(train_file: str, n_trees: int = 200, learning_rate: float = 0.1,
          max_depth: int = 6, k: int = 10) -> Ensemble:
    print(f"Loading {train_file}")
    labels, queries, _, features = load(train_file)
    print(f"{len(labels)} documents, {len(query_slices(queries))} queries")

    ensemble = Ensemble(learning_rate)
    model_scores = np.zeros(len(features))

    for i in range(n_trees):
        start = time.perf_counter()

        lambdas, weights = compute_lambdas(model_scores, labels, queries, k)

        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(features, lambdas)
        newton_leaf_values(tree, features, lambdas, weights)
        ensemble.add(tree)

        model_scores += learning_rate * tree.predict(features)

        elapsed = time.perf_counter() - start
        print(f"tree {i + 1:3d}/{n_trees}  "
              f"train NDCG@{k} = {mean_ndcg(model_scores, labels, queries, k):.4f}  "
              f"({elapsed:.2f}s)")

    return ensemble


def predict(model: Ensemble, test_file: str, out_file: str = "result.csv") -> None:
    _, queries, doc_ids, features = load(test_file)
    scores = model.predict(features)

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "score", "doc_id"])
        writer.writerows(zip(queries, scores, doc_ids))
    print(f"Wrote {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-t", "--train", required=True, help="training CSV")
    parser.add_argument("-p", "--predict", help="CSV to score with the trained model")
    parser.add_argument("-n", "--n-trees", type=int, default=200)
    parser.add_argument("-r", "--learning-rate", type=float, default=0.1)
    parser.add_argument("-d", "--max-depth", type=int, default=6)
    parser.add_argument("-k", type=int, default=10, help="NDCG truncation")
    args = parser.parse_args()

    model = learn(args.train, n_trees=args.n_trees,
                  learning_rate=args.learning_rate,
                  max_depth=args.max_depth, k=args.k)

    if args.predict:
        predict(model, args.predict)


if __name__ == "__main__":
    main()
