import csv
import math
import numpy as np
from optparse import OptionParser
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from itertools import chain
import time

class Ensemble:
    def __init__(self, rate):
        self.trees = []
        self.rate = rate

    def __len__(self):
        return len(self.trees)

    def add(self, tree):
        self.trees.append(tree)

    def eval_one(self, object):
        return self.eval([object])[0]

    def eval(self, objects):
        results = np.zeros(len(objects))
        for tree in self.trees:
            results += tree.predict(objects) * self.rate
        return results

    def remove(self, number):
        self.trees = self.trees[:-number]


def groupby(score, query):
    result = []
    this_query = None
    for s, q in zip(score, query):
        if q != this_query:
            result.append([])
            this_query = q
        result[-1].append(s)
    return [np.array(r) for r in result]


def point_dcg(arg):
    i, label = arg
    return (2 ** label - 1) / math.log(i + 2, 2)


def dcg(scores):
    return sum(map(point_dcg, enumerate(scores)))


def ndcg(page, k=10):
    model_top = page[:k]
    true_top = np.sort(page)[::-1][:k]

    max_dcg = dcg(true_top)
    model_dcg = dcg(model_top)

    if max_dcg == 0:
        return 1

    return model_dcg / max_dcg


def score(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    model_pages = groupby(prediction, query)

    total_ndcg = []

    for true_page, model_page in zip(true_pages, model_pages):
        page = true_page[np.argsort(model_page)[::-1]]
        total_ndcg.append(ndcg(page, k))

    return sum(total_ndcg) / len(total_ndcg)


def query_lambdas(page, k=10):
    true_page, model_page = page
    size = len(true_page)

    # position of each document in the model's current ranking (0 = best)
    sorted_idx = np.argsort(model_page)[::-1]
    rank = np.empty(size, dtype=int)
    rank[sorted_idx] = np.arange(size)

    idcg = dcg(np.sort(true_page)[::-1][:k])
    if idcg == 0:
        return np.zeros(size)

    gain = 2.0 ** true_page - 1
    discount = 1.0 / np.log2(rank + 2.0)

    lambdas = np.zeros(size)

    for i in range(size):
        for j in range(size):
            if true_page[i] > true_page[j]:
                # change in DCG if documents i and j swap positions
                delta_ndcg = abs((gain[i] - gain[j])
                                 * (discount[i] - discount[j])) / idcg

                rho = 1 / (1 + math.exp(model_page[i] - model_page[j]))

                lam = rho * delta_ndcg

                lambdas[j] -= lam
                lambdas[i] += lam
    return lambdas


def compute_lambdas(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    model_pages = groupby(prediction, query)

    print(len(true_pages), "pages")

    pool = Pool()
    lambdas = pool.map(query_lambdas, zip(true_pages, model_pages))
    return list(chain(*lambdas))


def mart_responces(prediction, true_score):
    return true_score - prediction


def learn(train_file, n_trees=10, learning_rate=0.1, k=10, validate=False):
    print("Loading train file")
    train = np.loadtxt(train_file, delimiter=",", skiprows=1)

    scores = train[:, 0]
    queries = train[:, 1]
    features = train[:, 3:]

    ensemble = Ensemble(learning_rate)

    print("Training starts...")
    model_output = np.zeros(len(features))

    for i in range(n_trees):
        print(" Iteration: " + str(i + 1))

        # Compute pseudo responses (lambdas)
        # which act as training label for document
        start = time.time()
        print("  --generating labels")
        lambdas = compute_lambdas(model_output, scores, queries, k)
        print("  --done", str(time.time() - start) + " sec")

        # create tree and append it to the model
        print("  --fitting tree")
        start = time.time()
        tree = DecisionTreeRegressor(max_depth=6)
        tree.fit(features, lambdas)

        print("  ---done", str(time.time() - start) + " sec")
        print("  --adding tree to ensemble")
        ensemble.add(tree)

        # update model score
        print("  --generating step prediction")
        prediction = tree.predict(features)

        print("  --updating full model output")
        model_output += learning_rate * prediction

        # train_score
        start = time.time()
        print("  --scoring on train")
        train_score = score(model_output, scores, queries, k)
        print("  --iteration train score " + str(train_score) + ", took " + str(time.time() - start) + "sec to calculate")

    print("Finished sucessfully.")
    print("------------------------------------------------")
    return ensemble


def predict(model, fn):
    predict = np.loadtxt(fn, delimiter=",", skiprows=1)

    queries = predict[:, 1]
    doc_id  = predict[:, 2]
    features = predict[:, 3:]

    results = model.eval(features)
    with open("result.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for line in zip(queries, results, doc_id):
            writer.writerow(line)
    return "OK"


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--train", action="store", type="string", dest="train_file")
    parser.add_option("-v", "--validation", action="store_true", dest="validate")
    parser.add_option("-p", "--predict", action="store", type="string", dest="predict_file")

    options, args = parser.parse_args()

    model = learn(options.train_file,
                  validate = options.validate,
                  n_trees = 200)

    if options.predict_file:
        predict(model, options.predict_file)
