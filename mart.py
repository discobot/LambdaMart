import math
import numpy as np
import math
# import pandas
from optparse import OptionParser
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
from copy import deepcopy
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
    this_list = -1
    for s, q in zip(score, query):
        if q != this_query:
            result.append([])
            this_query = q
            this_list += 1
        result[this_list].append(s)
    result = map(np.array, result)
    return result


def compute_point_dcg(arg):
    rel, i = arg
    return (2 ** rel - 1) / math.log(i + 2, 2)


def compute_point_dcg2(arg):
    rel, i = arg
    if i == 0:
        return rel
    else:
        return rel / (math.log(1 + i, 2))
    return


def compute_dcg(array):
    dcg = map(compute_point_dcg, zip(array, range(len(array))))
    return sum(dcg)


def compute_ndcg(page, k=10):
    idcg = compute_dcg(np.sort(page)[::-1][:k])
    dcg = compute_dcg(page[:k])

    if idcg == 0:
        return 1

    return dcg / idcg


def ndcg(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    pred_pages = groupby(prediction, query)

    total_ndcg = []
    for q in range(len(true_pages)):
        total_ndcg.append(compute_ndcg(true_pages[q][np.argsort(pred_pages[q])[::-1]], k))
    return sum(total_ndcg) / len(total_ndcg)


def query_lambdas(page):
    true_page, pred_page = page
    worst_order = np.argsort(true_page)
    true_page = true_page[worst_order]
    pred_page = pred_page[worst_order]

    page = true_page[np.argsort(pred_page)]
    idcg = compute_dcg(np.sort(page)[::-1])
    position_score = np.zeros((len(true_page), len(true_page)))

    for i in xrange(len(true_page)):
        for j in xrange(len(true_page)):
            position_score[i, j] = compute_point_dcg((page[i], j))

    lambdas = np.zeros(len(true_page))

    for i in xrange(len(true_page)):
        for j in xrange(len(true_page)):
                if page[i] > page[j]:

                    delta_dcg = position_score[i][j] - position_score[i][i]
                    delta_dcg += position_score[j][i] - position_score[j][j]

                    delta_ndcg = abs(delta_dcg / idcg)

                    rho = 1 / (1 + math.exp(page[i] - page[j]))
                    lam = rho * delta_ndcg

                    lambdas[i] -= lam
                    lambdas[j] += lam
    return lambdas


def compute_lambdas(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    pred_pages = groupby(prediction, query)

    print len(true_pages), "pages"

    pool = Pool()
    lambdas = pool.map(query_lambdas, zip(true_pages, pred_pages))
    return list(chain(*lambdas))


def mart_responces(prediction, true_score):
    return true_score - prediction


def learn(train_file, n_trees=10, learning_rate=0.1, k=10):
    print "Loading train file"
    train = np.loadtxt(train_file, delimiter=",", skiprows=1)
    # validation = np.loadtxt(validation_file, delimiter=",", skiprows=1)

    scores = train[:, 0]
    # val_scores = train[:, 0]

    queries = train[:, 1]
    # val_queries = validation[:, 1]

    features = train[:, 3:]
    # val_features = validation[:, 3:]

    ensemble = Ensemble(learning_rate)

    print "Training starts..."
    model_output = np.array([float(0)] * len(features))
    # val_output = np.array([float(0)] * len(validation))

    # print model_output
    # best_validation_score = 0
    time.clock()
    for i in range(n_trees):
        print " Iteration: " + str(i + 1)

        # Compute psedo responces (lambdas)
        # witch act as training label for document
        start = time.clock()
        print "  --generating labels"
        # lambdas = compute_lambdas(model_output, scores, queries, k)
        lambdas = mart_responces(model_output, scores)
        print "  --done", str(time.clock() - start) + "sec"

        # create tree and append it to the model
        print "  --fitting tree"
        start = time.clock()
        tree = DecisionTreeRegressor(max_depth=2)
        # print "Distinct lambdas", set(lambdas)
        tree.fit(features, lambdas)

        print "  ---done", str(time.clock() - start) + "sec"
        print "  --adding tree to ensemble"
        ensemble.add(tree)

        # update model score
        print "  --generating step prediction"
        prediction = tree.predict(features)
        # print "Distinct answers", set(prediction)

        print "  --updating full model output"
        model_output += learning_rate * prediction
        # print set(model_output)

        # train_score
        start = time.clock()
        print "  --scoring on train"
        train_score = ndcg(model_output, scores, queries, 10)
        print "  --iteration train score " + str(train_score) + ", took " + str(time.clock() - start) + "sec to calculate"

        # # validation score
        # print "  --scoring on validation"
        # val_output += learning_rate * tree.predict(val_features)
        # val_score = ndcg(val_output, val_scores, val_queries, 10)

        # print "  --iteration validation score " + str(val_score)

        # if(validation_score > best_validation_score):
        #         best_validation_score = validation_score
        #         best_model_len = len(ensemble)

        # # have we assidently break the celling?
        # if (best_validation_score > 0.9):
        #     break

    # rollback to best
    # if len(ensemble) > best_model_len:
        # ensemble.remove(len(ensemble) - best_model_len)

    # finishing up
    # print "final quality evaluation"
    train_score = compute_ndcg(ensemble.eval(features), scores)
    # test_score = compute_ndcg(ensemble.eval(validation), validation_score)

    # print "train %s, test %s" % (train_score, test_score)
    print "Finished sucessfully."
    print "------------------------------------------------"
    return ensemble


def evaluate(model, fn):
    predict = np.loadtxt(fn, delimiter=",", skiprows=1)

    queries = predict[:, 1]
    doc_id  = predict[:, 2]
    features = predict[:, 3:]

    results = model.eval(features)
    writer = csv.writer(open("result.csv"))
    for line in zip(queries, results, doc_id):
            writer.writerow(line)
    return "OK"


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--train", action="store", type="string", dest="train_file")
    # parser.add_option("-v", "--validation", action="store", type="string", dest="val_file")
    parser.add_option("-p", "--predict", action="store", type="string", dest="predict_file")
    options, args = parser.parse_args()
    iterations = 30
    learning_rate = 0.001

    model = learn(options.train_file,
                  # options.val_file,
                  n_trees=200)
    evaluate(model, options.predict_file)

