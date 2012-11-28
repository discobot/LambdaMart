import math
import numpy as np
# import pandas
from optparse import OptionParser
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict


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
        results = []
        for tree in trees:
            results += tree.predict(objects) * self.rate
        return results

    def remove(self, number):
        self.trees = self.trees[:-number]


def compute_point_dcg(arg):
    rel, i = arg
    return (2 ** rel - 1) / math.log(i + 2, 2)


def compute_dcg(array):
    dcg = sum(map(compute_point_dcg, zip(array, range(len(array)))))

    return dcg


def compute_ndcg(prediction, true, k):
    prediction = prediction
    true = true

    true = sorted(zip(prediction, true), reverse=True, key=lambda x: x[0])
    true = [tr[1] for tr in true]

    idcg = compute_dcg(sorted(true, reverse=True)[:k])
    dcg  = compute_dcg(true[:k])
    if idcg == 0:
        return 1
    return dcg / idcg


def ndcg(prediction, true_score, queries, k=10):
    print prediction
    list_prediction = defaultdict(list)
    list_true_score = defaultdict(list)

    for query, predicted, score in zip(queries, prediction, true_score):
        list_true_score[query].append(score)
        list_prediction[query].append(predicted)

    total_ndcg = []
    for query in list_prediction:
        total_ndcg.append(compute_ndcg(list_prediction[query], list_true_score[query], k))

    return sum(total_ndcg) / len(total_ndcg)


def compute_lambdas(lambdas):
    return lambdas


def learn(train_file, ntrees=3, learning_rate=0.1, val_file=None):
    print "Loading train file"
    train = np.loadtxt(train_file, delimiter=",", skiprows=1)
    scores = train[:, 0]
    queries = train[:, 1]
    features = train[:, 2:]
    ensemble = Ensemble(learning_rate)

    print "Training starts..."
    model_output = np.array([0] * len(features))
    # validation_output = np.array([0] * len(validation))
    # best_validation_score = 0
    lambdas = np.array(scores)

    for i in range(ntrees):
        print "  Iteration: " + str(i)
        # Compute psedo responces (lambdas)
        # witch act as training label for document
        lambdas = compute_lambdas(lambdas)

        # create tree and append it to the model
        tree = DecisionTreeRegressor()
        tree.fit(features, lambdas)
        ensemble.add(tree)

        # update model score
        predictions = tree.predict(features)
        print features

        for obj in range(len(model_output)):
            model_output += learning_rate * predictions

        # train_score
        train_score = ndcg(model_output, scores, queries, 10)
        print "    iteration train score " + str(train_score)

        # validation score
        # if len(validation) != 0:
        #     for obj in range(len(validation_output)):
        #         validation_output[i] += learning_rate * tree.eval(validation[i])

        #     validation_score = compute_ndcg(validation_output, validation_target)
        #     print "    iteration validation score " + str(validation_score)
        #     if(validation_score > best_validation_score):
        #         best_validation_score = validation_score
        #         best_model_len = len(ensemble)

        # # have we assidently break the celling?
        # if (best_validation_score > 0.9):
        #     break

    # rollback to best
    # if len(ensemble) > best_model_len:
        # ensemble.remove(len(ensemble) - best_model_len)

    # finishing up
    print "final quality evaluation"
    # train_score = compute_ndcg(ensemble.eval(features), scores)
    # test_score = compute_ndcg(ensemble.eval(validation), validation_score)

    # print "train %s, test %s" % (train_score, test_score)
    print "Finished sucessfully."
    print "------------------------------------------------"
    return ensemble


def evaluate(model, features):
    return model.eval(features)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--train", action="store", type="string", dest="train_file")
    # parser.add_option("-v", "--validation", action="store", type="string", dest="val_file")
    options, args = parser.parse_args()
    iterations = 30
    learning_rate = 0.001

    model = learn(options.train_file, )





