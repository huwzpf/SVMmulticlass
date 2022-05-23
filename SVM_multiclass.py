import copy
import random

import numpy as np
from SMO import SMO


class SVM_multiclass:
    def __init__(self, x, y, k, c, gamma):
        args = copy.deepcopy(x)
        labels = np.negative(np.ones((len(y), k)))
        for i in range(len(y)):
            labels[i, y[i]] = 1
            for j in range(x.shape[1]):
                if args[i, j] == 0:
                    args[i, j] = -1
        sample_size = 500
        self.svms = []
        for i in range(k):
            a = random.sample(list(np.where(y == i)[0]), sample_size)
            b = random.sample(list(np.where(y != i)[0]), sample_size)
            self.svms.append(SMO(np.array([args[j, :] for j in a+b]), np.array([labels[j, i] for j in a+b]), c, gamma))
        # self.svms = [SMO(x, labels[:, i], c, gamma) for i in range(k)]
        print("done creating datasets")
        for svm in self.svms:
            svm.train(0.5, 50)
            print("trained")

        print("training done")
        print(f"Error rate on train set : {self.test(x[:1000], y[:1000], k)}")

    def test(self, x, y, k, prnt=False):
        count = 0
        outputs = np.zeros((len(y), k))
        results = np.zeros(len(y))
        for i in range(len(y)):
            for j in range(k):
                outputs[i, j] = self.svms[j].hypothesis(x[i, :])
            results[i] = np.argmax(outputs[i, :])
            if results[i] != y[i]:
                count += 1
            if prnt:
                print(results[i], y[i])
        return count/len(y)
