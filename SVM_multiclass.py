import copy
import random

import numpy as np
from SMO import SMO


class SVM_multiclass:
    def __init__(self, x, y, k, c, gamma, tol, max_iters):
        args = copy.deepcopy(x).astype(np.float64)
        labels = np.negative(np.ones((len(y), k)))
        for i in range(len(y)):
            labels[i, y[i]] = 1

        args /= (255 / 2)
        args -= 1
        self.svms = []
        for i in range(k):
            a = list(np.where(y == i)[0])
            sample_size = len(a)
            b = random.sample(list(np.where(y != i)[0]), sample_size)
            idx = a + b
            np.random.shuffle(idx)
            self.svms.append(SMO(np.array([args[j, :] for j in idx]), np.array([labels[j, i] for j in idx]), c, gamma))
        print("done creating datasets")
        i = 0
        for svm in self.svms:
            svm.train(tol, max_iters, i)
            i += 1
            print(f"trained, accuracy : {svm.plot()}")

        print("training done")
        print(f"Error rate on train set : {self.test(args, y, k)}")

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
