import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt


class SMO:
    def __init__(self, x, y, c, gamma):
        self.b = 0
        self.alpha = np.zeros((x.shape[0], 1))
        self.features = x
        self.labels = y.reshape(len(y), 1)
        self.c = c
        self.gamma = gamma
        self.eps = 10**(-3)
        self.not_0_not_c_alphas = set()
        self.other_alphas = set(range(len(self.alpha)))
        self.err_cache = {}

    def get_error(self, idx):
        if idx in self.err_cache:
            return self.err_cache[idx]
        else:
            er = self.calculate_error(idx)
            self.err_cache[idx] = er
            return er

    @staticmethod
    def kernel_function(x, z, gamma):
        k = np.transpose(x-z).dot(x-z)/-gamma
        return np.exp(k)

    def objective_function_at_bounds(self, i, j, k_ii, k_jj, k_ij, low, high):

        s = self.labels[i] * self.labels[j]
        f1 = self.labels[i] * (self.get_error(i) + self.b) - self.alpha[i] * k_ii - s * self.alpha[j] * k_ij
        f2 = self.labels[j] * (self.get_error(j) + self.b) - s * self.alpha[i] * k_ij - self.alpha[j] * k_jj
        l1 = self.alpha[i] + s * (self.alpha[j] - low)
        h1 = self.alpha[i] + s * (self.alpha[j] - high)
        fl = l1 * f1 + low * f2 + 0.5 * l1*l1*k_ii + 0.5 * low * low * k_jj + s * low * l1 * k_ij
        fh = h1 * f1 + high * f2 + 0.5 * h1 * h1 * k_ii + 0.5 * high * high * k_jj + s * high * h1 * k_ij
        return fl, fh

    def hypothesis(self, x):
        ret = np.transpose(np.multiply(self.alpha, self.labels))\
                  .dot(np.array([self.kernel_function(a, x, self.gamma) for a in self.features])
                       .reshape(self.features.shape[0], 1)) + self.b
        return ret

    def calculate_constrains(self, i, j):
        # return L, H
        if self.labels[i] == self.labels[j]:
            return max(0.0, float(self.alpha[j] + self.alpha[i] - self.c)), min(self.c, float(self.alpha[j] + self.alpha[i]))
        else:
            return max(0.0, float(self.alpha[j] - self.alpha[i])), min(self.c, float(self.c + self.alpha[j] - self.alpha[i]))

    def calculate_error(self, i):
        return self.hypothesis(self.features[i]) - self.labels[i]



    def update_alpha_j(self, i, j, eta):
        ret = self.alpha[j] - float(self.labels[j] * (self.get_error(i) - self.get_error(j))) / eta
        L, H = self.calculate_constrains(i, j)
        if ret > H:
            ret = H
        elif ret < L:
            ret = L
        return ret
    def calculate_b(self, i, j, a_i_old, a_j_old):

        b_1 = self.b - self.get_error(i) - self.labels[i] * (self.alpha[i] - a_i_old) * \
              self.kernel_function(self.features[i], self.features[i], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[i], self.features[j], self.gamma)


        b_2 = self.b - self.get_error(j) - self.labels[i] * (self.alpha[i] - a_i_old) *\
              self.kernel_function(self.features[i], self.features[j], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[j], self.features[i], self.gamma)

        if 0 < self.alpha[i] < self.c:
            self.b = b_1
        elif 0 < self.alpha[j] < self.c:
            self.b = b_2
        else:
            self.b = (b_1 + b_2) / 2

    def choice_cheuristic(self, i):
        e = self.get_error(i)
        if e >= 0:
            return min(self.err_cache, key=lambda k: self.err_cache[k])
        else:
            return max(self.err_cache, key=lambda k: self.err_cache[k])

    def select_j(self, i):
        random.seed(time.time())
        j = random.randint(0, self.features.shape[0]-1)

        while i == j:
            j = random.randint(0, self.features.shape[0]-1)

        return j

    def take_step(self, i, j):
        if i == j:
            return 0
        l, h = self.calculate_constrains(i, j)
        if l == h:
            return 0
        old_a_i = copy.deepcopy(self.alpha[i])
        old_a_j = copy.deepcopy(self.alpha[j])

        k_ii = self.kernel_function(self.features[i], self.features[i], self.gamma)
        k_jj = self.kernel_function(self.features[j], self.features[j], self.gamma)
        k_ij = self.kernel_function(self.features[i], self.features[j], self.gamma)

        eta = 2*k_ij - k_ii - k_jj

        if eta < 0:
            new_a_j = self.update_alpha_j(i, j, eta)
        else:
            Lobj, Hobj = self.objective_function_at_bounds(i, j, k_ii, k_jj, k_ij, l, h)
            if Lobj < Hobj - self.eps:
                new_a_j = l
            elif Lobj > Hobj + self.eps:
                new_a_j = h
            else:
                new_a_j = self.alpha[j]

        if abs(self.alpha[j] - new_a_j) < self.eps * (self.alpha[j] + new_a_j + self.eps):
            return 0

        self.alpha[j] = new_a_j

        self.check_idx_bounds(j)

        self.alpha[i] += self.labels[i] * self.labels[j] * (old_a_j - self.alpha[j])

        self.check_idx_bounds(i)

        self.calculate_b(i, j, old_a_i, old_a_j)
        return 1

    def check_idx_bounds(self, j):
        if self.alpha[j] != 0 and self.alpha[j] != self.c:
            self.not_0_not_c_alphas.add(j)
            if j in self.other_alphas:
                self.other_alphas.remove(j)
            if j not in self.err_cache:
                self.err_cache[j] = self.calculate_error(j)
        else:
            if j in self.err_cache:
                self.err_cache.pop(j)

    def examine_example(self, i, tol):
        E_i = self.get_error(i)
        if (self.labels[i] * E_i < -tol and self.alpha[i] < self.c) or \
                (self.labels[i] * E_i > tol and self.alpha[i] > 0):
            if len(self.err_cache) != 0:
                idx = self.choice_cheuristic(i)
                if self.take_step(i, idx) == 1:
                    return 1

            for idx in self.not_0_not_c_alphas:
                if self.take_step(i, idx) == 1:
                    return 1

            for idx in self.other_alphas:
                if self.take_step(i, idx) == 1:
                    return 1

        return 0

    def train(self, tol, max_iters=15):
        iters = 0
        changed_alphas = 0
        examine_all = True
        while iters < max_iters and (changed_alphas > 0 or examine_all):
            print(iters)
            changed_alphas = 0
            if examine_all:
                for i in range(self.features.shape[0]):
                    changed_alphas += self.examine_example(i, tol)
            else:
                set_cpy = copy.copy(self.not_0_not_c_alphas)
                for i in set_cpy:
                    changed_alphas += self.examine_example(i, tol)

            if examine_all:
                examine_all = False
            elif changed_alphas == 0:
                examine_all = True
            iters += 1

        print(f"\n\n done in {iters} \n\n")

    def plot(self):
        count = 0
        for i in range(self.labels.shape[0]):
            prediction = self.hypothesis(self.features[i])
            if prediction >= 0 and self.labels[i] >= 0 or prediction < 0 and self.labels[i] < 0:
                count += 1
            print(prediction, self.labels[i])
        return count/len(self.labels)



