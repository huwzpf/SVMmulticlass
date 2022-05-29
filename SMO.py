import numpy as np
import random
import time
import copy
import logging

class SMO:
    class AlphaMetadata:
        def __init__(self):
            self.bound = True
            self.error_cached = False

    def __init__(self, x, y, c, gamma, km):
        self.b = 0
        self.alpha = np.zeros((x.shape[0], 1))
        self.features = x
        self.labels = y.reshape(len(y), 1)
        self.c = c
        self.gamma = gamma
        self.eps = 10**(-3)
        self.unbound_alphas = set()
        self.bound_alphas = set(range(len(self.alpha)))
        self.err_cache = {}
        self.unbound_err_cache = {}
        self.alpha_metadata = [SMO.AlphaMetadata() for _ in self.alpha]

        if km is None:
            self.train_hypothesis = self.no_km_hypothesis
        else:
            self.kernel_matrix = km
            self.train_hypothesis = self.km_hypothesis


    def get_error(self, idx):
        if self.alpha_metadata[idx].error_cached:
            return self.err_cache[idx]
        else:
            er = self.calculate_error(idx)
            self.err_cache[idx] = er
            self.alpha_metadata[idx].error_cached = True
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
        # hypothesis used on x that is not in kernel matrix
        ret = np.transpose(np.multiply(self.alpha, self.labels))\
                  .dot(np.array([self.kernel_function(a, x, self.gamma) for a in self.features])
                       .reshape(self.features.shape[0], 1)) + self.b
        return ret

    def no_km_hypothesis(self, i):
        ret = np.transpose(np.multiply(self.alpha, self.labels)) \
                  .dot(np.array([self.kernel_function(a, self.features[i], self.gamma) for a in self.features])
                       .reshape(self.features.shape[0], 1)) + self.b
        return ret

    def km_hypothesis(self, i):
        # hypothesis from kernel matrix, used while training
        ret = np.transpose(np.multiply(self.alpha, self.labels)).dot(self.kernel_matrix[:, i]) + self.b
        return ret

    def calculate_constrains(self, i, j):
        # return L, H
        if self.labels[i] == self.labels[j]:
            return max(0.0, float(self.alpha[j] + self.alpha[i] - self.c)), min(self.c, float(self.alpha[j] + self.alpha[i]))
        else:
            return max(0.0, float(self.alpha[j] - self.alpha[i])), min(self.c, float(self.c + self.alpha[j] - self.alpha[i]))

    def calculate_error(self, i):
        return self.train_hypothesis(i) - self.labels[i]

    def update_alpha_j(self, i, j, eta):
        ret = self.alpha[j] - float(self.labels[j] * (self.get_error(i) - self.get_error(j))) / eta
        L, H = self.calculate_constrains(i, j)
        if ret >= H:
            ret = H
        elif ret <= L:
            ret = L
        return ret

    def calculate_b(self, i, j, a_i_old, a_j_old):
        b_1 = self.b - self.get_error(i) - self.labels[i] * (self.alpha[i] - a_i_old) * \
              self.kernel_function(self.features[i], self.features[i], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[i], self.features[j], self.gamma)

        b_2 = self.b - self.get_error(j) - self.labels[i] * (self.alpha[i] - a_i_old) *\
              self.kernel_function(self.features[i], self.features[j], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[j], self.features[j], self.gamma)

        if 0 < self.alpha[i] < self.c:
            self.b = b_1
        elif 0 < self.alpha[j] < self.c:
            self.b = b_2
        else:
            self.b = (b_1 + b_2) / 2

    def choice_heuristic(self, i):
        e = self.get_error(i)
        if e >= 0:
            return min(self.unbound_err_cache, key=lambda k: self.unbound_err_cache[k])
        else:
            return max(self.unbound_err_cache, key=lambda k: self.unbound_err_cache[k])

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

        if abs(self.alpha[j] - new_a_j) < self.eps:
            return 0

        self.alpha[j] = new_a_j

        self.check_idx_bounds(j)

        self.alpha[i] += self.labels[i] * self.labels[j] * (old_a_j - self.alpha[j])

        self.check_idx_bounds(i)

        self.calculate_b(i, j, old_a_i, old_a_j)
        return 1

    def check_idx_bounds(self, j):

        if self.alpha_metadata[j].error_cached:
            self.err_cache.pop(j)
            self.alpha_metadata[j].error_cached = False

        if self.alpha[j] != 0 and self.alpha[j] != self.c:
            self.unbound_err_cache[j] = self.get_error(j)
            if self.alpha_metadata[j].bound:
                self.bound_alphas.remove(j)
                self.unbound_alphas.add(j)
                self.alpha_metadata[j].bound = False

        elif not self.alpha_metadata[j].bound:
            self.bound_alphas.add(j)
            self.unbound_alphas.remove(j)
            self.unbound_err_cache.pop(j)
            self.alpha_metadata[j].bound = True



    def examine_example(self, i, tol):
        E_i = self.get_error(i)
        if (self.labels[i] * E_i < -tol and self.alpha[i] < self.c) or \
                (self.labels[i] * E_i > tol and self.alpha[i] > 0):
            if len(self.unbound_err_cache) != 0:
                idx = self.choice_heuristic(i)
                if self.take_step(i, idx) == 1:
                    return 1

            tmp_list = list(self.unbound_alphas)
            start = random.randint(0, len(tmp_list))
            for j in range(len(tmp_list)):
                if self.take_step(i, tmp_list[(j + start) % len(tmp_list)]) == 1:
                    return 1

            tmp_list = list(self.bound_alphas)
            start = random.randint(0, len(tmp_list))
            for j in range(len(tmp_list)):
                if self.take_step(i, tmp_list[(j + start) % len(tmp_list)]) == 1:
                    return 1

        return 0

    def train(self, tol, max_iters=15, ident=0):
        # create logger
        filehandler = logging.FileHandler('logfile' + str(ident) + '.log', 'a')
        formatter = logging.Formatter(
            '%(asctime)-15s::%(levelname)s::%(filename)s::%(funcName)s::%(lineno)d::%(message)s')
        filehandler.setFormatter(formatter)
        log = logging.getLogger()  # root logger - Good to get it only once.
        for hdlr in log.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logging.FileHandler):
                log.removeHandler(hdlr)
        log.addHandler(filehandler)  # set the new handler
        # set the log level to INFO, DEBUG as the default is ERROR
        log.setLevel(logging.INFO)
        cnt = 0
        iters = 0
        changed_alphas = 0
        examine_all = True
        print(f"starting train {ident}")
        logging.info(f"starting train {ident}")
        while iters < max_iters and (changed_alphas > 0 or examine_all):
            changed_alphas = 0
            prev_unbound_alphas = copy.copy(self.unbound_alphas)
            if examine_all:
                for i in range(self.features.shape[0]):
                    changed_alphas += self.examine_example(i, tol)
            else:
                set_cpy = copy.copy(self.unbound_alphas)
                for i in set_cpy:
                    changed_alphas += self.examine_example(i, tol)

            if examine_all:
                examine_all = False
            elif changed_alphas == 0:
                examine_all = True
            iters += 1
            logging.info(f"iter: {iters}, b: {self.b}, changed alphas: {changed_alphas}, unbound: {len(self.unbound_alphas)}, bound : {len(self.bound_alphas)}")
            print(f"iter: {iters}, b: {self.b}, changed alphas: {changed_alphas}, unbound: {len(self.unbound_alphas)}, bound : {len(self.bound_alphas)}")
            if prev_unbound_alphas == self.unbound_alphas:
                cnt += 1
                if cnt > 10:
                    print("exceededgit ")
                    break
            else:
                cnt = 0

        np.savetxt('alpha_'+str(ident)+'.csv', self.alpha, delimiter=",")
        np.savetxt('b_'+str(ident)+'.csv', self.b, delimiter=",")
        print(f"\n\n done in {iters} \n\n")

    def cost_function(self):
        # todo
        count = 0
        for i in range(self.labels.shape[0]):
            prediction = self.km_hypothesis(i)
            if prediction >= 0 and self.labels[i] >= 0 or prediction < 0 and self.labels[i] < 0:
                count += 1
            if i < 30:
                print(prediction,   self.labels[i])
        return count/len(self.labels)
