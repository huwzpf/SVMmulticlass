import numpy as np
import random
import time
import matplotlib.pyplot as plt


class SMO:
    def __init__(self, x, y, c, gamma):
        self.b = 0
        self.alpha = np.zeros((x.shape[0], 1))
        self.features = x
        self.labels = y.reshape(len(y), 1)
        self.c = c
        self.gamma = gamma

    @staticmethod
    def kernel_function(x, z, gamma):
        k = np.transpose(x-z).dot(x-z)/-gamma
        return np.exp(k)

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

    def calculate_eta(self, i, j):
        return 2 * self.kernel_function(self.features[i], self.features[j], self.gamma) \
               - self.kernel_function(self.features[i], self.features[i], self.gamma) -\
               self.kernel_function(self.features[j], self.features[j], self.gamma)

    def update_alpha_j(self, i, j):
        alpha_j_old = self.alpha[j]
        self.alpha[j] -= float(self.labels[j] * (self.calculate_error(i) - self.calculate_error(j))) / float(self.calculate_eta(i,j))
        L, H = self.calculate_constrains(i, j)

        if self.alpha[j] > H:
            self.alpha[j] = H
        elif self.alpha[j] < L:
            self.alpha[j] = L

    def calculate_b(self, i, j, a_i_old, a_j_old):
        b_1 = self.b - self.calculate_error(i) - self.labels[i] * (self.alpha[i] - a_i_old) * \
              self.kernel_function(self.features[i], self.features[i], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[i], self.features[j], self.gamma)

        b_2 = self.b - self.calculate_error(j) - self.labels[i] * (self.alpha[i] - a_i_old) *\
              self.kernel_function(self.features[i], self.features[j], self.gamma) - self.labels[j] *\
              (self.alpha[j] - a_j_old) * self.kernel_function(self.features[j], self.features[i], self.gamma)

    def select_j(self, i):
        random.seed(time.time())
        j = random.randint(0, self.features.shape[0]-1)

        while i == j:
            j = random.randint(0, self.features.shape[0]-1)

        return j

    def train(self, tol, max_n):
        n = 0
        while n < max_n:
            changed_alphas = 0
            for i in range(self.features.shape[0]):
                E_i = self.calculate_error(i)
                if (self.labels[i] * E_i < -tol and self.alpha[i] < self.c) or\
                        (self.labels[i] * E_i > tol and self.alpha[i] > 0):
                    j = self.select_j(i)
                    old_a_i = self.alpha[i]
                    old_a_j = self.alpha[j]
                    l, h = self.calculate_constrains(i, j)
                    if l == h:
                        continue
                    eta = self.calculate_eta(i, j)
                    if eta > 0:
                        continue
                    self.update_alpha_j(i, j)
                    if abs(self.alpha[j] - old_a_j) < 10 ** -5:
                        continue
                    # update alpha_i
                    self.alpha[i] += self.labels[i] * self.labels[j] * (old_a_j - self.alpha[j])
                    self.calculate_b(i, j, old_a_i, old_a_j)
                    changed_alphas += 1
            if changed_alphas == 0 :
                n += 1
            else:
                n = 0

    def plot(self):
        for i in range(self.labels.shape[0]):
            prediction = self.hypothesis(self.features[i])
            if self.labels[i] > 0:
                color = '#ff2200'
            else:
                color = '#1f77b4'
            plt.scatter(self.features[i, 0], self.features[i, 1], c=color)

        axes = plt.gca()
        (x_min, x_max) = axes.get_xlim()
        (y_min, y_max) = axes.get_ylim()
        # arbitrary number
        elements = len(self.labels) * 2
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, elements), np.linspace(y_min, y_max, elements))
        p = np.empty((elements, elements))
        for i in range(elements):
            for j in range(elements):
                k = np.array([x_grid[i, j], y_grid[i, j]])
                p[i, j] = self.hypothesis(k)
        plt.contour(x_grid, y_grid, p, levels=[0.0])

        plt.show()