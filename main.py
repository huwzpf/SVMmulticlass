import copy
import time
import mca
import pandas as pd
import numpy as np
import gzip
import random
from SMO import SMO
from SVM_multiclass import SVM_multiclass


def load_training_images(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def load_training_labels(file):
    with gzip.open(file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

def main():
    train_args = load_training_images('D:/Projects/Data/train-images-idx3-ubyte.gz')
    train_labels = load_training_labels('D:/Projects/Data/train-labels-idx1-ubyte.gz')
    test_args = load_training_images('D:/Projects/Data/t10k-images-idx3-ubyte.gz')
    test_labels = load_training_labels('D:/Projects/Data/t10k-labels-idx1-ubyte.gz')

    y = copy.copy(train_labels)
    x = copy.copy(train_args.reshape(train_args.shape[0], -1)).astype(np.float64)
    x /= (255/2)
    x -= 1
    y = np.where(y != 8, -1, 1)

    max_iters = 1000000
    reg_term = 1.0
    tol = 0.005
    gamma = 10000

    # test_svm(gamma, max_iters, reg_term, tol, x, y)

    # test_svm(gamma, max_iters, reg_term, tol, x, y)

    s = SVM_multiclass(train_args.reshape(train_args.shape[0], -1), train_labels, 10, reg_term, gamma, tol, max_iters)
    s.test(x, test_labels, 10)
    print("done!!!!")


def test_svm(gamma, max_iters, reg_term, tol, x, y):
    a = list(np.where(y == 1)[0])
    sample_size = len(a)
    b = random.sample(list(np.where(y == -1)[0]), sample_size)
    idx = a + b
    np.random.shuffle(idx)
    args = np.array([x[j, :] for j in idx])
    s = SMO(args, np.array([y[j] for j in idx]), reg_term, gamma)
    print("starting")
    start = time.time()
    s.train(tol, max_iters)
    end = time.time()
    print(f"accuracy :{s.plot()} in {end - start}")


if __name__ == '__main__':
    main()
    print("ddd")
    # data = pd.read_csv('D:/Projects/Data/data_logistic_1.csv').dropna()
    # x = data.iloc[:, :-1].to_numpy()
    # y = data.iloc[:, -1].to_numpy()
    #  for i in range(len(y)):
    #      if y[i] == 0:
    #          y[i] = -1
    # s = SMO(x, y, 100.0, 100)
    # s.train(0.05, 5000)
    # s.plot()
