import random
from SMO import *
import pandas as pd


def main():
    data = pd.read_csv('C:/Users/PCu/Data/data_logistic_1.csv').dropna()
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    s = SMO(x, y)
    pr = 0
    s.train(0.0000005, 50)
    for i in range(len(y)):
        if s.hypothesis(x[i]) < 0:
            pr = -1
        else:
            pr = 1
        print(pr, y[i])


if __name__ == '__main__':
    main()
