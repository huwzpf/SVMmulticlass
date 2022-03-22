import pandas as pd
from SMO import SMO


def main():
    data = pd.read_csv('C:/Users/PCu/Data/data_logistic_1.csv').dropna()
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    s = SMO(x, y, 0.1, 100)
    s.train(0.0000005, 50)
    s.plot()

if __name__ == '__main__':
    main()
