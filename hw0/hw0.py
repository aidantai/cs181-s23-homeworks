# CS181 HW0
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal
import seaborn
import pandas


# y = f(x_1,x_2) = exp(-(x_1-2)**2-(x_2-1)**2)
def problem_two():
    # data
    X_1 = np.linspace(0,6,25)
    X_2 = np.linspace(0,6,25)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    Y = np.exp(-(X_1-2)**2-(X_2-1)**2)

    fig = plt.figure() #figsize=(6, 6)
    ax = fig.add_subplot(111, projection='3d') #111 -> position
    ax.plot_surface(X_1, X_2, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def problem_three_one():
    # x = np.linspace(0, 240, 10)
    # y = np.linspace(0, 10, 10)
    # x, y = np.meshgrid(x, y)
    # pos = np.dstack((x, y))

    mean = [120, 4]
    cov = [[1.5, 1], [1, 1.5]]
    rv = multivariate_normal(mean, cov)
    samples = rv.rvs(size=500)
    
    s = []
    w = []
    for sample in samples:
        s_, w_ = sample
        s.append(s_)
        w.append(w_)
    data = {"S":s, "W":w};
    df = pandas.DataFrame(data=data)
    seaborn.histplot(df, x="S", y="W")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.contourf(x, y, rv.pdf(pos))
    plt.show()

def problem_three_two():
    mean = [120, 4]
    cov = [[1.5, 1], [1, 1.5]]
    rv = multivariate_normal(mean, cov)
    w = np.linspace(0, 10, 1001)
    p_118 = []
    p_122 = []
    for i in range(1001):
        p_118.append(rv.pdf([118,w[i]]))
        p_122.append(rv.pdf([122,w[i]]))


    # S = []
    # W = []
    # for sample in samples:
    #     s, w = sample
    #     S.append(s)
    #     W.append(w)
    # data = {"S":S, "W":W};
    # df = pandas.DataFrame(data=data)
    # seaborn.histplot(df, x="S", y="W")

    # fig = plt.figure(1, 1)
    fig, ax = plt.subplots(1, 1, figsize=[15,5])
    # ax = fig.add_subplot(111)
    ax.plot(w, p_118)
    ax.plot(w, p_122)
    plt.show()

def total_processing_sample():
    packages = np.random.poisson(72)
    processes = np.random.normal(86.4, 5.84, size=packages)
    total_processing_time = np.sum(processes)
    return total_processing_time

def problem_three_five():
    samples = []
    for i in range(1000):
        samples.append(total_processing_sample())
    mean = np.average(samples)
    stdv = np.std(samples)
    print(mean)
    print(stdv)


if __name__ == '__main__':
    # problem_three_one()
    # problem_three_two()
    problem_three_five()
    