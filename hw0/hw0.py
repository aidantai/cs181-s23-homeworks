# CS181 HW0
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal
import seaborn


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


def problem_three():
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    x, y = np.meshgrid(x, y)
    pos = np.dstack((x, y))

    mean = [120, 4]
    cov = [[1.5, 1],[1,1.5]]
    rv = multivariate_normal(mean, cov)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, rv.pdf(pos))
    plt.show()

if __name__ == '__main__':
    problem_three()
    
