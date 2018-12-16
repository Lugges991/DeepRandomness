import random
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """function to be integrated """
    y = 1 / 7**x
    return y


def MC_integrate(lb, ub, n):
    """perform MonteCarlo Integration


    @param  lb: lower bound of the integration
    @param  ub: upper bound of the integration
    @param  n: number of random samples
    """

    # initializing minimum and maximum y
    y_min = f(lb)
    y_max = y_min
    for i in np.linspace(lb, ub, 1000):
        # determine y_min and y_max
        if f(i) < y_min:
            y_min = f(i)
        if f(i) > y_max:
            y_max = f(i)

    # calculate the area where the graph is contained ( not the area under the
    # graph!
    interval_area = (ub - lb) * (y_max)

    # initialize plot
    plt.ion()
    # plot the graph of f(x)
    plt.plot(np.linspace(lb, ub, 100),
             f(np.linspace(lb, ub, 100)), color='b')

    # create an array to hold the 'hits' and one to hold the 'misses'
    hits = []
    misses = []

    # generate a random 2xn matrix P, drawn from the 'continuous uniform'
    # distribution over the stated interval
    x_rand = []
    y_rand = []

    for i in range(n):
        # generate a random point in the given interval
        x_rand.append((ub - lb) * random.random() + lb)
        y_rand.append((ub - lb) * random.random() + lb)

        # now check whether the Point P(x,y) lies under the graph of f(x)
        # therefore y < f(x)
        if y_rand[i] < f(x_rand[i]):
            hits.append((x_rand[i], y_rand[i]))
            plt.scatter(x_rand[i], y_rand[i], color='g')

        else:
            misses.append((x_rand[i], y_rand[i]))
            plt.scatter(x_rand[i], y_rand[i], color='r')
        # calculate the ratio of hits to whole number of samples
        ratio = len(hits) / n

        # calculate the area UNDER the graph aka the integral
        Area = ratio * interval_area
        plt.legend(['Area = {}'.format(Area)], loc=1,)
        plt.draw()
        plt.pause(0.0001)
    print('Area of f(x): ' + str(Area))


MC_integrate(0, 1, 1000)
