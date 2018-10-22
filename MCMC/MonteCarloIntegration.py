import numpy as np

"""function to be integrated """


def f(x):
    # y = 1 / np.sqrt(2 * np.pi) * np.exp(-(1 / 2) * np.square(x))
    y = np.sqrt(14 * np.square(x)) / 32**x
    return y


"""perform MonteCarlo Integration

    lb: lower bound of the integration
    ub: upper bound of the integration
    n: number of random samples
"""


def MC_integrate(lb, ub, n):
    # create an array to hold the 'hits' and one to hold the 'misses'
    hits = []
    misses = []

    # generate a random 2xn matrix P, drawn from the 'continuous uniform'
    # distribution over the stated interval
    x_rand = []
    y_rand = []
    y_min = f(lb)
    y_max = y_min
    for i in range(n):
        x_rand.append((ub - lb) * np.random.random_sample() + lb)
        y_rand.append((ub - lb) * np.random.random_sample() + lb)
        # now check whether the Point P(x,y) lies under the graph of f(x)
        # therefore y < f(x)
        if y_rand[i] < f(x_rand[i]):
            hits.append((x_rand, y_rand))
        else:
            misses.append((x_rand, y_rand))

        # determine y_min and y_max
        if f(x_rand[i]) < y_min:
            y_min = f(x_rand[i])
        if f(x_rand[i]) > y_max:
            y_max = f(x_rand[i])
    print(len(hits))
    print(y_max, y_min)
    ratio = len(hits) / n
    interval_area = (ub - lb) * (2)
    print(interval_area)
    Area = ratio * interval_area
    return Area


if __name__ == '__main__':
    print(MC_integrate(0, 1, 1000))
