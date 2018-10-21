import numpy as np

"""function to be integrated """


def f(x):
    y = 1 / np.sqrt(2 * np.pi) * np.exp(-(1 / 2) * np.square(x))
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
    P = np.zeros((2, n))
    y_min = f(lb)
    y_max = y_min
    for i in range(n):
        (P[0, i], P[1, i]) = (ub - lb) * np.random.random_sample((2, 1)) + lb
        # now check whether the Point P(x,y) lies under the graph of f(x)
        # therefore y < f(x)
        if P[1, i] < f(P[0, i]):
            hits.append((P[0, i], P[1, i]))
        else:
            misses.append((P[0, i], P[1, i]))

        # determine y_min and y_max
        if f(P[0, i]) < y_min:
            y_min = f(P[0, i])
        if f(P[0, i]) > y_max:
            y_max = f(P[0, i])
    print(len(hits))
    ratio = len(hits) / n
    interval_area = (ub - lb) * (y_max - y_min)
    Area = ratio * interval_area
    return Area


if __name__ == '__main__':
    pass
