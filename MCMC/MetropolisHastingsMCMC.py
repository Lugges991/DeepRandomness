import matplotlib.pyplot as plt
import numpy as np

"""define a target distribution"""


def circle(X):
    return (X[0, 0]-1)**2 + (X[0, 1]-2)**2 - 3**2


"""define any distribution as the proposal distribution"""


def proposal_distribution(X):
    mean = X.reshape(-1,)
    covariance = np.array([[1, 0], [0, 100]])
    return np.random.multivariate_normal(mean, covariance, 1)


def MetropolisHastingsMCMC(p, n):
    # initialize the array to hold the samples
    accepted = np.zeros((n, 2))

    # initialize X
    X = np.array([[0., 0.]])

    for i in range(n):
        # sample a candidate Y from the proposal distribution
        Y = proposal_distribution(X)
        # sample a uniformly distributed random variable on the interval 0,1
        U = np.random.rand()
        # if U is smaller than
        if U < p(Y) / p(X):
            # set Y as the new X and append it to the samples, else append the
            # old X to the samples
            X = Y
        accepted[i] = X
    return accepted


samples = MetropolisHastingsMCMC(circle, 10000)
plt.scatter(samples[:, 0], samples[:, 1])
plt.axis((-5, 5, -5, 5))
plt.show()
