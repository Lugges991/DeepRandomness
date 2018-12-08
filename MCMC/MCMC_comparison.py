import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def make_likelihood(variance):
    return tfp.distributions.MultivariateNormalDiag(
        scale_diag=tf.sqrt(variance))


def create_n_dim_gauss(n_dims, size=5000):
    """create a n-variate gaussian normal distribution with mean 0 and cov 1"""
    mean = np.zeros(n_dims)
    cov = np.diag(np.linspace(1, 100, n_dims))
    gauss = np.random.multivariate_normal(mean, cov, size)
    return mean, cov, gauss


def sample_simple_random(data, n):
    """sampling by randomly drawing from the given data"""
    return data[np.random.randint(0, n)]


def sample_MHMCMC(data):
    pass


def sample_RWMCMC():
    pass


def sample_HMC():
    likelihood = make_likelihood(true_variances)

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=1000,
        current_state=tf.zeros(dims),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=likelihood.log_prob,
            step_size=0.5,
            num_leapfrog_steps=2),
        num_burnin_steps=500)


if __name__ == '__main__':
    mean, cov, gauss = sample_simple_random()
