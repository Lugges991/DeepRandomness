import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
tfd = tfp.distributions


# Assume that the state is passed as a list of 1-d tensors `x` and `y`.
# Then the target log-density is defined as follows:
def target_log_prob(x, y, target, true_mean):
    # Stack the input tensors together
    z = tf.stack([x, y], axis=-1) - true_mean
    return target.log_prob(tf.squeeze(z))


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

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    num_results = 500
    num_chains = 100

    # Target distribution is defined through the Cholesky decomposition `L`:
    L = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=L)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Random Walk Metropolis with normal proposal for `num_results`
    # iterations for `num_chains` independent chains:
    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob,
            seed=54),
        num_burnin_steps=200,
        num_steps_between_results=1,  # Thinning.
        parallel_iterations=1)
    samples = tf.stack(samples, axis=-1)

    sample_mean = tf.reduce_mean(samples, axis=0)
    x = tf.squeeze(samples - sample_mean)
    sample_cov = tf.matmul(tf.transpose(x, [1, 2, 0]),
                           tf.transpose(x, [1, 0, 2])) / num_results

    mean_sample_mean = tf.reduce_mean(sample_mean)
    mean_sample_cov = tf.reduce_mean(sample_cov, axis=0)
    x = tf.reshape(sample_cov - mean_sample_cov, [num_chains, 2 * 2])
    cov_sample_cov = tf.reshape(tf.matmul(x, x, transpose_a=True) / num_chains,
                                shape=[2 * 2, 2 * 2])

    with tf.Session() as sess:
        [
            mean_sample_mean_,
            mean_sample_cov_,
            cov_sample_cov_,
        ] = sess.run([
            mean_sample_mean,
            mean_sample_cov,
            cov_sample_cov,
        ])

        print('Estimated mean: {}'.format(mean_sample_mean_))
        print('Estimated avg covariance: {}'.format(mean_sample_cov_))
        print('Estimated covariance of covariance: {}'.format(cov_sample_cov_))


def sample_HMC():
    pass
#     likelihood = make_likelihood(true_variances)
#
#     states, kernel_results = tfp.mcmc.sample_chain(
#         num_results=1000,
#         current_state=tf.zeros(dims),
#         kernel=tfp.mcmc.HamiltonianMonteCarlo(
#             target_log_prob_fn=likelihood.log_prob,
#             step_size=0.5,
#             num_leapfrog_steps=2),
#         num_burnin_steps=500)


if __name__ == '__main__':
    mean, cov, gauss = sample_simple_random()
