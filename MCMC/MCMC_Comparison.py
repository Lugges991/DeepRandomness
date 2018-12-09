import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
dtype = np.float32


class MCMC_Comp:

    def __init__(self, true_mean, true_cov, n_dims=2):
        self.n_dims = n_dims
        self.true_mean = true_mean
        self.true_cov = true_cov
        self.L = tf.linalg.cholesky(self.true_cov)
        self.target = tfd.MultivariateNormalTriL(
            loc=self.true_mean, scale_tril=self.L)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(self, x, y):
        # Stack the input tensors together
        z = tf.stack([x, y], axis=-1) - self.true_mean
        return self.target.log_prob(tf.squeeze(z))

    def sample_RWMCMC(self, num_results=1000, num_chains=100, num_burnin=200):

        # Initial state of the chain
        init_state = [np.ones([num_chains, 1], dtype=dtype),
                      np.ones([num_chains, 1], dtype=dtype)]

        # Run Random Walk Metropolis with normal proposal for `num_results`
        # iterations for `num_chains` independent chains:
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.target_log_prob,
                seed=54),
            num_burnin_steps=num_burnin,
            num_steps_between_results=1,  # Thinning.
            parallel_iterations=1)
        samples = tf.stack(samples, axis=-1)

        with tf.Session() as sess:
            _sample, _kernel_res = sess.run([samples, kernel_results])
        return _sample, _kernel_res

    def sample_HMC(self, num_results=1000, num_burnin=200):
        # Create state to hold updated `step_size`.
        step_size = tf.get_variable(
            name='step_size',
            initializer=1.,
            use_resource=True,  # For TFE compatibility.
            trainable=False)

        # Initialize the HMC transition kernel.
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob,
            num_leapfrog_steps=3,
            step_size=[step_size, step_size],
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

        # Run the chain (with burn-in).
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=int(num_results),
            num_burnin_steps=int(num_burnin),
            current_state=1.,
            kernel=hmc)

        with tf.Session() as sess:
            _sample, _kernel_res = sess.run([samples, kernel_results])
        return _sample, _kernel_res


if __name__ == '__main__':
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])

    comp = MCMC_Comp(true_mean, true_cov)
    rw_samples, rw_kernel = comp.sample_RWMCMC()
    hmc_samples, hmc_kernel = comp.sample_HMC()

    plt.subplot(2, 1, 1)
    plt.scatter(rw_samples[-1], range(len(rw_samples)), color='r')
    plt.title('Random-Walk Metropolis')
    plt.ylabel('Last position variable')
    plt.xlabel('Iteration')

    plt.scatter(hmc_samples[-1], range(len(hmc_samples[-1])), color='g')
    plt.title('Hamiltonian Monte Carlo')
    plt.ylabel('Last position variable')
    plt.xlabel('Iteration')
