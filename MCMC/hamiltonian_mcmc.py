import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# Target distribution is proportional to: `exp(-x (1 + x))`.


def unnormalized_log_prob(x):
    return -x - x**2.


dtype = np.float32
tfd = tfp.distributions
target = tfd.Normal(loc=dtype(0), scale=dtype(1))
# Create state to hold updated `step_size`.
step_size = tf.get_variable(
    name='step_size',
    initializer=1.,
    use_resource=True,  # For TFE compatibility.
    trainable=False)

# Initialize the HMC transition kernel.
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target.log_prob,
    num_leapfrog_steps=3,
    step_size=step_size,
    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

# Run the chain (with burn-in).
samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=int(5000),
    num_burnin_steps=int(1e3),
    current_state=1.,
    kernel=hmc)

# Initialize all constructed variables.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()
    samples_, kernel_results_ = sess.run([samples, kernel_results])
    print(len(samples_))
    print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
        samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))

    plt.scatter(range(len(samples_)), samples_, color='g')
    plt.title('Hamiltonian Monte Carlo')
    plt.ylabel('Last position variable')
    plt.xlabel('Iteration')
    plt.show()
