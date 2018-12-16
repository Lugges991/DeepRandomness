import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = np.float32

# define the target distribution
target = tfd.Normal(loc=dtype(0), scale=dtype(1))

# initialize the sample chain
samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=5000,
    current_state=dtype(1),
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target.log_prob,
        seed=42),
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

# operation to calculate the sample mean
sample_mean = tf.reduce_mean(samples, axis=0)
# operation to calculate the sample standard deviation
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))

# operation to initialize all variables for tensorflow
init = tf.global_variables_initializer()

# start a tensorflow session
with tf.Session() as sess:
    init.run()

    # run the random--walk algorithm and acquire the samples
    _samples, _kernel_results = sess.run([samples, kernel_results])
    # run the operations for sample mean and std
    [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

    # print results
    print('sample mean: ', _samples.mean())
    print('sample stddev: ', _samples.std())
    print('Estimated mean: {}'.format(sample_mean_))
    print('Estimated standard deviation: {}'.format(sample_std_))

    # plot samples
    plt.scatter(range(len(_samples)), _samples, color='g')
    plt.title('Random-Walk Metropolis')
    plt.ylabel('Last position variable')
    plt.xlabel('Iteration')
    plt.show()
