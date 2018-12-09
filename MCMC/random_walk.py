import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = np.float32

target = tfd.Normal(loc=dtype(0), scale=dtype(1))

samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=5000,
    current_state=dtype(1),
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target.log_prob,
        seed=42),
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    _samples, _kernel_results = sess.run([samples, kernel_results])
    [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

    print('sample mean: ', _samples.mean())
    print('sample stddev: ', _samples.std())
    print('Estimated mean: {}'.format(sample_mean_))
    print('Estimated standard deviation: {}'.format(sample_std_))
    print(samples.eval())

    plt.scatter(range(len(_samples)), _samples, color='g')
    plt.title('Random-Walk Metropolis')
    plt.ylabel('Last position variable')
    plt.xlabel('Iteration')
    plt.show()
