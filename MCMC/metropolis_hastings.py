import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = np.float32

target = tfd.Normal(loc=dtype(0), scale=dtype(1))

samples, _ = tfp.mcmc.sample_chain(
    num_results=5000,
    current_state=dtype(1),
    kernel=tfp.mcmc.MetropolisHastings,
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
with tf.Session() as sess:
    [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

    print('Estimated mean: {}'.format(sample_mean_))
    print('Estimated standard deviation: {}'.format(sample_std_))