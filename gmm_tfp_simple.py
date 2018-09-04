'''GMM with primitive tfp.distributions
    Using a simple model as in http://edwardlib.org/tutorials/unsupervised
    20180904
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from matplotlib import pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

print('Tensorflow version: ', tf.__version__)

# always to remembre to reset the default computational graph
tf.reset_default_graph()

dtype = np.float32
N = 500
D = 2
K = 3

# Generate the data
true_loc = np.array([[-5, 3],
                     [0, 0],
                     [2, 2]], dtype)
random = np.random.RandomState(seed=42)

true_hidden_component = random.randint(0, K, N)
count = [np.count_nonzero(true_hidden_component == k) * 1.0 / N
         for k in range(K)]
print(count)
observations = (true_loc[true_hidden_component] +
                random.randn(N, D).astype(dtype))

# plt.scatter(observations[:, 0], observations[:, 1], c=true_hidden_component)
# plt.savefig('./plots/gmm_tfp_data.png')

# PRIOR
rv_pi = tfd.Dirichlet(
    concentration=tf.ones(K),
    name='rv_pi'
)

rv_mu = tfd.Independent(
    tfd.Normal(
        loc=tf.zeros([K, D], dtype),
        scale=tf.ones([K, D], dtype)
    ),
    reinterpreted_batch_ndims=1,
    name='rv_mu'
)

rv_sigma = tfd.Independent(
    tfd.InverseGamma(
        concentration=tf.ones([K, D], dtype=dtype),
        rate=tf.ones([K, D], dtype=dtype)
    ),
    reinterpreted_batch_ndims=1,
    name='rv_sigma'
)

print('\n'.join(str(rv) for rv in [rv_pi, rv_mu, rv_sigma]))

# MODEL


def joint_log_prob(pi, mu, sigma):
    # new rv representing the observed data, which is drawn from a mixture
    rv_observations = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=pi),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma
        )
    )

    # a list of log_probs of all components in the joint
    log_probs_parts = [
        rv_observations.log_prob(observations),
        rv_pi.log_prob(pi)[..., tf.newaxis],
        rv_mu.log_prob(mu),
        rv_sigma.log_prob(sigma)
    ]
    sum_log_prob = tf.reduce_sum(tf.concat(log_probs_parts, axis=-1), axis=-1)
    return sum_log_prob


# ### Other way to define model using edward2

# def gmm_model():
#     rv_pi = ed.Dirichlet(concentration=tf.ones(K), name='rv_pi')
#     rv_mu = ed.Normal(loc=tf.zeros([K, D]), scale=tf.ones([K, D]), name='rv_mu')
#     rv_sigma = ed.InverseGamma(concentration=tf.ones([K, D]), rate=tf.ones([K, D]), name='rv_sigma')
#     rv_x = ed.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=rv_pi),
#         components_distribution=tfd.MultivariateNormalDiag(loc=rv_mu, scale_diag=rv_sigma),
#         sample_shape=N,
#         name='rv_x'
#     )
#     return rv_x, rv_pi, rv_mu, rv_sigma

# # log_p = ed.make_log_joint_fn(gmm_model)
# def log_p(pi, mu, sigma, observations):
#     rv_x, rv_pi, rv_mu, rv_sigma = gmm_model()
#     log_probs_parts = [
#         rv_x.distribution.log_prob(observations),
#         rv_pi.distribution.log_prob(pi)[..., tf.newaxis],
#         tf.reduce_sum(rv_mu.distribution.log_prob(mu), axis=0),
#         tf.reduce_sum(rv_sigma.distribution.log_prob(sigma), axis=0)
#     ]
#     sum_log_prob = tf.reduce_sum(tf.concat(log_probs_parts, axis=-1), axis=-1)
#     return sum_log_prob


# def joint_log_prob(pi, mu, sigma):
#     return log_p(pi, mu, sigma, observations)


# INFERENCE

def inference_map(num_epochs=2000, learning_rate=0.05):
    # here we can not create a distribution
    # eg: qpi = tfd.Dirichlet(concentration=...)
    # because it causes the error: cannot convert the dist. to tensor
    qpi = tf.nn.softplus(tf.Variable(np.ones(K, dtype=dtype) * 1.0 / K))
    qmu = tf.Variable(np.zeros([K, D], dtype=dtype))
    qsigma = tf.nn.softplus(tf.Variable(np.ones([K, D], dtype=dtype)))

    energy = - joint_log_prob(qpi, qmu, qsigma)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_process = optimizer.minimize(energy)
    init_process = tf.global_variables_initializer()
    losses = []

    with tf.Session() as sess:
        sess.run(init_process)
        for i in range(num_epochs):
            sess.run(training_process)
            if i % 10 == 0:
                losses.append(sess.run(energy))

        posterior_pi, posterior_mu = sess.run([qpi, qmu])

    print(posterior_pi)
    print(posterior_mu)

    plt.plot(range(len(losses)), losses)
    plt.savefig('./plots/gmm_tfp_simple_loss_map.png')


def inference_vi2(num_epochs=2000, learning_rate=0.05):
    # variational variables
    qpi_alpha = tf.nn.softplus(tf.Variable(np.zeros(K) * 1.0 / K, dtype=dtype))
    qmu_loc = tf.Variable(np.zeros([K, D]), dtype=dtype)
    qmu_scale = tf.nn.softplus(tf.Variable(np.ones([K, D]), dtype=dtype))
    qsigma_alpha = tf.nn.softplus(tf.Variable(np.ones([K, D]), dtype=dtype))
    qsigma_beta = tf.nn.softplus(tf.Variable(np.ones([K, D]), dtype=dtype))

    def variational_model(qpi_alpha, qmu_loc, qmu_scale, qsigma_alpha, qsigma_beta):
        qpi = ed.Dirichlet(concentration=qpi_alpha, name='qpi')
        # This model works well
        # qmu = ed.MultivariateNormalDiag(loc=qmu_loc, scale_diag=qmu_scale, name='qmu')
        qmu = ed.Normal(loc=qmu_loc, scale=qmu_scale, name='qmu')
        qsigma = ed.InverseGamma(concentration=qsigma_alpha, rate=qsigma_beta,
                                 name='qsigma')
        return qpi, qmu, qsigma

    log_q = ed.make_log_joint_fn(variational_model)

    def joint_log_variational_model(qpi, qmu, qsigma):
        return log_q(qpi_alpha=qpi_alpha, qmu_loc=qmu_loc, qmu_scale=qmu_scale,
                     qsigma_alpha=qsigma_alpha, qsigma_beta=qsigma_beta,
                     qpi=qpi, qmu=qmu, qsigma=qsigma)

    qpi, qmu, qsigma = variational_model(
        qpi_alpha, qmu_loc, qmu_scale, qsigma_alpha, qsigma_beta)
    energy = joint_log_prob(qpi, qmu, qsigma)
    entropy = - joint_log_variational_model(qpi, qmu, qsigma)
    elbo = energy + entropy

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_process = optimizer.minimize(- elbo)
    init_process = tf.global_variables_initializer()
    losses = []

    with tf.Session() as sess:
        sess.run(init_process)
        for i in range(num_epochs):
            sess.run(training_process)
            if i % 10 == 0:
                losses.append(sess.run(elbo))

        posterior_pi, posterior_mu, posterior_sigma = sess.run([
            qpi, qmu, qsigma
        ])

    print(posterior_mu)
    # plt.plot(range(len(losses)), losses)
    # plt.savefig('./plots/gmm_tfp_simples_loss_vi2.png')

    # Model criticism
    generate_process= tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=posterior_pi),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=posterior_mu,
            scale_diag=posterior_sigma
        )
    )

    with tf.Session() as sess:
        x_generated = generate_process.sample(500).eval()
    return x_generated



def inference_vi1(num_epochs=2000, learning_rate=0.05):
    def variational_model():
        qpi = tf.nn.softplus(tf.Variable(np.zeros(K) * 1.0 / K, dtype=dtype), name='qpi')
        qmu = tf.Variable(np.zeros([K, D]), dtype=dtype, name='qmu')
        qsigma = tf.nn.softplus(tf.Variable(np.ones([K, D]), dtype=dtype, name='qsigma'))
        return qpi, qmu, qsigma

    log_q = ed.make_log_joint_fn(variational_model)

    def joint_log_variational_model(qpi, qmu, qsigma):
        return log_q(qpi=qpi, qmu=qmu, qsigma=qsigma)

    qpi, qmu, qsigma = variational_model()
    energy = joint_log_prob(qpi, qmu, qsigma)
    entropy = - joint_log_variational_model(qpi, qmu, qsigma)
    elbo = energy + entropy

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_process = optimizer.minimize(- elbo)
    init_process = tf.global_variables_initializer()
    losses = []

    with tf.Session() as sess:
        sess.run(init_process)
        for i in range(num_epochs):
            sess.run(training_process)
            if i % 10 == 0:
                losses.append(sess.run(elbo))

        posterior_mu = sess.run([qmu])

    print(posterior_mu)
    plt.plot(range(len(losses)), losses)
    plt.savefig('./plots/gmm_tfp_simples_loss_vi1.png')


def main():
    # inference_map(num_epochs=2500, learning_rate=0.01)
    x_generated = inference_vi2(num_epochs=2000, learning_rate=0.075)

    plt.scatter(observations[:, 0], observations[:, 1], c=true_hidden_component, alpha=0.2)
    plt.scatter(x_generated[:, 0], x_generated[:, 1], marker='x')
    plt.savefig('./plots/gmm_tfp_simple_generate1.png')


if __name__ == '__main__':
    main()
