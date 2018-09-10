'''GMM with primitive tfp.distributions (20180904)
    Using a simple model as in http://edwardlib.org/tutorials/unsupervised
    Using PPCA model as mixture components (20180906)
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris, load_digits

tfd = tfp.distributions
tfb = tfp.bijectors

print('Tensorflow version: ', tf.__version__)

# always to remembre to reset the default computational graph
tf.reset_default_graph()

dtype = np.float32
N = 500  # number of data points
D = 2  # dimensionality in original space
M = 2  # dimensionality in reduced space
K = 3  # number of components in the mixture

# Generate the data
true_loc = np.array([[-5, 3],
                     [0, 0],
                     [2, 2]], dtype)
random = np.random.seed(seed=42)

# X = np.zeros((N, 2))
# step = 4. * np.pi / N
# for i in range(X.shape[0]):
#     x = i * step - 6.
#     X[i, 0] = x + np.random.normal(0, 0.1)
#     X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))
# true_hidden_component = []
# observations = X.astype(dtype)
# D = 2
# K = 8

# true_hidden_component = random.randint(0, K, N)
# count = [np.count_nonzero(true_hidden_component == k) * 1.0 / N
#          for k in range(K)]
# print(count)
# observations = (true_loc[true_hidden_component] +
#                 random.randn(N, D).astype(dtype))

observations, true_hidden_component = load_iris(return_X_y=True)
observations = observations.astype(dtype)
N, D = observations.shape

# plt.scatter(observations[:, 0], observations[:, 1])
# plt.savefig('./plots/gmm_tfp_data.png')
# plt.gcf().clear()

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

rv_z = tfd.Independent(
    tfd.Normal(
        loc=tf.zeros([K, N, M], dtype),
        scale=tf.ones([K, N, M], dtype)
    ),
    reinterpreted_batch_ndims=1,
    name='rv_z'
)

rv_w = tfd.Independent(
    tfd.Normal(
        loc=tf.zeros([K, D, M], dtype),
        scale=tf.ones([K, D, M], dtype)
    ),
    reinterpreted_batch_ndims=1,
    name='rv_w'
)

print('\n'.join(str(rv) for rv in [rv_pi, rv_mu, rv_sigma, rv_z, rv_w]))

# MODEL


def joint_log_prob(pi, mu, sigma, z, w):
    # new rv representing the observed data, which is drawn from a mixture
    rv_observations = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=pi),
        components_distribution=tfd.MultivariateNormalDiag(
            # UP! RETHINK here
            loc=tf.matmul(z, w, transpose_b=True) + mu,
            scale_diag=sigma
        )
    )
    print('RV Observation: ', rv_observations)

    # a list of log_probs of all components in the joint
    log_probs_parts = [
        rv_observations.log_prob(observations),
        rv_pi.log_prob(pi)[..., tf.newaxis],
        rv_mu.log_prob(mu),
        rv_sigma.log_prob(sigma),
        rv_z._log_prob(z),
        rv_w.log_prob(w)
    ]
    sum_log_prob = tf.reduce_sum(tf.concat(log_probs_parts, axis=-1), axis=-1)
    return sum_log_prob


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
    plt.gcf().clear()


def inference_vi2(num_epochs=2000, learning_rate=0.05):
    # variational variables
    qpi_alpha = tf.nn.softplus(tf.Variable(np.ones(K) * 1.0 / K, dtype=dtype))
    qmu_loc = tf.Variable(np.zeros([K, D]), dtype=dtype)
    # `20180905` try to make the 'variance' smaller (no evidence)
    qmu_scale = tf.nn.softplus(0.1 * tf.Variable(np.ones([K, D]), dtype=dtype))
    qsigma_alpha = tf.nn.softplus(
        0.5 * tf.Variable(np.ones([K, D]), dtype=dtype))
    qsigma_beta = tf.nn.softplus(
        0.5 * tf.Variable(np.ones([K, D]), dtype=dtype))

    qz_loc = tf.Variable(np.zeros([K, N, M]), dtype=dtype)
    qz_scale = tf.nn.softplus(tf.Variable(np.ones([K, N, M]), dtype=dtype))
    qw_loc = tf.Variable(np.zeros([K, D, M]), dtype=dtype)
    qw_scale = tf.nn.softplus(tf.Variable(np.ones([K, D, M]), dtype=dtype))

    def variational_model():
        qpi = ed.Dirichlet(concentration=qpi_alpha, name='qpi')
        # This model works well
        qmu = ed.MultivariateNormalDiag(
            loc=qmu_loc, scale_diag=qmu_scale, name='qmu')
        # qmu = ed.Normal(loc=qmu_loc, scale=qmu_scale, name='qmu')
        qsigma = ed.InverseGamma(concentration=qsigma_alpha, rate=qsigma_beta,
                                 name='qsigma')
        qz = ed.MultivariateNormalDiag(loc=qz_loc, scale_diag=qz_scale, name='qz')
        qw = ed.MultivariateNormalDiag(loc=qw_loc, scale_diag=qw_scale, name='qw')
        return qpi, qmu, qsigma, qz, qw

    log_q = ed.make_log_joint_fn(variational_model)

    def joint_log_variational_model(qpi, qmu, qsigma, qz, qw):
        return log_q(qpi=qpi, qmu=qmu, qsigma=qsigma, qz=qz, qw=qw)

    qpi, qmu, qsigma, qz, qw = variational_model()
    energy = joint_log_prob(qpi, qmu, qsigma, qz, qw)
    entropy = - joint_log_variational_model(qpi, qmu, qsigma, qz, qw)
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

    # print(posterior_mu)
    plt.plot(range(len(losses)), losses)
    plt.savefig('./plots/gmm_tfp_simples_loss_vi2.png')
    plt.gcf().clear()

    # Model criticism
    generate_process = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=posterior_pi),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=posterior_mu,
            scale_diag=posterior_sigma
        )
    )

    with tf.Session() as sess:
        # get cluster for observed data
        observations_pad = generate_process._pad_sample_dims(
            tf.convert_to_tensor(observations))
        log_prob_x = generate_process.components_distribution._log_prob(
            observations_pad)
        clusters = tf.argmax(log_prob_x, axis=1).eval()

        # test generate some new data
        n_samples = 250
        x_generated = generate_process.sample(n_samples)
        x_generated = x_generated.eval()
    return x_generated, clusters, posterior_pi, posterior_mu, posterior_sigma


def inference_vi1(num_epochs=2000, learning_rate=0.05):
    def variational_model():
        qpi = tf.nn.softplus(tf.Variable(
            np.zeros(K) * 1.0 / K, dtype=dtype), name='qpi')
        qmu = tf.Variable(np.zeros([K, D]), dtype=dtype, name='qmu')
        qsigma = tf.nn.softplus(tf.Variable(
            np.ones([K, D]), dtype=dtype, name='qsigma'))
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


def make_ellipses(means, covs, ax):
    # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    # colors = ['navy', 'turquoise', 'darkorange']
    import itertools
    color_iter = itertools.cycle(
        ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
    for n in range(len(covs)):
        color = next(color_iter)
        covariances = np.diag(covs[n][:2])
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(means[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)


def main():
    # inference_map(num_epochs=2500, learning_rate=0.01)
    x_generated, predicted_clusters, pi, mu, sigma = inference_vi2(
        num_epochs=5000, learning_rate=0.032)
    print('Cluster proportion: ', pi)

    # hcv = homogeneity_completeness_v_measure(
    #     labels_true=true_hidden_component, labels_pred=predicted_clusters)
    # print('MyGMM homogeneity={}, completeness={}, v_measure={}'.format(*hcv))

    _, ax = plt.subplots()
    plt.scatter(observations[:, 0], observations[:, 1], alpha=0.4)
    # plt.scatter(x_generated[:, 0], x_generated[:, 1], marker='x')
    plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='black', alpha=0.2)
    make_ellipses(means=mu, covs=sigma, ax=ax)

    plt.savefig('./plots/gmm_tfp_simple_generate1.png')
    plt.gcf().clear()


def gmm_sklearn():
    estimator = GaussianMixture(n_components=K, covariance_type='diag',
                                max_iter=50, random_state=0)
    estimator.fit(observations)
    predicted_clusters = estimator.predict(observations)

    hcv = homogeneity_completeness_v_measure(
        labels_true=true_hidden_component, labels_pred=predicted_clusters)
    print('SklearnGMM: homogeneity={}, completeness={}, v_measure={}'.format(*hcv))


if __name__ == '__main__':
    main()
    # gmm_sklearn()
    # SklearnGMM: homogeneity=0.603092532433, completeness=0.6219483973602182, v_measure=0.6123753498912625
