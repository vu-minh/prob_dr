"""Mixture of PPCA (with numpy)
    20180910
"""

import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.metrics import homogeneity_completeness_v_measure as hcv_measure
from sklearn.cluster import KMeans

from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt
from matplotlib import cm

dtype = np.float32

# Note the dimension of the tensors
# X: (D, N)
# Z: (M, N)
# pi: (K)
# sigma: (K)
# mu: (K, D)
# W: (K, D, M)


def initialize(X, N, D, M, K, variance=0.25):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X.T)
    mu = kmeans.cluster_centers_

    pi = np.ones([K, 1], dtype=dtype) / K
    sigma = np.ones([K, 1], dtype=dtype) * variance
    W = np.zeros([K, D, M], dtype=dtype)
    for k in range(K):
        W[k] = np.random.randn(D, M)

    return pi, sigma, mu, W


def log_pdf_ppca(X, W_k, mu_k, sigma_k):
    """Calculate the log density of each point in a dataset
    w.r.t a specific local PPCA model
    $p(x_n | mu_k, sigma_k) ~ N(x_n | mu_k, W_k*W_k^T + sigma_k I)$

    Args:
        X: (D, N)
        W_k: (D, M)
        mu_k: (D)
        sigma_k: (scalar)

    Returns:
        log_density: (N)
        T_inv: (M, M)
    """
    D, N = X.shape
    D, M = W_k.shape

    C = W_k @ W_k.T + sigma_k * np.eye(D)  # (D, D)
    T = W_k.T @ W_k + sigma_k * np.eye(M)  # (M, M)
    T_inv = np.linalg.inv(T)          # (M, M)
    # C_inv = 1.0 / sigma_k * (           # (D, D)
    #     np.ones(D) - W_k @ T_inv @ W_k.T
    # )
    # log_det = - 0.5 * np.log(np.linalg.det(C))  # scalar
    # coeff = -0.5 * D * np.log(2 * np.pi)  # scalar

    # X_centered = X - mu_k.reshape(D, 1)  # (D, N)
    # log_density = coeff + log_det \
    #     - 0.5 * (X_centered.T @ C_inv @ X_centered).sum(axis=0)  # (N, 1)

    log_density2 = multivariate_normal.logpdf(X.T, mean=mu_k, cov=C)
    # # print('my implementation', log_density.sum())
    # # print('scipy: ', log_density2.sum())

    return log_density2, T_inv


def posterior(X, pi, sigma, mu, W):
    """Responsibility

        Returns:
            R (K, N)
            T_inv (M, M)
    """
    D, N = X.shape
    K, D, M = W.shape

    log_R = np.zeros([K, N])
    T_inv = np.zeros([K, M, M])

    for k in range(K):
        log_p_k, T_inv_k = log_pdf_ppca(X, W[k], mu[k], sigma[k])
        log_R[k] = np.log(pi[k]) + log_p_k  # (N, )
        T_inv[k] = T_inv_k
    log_R_sum = logsumexp(log_R, axis=0).reshape(1, N)
    R = np.exp(log_R - log_R_sum)
    return R, T_inv


def score(X, pi, sigma, mu, W, R, T_inv):
    """Score: log likelihood of complete-data
        $log p(x, mu, sigma, pi)$

        Args:
            X: (D, N)
            pi: (K)
            sigma: (K)
            mu: (K, D)
            W: (K, D, M)
            R: (K, N)
            T_inv: (K, M, M)

        Returns:
            score: (scalar)
            Z: (K, M, N) latent position of each point w.r.t each cluster
    """
    D, N = X.shape
    K, D, M = W.shape

    Z = np.zeros([K, M, N])  # latent pos for each point w.r.t each component
    ZZ = np.zeros([K, M, M])
    for k in range(K):
        # calculate latent position E[z] (eq. 71)
        # $z_n = T_inv * W_k.T * (x_n - mu_k)$
        Z[k] = T_inv[k] @ W[k].T @ (X - mu[k].reshape(D, 1))
        # calculate E[zzT] (eq. 72)
        ZZ[k] = sigma[k] * T_inv[k] + Z[k] @ Z[k].T

    # calculate E[Lc] (eq. 81)
    lc = 0.0
    for n in range(N):
        t = 0.0
        for k in range(K):
            sigma_k = sigma[k][0]
            Xk = X[:, n].reshape(D, 1) - mu[k].reshape(D, 1)  # (D, 1)
            Zkn = Z[k, :, n].reshape([M, 1])

            t += np.log(pi[k])
            t -= 0.5 * D * np.log(sigma[k])
            t -= 0.5 * M * np.log(2 * np.pi)
            t -= 0.5 * np.asscalar(Zkn.T @ Zkn)
            t -= 0.5 / sigma_k * np.asscalar(Xk.T @ Xk)
            t += 1.0 / sigma[k] * np.asscalar(Zkn.T @ W[k].T @ Xk)
            t -= 0.5 / sigma[k] * np.diag(W[k].T @ W[k] @ ZZ[k]).sum()

        lc += R[k, n] * t

    return - lc, Z


def mppca(X, M, K, n_iters=100, tolerance=1e-4):
    D, N = X.shape
    print('Dataset: N={}, D={} --> M={}, K={}'.format(N, D, M, K))

    pi, sigma, mu, W = initialize(X, N, D, M, K)
    print('Init params: ')
    #print(pi, sigma, mu)
    print('Done init\n')

    scores = []

    for i in tqdm(range(n_iters)):
        # E-Step
        # calculate responsibility (eq. 21), using old params
        old_pi, old_sigma, old_mu, old_W = pi, sigma, mu, W
        R, T_inv = posterior(X, old_pi, old_sigma, old_mu, old_W)  # (K, N)
        assert R.shape == (K, N)
        assert T_inv.shape == (K, M, M)

        # update pi (eq. 22)
        K, N = R.shape
        N_k = np.sum(R, axis=1).reshape(K, 1)  # (K, )
        new_pi = N_k / N  # (K, )
        assert new_pi.shape == (K, 1)

        # update mu (eq. 23)
        new_mu = R @ X.T / N_k  # (K, D)
        assert new_mu.shape == (K, D)

        # M-Step
        new_sigma = np.ones([K, 1], dtype=dtype)
        new_W = np.ones([K, D, M], dtype=dtype)
        for k in range(K):
            # calculate Si (eq. 84)
            mu_k = new_mu[k].reshape(D, 1)
            R_k = R[k].reshape(1, N)

            X_centered = X - mu_k  # (D, N)
            cov = (R_k * X_centered) @ X_centered.T  # ((N, 1)*(N, D)) @ (D, N)
            S_k = 1.0 / (new_pi[k] * N) * cov
            assert S_k.shape == (D, D)

            SkWk = S_k @ old_W[k]  # (D, D) @ (D, M) -> (D, M)
            # update W (eq. 82)
            temp = np.linalg.inv(sigma[k]*np.eye(M) +  # (M, M)
                                 T_inv[k] @ old_W[k].T @ SkWk)  # (M, M)
            new_W[k] = SkWk @ temp  # (D, M) @ (M, M) -> (D, M)
            assert new_W[k].shape == (D, M)

            # update sigma (eq. 83)
            # (D, D) - (D, M) @ (M, M) @ (M, D)
            temp = S_k - SkWk @ T_inv[k] @ new_W[k].T
            new_sigma[k] = 1.0 / D * np.diag(temp).sum()  # tr(D, D) -> scalar

        # check covergence based on score
        logllh, Z = score(X, new_pi, old_sigma, new_mu, old_W, R, T_inv)
        scores.append(logllh)

        converge = False
        if not converge:
            # update param
            pi, sigma, mu, W = new_pi, new_sigma, new_mu, new_W

        # check NAN
        check_nan_all([pi, sigma, mu, W])

        # predicted cluster
        predicted_labels = np.argmax(R, axis=0)

        # latent position not centered (eq. 71)
        # $z_n = T_inv * W_k.T * x_n$
        Zk = np.zeros([M, N])
        for i in range(N):
            label_i = predicted_labels[i]
            W_i = W[label_i]  # (D, M)
            Zk[:, i] = T_inv[label_i] @ W_i.T @ (
                X[:, i])  # (M, M) @ (M, D) @ (D, 1)

    return pi, sigma, mu, W, R, Z, Zk, scores


def check_nan_all(params):
    for param in params:
        if np.isnan(param).any():
            raise Exception('Param values containts NAN')


def load_sin_curve(N=500, K=8, return_X_y=False):
    X = np.zeros((N, 2))
    step = 4. * np.pi / N
    for i in range(X.shape[0]):
        x = i * step - 6.
        X[i, 0] = x + np.random.normal(0, 0.1)
        X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))
    return {'data': X, 'target': [0]*N, 'target_names': [0]*K}


def load_dataset(id=0):
    load_funcs = [
        load_iris,
        load_digits,
        load_wine,
        load_breast_cancer,
        load_sin_curve
    ]
    dataset = load_funcs[id](return_X_y=False)
    X = dataset['data']
    y = dataset['target']
    K = len(dataset['target_names'])
    return X, y, K


def evaluate(dataset_name, id, selected_classes=None):
    if selected_classes is None:
        X, y, K = load_dataset(id)
    else:
        X, y, K = digits_some_classes(selected_classes)
        dataset_name = '{}_{}'.format(dataset_name, '-'.join(
            list(map(str, selected_classes))
        ))

    N = 2000
    M = 2
    X = X[:N].astype(np.float32).T
    y = y[:N]
    D, N = X.shape

    pi, sigma, mu, W, R, Z, Zk, scores = mppca(X, M, K, n_iters=50)

    predicted_labels = np.argmax(R, axis=0)
    hcv = hcv_measure(labels_true=y, labels_pred=predicted_labels)
    print('Clustering measures: '
          'hom={:.3f}, com={:.3f}, v-measure={:.3f}'.format(*hcv))

    plt.plot(range(len(scores)), scores)
    plt.savefig('./plots/mppca_scores_{}.png'.format(dataset_name))
    plt.gcf().clear()

    scatter_with_compare(Zk.T, y, predicted_labels, dataset_name)


def scatter_with_compare(X2d, y, predicted_labels, dataset_name):
    plt.scatter(X2d[:, 0], X2d[:, 1], marker='o', color='white', alpha=1.0,
                linewidths=1, s=64,
                cmap='tab10', edgecolors=cm.tab10(y))
    plt.scatter(X2d[:, 0], X2d[:, 1], c=predicted_labels, s=16, alpha=0.8,
                cmap='tab20b')
    plt.savefig('./plots/mppca_{}.png'.format(dataset_name))
    plt.gcf().clear()


def test_sin_curve(N=500):
    X, y, K = load_dataset(id=4)
    X = X.T
    pi, sigma, mu, W, R, Z, Zk, scores = mppca(X, M=2, K=6, n_iters=20)
    predicted_labels = np.argmax(R, axis=0)
    plt.scatter(X[0], X[1], c=predicted_labels, alpha=0.5)
    plt.savefig('./plots/mppca_np_sin_curve.png')


def digits_some_classes(selected_classes=[0, 1], num_datapoints=2000):
    x_train, y_train = load_digits(return_X_y=True)
    mask = [True if yclass in selected_classes else False for yclass in y_train]
    X = x_train[mask][:num_datapoints]
    y = y_train[mask][:num_datapoints]
    K = len(selected_classes)
    return X, y, K


if __name__ == '__main__':
    # datasets = [
    #     'IRIS', 'DIGITS', 'WINE', 'BREAST_CANCER'
    # ]
    # for dataset_id, dataset_name in enumerate(datasets):
    #     print('Dataset: {}'.format(dataset_name))
    #     evaluate(dataset_name, dataset_id)
    #     print()
    evaluate('DIGITS', id=1, selected_classes=[8, 0])
