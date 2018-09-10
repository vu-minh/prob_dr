"""Mixture of PPCA (with pytorch)
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

dtype = np.float32

# Note the dimension of the tensors
# X: (D, N)
# Z: (M, N)
# pi: (K)
# sigma: (K)
# mu: (K, D)
# W: (K, D, M)


def initialize(X, N, D, M, K, variance=1.0):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X.T)
    mu = kmeans.cluster_centers_

    pi = np.ones([K, 1], dtype=dtype) / K
    sigma = np.ones([K, 1], dtype=dtype) * 0.5
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


def score():
    """Score: log likelihood of complete-data
        $log p(x, mu, sigma, pi)$

        Returns:
            score: (scalar)
    """
    # calculate E[z] (eq. 71)
    # calculate E[zzT] (eq. 72)
    # calculate E[Lc] (eq. 81)
    pass


def em_loop(X, M, K, n_iters=100, tolerance=1e-4):
    D, N = X.shape
    print('Dataset: N={}, D={} --> M={}, K={}'.format(N, D, M, K))

    pi, sigma, mu, W = initialize(X, N, D, M, K)
    print('Init params: ')
    #print(pi, sigma, mu)
    print('Done init\n')

    for i in tqdm(range(n_iters)):
        # print('================\nIteration: {}'.format(i+1))
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
        converge = False
        if not converge:
            # update param
            pi, sigma, mu, W = new_pi, new_sigma, new_mu, new_W

        # print('Done iteration {}\n\n'.format(i+1))

        # check NAN
        check_nan_all([pi, sigma, mu, W])

        # predicted cluster
        predicted_labels = np.argmax(R, axis=0)

        # latent position (eq. 71)
        # $z_n = T_inv * W_k.T * (x_n - mu_k)$
        Z = np.zeros([M, N])
        for i in range(N):
            label_i = predicted_labels[i]
            mean_i = mu[label_i]
            W_i = W[label_i] # (D, M)
            Z[:, i] = T_inv[label_i] @ W_i.T @ (X[:, i]) # (M, M) @ (M, D) @ (D, 1)
            
    return pi, sigma, mu, W, R, Z


def check_nan_all(params):
    for param in params:
        if np.isnan(param).any():
            raise Exception('Param values containts NAN')


def main():
    N = 500
    M = 2
    K = 3
    X, y = load_iris(return_X_y=True)
    X = X[:N].astype(np.float32).T
    y = y[:N]
    D, N = X.shape

    pi, sigma, mu, W, R, Z = em_loop(X, M, K, n_iters=20)
    print(pi)
    # print(mu)
    predicted_labels = np.argmax(R, axis=0)
    hcv = hcv_measure(labels_true=y, labels_pred=predicted_labels)
    print(hcv)


    muZ = np.zeros([K, M])
    for k in range(K):
        muZ[k] = mu[k].reshape(1, D) @ W[k].reshape(D, M)
    print('muZ: ', muZ.shape)

    plt.scatter(Z[0], Z[1], c=y, alpha=0.3)
    # plt.scatter(Z[0], Z[1], c=predicted_labels, marker='*', alpha=0.2)
    # plt.scatter(muZ[:, 0], muZ[:, 1], marker='*', color='red')
    plt.savefig('./plots/ppca_np_digits0.png')
    plt.show()


if __name__ == '__main__':
    main()

# GMM Sklearn:  -2.05502477396482
# [[5.92570671 2.74947486 4.40355611 1.41204163]
#  [6.80608227 3.07023101 5.71889404 2.10305301]
#  [5.006      3.41800001 1.464      0.244     ]]
# [0.41200771 0.25465896 0.33333333]
# Measure:  (0.7695955651465264, 0.7859016861703576, 0.77766315794623)

# my code MNIST small:
# [[0.11814364]
#  [0.11351336]
#  [0.09849755]
#  [0.13053337]
#  [0.05953876]
#  [0.09761061]
#  [0.08124621]
#  [0.0980306 ]
#  [0.09794485]
#  [0.10494106]]
# (0.7939168638797219, 0.8002401950243518, 0.7970659884929948)