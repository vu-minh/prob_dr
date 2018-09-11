"""PCA and GMM using sklearn
"""


import numpy as np

from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.metrics import homogeneity_completeness_v_measure as hcv_measure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from matplotlib import pyplot as plt
from matplotlib import cm


def load_dataset(id=0):
    load_funcs = [
        load_iris,
        load_digits,
        load_wine,
        load_breast_cancer,
    ]
    dataset = load_funcs[id](return_X_y=False)
    X = dataset['data']
    y = dataset['target']
    K = len(dataset['target_names'])
    return X, y, K


def measure_cluster(labels_true, labels_pred, msg='Clustering measures'):
    print('\n' + msg + ':')
    hcv = hcv_measure(labels_true, labels_pred)
    print('Clustering measures: '
          'hom={:.3f}, com={:.3f}, v-measure={:.3f}'.format(*hcv))


def evaluate(dataset_name, id):
    X, y, K = load_dataset(id)

    gmm = GaussianMixture(n_components=K, covariance_type='diag')
    gmm.fit(X)
    predicted_labels1 = gmm.predict(X)
    measure_cluster(y, predicted_labels1, msg='GMM on the original dataset')

    pca = PCA(n_components=K)
    X2d = pca.fit_transform(X)
    gmm2 = GaussianMixture(n_components=K, covariance_type='diag')
    gmm2.fit(X2d)
    predicted_labels2 = gmm2.predict(X2d)
    measure_cluster(y, predicted_labels2, msg='PCA then GMM on reduced data')

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    plt.rcParams.update({'axes.titlesize':'xx-large'})
    
    # True label 
    axes[0].scatter(X2d[:, 0], X2d[:, 1], marker='o', color='white', alpha=1.0,
                linewidths=1, s=64,
                cmap='tab10', edgecolors=cm.tab10(y))
    axes[0].set_title('Border color = True labels')

    # GMM labels
    # axes[1].scatter(X2d[:, 0], X2d[:, 1], marker='o', color='white', alpha=0.6,
    #             linewidths=1, s=64,
    #             cmap='tab10', edgecolors=cm.tab10(y))
    axes[1].scatter(X2d[:, 0], X2d[:, 1], marker='o', alpha=1.0,
                linewidths=1, s=64,
                cmap='tab10', c=predicted_labels1)
    axes[1].set_title('Predicted labels with GMM on original data')

    # PCA -> GMM labels
    # axes[2].scatter(X2d[:, 0], X2d[:, 1], marker='o', color='white', alpha=0.6,
    #             linewidths=1, s=64,
    #             cmap='tab10', edgecolors=cm.tab10(y))                
    axes[2].scatter(X2d[:, 0], X2d[:, 1], c=predicted_labels2, s=64, alpha=1.0,
                cmap='tab10')
    axes[2].set_title('Predicted labels with GMM on reduced data by PCA')

    plt.tight_layout()
    plt.savefig('./plots/pca_gmm_{}.png'.format(dataset_name))


def main():
    datasets = [
        'IRIS', 'DIGITS', 'WINE', 'BREAST_CANCER'
    ]
    for dataset_id, dataset_name in enumerate(datasets):
        print('Dataset: {}'.format(dataset_name))
        evaluate(dataset_name, dataset_id)

    # evaluate('IRIS', 0)


if __name__ == '__main__':
    main()
