import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class MNISTDataSingleton(object):
    data = None
    pca = None

    X_train = None
    X_train_pca = None
    y_train = None

    X_test = None
    X_test_pca = None
    y_test = None

    def __init__(self, n_components=150):
        if MNISTDataSingleton.data is None:
            MNISTDataSingleton.data = fetch_mldata('MNIST original', data_home='./')
            X_train, X_test, y_train, y_test = train_test_split(MNISTDataSingleton.data['data'],
                                                                MNISTDataSingleton.data['target'],
                                                                test_size=0.2,
                                                                random_state=42)  # 42 is the answer after all..

            pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False).fit(X_train)
            MNISTDataSingleton.pca = pca

            MNISTDataSingleton.X_train = X_train
            MNISTDataSingleton.X_train_pca = pca.transform(X_train)
            MNISTDataSingleton.X_test = X_test
            MNISTDataSingleton.X_test_pca = pca.transform(X_test)

            MNISTDataSingleton.y_train = y_train
            MNISTDataSingleton.y_test = y_test


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def batch(iterable, batch_size):
    iter_len = len(iterable)
    for i in range(0, iter_len, batch_size):
        yield iterable[i: i+batch_size]