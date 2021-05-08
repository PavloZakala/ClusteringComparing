import time
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import silhouette_score

from tools import make_plot


def run_KMedoids_on_data_with_K(data_list, list_of_k: list = [], path="data"):
    for data_gener, k in zip(data_list, list_of_k):
        X, y = data_gener(path=path)
        kmedoids_alternate = KMedoids(n_clusters=k, random_state=0, method="alternate")
        start_alternate = time.time()
        kmedoids_alternate.fit(X)
        finish_alternate = time.time()

        kmedoids_pam = KMedoids(n_clusters=k, random_state=0, method="pam")
        start_pam = time.time()
        kmedoids_pam.fit(X)
        finish_pam = time.time()

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, kmedoids_alternate.labels_, axes=axes[0], title="Prediction Alternate")
        make_plot(X, kmedoids_pam.labels_, axes=axes[2], title="Prediction PAM")
        plt.show()

        print("[Alternate]: accuracy {:.4f}, time {:.3f} ".format(rand_score(kmedoids_alternate.labels_, y),
                                                                  finish_alternate - start_alternate))
        print("[PAM]: accuracy {:.4f}, time {:.3f}".format(rand_score(kmedoids_pam.labels_, y), finish_pam - start_pam))


def run_KMedoids_on_data(data_list, K_range, path="data"):
    for data_getter in data_list:
        X, y = data_getter(path=path)
        scores = []
        for K in K_range:
            kmedoids_alternate = KMedoids(n_clusters=K, random_state=0, method="alternate")
            kmedoids_alternate.fit(X)

            score = silhouette_score(X, kmedoids_alternate.labels_, metric='euclidean')
            scores.append(score)

        kmedoids_alternate = KMedoids(n_clusters=np.argmax(scores) + 2, random_state=0, method="alternate").fit(X)
        kmedoids_pam = KMedoids(n_clusters=np.argmax(scores) + 2, random_state=0, method="pam").fit(X)

        fig, axes = plt.subplots(2, 2, figsize=(9, 8))

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": K_list}), x="x", y="y",
                     ax=axes[0, 0])
        make_plot(X, y, axes=axes[0, 1], title="Target")
        make_plot(X, kmedoids_alternate.labels_, axes=axes[1, 0], title="Prediction Alternate")
        make_plot(X, kmedoids_pam.labels_, axes=axes[1, 1], title="Prediction PAM")

        plt.show()

        print("Best K = {}".format(np.argmax(scores) + 2))
        print("[Alternate]: accuracy {:.4f}".format(rand_score(kmedoids_alternate.labels_, y)))
        print("[PAM]: accuracy {:.4f}".format(rand_score(kmedoids_pam.labels_, y)))


def run_on_3d_data(X, y, K):
    kmedoids_alternate = KMedoids(n_clusters=K, random_state=0, method="alternate")
    start_alternate = time.time()
    kmedoids_alternate.fit(X)
    finish_alternate = time.time()

    kmedoids_pam = KMedoids(n_clusters=K, random_state=0, method="pam")
    start_pam = time.time()
    kmedoids_pam.fit(X)
    finish_pam = time.time()

    # Target
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    plt.show()

    # Predict Alternate
    print("[Predict Alternate]: accuracy {:.4f}, time {:.3f} ".format(rand_score(kmedoids_alternate.labels_, y),
                                                                      finish_alternate - start_alternate))
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    plt.show()

    # Predict PAM
    print("[Predict PAM]: accuracy {:.4f}, time {:.3f}".format(rand_score(kmedoids_pam.labels_, y),
                                                               finish_pam - start_pam))
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    plt.show()
