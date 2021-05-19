import os.path
import random
import time
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
import tracemalloc

from sklearn.base import clone
from matplotlib import pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import silhouette_score
from scipy.interpolate import make_interp_spline

from tools import make_plot


def run_KMedoids_on_data_with_K(data_list, list_of_k: list = [], path="data"):
    for i, (data_gener, k) in enumerate(zip(data_list, list_of_k)):
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
        plt.savefig(os.path.join(path, "kmediods{}.png".format(i)))
        plt.show()

        print("[Alternate]: accuracy {:.4f}, time {:.3f} ".format(rand_score(kmedoids_alternate.labels_, y),
                                                                  finish_alternate - start_alternate))
        print("[PAM]: accuracy {:.4f}, time {:.3f}".format(rand_score(kmedoids_pam.labels_, y), finish_pam - start_pam))


def run_KMedoids_on_data(data_list, K_range, path="data"):
    for i, data_getter in enumerate(data_list):
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

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": K_range}), x="x", y="y",
                     ax=axes[0, 0])
        make_plot(X, y, axes=axes[0, 1], title="Target")
        make_plot(X, kmedoids_alternate.labels_, axes=axes[1, 0], title="Prediction Alternate")
        make_plot(X, kmedoids_pam.labels_, axes=axes[1, 1], title="Prediction PAM")

        plt.savefig(os.path.join(path, "kmediodsFindK{}.png".format(i)))
        plt.show()

        print("Best K = {}".format(np.argmax(scores) + 2))
        print("[Alternate]: accuracy {:.4f}".format(rand_score(kmedoids_alternate.labels_, y)))
        print("[PAM]: accuracy {:.4f}".format(rand_score(kmedoids_pam.labels_, y)))


def run_on_3d_data(X, y, K, path="data"):
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


def evaluation_time_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_k: list = list(range(2, 30)),
        total_size: int = 2500,
        method="alternate",
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_k_range = np.linspace(min(finding_k), max(finding_k), 300)
    time_table_df["range"] = new_k_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        time_line = []
        for K_for_find in tqdm(finding_k):
            kmedoids = KMedoids(n_clusters=K_for_find, random_state=0, method=method)
            start_alternate = time.time()
            kmedoids.fit(X)
            finish_alternate = time.time()
            time_line.append(finish_alternate - start_alternate)

        time_table.append(time_line)
        gfg = make_interp_spline(finding_k, time_line, k=3)
        time_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.show()


def evaluation_time_of_working_by_size(
        real_k: int = 7,
        finding_k: list = [2, 3, 5, 8, 10, 15, 30],
        total_size: list = list(range(500, 7000, 500)),
        method="alternate",
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_size_range = np.linspace(min(total_size), max(total_size), 300)
    time_table_df["range"] = new_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        time_line = []
        for K_for_find in tqdm(finding_k):
            kmedoids = KMedoids(n_clusters=K_for_find, random_state=0, method=method)
            start_alternate = time.time()
            kmedoids.fit(X)
            finish_alternate = time.time()
            time_line.append(finish_alternate - start_alternate)

        time_table.append(time_line)
    time_table = np.array(time_table).T

    for row_by_k, K in zip(time_table, finding_k):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        time_table_df["K={}".format(K)] = gfg(new_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in finding_k], columns=total_size),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.show()



def evaluation_mem_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_k: list = list(range(2, 30)),
        total_size: int = 2500,
        method="alternate",
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_k_range = np.linspace(min(finding_k), max(finding_k), 300)
    mem_table_df["range"] = new_k_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        mem_line = []
        for K_for_find in tqdm(finding_k):
            kmedoids = KMedoids(n_clusters=K_for_find, random_state=0, method=method)
            tracemalloc.start()
            kmedoids.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
            mem_line.append(peak - current)

        mem_table.append(mem_line)

        gfg = make_interp_spline(finding_k, mem_line, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.show()


def evaluation_mem_of_working_by_size(
        real_k: int = 7,
        finding_k: list = [2, 3, 5, 8, 10, 15, 30],
        total_size: list = list(range(500, 7000, 500)),
        method="alternate",
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_size_range = np.linspace(min(total_size), max(total_size), 300)
    mem_table_df["range"] = new_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        mem_line = []
        for K_for_find in tqdm(finding_k):

            kmedoids = KMedoids(n_clusters=K_for_find, random_state=0, method=method)
            tracemalloc.start()
            kmedoids.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_line.append(peak - current)
        mem_table.append(mem_line)
    mem_table = np.array(mem_table).T

    for row_by_k, K in zip(mem_table, finding_k):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in finding_k], columns=total_size),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.show()


def check_init_dependency(method="pam", path="data"):
    from datasets import test_gaussian_data

    SIZE = 1500
    REAL_K = [2, 3, 5, 8, 10, 15, 30]
    scores_table = []
    for K in REAL_K:
        X, y = test_gaussian_data(SIZE, K)
        scores = []
        for i in tqdm(range(100)):
            kmedoids = KMedoids(n_clusters=K, random_state=i, init="build", method=method)
            kmedoids.fit(X)
            scores.append(rand_score(kmedoids.labels_, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.show()

def check_stability(method="pam", path="data"):
    from datasets import test_gaussian_data

    CLUSTER_SIZE = 30
    REAL_K = [2, 3, 5, 8, 10, 12]
    scores_table = []
    for K in REAL_K:
        # X, y = test_gaussian_data(SIZE, K)
        total_size = CLUSTER_SIZE * K
        delta_size = int(total_size * 0.1)
        scores = []

        for i in tqdm(range(100)):
            X, y = test_gaussian_data(total_size, K, random_state=i)

            kmedoids = KMedoids(n_clusters=K, random_state=0, init="build", method=method)
            kmedoids.fit(X)
            origin_score = rand_score(kmedoids.labels_, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.2 * 3 / K
            idx = np.random.choice(total_size, delta_size, replace=False)
            X[idx] = X[idx] + delta

            kmedoids = KMedoids(n_clusters=K, random_state=0, init="build", method=method)
            kmedoids.fit(X)
            delta_score = rand_score(kmedoids.labels_, y)

            scores.append(abs(delta_score - origin_score))

            # if (i + 1) % 100 == 0:
            #     fig, axes = plt.subplots(1, figsize=(9, 8))
            #
            #     make_plot(X, y, axes=axes, title="Target")
            #     plt.show()

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.show()


if __name__ == '__main__':
    # evaluation_time_of_working_by_k()
    # evaluation_time_of_working_by_size()
    # evaluation_mem_of_working_by_size()
    # evaluation_mem_of_working_by_k()
    # check_init_dependency()
    # check_stability()
    check_stability()

