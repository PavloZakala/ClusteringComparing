import os.path
import time
import tracemalloc

import hdbscan
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm

from tools import make_plot, make_3d_plot, update_table


def run_HDBSCAN_on_data_with_K(data_list, list_of_k: list = [], name="", path="data"):
    alg_time = {"hdbscan_time": []}
    alg_mem = {"hdbscan_mem": []}

    for i, (data_gener, min_cluster_size) in enumerate(zip(data_list, list_of_k)):
        X, y = data_gener(path=path)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        tracemalloc.start()
        start = time.time()
        clusterer.fit_predict(X)
        finish = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["hdbscan_time"].append(finish - start)
        alg_mem["hdbscan_mem"].append((peak - current) / 10 ** 6)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, clusterer.labels_, axes=axes[0], title="Prediction HDBSCAN")

        plt.savefig(os.path.join(path, "images", "hdbscan_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("[HDBSCAN]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(clusterer.labels_, y),
            finish - start,
            alg_mem["hdbscan_mem"][-1]
        ))

    path = os.path.join(path, "images", "data_{}.pkl".format(name.upper()))
    update_table({**alg_mem, **alg_time}, path)


def run_HDBSCAN_on_data(data_list, min_cluster_size_range, name="", path="data"):
    for i, data_getter in enumerate(data_list):
        X, y = data_getter(path=path)
        scores = []
        for min_cluster_size in min_cluster_size_range:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

            score = silhouette_score(X, clusterer.labels_, metric='euclidean')
            scores.append(score)
        best_min_cluster_size = min_cluster_size_range[np.argmax(scores)]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=best_min_cluster_size)

        fig, axes = plt.subplots(1, 3, figsize=(9, 8))

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": min_cluster_size_range}), x="x", y="y",
                     ax=axes[0])
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, clusterer.labels_, axes=axes[2], title="Prediction HDBSCAN")

        plt.savefig(os.path.join(path, "images", "hdbscan_find_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("Best min_cluster_size = {}".format(best_min_cluster_size))
        print("[HDBSCAN]: accuracy {:.4f}".format(rand_score(clusterer.labels_, y)))


def run_on_3d_data(X, y, min_cluster_size, data_name, path="data"):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    tracemalloc.start()
    start = time.time()
    clusterer.fit(X)
    finish = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Target
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "hdbscan_{}_PairGrid_Target.png".format(data_name)))
    plt.show()

    make_3d_plot(X, y)
    plt.savefig(os.path.join(path, "images", "hdbscan_{}_3d_Target.png".format(data_name)))
    plt.show()

    # Predict HDBSCAN
    print("[HDBSCAN]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(clusterer.labels_, y),
        finish - start,
        (peak - current) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = clusterer.labels_

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "hdbscan{}_PairGrid_Predict.png".format(data_name)))
    plt.show()

    make_3d_plot(X, clusterer.labels_)
    plt.savefig(os.path.join(path, "images", "hdbscan_{}_3d_predict.png".format(data_name)))
    plt.show()


def evaluation_time_of_working_by_min_cluster_size(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_min_cluster_size: list = [50, 100, 150, 200, 300],
        total_size: int = 2500,
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_min_cluster_size_range = np.linspace(min(finding_min_cluster_size), max(finding_min_cluster_size), 300)
    time_table_df["range"] = new_min_cluster_size_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        time_line = []
        for min_cluster_size in tqdm(finding_min_cluster_size):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            start = time.time()
            clusterer.fit(X)
            finish = time.time()
            time_line.append(finish - start)

        time_table.append(time_line)
        gfg = make_interp_spline(finding_min_cluster_size, time_line, k=3)
        time_table_df["K={}".format(K)] = gfg(new_min_cluster_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in real_k],
                             columns=finding_min_cluster_size), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "hdbscan_timeK.png"))
    plt.show()


def evaluation_time_of_working_by_size(
        real_k: int = 7,
        finding_min_cluster_size: list = [50, 100, 150, 200, 300],
        total_size: list = list(range(500, 7000, 500)),
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_min_cluster_size_range = np.linspace(min(finding_min_cluster_size), max(finding_min_cluster_size), 300)
    time_table_df["range"] = new_min_cluster_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        time_line = []
        for min_cluster_size in tqdm(finding_min_cluster_size):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            start_alternate = time.time()
            clusterer.fit(X)
            finish_alternate = time.time()
            time_line.append(finish_alternate - start_alternate)

        time_table.append(time_line)
    time_table = np.array(time_table).T

    for row_by_k, min_cluster_size in zip(time_table, finding_min_cluster_size):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        time_table_df["min_cluster_size={}".format(min_cluster_size)] = gfg(new_min_cluster_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["min_cluster_size={}".format(K) for K in finding_min_cluster_size],
                             columns=total_size),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "hdbscan_time_size.png"))
    plt.show()


def evaluation_mem_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_min_cluster_size: list = [50, 100, 150, 200, 300],
        total_size: int = 2500,
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_min_cluster_size_range = np.linspace(min(finding_min_cluster_size), max(finding_min_cluster_size), 300)
    mem_table_df["range"] = new_min_cluster_size_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        mem_line = []
        for min_cluster_size in tqdm(finding_min_cluster_size):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, random_state=0)
            tracemalloc.start()
            clusterer.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_line.append((peak - current) / 10 ** 6)

        mem_table.append(mem_line)

        gfg = make_interp_spline(finding_min_cluster_size, mem_line, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_min_cluster_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in real_k], columns=finding_min_cluster_size),
        ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "hdbscan_memK.png"))
    plt.show()


def evaluation_mem_of_working_by_size(
        real_k: int = 7,
        finding_min_cluster_size: list = [50, 100, 150, 200, 300],
        total_size: list = list(range(500, 7000, 500)),
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_size_range = np.linspace(min(total_size), max(total_size), 300)
    mem_table_df["range"] = new_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        mem_line = []
        for min_cluster_size in tqdm(finding_min_cluster_size):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, random_state=0)

            tracemalloc.start()
            clusterer.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_line.append((peak - current) / 10 ** 6)
        mem_table.append(mem_line)
    mem_table = np.array(mem_table).T

    for row_by_k, min_cluster_size in zip(mem_table, finding_min_cluster_size):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        mem_table_df["min_cluster_size ={}".format(min_cluster_size)] = gfg(new_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["min_cluster_size={}".format(K) for K in finding_min_cluster_size],
                             columns=total_size),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "hdbscan_mem_size.png"))
    plt.show()


def check_init_dependency(
        size=300,
        path="data",
        iter=100):
    from datasets import test_gaussian_data

    REAL_K = [2, 3, 5, 8, 10, 15, 30]
    scores_table = []
    for K in REAL_K:
        X, y = test_gaussian_data(size, K)
        scores = []
        for i in tqdm(range(iter)):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=size / K / 2, random_state=0)
            scores.append(rand_score(clusterer.labels_, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "hdbscan_init_dependency.png"))
    plt.show()

    path = os.path.join(path, "images", "init_dependency.pkl")
    update_table({
        "hdbscan_2": scores_table[0],
        "hdbscan_5": scores_table[2],
    }, path)


def check_stability(
        cluster_size=30,
        real_k=[2, 3, 5, 8, 10, 14],
        path="data",
        iter=100):
    from datasets import test_gaussian_data_v2

    scores_table = []
    for K in real_k:

        total_size = cluster_size * K
        delta_size = int(total_size * 0.1)
        scores = []

        for i in tqdm(range(iter)):
            X, y = test_gaussian_data_v2(total_size, K, random_state=i)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, random_state=0)

            origin_score = rand_score(clusterer.labels_, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.2 / K
            idx = np.random.choice(total_size, delta_size, replace=False)

            X[idx] = X[idx] + delta

            clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, random_state=0)

            delta_score = rand_score(clusterer.labels_, y)

            scores.append(abs(delta_score - origin_score))

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in real_k])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "hdbscan_stability.png"))
    plt.show()

    path = os.path.join(path, "images", "stability.pkl")
    update_table({
        "hdbscan_2": scores_table[0],
        "hdbscan_5": scores_table[2],
    }, path)


if __name__ == '__main__':
    K = [3, 3, 3, 4, 15, 31, 3, 4, 15]

    # run_KMedoids_on_data_with_K([get_data_blobs, get_data_blobs2, get_data_blobs3], K)

    # evaluation_time_of_working_by_k()
    # evaluation_time_of_working_by_size()
    # evaluation_mem_of_working_by_size()
    # evaluation_mem_of_working_by_k()
    # check_init_dependency()
    # check_stability("pam")

    # from datasets import get_gaussian_data_3d
    #
    # X, y = get_gaussian_data_3d()
    # run_on_3d_data(X, y, 3, "Gaussian")
