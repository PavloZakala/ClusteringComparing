import os.path
import time
import tracemalloc

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyclustering.cluster.cure import cure
from scipy.interpolate import make_interp_spline
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm

from tools import make_plot, make_3d_plot, update_table


def convert_cluster_list_to_labels(list_of_clusters, y):
    labels = np.zeros_like(y)
    for i, claster_indexes in enumerate(list_of_clusters):
        labels[claster_indexes] = i + 1
    return labels


def run_CURE_on_data_with_K(data_list, list_of_k: list = [], name="", path="data"):
    alg_time = {"cure_time": []}
    alg_mem = {"cure_mem": []}

    for i, (data_gener, k) in enumerate(zip(data_list, list_of_k)):
        X, y = data_gener(path=path)

        tracemalloc.start()
        start = time.time()
        clusterer = cure(X, k, 5, 0.5, True)
        finish = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["cure_time"].append(finish - start)
        alg_mem["cure_mem"].append((peak - current) / 10 ** 6)

        labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, labels, axes=axes[0], title="Prediction CURE")

        plt.savefig(os.path.join(path, "images", "cure{}{}.png".format(name.upper(), i)))
        plt.show()

        print("[CURE]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(labels, y),
            finish - start,
            alg_mem["cure_mem"][-1]
        ))

    path = os.path.join(path, "images", "data_{}.pkl".format(name.upper()))
    update_table({**alg_mem, **alg_time}, path)


def run_CURE_on_data(data_list, K_range, name="", path="data"):
    for i, data_getter in enumerate(data_list):
        X, y = data_getter(path=path)
        scores = []
        for K in K_range:
            clusterer = cure(X, K, 5, 0.5, True)
            labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

            score = silhouette_score(X, labels, metric='euclidean')
            scores.append(score)

        clusterer = cure(X, np.argmax(scores) + 2, 5, 0.5, True)
        labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

        fig, axes = plt.subplots(1, 3, figsize=(9, 8))

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": K_range}), x="x", y="y",
                     ax=axes[0])
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, labels, axes=axes[2], title="Prediction CURE")

        plt.savefig(os.path.join(path, "images", "cure_findK_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("Best K = {}".format(np.argmax(scores) + 2))
        print("[CURE]: accuracy {:.4f}".format(rand_score(labels, y)))


def run_on_3d_data(X, y, K, data_name, path="data"):
    tracemalloc.start()
    start = time.time()
    clusterer = cure(X, K, 5, 0.5, True)
    finish = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Target
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

    plt.savefig(os.path.join(path, "images", "cure_{}_PairGrid_Target.png".format(data_name)))
    plt.show()

    make_3d_plot(X, y)
    plt.savefig(os.path.join(path, "images", "cure_{}_3d_Target.png".format(data_name)))
    plt.show()

    # Predict Alternate
    print("[CURE]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(labels, y),
        finish - start,
        (peak - current) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = labels

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "cure_{}_PairGrid_Predict.png".format(data_name)))
    plt.show()

    make_3d_plot(X, labels)
    plt.savefig(os.path.join(path, "images", "cure_{}_3d_Predict.png".format(data_name)))
    plt.show()


def evaluation_time_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_k: list = list(range(2, 30)),
        total_size: int = 2500,
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
            start = time.time()
            cure(X, K_for_find, 5, 0.5, True)
            finish = time.time()
            time_line.append(finish - start)

        time_table.append(time_line)
        gfg = make_interp_spline(finding_k, time_line, k=3)
        time_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "cure_timeK.png"))
    plt.show()


def evaluation_time_of_working_by_size(
        real_k: int = 7,
        finding_k: list = [2, 3, 5, 8, 10, 15, 30],
        total_size: list = list(range(500, 7000, 500)),
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
            start_alternate = time.time()
            cure(X, K_for_find, 5, 0.5, True)
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

    plt.savefig(os.path.join(path, "images", "cure_time_size.png"))
    plt.show()


def evaluation_mem_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_k: list = list(range(2, 30)),
        total_size: int = 2500,
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
            tracemalloc.start()
            cure(X, K_for_find, 5, 0.5, True)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_line.append((peak - current) / 10 ** 6)

        mem_table.append(mem_line)

        gfg = make_interp_spline(finding_k, mem_line, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "cure_memK.png"))
    plt.show()


def evaluation_mem_of_working_by_size(
        real_k: int = 7,
        finding_k: list = [2, 3, 5, 8, 10, 15, 30],
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

        for K_for_find in tqdm(finding_k):
            tracemalloc.start()
            cure(X, K_for_find, 5, 0.5, True)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_line.append((peak - current) / 10 ** 6)
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

    plt.savefig(os.path.join(path, "images", "cure_mem_size.png"))
    plt.show()


def check_init_dependency(
        size=300,
        path="data",
        iter=100):
    from datasets import test_gaussian_data

    REAL_K = [2, 3, 5, 8, 10, 15]
    scores_table = []
    for K in REAL_K:
        X, y = test_gaussian_data(size, K)
        scores = []
        for i in tqdm(range(iter)):
            clusterer = cure(X, K, 5, 0.5, True)
            labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)
            scores.append(rand_score(labels, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "cure_init_dependency.png"))
    plt.show()

    path = os.path.join(path, "images", "init_dependency.pkl")
    update_table({
        "cure_2": scores_table[0],
        "cure_5": scores_table[2],
    }, path)


def check_stability(cluster_size=30,
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

            clusterer = cure(X, K, 5, 0.5, True)
            labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

            origin_score = rand_score(labels, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.3 / K
            idx = np.random.choice(total_size, delta_size, replace=False)

            X[idx] = X[idx] + delta

            clusterer = cure(X, K, 5, 0.5, True)
            labels = convert_cluster_list_to_labels(clusterer.get_clusters(), y)

            delta_score = rand_score(labels, y)

            scores.append(abs(delta_score - origin_score))

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in real_k])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "cure_stability.png"))
    plt.show()

    path = os.path.join(path, "images", "stability.pkl")
    update_table({
        "cure_{}_2": scores_table[0],
        "cure_{}_5": scores_table[2],
    }, path)


if __name__ == '__main__':
    pass
