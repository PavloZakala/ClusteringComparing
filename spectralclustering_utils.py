import os.path
import time
import tracemalloc

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm

from tools import make_plot, make_3d_plot, update_table


def run_SpectralClustering_on_data_with_K(data_list, list_of_k: list = [], name="", path="data"):
    alg_time = {"spectral_clustering_time": []}
    alg_mem = {"spectral_clustering_mem": []}

    for i, (data_gener, k) in enumerate(zip(data_list, list_of_k)):
        X, y = data_gener(path=path)

        clusterer = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=0)
        tracemalloc.start()
        start = time.time()
        clusterer.fit(X)
        finish = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["spectral_clustering_time"].append(finish - start)
        alg_mem["spectral_clustering_mem"].append((peak - current) / 10 ** 6)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        make_plot(X, y, axes=axes[0], title="Target")
        make_plot(X, clusterer.labels_, axes=axes[1], title="Prediction Spectral Clustering")

        plt.savefig(os.path.join(path, "images", "spectral_clustering_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("[Spectral Clustering]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(clusterer.labels_, y),
            finish - start,
            alg_mem["spectral_clustering_mem"][-1]
        ))

    path = os.path.join(path, "images", "data_{}.pkl".format(name.upper()))
    update_table({**alg_mem, **alg_time}, path)


def run_SpectralClustering_on_data(data_list, K_range, name="", path="data"):
    for i, data_getter in enumerate(data_list):
        X, y = data_getter(path=path)
        scores = []
        for K in K_range:
            clusterer = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=0)
            clusterer.fit(X)

            if max(clusterer.labels_) - min(clusterer.labels_) > 1:
                score = silhouette_score(X, clusterer.labels_, metric='euclidean')
                scores.append(score)
            else:
                scores.append(0.0)
        clusterer = SpectralClustering(n_clusters=np.argmax(scores) + 2, affinity="nearest_neighbors", random_state=0)
        clusterer.fit(X)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": K_range}), x="x", y="y",
                     ax=axes[0])
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, clusterer.labels_, axes=axes[2], title="Prediction Spectral Clustering")

        plt.savefig(os.path.join(path, "images", "spectral_clustering_findK_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("Best K = {}".format(np.argmax(scores) + 2))
        print("[SpectralClustering]: accuracy {:.4f}".format(rand_score(clusterer.labels_, y)))


def run_on_3d_data(X, y, K, data_name, path="data"):
    clusterer = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=0)
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

    plt.savefig(os.path.join(path, "images", "spectral_clustering_{}_PairGrid_Target.png".format(data_name)))
    plt.show()

    make_3d_plot(X, y)
    plt.savefig(os.path.join(path, "images", "spectral_clustering_{}_3d_Target.png".format(data_name)))
    plt.show()

    # Predict Alternate
    print("[Spectral Clustering]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(clusterer.labels_, y),
        finish - start,
        (peak - current) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = clusterer.labels_

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "spectral_clustering_{}_PairGrid_Predict.png".format(data_name)))
    plt.show()

    make_3d_plot(X, clusterer.labels_)
    plt.savefig(os.path.join(path, "images", "spectral_clustering_{}_3d_Predict.png".format(data_name)))
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

    for K in tqdm(real_k):
        X, y = test_gaussian_data(total_size, K)
        time_line = []
        for K_for_find in finding_k:
            clusterer = SpectralClustering(n_clusters=K_for_find, affinity="nearest_neighbors", random_state=0)
            start = time.time()
            clusterer.fit(X)
            finish = time.time()
            time_line.append(finish - start)

        time_table.append(time_line)
        gfg = make_interp_spline(finding_k, time_line, k=3)
        time_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "spectral_clustering_timeK.png"))
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

    for size in tqdm(total_size):
        X, y = test_gaussian_data(size, real_k)
        time_line = []
        for K_for_find in finding_k:
            clusterer = SpectralClustering(n_clusters=K_for_find, affinity="nearest_neighbors", random_state=0)
            start_alternate = time.time()
            clusterer.fit(X)
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

    plt.savefig(os.path.join(path, "images", "spectral_clustering_time_size.png"))
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

    for K in tqdm(real_k):
        X, y = test_gaussian_data(total_size, K)
        mem_line = []
        for K_for_find in finding_k:
            clusterer = SpectralClustering(n_clusters=K_for_find, affinity="nearest_neighbors", random_state=0)

            tracemalloc.start()
            clusterer.fit(X)
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

    plt.savefig(os.path.join(path, "images", "spectral_clustering_memK.png"))
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

    for size in tqdm(total_size):
        X, y = test_gaussian_data(size, real_k)
        mem_line = []
        for K_for_find in finding_k:
            clusterer = SpectralClustering(n_clusters=K_for_find, affinity="nearest_neighbors", random_state=0)

            tracemalloc.start()
            clusterer.fit(X)
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

    plt.savefig(os.path.join(path, "images", "spectral_clustering_mem_size.png"))
    plt.show()


def check_init_dependency(
        size=300,
        path="data",
        iter=100):
    from datasets import test_gaussian_data

    REAL_K = [2, 3, 5, 8, 10, 15]
    scores_table = []
    for K in tqdm(REAL_K):
        X, y = test_gaussian_data(size, K)
        scores = []
        for i in range(iter):
            clusterer = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=i)
            scores.append(rand_score(clusterer.labels_, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "spectral_clustering_init_dependency.png"))
    plt.show()

    path = os.path.join(path, "images", "init_dependency.pkl")
    update_table({
        "spectral_clustering_2": scores_table[0],
        "spectral_clustering_5": scores_table[2],
    }, path)


def check_stability(
        cluster_size=30,
        real_k=[2, 3, 5, 8, 10, 14],
        path="data",
        iter=100):
    from datasets import test_gaussian_data_v2

    scores_table = []
    for K in tqdm(real_k):

        total_size = cluster_size * K
        delta_size = int(total_size * 0.1)
        scores = []

        for i in range(iter):
            X, y = test_gaussian_data_v2(total_size, K, random_state=i)

            clusterer = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=0)

            origin_score = rand_score(clusterer.labels_, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.3 / K
            idx = np.random.choice(total_size, delta_size, replace=False)

            X[idx] = X[idx] + delta

            clusterer = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=0)

            delta_score = rand_score(clusterer.labels_, y)

            scores.append(abs(delta_score - origin_score))

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in real_k])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "spectral_clustering_stability.png"))
    plt.show()

    path = os.path.join(path, "images", "stability.pkl")
    update_table({
        "spectral_clustering_2": scores_table[0],
        "spectral_clustering_5": scores_table[2],
    }, path)


if __name__ == '__main__':
    from datasets import GAUSSIAN_BLOBS_DATA, GAUSSIAN_BLOBS_K
    from datasets import UNBALANCED_GAUSSIAN_BLOBS_DATA, UNBALANCED_GAUSSIAN_BLOBS_K
    from datasets import CUBES_RECT_PARALLEL_DATA, CUBES_RECT_PARALLEL_K
    from datasets import NON_SPHERICAL_DATA, NON_SPHERICAL_K
    from datasets import OTHER_FORMS_DATA, OTHER_FORMS_K

    for data_list, k_range, data_name in [
        (GAUSSIAN_BLOBS_DATA, GAUSSIAN_BLOBS_K, "GAUSSIAN_BLOBS"),
        (UNBALANCED_GAUSSIAN_BLOBS_DATA, UNBALANCED_GAUSSIAN_BLOBS_K, "UNBALANCED_GAUSSIAN_BLOBS"),
        (CUBES_RECT_PARALLEL_DATA, CUBES_RECT_PARALLEL_K, "CUBES_RECT_PARALLEL"),
        (NON_SPHERICAL_DATA, NON_SPHERICAL_K, "NON_SPHERICAL"),
        (OTHER_FORMS_DATA, OTHER_FORMS_K, "OTHER_FORMS"),
    ]:
        run_SpectralClustering_on_data_with_K(NON_SPHERICAL_DATA, NON_SPHERICAL_K, "NON_SPHERICAL")

        # run_SpectralClustering_on_data(GAUSSIAN_BLOBS_DATA, list(range(2, 40)))
