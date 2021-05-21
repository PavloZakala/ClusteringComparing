import os.path
import time
import tracemalloc

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm

from tools import make_plot, make_3d_plot, update_table


def run_MeanShift_on_data_with_K(data_list, list_of_bandwidth: list = [], name="", path="data"):
    alg_time = {"meanshift_time": []}
    alg_mem = {"meanshift_mem": []}

    for i, (data_gener, bandwidth) in enumerate(zip(data_list, list_of_bandwidth)):
        X, y = data_gener(path=path)

        meanshift = MeanShift(bandwidth=bandwidth)

        tracemalloc.start()
        start = time.time()
        meanshift.fit(X)
        finish = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["meanshift_time"].append(finish - start)
        alg_mem["meanshift_mem"].append((peak - current) / 10 ** 6)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, meanshift.labels_, axes=axes[0], title="Prediction MeanShift")

        plt.savefig(os.path.join(path, "images", "meanshift_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("[MeanShift]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(meanshift.labels_, y),
            finish - start,
            alg_mem["meanshift_mem"][-1]
        ))

    path = os.path.join(path, "images", "data_{}.pkl".format(name.upper()))
    update_table({**alg_mem, **alg_time}, path)


def run_MeanShift_on_data(data_list, BandWidth_range, name="", path="data"):
    for i, data_getter in enumerate(data_list):
        X, y = data_getter(path=path)
        scores = []
        for bandwidth in BandWidth_range:
            meanshift = MeanShift(bandwidth=bandwidth)
            meanshift.fit(X)

            if max(meanshift.labels_) - min(meanshift.labels_) > 1 and \
               max(meanshift.labels_) < len(y) - 1:
                score = silhouette_score(X, meanshift.labels_, metric='euclidean')
                scores.append(score)
            else:
                scores.append(0.0)
        bandwidth = BandWidth_range[np.argmax(scores)]

        meanshift = MeanShift(bandwidth=bandwidth).fit(X)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        sns.lineplot(data=pd.DataFrame(data={"y": scores, "x": BandWidth_range}), x="x", y="y",
                     ax=axes[0])
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, meanshift.labels_, axes=axes[2], title="Prediction MeanShift")

        plt.savefig(os.path.join(path, "images", "meanshift_find_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("Best bandwidth = {}".format(bandwidth))
        print("[MeanShift]: accuracy {:.4f}".format(rand_score(meanshift.labels_, y)))


def run_on_3d_data(X, y, bandwidth, data_name, path="data"):
    meanshift = MeanShift(bandwidth=bandwidth)

    tracemalloc.start()
    start = time.time()
    meanshift.fit(X)
    finish = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Target
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "meanshift_{}_PairGrid_Target.png".format(data_name)))
    plt.show()

    make_3d_plot(X, y)
    plt.savefig(os.path.join(path, "images", "meanshift_{}_3d_Target.png".format(data_name)))
    plt.show()

    # Predict MeanShift
    print("[MeanShift]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(meanshift.labels_, y),
        finish - start,
        (peak - current) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = meanshift.labels_

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "meanshift_{}_PairGrid_Predict.png".format(data_name)))
    plt.show()

    make_3d_plot(X, meanshift.labels_)
    plt.savefig(os.path.join(path, "images", "meanshift_{}_3d_Predict.png".format(data_name)))
    plt.show()


def evaluation_time_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_bandwidth: list = [0.5, 1.0, 1.5, 2.0, 2.5],
        total_size: int = 2500,
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_bandwidth_range = np.linspace(min(finding_bandwidth), max(finding_bandwidth), 200)
    time_table_df["range"] = new_bandwidth_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        time_line = []
        for bandwidth in tqdm(finding_bandwidth):
            meanshift = MeanShift(bandwidth=bandwidth)
            start = time.time()
            meanshift.fit(X)
            finish = time.time()
            time_line.append(finish - start)

        time_table.append(time_line)
        gfg = make_interp_spline(finding_bandwidth, time_line, k=3)
        time_table_df["K={}".format(K)] = gfg(new_bandwidth_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=time_table, index=["K={}".format(K) for K in real_k], columns=finding_bandwidth),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "meanshift_timeK.png"))
    plt.show()


def evaluation_time_of_working_by_size(
        real_k: int = 7,
        finding_bandwidth: list = [0.5, 1.0, 1.5, 2.0, 2.5],
        total_size: list = list(range(200, 1000, 200)),
        path="data"):
    from datasets import test_gaussian_data

    time_table_df = pd.DataFrame()
    time_table = []

    new_size_range = np.linspace(min(total_size), max(total_size), 300)
    time_table_df["range"] = new_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        time_line = []
        for bandwidth in tqdm(finding_bandwidth):
            meanshift = MeanShift(bandwidth=bandwidth)
            start = time.time()
            meanshift.fit(X)
            finish = time.time()
            time_line.append(finish - start)

        time_table.append(time_line)
    time_table = np.array(time_table).T

    for row_by_k, bandwidth in zip(time_table, finding_bandwidth):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        time_table_df["bandwidth={}".format(bandwidth)] = gfg(new_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        pd.DataFrame(data=time_table, index=["bandwidth={}".format(K) for K in finding_bandwidth], columns=total_size),
        ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(time_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "meanshift_time_size.png"))
    plt.show()


def evaluation_mem_of_working_by_k(
        real_k: list = [2, 3, 5, 8, 10, 15, 30],
        finding_bandwidth: list = [0.5, 1.0, 1.5, 2.0, 2.5],
        total_size: int = 2500,
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_bandwidth_range = np.linspace(min(finding_bandwidth), max(finding_bandwidth), 300)
    mem_table_df["range"] = new_bandwidth_range

    for K in real_k:
        X, y = test_gaussian_data(total_size, K)
        mem_line = []
        for bandwidth in tqdm(finding_bandwidth):
            meanshift = MeanShift(bandwidth=bandwidth)

            tracemalloc.start()
            meanshift.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_line.append((peak - current) / 10 ** 6)

        mem_table.append(mem_line)

        gfg = make_interp_spline(finding_bandwidth, mem_line, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_bandwidth_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in real_k], columns=finding_bandwidth),
                ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "meanshift_memK.png"))
    plt.show()


def evaluation_mem_of_working_by_size(
        real_k: int = 7,
        finding_bandwidth: list = [0.5, 1.0, 1.5, 2.0, 2.5],
        total_size: list = list(range(200, 1000, 200)),
        path="data"):
    from datasets import test_gaussian_data

    mem_table_df = pd.DataFrame()
    mem_table = []

    new_size_range = np.linspace(min(total_size), max(total_size), 300)
    mem_table_df["range"] = new_size_range

    for size in total_size:
        X, y = test_gaussian_data(size, real_k)
        mem_line = []
        for bandwidth in tqdm(finding_bandwidth):
            meanshift = MeanShift(bandwidth=bandwidth)
            tracemalloc.start()
            meanshift.fit(X)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_line.append((peak - current) / 10 ** 6)

        mem_table.append(mem_line)
    mem_table = np.array(mem_table).T

    for row_by_k, bandwidth in zip(mem_table, finding_bandwidth):
        gfg = make_interp_spline(total_size, row_by_k, k=3)
        mem_table_df["bandwidth={}".format(bandwidth)] = gfg(new_size_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        pd.DataFrame(data=mem_table, index=["bandwidth={}".format(K) for K in finding_bandwidth], columns=total_size),
        ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "meanshift_mem_size.png"))
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
            meanshift = MeanShift(bandwidth=1.5)
            meanshift.fit(X)
            scores.append(rand_score(meanshift.labels_, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "meanshift_init_dependency.png"))
    plt.show()

    path = os.path.join(path, "images", "init_dependency.pkl")
    update_table({
        "meanshift_2": scores_table[0],
        "meanshift_5": scores_table[2],
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

            meanshift = MeanShift(bandwidth=1.5)
            meanshift.fit(X)
            origin_score = rand_score(meanshift.labels_, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.3 / K
            idx = np.random.choice(total_size, delta_size, replace=False)

            X[idx] = X[idx] + delta

            meanshift = MeanShift(bandwidth=1.5)
            meanshift.fit(X)
            delta_score = rand_score(meanshift.labels_, y)

            scores.append(abs(delta_score - origin_score))

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "meanshift_stability.png"))
    plt.show()

    path = os.path.join(path, "images", "stability.pkl")
    update_table({
        "meanshift_2": scores_table[0],
        "meanshift_5": scores_table[2],
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
        # run_MeanShift_on_data_with_K(GAUSSIAN_BLOBS_DATA, [2.0, 2.0, 1.1,
        #                                                    1.1, 1.0, 1.0,
        #                                                    2.0, 1.0, 1.0], "GAUSSIAN_BLOBS")

        BandWidth_range = list(np.arange(0.1, 1.4, 0.2)) + list(np.arange(2, 14, 1))

        run_MeanShift_on_data(OTHER_FORMS_DATA, BandWidth_range, "OTHER_FORMS")
