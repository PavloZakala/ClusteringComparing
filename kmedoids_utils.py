import os.path
import time
import tracemalloc

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from tools import make_plot, make_3d_plot, update_table


def run_KMedoids_on_data_with_K(data_list, list_of_k, name="", path="data"):
    alg_time = {"kmedoids_pam_time": [], "kmedoids_alternate_time": []}
    alg_mem = {"kmedoids_pam_mem": [], "kmedoids_alternate_mem": []}

    for i, (data_gener, k) in enumerate(zip(data_list, list_of_k)):
        X, y = data_gener(path=path)
        kmedoids_alternate = KMedoids(n_clusters=k, random_state=0, method="alternate")

        tracemalloc.start()
        start_alternate = time.time()
        kmedoids_alternate.fit(X)
        finish_alternate = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["kmedoids_alternate_time"].append(finish_alternate - start_alternate)
        alg_mem["kmedoids_alternate_mem"].append((peak - current) / 10 ** 6)

        kmedoids_pam = KMedoids(n_clusters=k, random_state=0, method="pam")

        tracemalloc.start()
        start_pam = time.time()
        kmedoids_pam.fit(X)
        finish_pam = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        alg_time["kmedoids_pam_time"].append(finish_pam - start_pam)
        alg_mem["kmedoids_pam_mem"].append((peak - current) / 10 ** 6)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        make_plot(X, y, axes=axes[1], title="Target")
        make_plot(X, kmedoids_alternate.labels_, axes=axes[0], title="Prediction Alternate")
        make_plot(X, kmedoids_pam.labels_, axes=axes[2], title="Prediction PAM")

        plt.savefig(os.path.join(path, "images", "k-mediods_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("[Alternate]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(kmedoids_alternate.labels_, y),
            finish_alternate - start_alternate,
            alg_mem["kmedoids_pam_mem"][-1]
        ))
        print("[PAM]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
            rand_score(kmedoids_pam.labels_, y),
            finish_pam - start_pam,
            alg_mem["kmedoids_alternate_mem"][-1]
        ))

    path = os.path.join(path, "images", "data_{}.pkl".format(name.upper()))
    update_table({**alg_mem, **alg_time}, path)


def run_KMedoids_on_data(data_list, K_range, name="", path="data"):
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

        plt.savefig(os.path.join(path, "images", "k-mediods_findK_{}_{}.png".format(name.upper(), i)))
        plt.show()

        print("Best K = {}".format(np.argmax(scores) + 2))
        print("[Alternate]: accuracy {:.4f}".format(rand_score(kmedoids_alternate.labels_, y)))
        print("[PAM]: accuracy {:.4f}".format(rand_score(kmedoids_pam.labels_, y)))


def run_on_3d_data(X, y, K, data_name, path="data"):
    kmedoids_alternate = KMedoids(n_clusters=K, random_state=0, method="alternate")

    tracemalloc.start()
    start_alternate = time.time()
    kmedoids_alternate.fit(X)
    finish_alternate = time.time()
    current_alternate, peak_alternate = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    kmedoids_pam = KMedoids(n_clusters=K, random_state=0, method="pam")

    tracemalloc.start()
    start_pam = time.time()
    kmedoids_pam.fit(X)
    finish_pam = time.time()
    current_pam, peak_pam = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Target
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_PairGrid_target.png".format(data_name)))
    plt.show()

    make_3d_plot(X, y)
    plt.savefig(os.path.join(path, "images", "k-mediods_{}3d_target.png".format(data_name)))
    plt.show()

    # Predict Alternate
    print("[Predict Alternate]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(kmedoids_alternate.labels_, y),
        finish_alternate - start_alternate,
        (peak_alternate - current_alternate) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = kmedoids_alternate.labels_

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_PairGrid_Alternate.png".format(data_name)))
    plt.show()

    make_3d_plot(X, kmedoids_alternate.labels_)
    plt.savefig(os.path.join(path, "images", "k-mediods_{}_3d_Alternate.png".format(data_name)))
    plt.show()

    # Predict PAM
    print("[Predict PAM]: accuracy {:.4f}, time {:.3f}, mem {:.3f}Mb".format(
        rand_score(kmedoids_pam.labels_, y),
        finish_pam - start_pam,
        (peak_pam - current_pam) / 10 ** 6
    ))
    df = pd.DataFrame(data=X)
    df["cluster"] = kmedoids_pam.labels_

    g = sns.PairGrid(df, hue="cluster", palette="deep")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_PairGrid_PAM.png".format(data_name)))
    plt.show()

    make_3d_plot(X, kmedoids_pam.labels_)
    plt.savefig(os.path.join(path, "images", "k-mediods_{}_3d_PAM.png".format(data_name)))
    plt.show()

    path = os.path.join(path, "images", "data_{}.pkl".format(data_name.upper()))
    update_table({
        "kmedoids_pam_time": [finish_pam - start_pam],
        "kmedoids_alternate_time": [finish_alternate - start_alternate],
        "kmedoids_pam_mem": [(peak_pam - current_pam) / 10 ** 6],
        "kmedoids_alternate_mem": [(peak_alternate - current_alternate) / 10 ** 6]
    }, path)


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

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_timeK.png".format(method.upper())))
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

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_time_size.png".format(method.upper())))
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
            mem_line.append((peak - current) / 10 ** 6)

        mem_table.append(mem_line)

        gfg = make_interp_spline(finding_k, mem_line, k=3)
        mem_table_df["K={}".format(K)] = gfg(new_k_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pd.DataFrame(data=mem_table, index=["K={}".format(K) for K in real_k], columns=finding_k), ax=axes[0])

    sns.lineplot(x='range', y='value', hue='variable',
                 data=pd.melt(mem_table_df, ['range']), ax=axes[1])

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_memK.png".format(method.upper())))
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

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_mem_size.png".format(method.upper())))
    plt.show()


def check_init_dependency(
        size=1500,
        method="pam",
        path="data",
        iter=100):
    from datasets import test_gaussian_data

    REAL_K = [2, 3, 5, 8, 10, 15]
    scores_table = []
    for K in REAL_K:
        X, y = test_gaussian_data(size, K)
        scores = []
        for i in tqdm(range(iter)):
            kmedoids = KMedoids(n_clusters=K, random_state=i, init="build", method=method)
            kmedoids.fit(X)
            scores.append(rand_score(kmedoids.labels_, y))
        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in REAL_K])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_init_dependency.png".format(method.upper())))
    plt.show()

    path = os.path.join(path, "images", "init_dependency.pkl")
    update_table({
        "kmedoids_{}_2".format(method): scores_table[0],
        "kmedoids_{}_5".format(method): scores_table[2],
    }, path)


def check_stability(
        cluster_size=30,
        real_k=[2, 3, 5, 8, 10, 14],
        method="pam",
        path="data",
        iter=100
):
    from datasets import test_gaussian_data_v2

    scores_table = []
    for K in real_k:

        total_size = cluster_size * K
        delta_size = int(total_size * 0.1)
        scores = []

        for i in tqdm(range(iter)):
            X, y = test_gaussian_data_v2(total_size, K, random_state=i)

            kmedoids = KMedoids(n_clusters=K, random_state=0, init="build", method=method)
            kmedoids.fit(X)
            origin_score = rand_score(kmedoids.labels_, y)

            delta = (2 * np.random.rand(delta_size, 2) - 1.0) * 0.2 / K
            idx = np.random.choice(total_size, delta_size, replace=False)

            X[idx] = X[idx] + delta

            kmedoids = KMedoids(n_clusters=K, random_state=0, init="build", method=method)
            kmedoids.fit(X)
            delta_score = rand_score(kmedoids.labels_, y)

            scores.append(abs(delta_score - origin_score))

        scores_table.append(scores)

    data = pd.DataFrame(data=np.array(scores_table).T, columns=["K={}".format(K) for K in real_k])
    sns.boxplot(data=data, orient="h", palette="Set2")

    plt.savefig(os.path.join(path, "images", "k-mediods_{}_stability.png".format(method.upper())))
    plt.show()

    path = os.path.join(path, "images", "stability.pkl")
    update_table({
        "kmedoids_{}_2".format(method): scores_table[0],
        "kmedoids_{}_5".format(method): scores_table[2],
    }, path)


if __name__ == '__main__':
    # for data_list, k_range, data_name in [
    #     (GAUSSIAN_BLOBS_DATA, GAUSSIAN_BLOBS_K, "GAUSSIAN_BLOBS"),
    #     (UNBALANCED_GAUSSIAN_BLOBS_DATA, UNBALANCED_GAUSSIAN_BLOBS_K, "UNBALANCED_GAUSSIAN_BLOBS"),
    #     (CUBES_RECT_PARALLEL_DATA, CUBES_RECT_PARALLEL_K, "CUBES_RECT_PARALLEL"),
    #     (NON_SPHERICAL_DATA, NON_SPHERICAL_K, "NON_SPHERICAL"),
    #     (OTHER_FORMS_DATA, OTHER_FORMS_K, "OTHER_FORMS"),
    # ]:
    #     run_KMedoids_on_data_with_K(data_list, k_range, data_name)
    #     run_KMedoids_on_data(data_list, k_range, data_name)

    # X, y = get_gaussian_data_3d()
    # run_on_3d_data(X, y, K=3, data_name="GAUSSIAN_BLOBS")
    #
    # X, y = get_data_cube_3d()
    # run_on_3d_data(X, y, K=4, data_name="UBES_RECT_PARALLEL")

    # # Alternate
    # evaluation_time_of_working_by_k()
    # evaluation_time_of_working_by_size()
    # evaluation_mem_of_working_by_k()
    # evaluation_mem_of_working_by_size()
    #
    # # PAM
    # evaluation_time_of_working_by_k(total_size=1500, method="pam")
    # evaluation_time_of_working_by_size()
    # evaluation_mem_of_working_by_k()
    # evaluation_mem_of_working_by_size()

    # # Alternate
    # check_init_dependency(method="alternate")
    # check_stability(method="alternate")
    #
    # PAM
    check_init_dependency(method="pam")
    check_stability(method="pam")
