import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets


def __get_data_by_file(file_name):
    X = []
    y = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            x1, x2, y1 = line.split()
            X.append([float(x1), float(x2)])
            y.append(int(y1))

    return np.array(X), np.array(y)


#  *********************** Gaussian blobs ***********************

def __gaussian_data(n_samples=1500, k=3, dim=2, cluster_std=None, locations=None, rotate=None, random_state=42):
    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    if cluster_std is None:
        cluster_std = np.random.rand(k, 2) * np.array([[1.0, 0.5]])
    if locations is None:
        locations = np.random.rand(k, 2) * 2.0 - 1.0
    if rotate is None:
        rotate = np.random.rand(k) * np.pi / 2.0

    X, y = [], []
    for i, (size, std, loc, alpha) in enumerate(zip(cluster_sizes, cluster_std, locations, rotate)):
        transformation = [[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]

        Xl = np.random.normal(size=(size, dim), scale=std) @ transformation + loc
        yl = np.ones((size,), dtype=np.int32) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


def get_data_blobs(n_samples=1500, **kwargs):
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=8)
    return X, y + 1


def get_data_blobs2(n_samples=1500, **kwargs):
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=360, cluster_std=1.8)
    return X, y + 1


def get_data_blobs3(n_samples=1500, **kwargs):
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = X @ transformation
    return X_aniso, y + 1


def get_data_blobs4(n_samples=1500, **kwargs):
    return __gaussian_data(n_samples, 4, cluster_std=[[1.0, 0.3], [1.0, 0.2], [.5, 0.3], [0.8, 0.25]],
                           locations=[[1.0, -1.2], [-1.0, 1.2], [-1.3, -0.5], [1.0, 2.2]],
                           rotate=[np.pi / 4, -np.pi / 4, 0, -np.pi / 6])


def get_data_d31(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "D31.txt"))


def get_data_R15(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "R15.txt"))


def get_data_blobs2_noise(n_samples=1500, **kwargs):
    noise_size = int(n_samples * 0.03)

    X, y = get_data_blobs2(n_samples - noise_size, **kwargs)
    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


def get_data_blobs4_noise(n_samples=1500, **kwargs):
    noise_size = int(n_samples * 0.03)

    X, y = get_data_blobs4(n_samples - noise_size, **kwargs)
    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


def get_data_R15_noise(**kwargs):
    X, y = get_data_R15(**kwargs)
    noise_size = int(len(X) * 0.03)

    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


def get_gaussian_data_3d(n_samples=1500, random_state=14, **kwargs):
    k = 3
    dim = 3

    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    cluster_std = [[1.0, 0.3, 0.2], [1.0, 0.2, 0.3], [.5, 0.3, 0.5]]
    locations = [[1.0, -1.2, -1.], [-1.0, 1.2, 1], [-1.3, -1.5, 1]]
    rotate = [[np.pi / 4, np.pi / 3], [2 * np.pi / 3, np.pi / 6], [-np.pi / 3, 5 * np.pi / 6]]

    if cluster_std is None:
        cluster_std = np.random.rand(k, 2) * np.array([[1.0, 0.5]])
    if locations is None:
        locations = np.random.rand(k, 2) * 2.0 - 1.0
    if rotate is None:
        rotate = np.random.rand(k) * np.pi / 2.0

    X, y = [], []
    for i, (size, std, loc, alpha) in enumerate(zip(cluster_sizes, cluster_std, locations, rotate)):
        transformation = np.array([[np.cos(alpha[0]), np.sin(alpha[0]), 0.0],
                                   [-np.sin(alpha[0]), np.cos(alpha[0]), 0.0],
                                   [0.0, 0.0, 1.0]])

        transformation = transformation @ np.array([[np.cos(alpha[1]), 0.0, np.sin(alpha[1])],
                                                    [0.0, 1.0, 0.0],
                                                    [-np.sin(alpha[1]), 0.0, np.cos(alpha[1])]])

        Xl = np.random.normal(size=(size, dim), scale=std) @ transformation + loc
        yl = np.ones((size,)) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


GAUSSIAN_BLOBS_DATA = [get_data_blobs, get_data_blobs2, get_data_blobs3,
                       get_data_blobs4, get_data_R15, get_data_d31,
                       get_data_blobs2_noise, get_data_blobs4_noise, get_data_R15_noise]


# *********************** Unbalance gaussian blobs ***********************


def __unbalance_data(n_samples=1500, k=3, cluster_coef=0.65, cluster_std=1.8, random_state=42):
    cluster_sizes = []
    for i in range(k - 1):
        local = int(n_samples * (1 - cluster_coef))
        cluster_sizes.append(local)
        n_samples = n_samples - local

    cluster_sizes.append(n_samples)
    X, y = [], []
    if type(cluster_std) == list:
        for i, (size, std) in enumerate(zip(cluster_sizes, cluster_std)):
            Xl, yl = datasets.make_blobs(n_samples=size, centers=1,
                                         random_state=i + random_state, cluster_std=std)
            X += Xl.tolist()
            y += (yl + i + 1).tolist()
    else:
        for i, size in enumerate(cluster_sizes):
            Xl, yl = datasets.make_blobs(n_samples=size, centers=1,
                                         random_state=i + random_state, cluster_std=cluster_std)
            X += Xl.tolist()
            y += (yl + i + 1).tolist()

    return np.array(X), np.array(y)


def get_data_unbalance1(**kwargs):
    return __unbalance_data(k=3, cluster_coef=0.65, cluster_std=1.0, random_state=234)


def get_data_unbalance2(**kwargs):
    return __unbalance_data(k=5, cluster_coef=0.65, cluster_std=1.3, random_state=34)


def get_data_unbalance3(**kwargs):
    return __unbalance_data(k=5, cluster_coef=0.45, cluster_std=1.3, random_state=34)


def get_data_unbalance4(**kwargs):
    return __unbalance_data(k=5, cluster_coef=0.55, cluster_std=[1.3, 0.85, 1.8, 1.0, 1.1], random_state=34)


def get_data_unbalance5(**kwargs):
    return __unbalance_data(k=10, cluster_coef=0.65, cluster_std=1.3, random_state=34)


def get_data_unbalance6(**kwargs):
    n_samples = 1500
    noise_size = int(n_samples * 0.03)
    n_samples = n_samples - noise_size

    X, y = __unbalance_data(n_samples=n_samples, k=5, cluster_coef=0.55, cluster_std=[1.3, 0.85, 1.8, 1.0, 1.1],
                            random_state=34)

    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


UNBALANCED_GAUSSIAN_BLOBS_DATA = [get_data_unbalance1, get_data_unbalance2, get_data_unbalance3,
                                  get_data_unbalance4, get_data_unbalance5, get_data_unbalance6]


# *********************** Cubes, Rectangles, Parallelepiped ***********************

def get_data_cube(n_samples=1500, **kwargs):
    return np.random.rand(n_samples, 2), np.ones((n_samples,), dtype=np.int32)


def get_data_cube2(n_samples=1500, random_state=22, **kwargs):
    k = 4

    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    X, y = [], []
    for i, size in enumerate(cluster_sizes):
        cx, cy = np.random.rand(2)

        loc = np.random.rand(2) * 1.5 - 0.75

        transformation = np.eye(2)
        transformation = transformation @ np.array([[cx, 0.0], [0.0, cy]])

        Xl = (np.random.rand(size, 2) - 0.5) @ transformation + loc
        yl = np.ones((size,)) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


def get_data_cube3(n_samples=1500, random_state=45, **kwargs):
    k = 4

    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    X, y = [], []
    for i, size in enumerate(cluster_sizes):
        cx, cy = np.random.rand(2)

        loc = np.random.rand(2) * 1.5 - 0.75

        transformation = np.eye(2)
        transformation = transformation @ np.array([[cx, 0.0], [0.0, cy]])
        transformation = transformation @ np.array([[1.0, 1.5], [0.0, 1.0]])

        Xl = (np.random.rand(size, 2) - 0.5) @ transformation + loc
        yl = np.ones((size,)) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


def get_data_cube4(n_samples=1500, random_state=42, **kwargs):
    k = 4

    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    X, y = [], []
    for i, size in enumerate(cluster_sizes):
        alpha = np.random.rand(1)[0] * np.pi / 2.0

        cx, cy = np.random.rand(2)

        loc = np.random.rand(2) * 1.5 - 0.75

        transformation = np.eye(2)
        transformation = transformation @ np.array([[cx, 0.0], [0.0, cy]])
        transformation = transformation @ np.array([[1.0, 1.5], [0.0, 1.0]])
        transformation = transformation @ np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])

        Xl = (np.random.rand(size, 2) - 0.5) @ transformation + loc
        yl = np.ones((size,)) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


def get_data_cube5(n_samples=1500, random_state=18, **kwargs):
    noise_size = int(n_samples * 0.03)
    n_samples = n_samples - noise_size

    X, y = get_data_cube2(n_samples, random_state)

    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


def get_data_cube6(n_samples=1500, random_state=73, **kwargs):
    noise_size = int(n_samples * 0.03)
    n_samples = n_samples - noise_size

    X, y = get_data_cube4(n_samples, random_state)

    min_v = X.min(0)
    max_v = X.max(0)

    noise = np.random.rand(noise_size, 2) * (max_v - min_v) + min_v
    y_noise = np.zeros((noise_size,), dtype=np.int32)

    return np.concatenate((noise, X)), np.concatenate((y_noise, y))


CUBES_RECT_PARALLEL_DATA = [get_data_cube, get_data_cube2, get_data_cube3,
                            get_data_cube4, get_data_cube5, get_data_cube6]


def get_data_cube_3d(n_samples=1500, random_state=123, **kwargs):
    k = 4

    cluster_sizes = [n_samples // k] * (k - 1)
    cluster_sizes.append(n_samples - sum(cluster_sizes))
    np.random.seed(random_state)

    X, y = [], []
    for i, size in enumerate(cluster_sizes):
        alpha, beta = np.random.rand(2) * np.pi / 2.0

        cx, cy, cz = np.random.rand(3)

        loc = np.random.rand(3) * 1.5 - 0.75

        transformation = np.eye(3)
        transformation = transformation @ np.array([[cx, 0.0, .0], [0.0, cy, .0], [0.0, .0, cz]])
        transformation = transformation @ np.array([[np.cos(alpha), np.sin(alpha), .0],
                                                    [-np.sin(alpha), np.cos(alpha), .0],
                                                    [.0, .0, 1.0]])
        transformation = transformation @ np.array([[np.cos(beta), 0.0, np.sin(beta)],
                                                    [0.0, 1.0, 0.0],
                                                    [-np.sin(beta), 0.0, np.cos(beta)]])

        Xl = (np.random.rand(size, 3) - 0.5) @ transformation + loc
        yl = np.ones((size,)) * (i + 1)

        X += Xl.tolist()
        y += yl.tolist()

    return np.array(X), np.array(y)


# *********************** Non-spherical ***********************

def get_data_circle(n_samples=1500, **kwargs):
    X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    return X, y + 1


def get_data_moons(n_samples=1500, **kwargs):
    X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    return X, y + 1


def get_data_jain(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "jain.txt"))


def get_data_pathbased(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "pathbased.txt"))


def get_data_spiral(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "spiral.txt"))


NON_SPHERICAL_DATA = [get_data_circle, get_data_moons, get_data_jain, get_data_pathbased, get_data_spiral]


# *********************** Other forms ***********************

def get_data_aggregation(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "Aggregation.txt"))


def get_data_compound(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "Compound.txt"))


def get_data_flame(**kwargs):
    return __get_data_by_file(os.path.join(kwargs.get("path", "data"), "flame.txt"))


OTHER_FORMS_DATA = [get_data_aggregation, get_data_compound, get_data_flame]

if __name__ == '__main__':
    import seaborn as sns
    import pandas as pd

    sns.set()


    def make_plot(X, y, axes=None, title=""):

        df = pd.DataFrame(data=X)
        df["cluster"] = y

        sns.scatterplot(data=df, x=0, y=1, hue="cluster", ax=axes, palette="deep", legend=False)
        if axes:
            axes.set_title(title)
        else:
            plt.title(title)

    for X, y in [get_data_unbalance5()]:
        # For 2d plots
        make_plot(X, y)

        # # For 3d plot, Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='o')

        # # For Nd plots PairGrid
        # df = pd.DataFrame(data=X)
        # df["cluster"] = y
        #
        # g = sns.PairGrid(df, hue="cluster")
        # g.map_diag(sns.histplot)
        # g.map_offdiag(sns.scatterplot)

        plt.show()
