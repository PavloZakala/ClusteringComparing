import os
import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl


def color_generation(N):
    colors = ["#000000"]
    random.seed(0)
    for i in range(N):
        colors.append("#{:06x}".format(random.randint(0, 16777215)))
    return colors


def make_plot(X, y, axes=None, title=""):
    df = pd.DataFrame(data=X)
    df["cluster"] = y

    sns.scatterplot(data=df, x=0, y=1, hue="cluster", ax=axes, palette="deep", legend=False)
    if axes:
        axes.set_title(title)
    else:
        plt.title(title)

def make_3d_plot(X, y):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)

    cmap = ListedColormap(sns.color_palette().as_hex())

    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, c=y, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def update_table(new_data:dict, file_path:str):
    data = {}

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pkl.load(f)

    data.update(new_data)

    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

if __name__ == '__main__':
    color_generation(31)
