import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    color_generation(31)
