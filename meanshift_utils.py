from sklearn.cluster import MeanShift
import time
import seaborn as sns
from datasets import GAUSSIAN_BLOBS_DATA
from matplotlib import pyplot as plt
from tools import make_plot

from sklearn.metrics.cluster import rand_score

if __name__ == '__main__':
    for data_gener in GAUSSIAN_BLOBS_DATA:
        X, y = data_gener()
        clusterer = MeanShift(bandwidth=2)
        start = time.time()
        clusterer.fit(X)
        finish = time.time()

        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        make_plot(X, y, axes=axes[0], title="Target")
        make_plot(X, clusterer.labels_, axes=axes[1], title="Prediction Mean Shift")
        plt.show()

        print("[Mean Shift]: accuracy {:.4f}, time {:.3f} ".format(rand_score(clusterer.labels_, y), finish - start))