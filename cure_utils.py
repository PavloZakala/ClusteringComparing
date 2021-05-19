from pyclustering.cluster.cure import cure
import time
import seaborn as sns
from datasets import GAUSSIAN_BLOBS_DATA
from matplotlib import pyplot as plt
from tools import make_plot
from sklearn.metrics.cluster import rand_score

if __name__ == '__main__':
    K = [3, 3, 3, 4, 15, 31, 3, 4, 15]

    for data_gener, k in zip(GAUSSIAN_BLOBS_DATA, K):
        X, y = data_gener()
        clusterer = cure(X, k, 5, 0.5, True)
        start = time.time()
        clusterer.process();
        finish = time.time()

        list_of_clusters = clusterer.get_clusters()

        labels = np.zeros_like(y)
        for i, claster_indexes in enumerate(list_of_clusters):
            labels[claster_indexes] = i + 1

        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        make_plot(X, y, axes=axes[0], title="Target")
        make_plot(X, labels, axes=axes[1], title="Prediction HDBSCAN")
        plt.show()

        print("[CURE]: accuracy {:.4f}, time {:.3f} ".format(rand_score(labels, y), finish - start))