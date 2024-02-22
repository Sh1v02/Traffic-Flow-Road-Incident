import os

from matplotlib import pyplot as plt


class PlotGraphs:
    def __init__(self):
        pass

    def plot_graph(self, x_values, y_values, name, save_dir=None, labels=None):
        labels = labels if labels else ["Steps", "Reward"]
        plt.figure(figsize=(8, 4))
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
        plt.plot(x_values, y_values, linestyle='-', color='b', linewidth=1.0)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + '/' + name + '.png')