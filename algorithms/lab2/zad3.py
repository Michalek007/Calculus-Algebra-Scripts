from typing import List
from matplotlib import pyplot as plt


class Graph:
    def __init__(self, mode: bool = False, dpi: int = 150, **default_values):
        self.mode = mode
        self.dpi = dpi
        self.default_values = default_values

    def create_graph(self, x, y, title: str, x_lim: list = None, y_lim: list = None, x_label: str = None,
                     y_label: str = None, label: list = None, continuous=True, values: List[list] = None):

        if label is None:
            label = self.default_values.get('label')
        if x_label is None:
            x_label = self.default_values.get('x_label')
        if y_label is None:
            y_label = self.default_values.get('y_label')
        if x_lim is None:
            x_lim = self.default_values.get('x_lim')
        if y_lim is None:
            y_lim = self.default_values.get('y_lim')

        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)
        plt.grid()
        plt.title(title)
        # plt.axis('off')
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if continuous:
            plt.plot(x, y, label=label[0] if label else None)
        else:
            plt.plot(x, y, '.', label=label[0] if label else None)
        if values:
            for i in range(len(values)):
                plt.plot(values[i][0], values[i][1], label=label[i + 1] if label else None)
            plt.legend(loc="upper right")
        if self.mode:
            plt.savefig(title + '.png', dpi=self.dpi)
            plt.clf()
        else:
            plt.show()


graph = Graph()
