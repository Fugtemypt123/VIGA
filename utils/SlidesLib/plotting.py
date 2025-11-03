# Call Matplotlib Library to draw graphs (Bar/Plot...)
import matplotlib.pyplot as plt
from llm import *
class Plotting:
    def bar_plot(self, data: dict, title: str, xlabel: str, ylabel: str, output_path: str = 'bar_plot.png'):
        """
        Create a bar plot.

        :param data: Dictionary containing data to plot (keys as labels, values as heights).
        :param title: Title of the plot.
        :param xlabel: Label for the X-axis.
        :param ylabel: Label for the Y-axis.
        :param output_path: Path to save the plot image.
        """
        labels = list(data.keys())
        heights = list(data.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, heights, color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def line_plot(self, data: dict, title: str, xlabel: str, ylabel: str, output_path: str = 'line_plot.png'):
        """
        Create a line plot.

        :param data: Dictionary containing data to plot (keys as x-values, values as y-values).
        :param title: Title of the plot.
        :param xlabel: Label for the X-axis.
        :param ylabel: Label for the Y-axis.
        :param output_path: Path to save the plot image.
        """
        x_values = list(data.keys())
        y_values = list(data.values())

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path
    
    def get_plot(self, data):
        instruction = ""
        