from typing import Callable, List, Union
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt

def generate_author_matrix(authors: pd.DataFrame, indexer: Callable[[int, int], int]) -> pd.DataFrame:
    author_ids: np.ndarray[int] = authors.index.values
    author_matrix: pd.DataFrame = pd.DataFrame(index=author_ids, columns=author_ids)
    
    for author_a in author_ids:
        for author_b in author_ids:
            author_matrix.loc[author_a, author_b] = int(indexer(author_a, author_b))
    
    return author_matrix

def save_author_matrix(authors: pd.DataFrame, indexer: Callable[[int, int], int], cmap: List[str], vmin: int = 0, vmax: int = 1, linecolor: str = 'whitesmoke', xlabel: str = "", ylabel: str = "", ticks: List[List[Union[float, str]]] = [], directory: str = ".", path: str = "results"):
    author_matrix: pd.DataFrame = generate_author_matrix(authors, indexer)
        
    axes: Axes = sns.heatmap(author_matrix,
        cmap = cmap, vmin = vmin, vmax = vmax,
        linewidths = 1, linecolor = linecolor, square = True)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    colorbar: Colorbar = axes.collections[-1].colorbar
    if ticks:
        colorbar.set_ticks(ticks[0])
        colorbar.set_ticklabels(ticks[1])
    plt.show()
    plt.savefig(f"{directory}/{path}.png", bbox_inches="tight")
    colorbar.remove()
