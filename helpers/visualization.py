"""
Class for visualizing point cloud data
"""

import plotly.express as px
import pandas as pd
import numpy as np
from skimage import exposure

class Visualization():
    def __init__(self, df : pd.DataFrame):
        self.df = df
        self.scatter = []
    def get_3d_scatter(self) -> px.scatter_3d:
        """
        Gets the interactive 3D scatter plot which can be visualized with the show() method
        :return: plotly scatter plot
        """
        scatter = px.scatter_3d(self.df,
                                x = self.df['x'],
                                y = self.df['y'],
                                z = self.df['z'],
                                opacity = 0.5)
        #                        template = "plotly_dark")
        fig_traces = []
        for trace in range(len(scatter["data"])):
            fig_traces.append(scatter["data"][trace])

        for traces in fig_traces:
            scatter.append_trace(traces, row = 1, col = 1)

        scatter.update_traces(marker = dict(size = 0.5,
                                            color = 'cyan'),
                              selector = dict(mode = 'markers'))
        scatter.update_layout(height = 400, width = 700, template='plotly_dark')
        scatter.update_xaxes(showgrid=False)
        scatter.update_yaxes(showgrid=False)
        self.scatter = scatter
        return scatter

    def get_2d_scatter(self) -> px.scatter:
        """
        Gets the interactive 3D scatter plot which can be visualized with the show() method
        :return: plotly scatter plot
        """
        if 'z' in self.df.columns:
            raise ValueError(f"Please use a 2D dataset")
        else:
            scatter = px.scatter(self.df,
                                 x = self.df['x'],
                                 y = self.df['y'],
                                 opacity = 0.5)
            self.scatter = scatter
        return scatter

    def loc2img(self, pixel_size : int = 10):
        """
        Method taken from P. Kollmannsberger for reconstructing 2D localizations into an image
        :param x:
        :param y:
        :param pixel_size: size of the pixel for visualization
        :return: image
        """
        x = self.df['x']
        y = self.df['y']
        xbins = np.arange(x.min(), x.max()+1, pixel_size)
        ybins = np.arange(y.min(), y.max()+1, pixel_size)
        img, xe, ye = np.histogram2d(y, x, bins=(ybins, xbins))
        equalized = exposure.equalize_hist(img)
        return img, equalized
