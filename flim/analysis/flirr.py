#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:11:28 2020

@author: khs3z
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import numpy as np
import pandas as pd
import seaborn as sns
from flim.analysis.barplots import grouped_meanbarplot
from flim.plugin import AbstractPlugin
from flim.gui.dialogs import BasicAnalysisConfigDlg
import wx
import matplotlib
import matplotlib.pyplot as plt
import itertools
from importlib_resources import files
import flim.resources
from flim.plugin import plugin


@plugin(plugintype="Plot")
class FLIRRPlot(AbstractPlugin):
    def __init__(self, name="FLIRR Plot", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_icon(self):
        source = files(flim.resources).joinpath("flirr_analysis1.png")
        return wx.Bitmap(str(source))

    def get_required_categories(self):
        return ["any"]

    def get_required_features(self):
        return []

    def run_configuration_dialog(self, parent, data_choices={}):
        selgrouping = self.params["grouping"]
        selfeatures = self.params["features"]
        dlg = BasicAnalysisConfigDlg(
            parent,
            "FLIRR Plot",
            input=self.input,
            enablefeatures=False,
            selectedgrouping=selgrouping,
            selectedfeatures=selfeatures,
            autosave=self.params["autosave"],
            working_dir=self.params["working_dir"],
        )
        if dlg.ShowModal() == wx.ID_OK:
            results = dlg.get_selected()
            self.params.update(results)
            return self.params
        else:
            return None

    def output_definition(self):
        return {f"FLIRR Plot: {self.params['gouping']}": matplotlib.figure.Figure}

    def execute(self):
        data = list(self.input.values())[0]
        grouping = self.params["grouping"]
        features = self.params["features"]
        results = {}
        logging.debug(f"\tcreating FLIRR plot, grouped by {grouping}")
        # fig, ax = plt.subplots()
        fig = self.flirrplot(
            data,
            grouping=grouping,
        )  # , facecolors='none', edgecolors='r')
        results[f"FLIRR Plot: {self.params['grouping']}"] = fig
        return results

    def flirrplot(
        self,
        data,
        grouping=[],
        dropna=True,
    ):
        all_features = [c for c in data.select_dtypes(include=np.number)]
        flirr_label = [c for c in all_features if "FLIRR" in c.upper()][0]
        scatter_features = [
            c
            for c in data.select_dtypes(include=np.number)
            if "FLIRR" not in c.upper()
            and "/" not in c
            and (
                ("FAD" in c and "a1" in c and "%" in c)
                or ("NAD" in c and "a2" in c and "%" in c)
            )
        ]

        fig, ax = plt.subplots()
        ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2, fig=fig)
        ax2 = plt.subplot2grid((3, 5), (0, 2), colspan=1, fig=fig)
        ax3 = plt.subplot2grid((3, 5), (0, 3), colspan=1, fig=fig)
        ax4 = plt.subplot2grid((3, 5), (0, 4), colspan=1, fig=fig)
        ax5 = plt.subplot2grid((3, 5), (1, 2), colspan=3, fig=fig)
        ax6 = plt.subplot2grid((3, 5), (2, 0), colspan=5, fig=fig)
        sns.scatterplot(
            ax=ax1,
            data=data,
            x=scatter_features[0],
            y=scatter_features[1],
            hue=grouping[0] if len(grouping) > 0 else None,
        )
        sns.kdeplot(
            ax=ax2,
            data=data,
            x=scatter_features[0],
            hue=grouping[0] if len(grouping) > 0 else None,
            legend=False,
        )
        sns.kdeplot(
            ax=ax3,
            data=data,
            x=scatter_features[1],
            hue=grouping[0] if len(grouping) > 0 else None,
            legend=False,
        )
        sns.kdeplot(
            ax=ax4,
            data=data,
            x=flirr_label,
            hue=grouping[0] if len(grouping) > 0 else None,
            legend=False,
        )
        sns.lineplot(
            ax=ax5,
            data=data,
            x=grouping[0],
            y=flirr_label,
            # ci=self.params["ci"],
            # err_style=self.params["err_style"],
            # markers=self.params["markers"],
            style=grouping[1] if len(grouping) > 1 else None,
            legend=True,
        )
        grouped_meanbarplot(
            data,
            [flirr_label],
            ax=ax6,
            title=None,
            bartype="single",
            categories=grouping,
            dropna=True,
            pivot_level=1,
            orientation="vertical",
            error_bar="+",
            error_type="sem",
            legend=False,
        )
        fig = ax1.get_figure()
        fig.tight_layout()
        return fig
