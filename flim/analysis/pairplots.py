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
import pandas as pd
import seaborn as sns
from flim.plugin import AbstractPlugin
from flim.gui.dialogs import BasicAnalysisConfigDlg
import wx
import matplotlib
import matplotlib.pyplot as plt
import itertools
from importlib_resources import files
import flim.resources
from flim.plugin import plugin


class PairPlotConfigDlg(BasicAnalysisConfigDlg):
    def __init__(
        self,
        parent,
        title,
        input=None,
        selectedgrouping=["None"],
        selectedfeatures="All",
        x_vars=[],
        y_vars=[],
        diag_kind="KDE",
        corner=False,
        hist_bins=20,
        autosave=True,
        working_dir="",
    ):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.diag_kind = diag_kind
        self.corner = corner
        self.hist_bins = hist_bins
        super().__init__(
            parent,
            title,
            input=input,
            selectedgrouping=selectedgrouping,
            selectedfeatures=selectedfeatures,
            optgridrows=1,
            optgridcols=0,
            autosave=autosave,
            working_dir=working_dir,
        )

    def get_option_panels(self):
        data = list(self.input.values())[0]
        osizer = wx.BoxSizer(wx.HORIZONTAL)
        diag_opts = ["KDE", "Histogram"]
        sel_diag = self.diag_kind
        if sel_diag not in diag_opts:
            sel_diag = diag_opts[0]
        self.diag_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=sel_diag,
            choices=diag_opts,
        )
        self.corner_cb = wx.CheckBox(self.panel, id=wx.ID_ANY, label="Corner")

        osizer.Add(
            wx.StaticText(self.panel, label="Diagonal Plot"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        osizer.Add(
            self.diag_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )
        osizer.Add(self.corner_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        return [osizer]

    def _get_selected(self):
        params = super()._get_selected()
        params["x_vars"] = self.selectedfeatures  # self.x_vars
        params["y_vars"] = self.selectedfeatures  # self.y_vars
        params["diag_kind"] = self.diag_combobox.GetValue()
        params["corner"] = self.corner_cb.GetValue()
        params["hist_bins"] = self.hist_bins  # self.hist_bins_spinner.GetValue())
        return params


@plugin(plugintype="Plot")
class PairPlot(AbstractPlugin):
    def __init__(self, name="Pair Plot", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_icon(self):
        source = files(flim.resources).joinpath("pair-new-back3.png")
        return wx.Bitmap(str(source))

    def get_required_categories(self):
        return []

    def get_required_features(self):
        return ["any", "any"]

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "x_vars": [],  # "KDE"
                "y_vars": [],  # "KDE"
                "diag_kind": "Histogram",  # "KDE"
                "corner": False,
                "hist_bins": 20,
            }
        )
        return params

    def run_configuration_dialog(self, parent, data_choices={}):
        selgrouping = self.params["grouping"]
        selfeatures = self.params["features"]
        dlg = PairPlotConfigDlg(
            parent,
            "Pair Plot",
            input=self.input,
            selectedgrouping=selgrouping,
            selectedfeatures=selfeatures,
            autosave=self.params["autosave"],
            working_dir=self.params["working_dir"],
            x_vars=self.params["x_vars"],
            y_vars=self.params["y_vars"],
            diag_kind=self.params["diag_kind"],
            hist_bins=self.params["hist_bins"],
            corner=self.params["corner"],
        )
        if dlg.ShowModal() == wx.ID_OK:
            results = dlg.get_selected()
            self.params.update(results)
            return self.params
        else:
            return None

    def output_definition(self):
        return {f"Pair Plot: {self.params['gouping']}": matplotlib.figure.Figure}

    def execute(self):
        data = list(self.input.values())[0]
        features = self.params["features"]
        x_vars = self.params["x_vars"]
        if not x_vars or len(x_vars) == 0:
            x_vars = features
        y_vars = self.params["y_vars"]
        if not y_vars or len(y_vars) == 0:
            y_vars = features
        hue = self.params["grouping"][0] if len(self.params["grouping"]) > 0 else None
        corner = self.params["corner"]
        results = {}
        logging.debug(f"\tcreating pair plot for {features}, grouped by {hue}")
        # fig, ax = plt.subplots()
        fig = self.grouped_pairplot(
            data,
            x_vars=x_vars,
            y_vars=y_vars,
            hue=hue,
            corner=corner,
        )  # , facecolors='none', edgecolors='r')
        results[f"Pair Plot: {self.params['grouping']}"] = fig
        return results

    def grouped_pairplot(
        self,
        data,
        x_vars,
        y_vars,
        hue=None,
        corner=False,
        dropna=True,
    ):
        # g = sns.pairplot(data, x_vars=features, y_vars=features, hue=hue, dropna=dropna)
        g = sns.PairGrid(
            data,
            diag_sharey=False,
            x_vars=x_vars,
            y_vars=y_vars,
            hue=hue,
            dropna=dropna,
            corner=corner,
        )
        if self.params["diag_kind"] == "Histogram":
            g.map_diag(sns.histplot, bins=self.params["hist_bins"])
        else:
            g.map_diag(sns.kdeplot)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.scatterplot)  # sns.kdeplot
        g.add_legend()
        return g.fig
