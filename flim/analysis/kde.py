#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import math
from flim.plugin import AbstractPlugin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import numpy as np
import pandas as pd
from flim.gui.dialogs import BasicAnalysisConfigDlg
import wx
from importlib_resources import files
import flim.resources
from prefect import Task
from flim.plugin import plugin, ALL_FEATURES


default_linestyles = ["-", "--", ":", "-."]


@plugin(plugintype="Plot")
class KDE(AbstractPlugin, Task):
    def __init__(self, name="KDE", **kwargs):
        # classifier=None, importancehisto=True, n_estimators=100, test_size=0.3, **kwargs):
        super().__init__(name=name, **kwargs)

    def get_required_categories(self):
        return []

    def get_icon(self):
        source = files(flim.resources).joinpath("kde.png")
        return wx.Bitmap(str(source))

    def get_required_features(self):
        return ["any"]

    def output_definition(self):
        data = list(self.input.values())[0]
        features = self.params["features"]
        if features == ALL_FEATURES and isinstance(data, pd.DataFrame):
            features = list(data.select_dtypes(np.number).columns.values)
        return_type = {"Table: KDE {header}": pd.DataFrame for feature in features}
        if self.params["single_plot"]:
            return_type["Plot: KDE"] = matplotlib.figure.Figure
        else:
            return_type.update(
                {f"Plot: KDE {feature}": matplotlib.figure.Figure for feature in features}
            )
        return return_type

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "single_plot": False,
                "col_wrap": 5,  # auto
                "return_plot": True,
            }
        )
        return params

    def get_mapped_parameters(self):
        if self.params["single_plot"]:
            return super().get_mapped_parameters()
        else:
            parallel_params = []
            for i, f in enumerate(self.params["features"]):
                pair_param = self.params.copy()
                pair_param["features"] = [f]
                parallel_params.append(pair_param)
            return parallel_params

    def run_configuration_dialog(self, parent, data_choices={}):
        selgrouping = self.params["grouping"]
        selfeatures = self.params["features"]
        dlg = BasicAnalysisConfigDlg(
            parent,
            f"Configuration: {self.name}",
            input=self.input,
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

    def execute(self):
        data = list(self.input.values())[0]
        results = {}
        features = self.params["features"]
        if features == ALL_FEATURES:
            features = list(data.select_dtypes(np.number).columns.values)
        if self.params["single_plot"] and len(features) > 1:
            n = len(features)
            if self.params["col_wrap"] == 0:
                # auto
                cols = math.ceil(math.sqrt(n))
            else:
                # fixed number of cols
                cols = min(self.params["col_wrap"], n)
            rows = math.ceil(float(n) / cols)
            fig, axs = plt.subplots(rows, cols)
            # remove unused axes objects in 2d grid
            for j in range(n, rows * cols):
                if axs.ndim == 2:
                    fig.delaxes(axs[j // cols][j % cols])
            self._add_picker(fig)
        for i, header in enumerate(sorted(features)):
            if self.params["single_plot"] and len(features) > 1:
                if axs.ndim == 2:
                    ax = axs[i // cols][i % cols]
                else:
                    ax = axs[i]
            else:
                fig, ax = plt.subplots()
                self._add_picker(fig)
            bins = 100
            cdata = data[header].replace([np.inf, -np.inf], np.nan).dropna()
            len(cdata)
            minx = cdata.min()  # hconfig[0]
            maxx = cdata.max()  # hconfig[1]
            logging.debug(f"Creating kde plot for {str(header)}, bins={str(bins)}")
            kde_data = self.grouped_kdeplot(
                data,
                header,
                groups=self.params["grouping"],
                clip=(minx, maxx),
                ax=ax,
                return_plot=self.params.get("return_plot"),
                show_legend=not self.params["single_plot"] or i == cols - 1,
            )  # bins=bins, hist=False,
            if not np.isinf([minx, maxx]).any() and not np.isnan([minx, maxx]).any():
                ax.set_xlim(minx, maxx)
            if self.params.get("return_plot") and not self.params["single_plot"]:
                fig.tight_layout()
                results[f"Plot: KDE {header}"] = fig
            #results[f"Table: KDE {header}"] = kde_data
        if self.params.get("return_plot") and self.params["single_plot"]:
            fig.tight_layout()
            results[f"Plot: KDE"] = fig
        return results

    def grouped_kdeplot(
        self,
        data,
        column,
        title=None,
        ax=None,
        show_legend=True,
        return_plot=True,
        groups=[],
        dropna=True,
        linestyles=None,
        pivot_level=1,
        **kwargs,
    ):
        if data is None or not column in data.columns.values:
            return None, None
        if groups is None:
            groups = []

        newkwargs = kwargs.copy()
        newkwargs["ax"] = ax
        newkwargs["warn_singular"] = False

        cols = [c for c in groups]
        cols.append(column)
        if dropna:
            data = data[cols].dropna(how="any", subset=[column])
        df = pd.DataFrame()
        if len(groups) > 0:
            gs = data.groupby(groups)
            styles = []
            if linestyles is None and len(groups) == 2:
                uniquevalues = [data[g].unique() for g in groups]
                if len(uniquevalues[0]) <= len(sns.color_palette()) and len(
                    uniquevalues[1]
                ) <= len(default_linestyles):
                    colors = [c for c in sns.color_palette()[: len(uniquevalues[0])]]
                    linestyles = [ls for ls in default_linestyles[: len(uniquevalues[1])]]
                    for c in colors:
                        for ls in linestyles:
                            styles.append({"color": c, "linestyle": ls})
            logging.debug(f"styles={styles}")
            index = 0
            labels = []
            for name, groupdata in gs:
                if len(groupdata[column]) > 0:
                    name_fixed = self._fix_label(name)
                    if len(styles) > index:
                        newkwargs["color"] = styles[index]["color"]
                        newkwargs["linestyle"] = styles[index]["linestyle"]
                    try:
                        kde = sns.kdeplot(groupdata[column], **newkwargs)
                        x, y = kde.get_lines()[-1].get_data()
                        df[name_fixed + "_x"] = x
                        df[name_fixed + "_y"] = y
                    except Exception as e:
                        logging.error(e)
                    labels.append(name_fixed)
                index += 1
            no_legendcols = len(groups) // 30 + 1
            if show_legend:
                ax.legend(
                    labels=labels,
                    loc="upper left",
                    title=", ".join(groups),
                    bbox_to_anchor=(1.0, 1.0),
                    fontsize="small",
                    ncol=no_legendcols,
                )
        else:
            try:
                kde = sns.kdeplot(data[column], **newkwargs)
                x, y = kde.get_lines()[-1].get_data()
                df["ungrouped_x"] = x
                df["ungrouped_y"] = y
            except Exception as e:
                logging.error(e)
        ax.autoscale(enable=True, axis="y")
        ax.set_ylim(0, None)
        # if title is None:
        #    title = column.replace("\n", " ")
        #    if len(groups) > 0:
        #        title = f"{title}"  # " grouped by {groups}"
        # if len(title) > 0:
        #    ax.set_title(title)

        return df
