#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import wx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flim.plugin import AbstractPlugin
from flim.gui.dialogs import BasicAnalysisConfigDlg
import wx
from wx.lib.masked import NumCtrl
from importlib_resources import files
import flim.resources
from flim.plugin import plugin


class PCAnalysisConfigDlg(BasicAnalysisConfigDlg):
    def __init__(
        self,
        parent,
        title,
        input=None,
        description=None,
        selectedgrouping=["None"],
        selectedfeatures="All",
        keeporig=False,
        keepstd=True,
        explainedhisto=False,
        n_components=None,
        autosave=True,
        working_dir="",
    ):
        self.keeporig = keeporig
        self.keepstd = keepstd
        self.explainedhisto = explainedhisto
        self.n_components = n_components
        super().__init__(
            parent,
            title,
            input=input,
            description=description,
            selectedgrouping=selectedgrouping,
            selectedfeatures=selectedfeatures,
            optgridrows=2,
            optgridcols=1,
            autosave=autosave,
            working_dir=working_dir,
        )

    def get_option_panels(self):
        data = list(self.input.values())[0]
        allfeatures = data.select_dtypes(
            include=["number"], exclude=["category"]
        ).columns.values
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.n_components_input = NumCtrl(
            self.panel,
            wx.ID_ANY,
            min=0.0,
            max=float(len(allfeatures)),
            value=self.n_components,
            fractionWidth=3,
        )
        sizer.Add(
            wx.StaticText(self.panel, label="N-components"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        sizer.Add(
            self.n_components_input, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        self.keeporig_cb = wx.CheckBox(
            self.panel, id=wx.ID_ANY, label="Include original data"
        )
        self.keeporig_cb.SetValue(self.keeporig)
        sizer.Add(self.keeporig_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.keepstd_cb = wx.CheckBox(
            self.panel, id=wx.ID_ANY, label="Include standardized data"
        )
        self.keepstd_cb.SetValue(self.keepstd)
        sizer.Add(self.keepstd_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.explainedhisto_cb = wx.CheckBox(
            self.panel, id=wx.ID_ANY, label="Explained histogram"
        )
        self.explainedhisto_cb.SetValue(self.explainedhisto)
        sizer.Add(
            self.explainedhisto_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        return [sizer]

    def _get_selected(self):
        n_comps = self.n_components_input.GetValue()
        if n_comps >= 1.0:
            # convert float to int
            n_comps = int(n_comps)
        params = super()._get_selected()
        params["keeporig"] = self.keeporig_cb.GetValue()
        params["keepstd"] = self.keepstd_cb.GetValue()
        params["n_components"] = n_comps
        params["explainedhisto"] = self.explainedhisto_cb.GetValue()

        return params


@plugin(plugintype="Analysis")
class PCAnalysis(AbstractPlugin):
    def __init__(self, name="PCA", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_description(self):
        return (
            "Specifiy PCA components to retain:\n\tleave empty:   retain all PCA"
            " components.\n\t0.0 < n < 1.0 (float):   retain PCA components that"
            " explain specified fraction of observed variance.\n\t1 <= n <= # features"
            " (integer):   retain first n PCA components."
        )

    def get_required_categories(self):
        return []

    def get_icon(self):
        source = files(flim.resources).joinpath("pca.png")
        return wx.Bitmap(str(source))

    def get_required_features(self):
        return ["any", "any"]

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "keeporig": False,
                "keepstd": True,
                "explainedhisto": False,
                "n_components": 0.999,
            }
        )
        return params

    def run_configuration_dialog(self, parent, data_choices={}):
        dlg = PCAnalysisConfigDlg(
            parent,
            f"Configuration: {self.name}",
            input=self.input,
            description=self.get_description(),
            selectedgrouping=self.params["grouping"],
            selectedfeatures=self.params["features"],
            keeporig=self.params["keeporig"],
            keepstd=self.params["keepstd"],
            explainedhisto=self.params["explainedhisto"],
            n_components=self.params["n_components"],
            autosave=self.params["autosave"],
            working_dir=self.params["working_dir"],
        )
        if dlg.ShowModal() == wx.ID_CANCEL:
            dlg.Destroy()
            return  # implicit None
        params = dlg.get_selected()
        self.configure(**params)
        return self.params

    def output_definition(self):
        if self.params["explainedhisto"]:
            return {
                "Table: PCA Components": None,
                "Table: PCA explained": None,
                "Plot: PCA Explained": None,
            }
        else:
            return {"Table: PCA Components": None, "Table: PCA explained": None}

    def execute(self):
        data = list(self.input.values())[0]
        features = self.params["features"]
        data_no_na = data[features].dropna(how="any", axis=0).reset_index()
        if len(features) == 1:
            # reshape 1d array
            data_no_class = data_no_na.values.reshape((-1, 1))
        else:
            data_no_class = data_no_na[features].values
        scaler = StandardScaler()
        scaler.fit(data_no_class)
        standard_data = scaler.transform(data_no_class)

        pca_params = {k: self.params[k] for k in self.params if k in ["n_components"]}
        pca = PCA(**pca_params)
        principalComponents = pca.fit_transform(standard_data)

        pca_df = pd.DataFrame(
            data=principalComponents,
            columns=["PC %d" % x for x in range(1, principalComponents.shape[1] + 1)],
        )
        pca_eigen_df = pd.DataFrame(
            data=pca.components_,
            columns=features,
        )
        if self.params["keeporig"] and self.params["keepstd"]:
            standard_df = pd.DataFrame(
                data=standard_data, columns=["%s\nstandardized" % c for c in features]
            )
            pca_df = pd.concat(
                [
                    data.select_dtypes(include="category"),
                    data[features],
                    standard_df,
                    pca_df,
                ],
                axis=1,
            )  # .reset_index(drop=True)
        elif self.params["keeporig"]:
            pca_df = pd.concat(
                [data.select_dtypes(include="category"), data[features], pca_df], axis=1
            )  # .reset_index(drop=True)
        elif self.params["keepstd"]:
            standard_df = pd.DataFrame(
                data=standard_data, columns=["%s\nstandardized" % f for f in features]
            )
            pca_df = pd.concat(
                [data.select_dtypes(include="category"), standard_df, pca_df], axis=1
            )  # .reset_index(drop=True)
        else:
            pca_df = pd.concat(
                [data.select_dtypes(include="category"), pca_df], axis=1
            )  # .reset_index(drop=True)
        pca_comp_label = "PC"
        explained_label = "explained var ratio"
        pca_explained_df = pd.DataFrame(
            data={
                pca_comp_label: [
                    str(c) for c in range(1, len(pca.explained_variance_ratio_) + 1)
                ],
                explained_label: pca.explained_variance_ratio_,
            }
        )
        pca_explained_df[pca_comp_label] = pca_explained_df[pca_comp_label].astype(
            "category"
        )

        results = {
            "Table: PCA Components": pca_df,
            "Table: PCA Explained": pca_explained_df,
            "Table: PCA Eigen": pca_eigen_df,
        }
        if self.params["explainedhisto"]:
            plot = pca_explained_df.set_index(pca_comp_label).plot.bar()
            results["Plot: PCA Explained"] = plot.get_figure()
        return results
