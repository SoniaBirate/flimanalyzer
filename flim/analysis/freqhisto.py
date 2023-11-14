#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import numpy as np
import pandas as pd
from flim.plugin import AbstractPlugin
from flim.gui.dialogs import BasicAnalysisConfigDlg
import wx
import matplotlib.pyplot as plt
from importlib_resources import files
import flim.resources
from flim.plugin import plugin


default_linestyles = ["-", "--", ":", "-."]


class FreqHistoConfigDlg(BasicAnalysisConfigDlg):
    def __init__(
        self,
        parent,
        title,
        input=None,
        selectedgrouping=["None"],
        selectedfeatures="All",
        bins=20,
        stacked=False,
        cumulative=False,
        histtype="step",
        datatable=False,
        featuresettings={},
        settingspecs={},
        autosave=True,
        working_dir="",
    ):
        self.bins = bins
        self.stacked = stacked
        self.cumulative = cumulative
        self.histtype = histtype
        self.datatable = datatable

        super().__init__(
            parent,
            title,
            input=input,
            selectedgrouping=selectedgrouping,
            selectedfeatures=selectedfeatures,
            optgridrows=0,
            optgridcols=1,
            enablefeatsettings=True,
            featuresettings=featuresettings,
            settingspecs=settingspecs,
            autosave=autosave,
            working_dir=working_dir,
        )

    def get_option_panels(self):
        optsizer = wx.BoxSizer(wx.HORIZONTAL)

        histtype_opts = ["bar", "barstacked", "step", "stepfilled"]
        sel_histtype = self.histtype
        if sel_histtype not in histtype_opts:
            sel_histtype = histtype_opts[0]
        self.histtype_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=sel_histtype,
            choices=histtype_opts,
        )
        optsizer.Add(
            wx.StaticText(self.panel, label="Type"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        optsizer.Add(
            self.histtype_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        self.stacked_cb = wx.CheckBox(self.panel, wx.ID_ANY, label="Stacked")
        self.stacked_cb.SetValue(self.stacked)
        optsizer.Add(
            self.stacked_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        self.cumulative_cb = wx.CheckBox(self.panel, wx.ID_ANY, label="Cumulative")
        self.cumulative_cb.SetValue(self.cumulative)
        optsizer.Add(
            self.cumulative_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        self.showdata_cb = wx.CheckBox(self.panel, wx.ID_ANY, label="Binned data table")
        self.showdata_cb.SetValue(self.datatable)
        optsizer.Add(
            self.showdata_cb, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        return [optsizer]

    def _get_selected(self):
        params = super()._get_selected()
        params["stacked"] = self.stacked_cb.GetValue()
        params["cumulative"] = self.cumulative_cb.GetValue()
        params["histtype"] = self.histtype_combobox.GetValue()
        params["datatable"] = self.showdata_cb.GetValue()
        return params


@plugin(plugintype="Plot")
class FreqHisto(AbstractPlugin):
    def __init__(self, name="Frequency Histogram", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_required_categories(self):
        return []

    def get_required_features(self):
        return ["any"]

    def get_icon(self):
        source = files(flim.resources).joinpath("histogram.png")
        return wx.Bitmap(str(source))

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params["bins"] = 30
        params["density"] = False
        params["cumulative"] = False
        params["histtype"] = "step"  # 'bar', 'barstacked', 'step', 'stepfilled'
        params["stacked"] = False
        params["datatable"] = False
        params["featuresettings"] = {
            "trp t1": [0, 8000, 81, ["Treatment"]],
            "trp t2": [0, 8000, 81, ["Treatment"]],
            "trp tm": [0, 4000, 81, ["Treatment"]],
            "trp a1[%]": [0, 100, 21, ["Treatment"]],
            "trp a2[%]": [0, 100, 21, ["Treatment"]],
            "trp a1[%]/a2[%]": [0, 2, 81, ["Treatment"]],
            "trp E%1": [0, 100, 21, ["Treatment"]],
            "trp E%2": [0, 100, 21, ["Treatment"]],
            "trp E%3": [0, 100, 21, ["Treatment"]],
            "trp r1": [0, 100, 21, ["Treatment"]],
            "trp r2": [0, 100, 21, ["Treatment"]],
            "trp r3": [0, 100, 21, ["Treatment"]],
            "trp chi": [0, 4.7, 81, ["Treatment"]],
            "trp photons": [0, 160, 81, ["Treatment"]],
            "NAD(P)H t1": [0, 800, 81, ["Treatment"]],
            "NAD(P)H t2": [0, 400, 81, ["Treatment"]],
            "NAD(P)H tm": [0, 2000, 81, ["Treatment"]],
            "NAD(P)H photons": [0, 2000, 81, ["Treatment"]],
            "NAD(P)H a2[%]": [0, 100, 51, ["Treatment"]],
            #                    'NAD(P)H %': [0,99,34,['Treatment']],
            "NADPH %": [0, 99, 34, ["Treatment"]],
            "NADH %": [0, 100, 51, ["Treatment"]],
            "NAD(P)H/NADH": [0, 3, 31, ["Treatment"]],
            "NAD(P)H chi": [0.7, 4.7, 81, ["Treatment"]],
            "FAD t1": [0, 4000, 81, ["Treatment"]],
            "FAD t2": [1000, 7000, 81, ["Treatment"]],
            "FAD tm": [0, 4000, 81, ["Treatment"]],
            "FAD a1[%]": [0, 100, 51, ["Treatment"]],
            "FAD a2[%]": [0, 100, 21, ["Treatment"]],
            "FAD a1[%]/a2[%]": [0, 16, 81, ["Treatment"]],
            "FAD chi": [0.7, 4.7, 81, ["Treatment"]],
            "FAD photons": [0, 800, 81, ["Treatment"]],
            "FLIRR": [0, 2.4, 81, ["Treatment"]],
            "NADPH a2/FAD a1": [0, 10, 101, ["Treatment"]],
        }
        return params

    def get_mapped_parameters(self):
        parallel_params = []
        for f in self.params["features"]:
            pair_param = self.params.copy()
            pair_param["features"] = [f]
            parallel_params.append(pair_param)
        return parallel_params

    def run_configuration_dialog(self, parent, data_choices={}):
        data = list(self.input.values())[0]
        selgrouping = self.params["grouping"]
        selfeatures = self.params["features"]
        histmax = data.iloc[:, 1:].max(axis=1).max()
        # defines how to get input for values
        binspecs = { #revised bins so that they can accept negative vals
            "bins": [wx.SpinCtrl, {"min": 1, "max": 500, "initial": 30}],
            "min": [
                wx.SpinCtrlDouble,
                {"min": -100, "max": histmax, "initial": 0, "inc": 0.1},
            ],
            "max": [
                wx.SpinCtrlDouble,
                {"min": -100, "max": histmax, "initial": histmax, "inc": 0.1},
            ],
        }
        dlg = FreqHistoConfigDlg(
            parent,
            f"Configuration: {self.name}",
            input=self.input,
            selectedgrouping=selgrouping,
            selectedfeatures=selfeatures,
            bins=self.params["bins"],
            stacked=self.params["stacked"],
            cumulative=self.params["cumulative"],
            histtype=self.params["histtype"],
            datatable=self.params["datatable"],
            featuresettings=self.params["featuresettings"],
            settingspecs=binspecs,
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
        density = self.params["density"]
        histtype = self.params["histtype"]
        cumulative = self.params["cumulative"]
        self.stacked = self.params["stacked"]
        self.datatable = self.params["datatable"]
        bins = 30
        for header in sorted(self.params["features"]):
            mrange = (data[header].min(), data[header].max())
            try:
                hconfig = self.params["featuresettings"][header]
                mrange = (hconfig["min"], hconfig["max"])
                bins = hconfig["bins"]
            except:
                logging.debug("\tmissing binning parameters, using defaults.")
                self.params["featuresettings"][header] = {
                    "min": mrange[0],
                    "max": mrange[1],
                    "bins": 30,
                }
            logging.debug(
                f"\tcreating frequency histogram plot for {header} with {bins} bins,"
                f" range {mrange}"
            )
            # categories = [col for col in self.flimanalyzer.get_importer().get_parser().get_regexpatterns()]
            #            fig, ax = MatplotlibFigure()
            # fig = plt.figure(FigureClass=MatplotlibFigure)
            # ax = fig.add_subplot(111)
            binvalues, binedges, groupnames, fig, ax = self.histogram(
                data,
                header,
                groups=self.params["grouping"],
                normalize=100,
                range=mrange,
                bins=bins,
                histtype=histtype,
                cumulative=cumulative,
                density=density,
                stacked=self.stacked,
                alpha=0.5,
            )

            results[f"Frequency Histo Plot: {header}"] = fig

            if self.datatable:
                df = pd.DataFrame()
                df["bin edge low"] = binedges[:-1]
                df["bin edge high"] = binedges[1:]
                if len(binvalues.shape) == 1:
                    df[groupnames[0]] = binvalues
                else:
                    for i in range(len(binvalues)):
                        df[groupnames[i]] = binvalues[i]
                df.reset_index()
                # bindata = core.plots.bindata(binvalues,binedges, groupnames)
                # bindata = bindata.reset_index()
                results[f"Frequency Histo Table: {header}"] = df
        return results

    def histogram(
        self,
        data,
        column,
        title=None,
        groups=[],
        normalize=None,
        titlesuffix=None,
        **kwargs,
    ):
        # plt.rcParams.update({'figure.autolayout': True})

        if data is None or not column in data.columns.values:
            return None, None
        fig, ax = plt.subplots(constrained_layout=True)
        if groups is None:
            groups = []

        newkwargs = kwargs.copy()
        # newkwargs.update({'range':(minx,maxx)})
        totalcounts = data[column].dropna(axis=0, how="all").count()
        pltdata = []
        weights = []
        groupnames = []
        if len(groups) == 0:
            groupnames.append("all")
            pltdata = data[column].values
            newkwargs.update({"label": "all"})
            if normalize is not None:
                weights = (
                    np.ones_like(data[column].values) / float(totalcounts) * normalize
                )
        else:
            groupeddata = data.groupby(groups, observed=True)
            newkwargs.update({"label": list(groupeddata.groups)})
            for name, group in groupeddata:
                if len(group[column]) > 0:
                    groupnames.append(name)
                    pltdata.append(group[column].values)
                    totalcount = len(data)
                    if not self.stacked:
                        totalcounts = group[column].count()
                    if normalize is not None:
                        weights.append(
                            np.ones_like(group[column].values)
                            / float(totalcounts)
                            * normalize
                        )
        if normalize is not None:
            slabel = ""
            if not self.stacked:
                slabel = "in each group "
            if normalize == 100:
                ax.set_ylabel(f"relative counts {slabel}[%]")
                groupnames = [f"{n} [rel %]" for n in groupnames]
            else:
                ax.set_ylabel("relative counts (norm. to %.1f)" % normalize)

            newkwargs.update(
                {
                    "weights": weights,
                    #'density':False,
                }
            )
        else:
            ax.set_ylabel("counts")
        #    if newkwargs[range] is not None:
        #        ax.set_xlim(newkwargs[range[0]],newkwargs[range[1]])
        ax.set_xlabel(column)

        if title is None:
            title = column.replace("\n", " ")  # .encode('utf-8')
            if len(groups) > 0:
                title = f"{title} grouped by {groups}"
        if len(title) > 0:
            ax.set_title(title)

        fig.set_size_inches(8, 8)

        binvalues, binedges, patches = ax.hist(pltdata, **newkwargs)
        if len(groups) > 0 and len(binvalues) > 1:
            h, labels = ax.get_legend_handles_labels()
            # labels = [l.encode('ascii','ignore').split(',')[1].strip(' \)') for l in labels]
            labels = [
                l.replace("'", "").replace("(", "").replace(")", "") for l in labels
            ]
            no_legendcols = len(binvalues) // 30 + 1
            # chartbox = ax.get_position()
            # ax.set_position([chartbox.x0, chartbox.y0, chartbox.width* (1-0.2 * no_legendcols), chartbox.height])
            ax.legend(
                labels=labels,
                loc="upper left",
                title=", ".join(groups),
                bbox_to_anchor=(1.0, 1.0),
                fontsize="small",
                ncol=no_legendcols,
            )

        # plt.rcParams.update({'figure.autolayout': False})
        self._add_picker(fig)

        return (
            np.array(binvalues),
            binedges,
            groupnames,
            fig,
            ax,
        )
