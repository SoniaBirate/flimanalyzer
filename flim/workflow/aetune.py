import logging
import inspect
import itertools
import numpy as np
import os
import pandas as pd
import prefect
import re
import wx
import matplotlib.pyplot as plt

from prefect import Task, Flow, Parameter, task, unmapped, flatten
from prefect.tasks.core.constants import Constant
from prefect.tasks.core.collections import List
from prefect.executors import DaskExecutor
from prefect.executors.base import Executor
from prefect.engine.results import LocalResult

from importlib_resources import files

import flim.analysis.ml.autoencoder as autoencoder
import flim.resources
from flim import utils
from flim.plugin import plugin, perm, product, select
from flim.plugin import AbstractPlugin, DataBucket, ALL_FEATURES
from flim.data.pivotdata import Pivot
from flim.data.unpivotdata import UnPivot
from flim.data.filterdata import Filter
from flim.data.concatdata import Concatenator
from flim.data.mergedata import Merger
from flim.analysis.aerun import RunAE
from flim.analysis.aetraining import AETraining, AETrainingConfigDlg
from flim.analysis.aesimulate import AESimulate
from flim.analysis.barplots import BarPlot
from flim.analysis.heatmap import Heatmap
from flim.analysis.kde import KDE
from flim.analysis.kmeans import KMeansClustering
from flim.analysis.ksstats import KSStats
from flim.analysis.lineplots import LinePlot
from flim.analysis.seriesanalyzer import SeriesAnalyzer
from flim.analysis.summarystats import SummaryStats
from flim.analysis.relativechange import RelativeChange
from flim.analysis.pca import PCAnalysis
from flim.analysis.scatterplots import ScatterPlot
from flim.analysis.barplots import BarPlot
from flim.analysis.lineplots import LinePlot
from flim.gui.dialogs import BasicAnalysisConfigDlg
from flim.workflow.basicflow import AbsWorkFlow


class AESimTuneConfigDlg(BasicAnalysisConfigDlg):
    def __init__(
        self,
        parent,
        title,
        input={},
        data_choices={},
        description=None,
        selectedgrouping=["None"],
        selectedfeatures="All",
        epoches=20,
        batch_size=200,
        learning_rate=1e-4,
        weight_decay=1e-7,
        timeseries="",
        model="",
        device="cpu",
        rescale=False,
        checkpoint_interval=20,
        autosave=True,
        working_dir=os.path.expanduser("~"),
    ):
        data = next(iter(input.values()))
        self.timeseries_opts = data.select_dtypes(include=["category"]).columns.values
        self.timeseries = timeseries
        self.epoches = epoches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.model_opts = [a for a in autoencoder.get_autoencoder_classes()]
        self.working_dir = working_dir
        self.device = device
        self.rescale = rescale
        self.checkpoint_interval = checkpoint_interval
        super().__init__(
            parent,
            title,
            input=input,
            data_choices=data_choices,
            description=description,
            selectedgrouping=selectedgrouping,
            selectedfeatures=selectedfeatures,
            optgridrows=0,
            optgridcols=1,
            autosave=autosave,
            working_dir=working_dir,
        )
        self._update_model_info(None)

    def get_option_panels(self):
        epoches_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.epoches_spinner = wx.SpinCtrl(
            self.panel, wx.ID_ANY, min=1, max=500, initial=self.epoches
        )
        epoches_sizer.Add(
            wx.StaticText(self.panel, label="Epoches"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        epoches_sizer.Add(
            self.epoches_spinner, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        batch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.batchsize_spinner = wx.SpinCtrl(self.panel,wx.ID_ANY,min=1,max=4096,initial=self.batch_size)
        self.batchsize_input = wx.TextCtrl(
            self.panel, wx.ID_ANY, value=utils.to_str_sequence(self.batch_size)
        )  # NumCtrl(self.panel,wx.ID_ANY, min=0.0, max=1.0, value=self.learning_rate, fractionWidth=10)
        batch_sizer.Add(
            wx.StaticText(self.panel, label="Batch Size"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        batch_sizer.Add(
            self.batchsize_input, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        checkpoint_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.checkpoint_spinner = wx.SpinCtrl(
            self.panel, wx.ID_ANY, min=1, max=500, initial=self.checkpoint_interval
        )
        checkpoint_sizer.Add(
            wx.StaticText(self.panel, label="Save interval"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        checkpoint_sizer.Add(
            self.checkpoint_spinner, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        spinner_sizer = wx.BoxSizer(wx.VERTICAL)
        spinner_sizer.Add(epoches_sizer)
        spinner_sizer.Add(batch_sizer)
        spinner_sizer.Add(checkpoint_sizer)

        learning_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.learning_input = wx.TextCtrl(
            self.panel, wx.ID_ANY, value=utils.to_str_sequence(self.learning_rate)
        )  # NumCtrl(self.panel,wx.ID_ANY, min=0.0, max=1.0, value=self.learning_rate, fractionWidth=10)
        learning_sizer.Add(
            wx.StaticText(self.panel, label="Learning Rate"),
            1,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        learning_sizer.Add(
            self.learning_input, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        weight_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.weight_input = wx.TextCtrl(
            self.panel, wx.ID_ANY, value=utils.to_str_sequence(self.weight_decay)
        )  # NumCtrl(self.panel,wx.ID_ANY, min=0.0, max=1.0, value=self.weight_decay, fractionWidth=10)
        weight_sizer.Add(
            wx.StaticText(self.panel, label="Weight Decay"),
            1,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        weight_sizer.Add(
            self.weight_input, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        float_sizer = wx.BoxSizer(wx.VERTICAL)
        float_sizer.Add(learning_sizer)
        float_sizer.Add(weight_sizer)

        device_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.device_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=self.device,
            choices=["cpu", "cuda"],
        )
        self.rescale_checkbox = wx.CheckBox(
            self.panel, wx.ID_ANY, label="Rescale decoded"
        )
        self.rescale_checkbox.SetValue(self.rescale)
        device_sizer.Add(
            wx.StaticText(self.panel, label="Device"),
            1,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        device_sizer.Add(
            self.device_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )

        rescale_sizer = wx.BoxSizer(wx.VERTICAL)
        rescale_sizer.Add(device_sizer)
        rescale_sizer.Add(self.rescale_checkbox)

        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(spinner_sizer)
        top_sizer.Add(float_sizer)
        top_sizer.Add(rescale_sizer)

        timeseries_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sel_timeseries = self.timeseries
        if sel_timeseries not in self.timeseries_opts:
            sel_timeseries = self.timeseries_opts[0]
        self.timeseries_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=sel_timeseries,
            choices=self.timeseries_opts,
        )
        timeseries_sizer.Add(
            wx.StaticText(self.panel, label="Time Series"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        timeseries_sizer.Add(
            self.timeseries_combobox,
            0,
            wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL,
            5,
        )

        sel_model = self.model
        if sel_model not in self.model_opts:
            sel_model = self.model_opts[0]
        self.model_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=sel_model,
            choices=self.model_opts,
        )
        self.model_combobox.Bind(wx.EVT_COMBOBOX, self._update_model_info)

        # self.workingdirtxt = wx.StaticText(self.panel, label=self.working_dir)
        # browsebutton = wx.Button(self.panel, wx.ID_ANY, "Choose...")
        # browsebutton.Bind(wx.EVT_BUTTON, self._on_browse)

        timeseries_sizer.Add(
            wx.StaticText(self.panel, label="Model Architecture"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        timeseries_sizer.Add(
            self.model_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )
        timeseries_sizer.Add(
            wx.StaticText(self.panel, label="Working Directory"),
            1,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        # timeseries_sizer.Add(self.workingdirtxt, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        # timeseries_sizer.Add(
        #    browsebutton, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        # )

        descr_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_descr = wx.TextCtrl(
            self.panel,
            value="None",
            size=(400, 100),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.SUNKEN_BORDER,
        )
        self.model_layers = wx.TextCtrl(
            self.panel,
            value="None",
            size=(400, 100),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.SUNKEN_BORDER,
        )
        descr_sizer.Add(self.model_descr, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        descr_sizer.Add(
            self.model_layers, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5
        )
        # descr_sizer.Add(wx.StaticText(self, label="Weight Decay"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        # descr_sizer.Add(self.weight_input, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5)

        all_sizer = wx.BoxSizer(wx.VERTICAL)
        all_sizer.Add(top_sizer)
        all_sizer.Add(timeseries_sizer)
        all_sizer.Add(descr_sizer)

        return [all_sizer]

    def _on_feature_selection(self, event):
        self._update_model_info(None)

    def OnSelectAll(self, event):
        super().OnSelectAll(event)
        self._update_model_info(None)

    def OnDeselectAll(self, event):
        super().OnDeselectAll(event)
        self._update_model_info(None)

    def _update_model_info(self, event):
        modelname = self.model_combobox.GetValue()
        # batch_size = self.batchsize_spinner.GetValue()
        aeclasses = autoencoder.get_autoencoder_classes()
        sel_features = [
            self.allfeatures[key] for key in self.cboxes if self.cboxes[key].GetValue()
        ]
        ae = autoencoder.create_instance(
            aeclasses[modelname], nb_param=len(sel_features)
        )
        if ae:
            descr = ae.get_description()
            layer_txt = "\n".join(
                str(ae).split("\n")[1:-1]
            )  # strip first and last line
        else:
            descr = "No input features or model defined"
            layer_txt = descr
        self.model_descr.Replace(0, self.model_descr.GetLastPosition(), descr)
        self.model_layers.Replace(0, self.model_layers.GetLastPosition(), layer_txt)

    def _on_browse(self, event):
        dirname = self.workingdirtxt.GetLabel()
        with wx.DirDialog(
            self,
            "Working Directory",
            dirname,
            wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        ) as dirDialog:
            # dirDialog.SetPath(dirname)
            # fileDialog.SetFilename(fname)
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return
            dirname = dirDialog.GetPath()
            self.workingdirtxt.SetLabel(dirname)

    def _get_selected(self):
        params = super()._get_selected()
        params["epoches"] = self.epoches_spinner.GetValue()
        params["batch_size"] = [
            int(i) for i in self.batchsize_input.GetValue().split(",")
        ]
        params["learning_rate"] = [
            float(i) for i in self.learning_input.GetValue().split(",")
        ]
        params["weight_decay"] = [
            float(i) for i in self.weight_input.GetValue().split(",")
        ]
        params["timeseries"] = self.timeseries_combobox.GetValue()
        params["working_dir"] = self.workingdirtxt.GetLabel()
        params["model"] = self.model_combobox.GetValue()
        params["device"] = self.device_combobox.GetValue()
        params["rescale"] = self.rescale_checkbox.GetValue()
        params["checkpoint_interval"] = self.checkpoint_spinner.GetValue()
        return params


@plugin(plugintype="Workflow")
class AEWorkflow(AbsWorkFlow):
    def __init__(self, name="FLIM Data Simulation Tuning", **kwargs):
        super().__init__(name=name, **kwargs)
        # self.executor = None #DaskExecutor(address="tcp://172.18.75.87:8786")

    def get_required_features(self):
        return ["any"]

    # def get_icon(self):
    #     source = files(flim.resources).joinpath("heatmap.png")
    #     return wx.Bitmap(str(source))
    
    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "grouping": ["Treatment", "FOV", "Cell"],
                "features": [
                    "FAD a1",
                    "FAD a2",
                    "FAD t1",
                    "FAD t2",
                    "FAD photons",
                    "NAD(P)H a1",
                    "NAD(P)H a2",
                    "NAD(P)H t1",
                    "NAD(P)H t2",
                    "NAD(P)H photons",
                ],
                "timeseries": "Treatment",
                "sets": 4,
                "epoches": 6,  # 20,
                "learning_rate": [
                    0.00001,
                    0.00002,
                ],  # , 0.00005, 0.0001, 0.00011, 0.00015, 0.0002],
                "weight_decay": [1e-7, 1e-8],
                "batch_size": [128, 64],
                "working_dir": os.path.join(
                    os.path.expanduser("~"), "FLIMAnalyzerResults"
                ),
                "model": "AE Simulator 1-6",
                "device": "cpu",
                "rescale": True,
                "checkpoint_interval": 2,  # 20
            }
        )
        return params

    def run_configuration_dialog(self, parent, data_choices={}):
        dlg = AESimTuneConfigDlg(
            parent,
            f"Configuration: {self.name}",
            input=self.input,
            description=self.get_description(),
            selectedgrouping=self.params["grouping"],
            selectedfeatures=self.params["features"],
            epoches=self.params["epoches"],
            batch_size=self.params["batch_size"],
            weight_decay=self.params["weight_decay"],
            learning_rate=self.params["learning_rate"],
            timeseries=self.params["timeseries"],
            model=self.params["model"],
            device=self.params["device"],
            rescale=self.params["rescale"],
            checkpoint_interval=self.params["checkpoint_interval"],
            autosave=self.params["autosave"],
            working_dir=self.params["working_dir"],
        )
        if dlg.ShowModal() == wx.ID_CANCEL:
            dlg.Destroy()
            return  # implicit None
        params = dlg.get_selected()
        self.configure(**params)
        return self.params

    def construct_flow(self, executor, result):
        checkpoint_interval = self.params["checkpoint_interval"]

        data = list(self.input.values())[0]
        sel_features = self.params["features"]
        all_features = [c for c in data.select_dtypes(include=np.number)]

        batch_sizes = [f"b{i}" for i in self.params["batch_size"]]
        rates = [f"l{i}" for i in self.params["learning_rate"]]
        decays = [f"d{i}" for i in self.params["weight_decay"]]
        combinations = [
            utils.clean(str(i))
            for i in list(itertools.product(batch_sizes, rates, decays))
        ]
        batch_sizes, rates, decays = utils.permutate(
            self.params["batch_size"],
            self.params["learning_rate"],
            self.params["weight_decay"],
        )

        add_noise = False

        # listtask = List()
        datatask = DataBucket(name="Input")
        filtertask = Filter()
        aetraintask_10 = AETraining()
        #    learning_rate=rates,
        #    weight_decay=decays,
        #    batch_size=batch_sizes,
        #    create_plots=False,
        #    checkpoint_interval=checkpoint_interval,
        #    )

        concattask = Concatenator()
        summarytask = SummaryStats()
        unpivottask = UnPivot()
        lineplottask = LinePlot()
        barplottask = BarPlot()
        simtask = AESimulate()
        heatmaptask = Heatmap()
        kdetask = KDE()

        with Flow(f"{self.name}", executor=executor, result=result) as flow:
            working_dir = Parameter("working_dir", self.params["working_dir"])
            modelfile = f'AESim-{len(self.params["features"])}'

            timeseries = Parameter("timeseries", default=self.params["timeseries"])
            grouping = Parameter("grouping", default=self.params["grouping"])
            epoches = Parameter("epoches", default=self.params["epoches"])
            rates = Parameter("learning rate", default=rates)
            decays = Parameter("weight decay", default=decays)
            batch_sizes = Parameter("batch size", default=batch_sizes)
            train_features = Parameter(
                f'features: {len(self.params["features"])}',
                default=sel_features,
            )
            model = Parameter("model", default=self.params["model"])
            add_noise = Parameter("add noise", default=add_noise)
            sets = Parameter("sets", default=self.params["sets"])

            input = datatask(
                name="Input",
                input=self.input,
                input_select=[0],
                task_tags="Input",
            )

            input_filtered = filtertask(
                input=input,
                input_select=[0],
                task_tags="Input_Filtered",
            )

            heatmapexp = heatmaptask(
                input=input_filtered,
                input_select=unmapped([0]),
                grouping=unmapped([]),
                features=unmapped(train_features),
                corr_type=unmapped("spearman"),
                task_tags="Input_Filtered",
            )

            kdeexp = kdetask(
                input=input_filtered,
                input_select=unmapped([0]),
                grouping=unmapped(["Treatment"]),
                features=unmapped(train_features),
                task_tags="Input_Filtered",
            )

            aeresults = aetraintask_10.map(
                input=unmapped(input_filtered),
                input_select=unmapped([0]),
                grouping=unmapped(grouping),
                features=unmapped(train_features),
                epoches=unmapped(epoches),
                learning_rate=rates,
                weight_decay=decays,
                batch_size=batch_sizes,
                checkpoint_interval=unmapped(checkpoint_interval),
                timeseries=unmapped(timeseries),
                rescale=unmapped(True),
                model=unmapped(model),
                working_dir=unmapped(working_dir),
                modelfile=unmapped(modelfile),
                create_plots=unmapped(False),
                checkpoint=unmapped(True),
                task_tags=combinations,
            )

            concatresults = concattask(
                input=aeresults,
                input_select=["Table: AE Loss"],
                type=False,
                task_tags="AE1_Training_Summary",
            )

            summaryresults = summarytask(
                input=concatresults,
                input_select=[0],
                grouping=["Source", "Batch Size", "Learning Rate", "Weight Decay"],
                features=["Training Loss", "Validation Loss", "Model File"],
                singledf=True,
                aggs=["min"],
                task_tags="AE1_Training_Summary",
            )

            unpivotresults = unpivottask(
                input=concatresults,
                input_select=[0],
                features=["Training Loss", "Validation Loss"],
                category_name="Loss Type",
                feature_name="Loss Value",
                task_tags="AE1_Training_Summary",
            )

            lineplotresults = lineplottask(
                input=unpivotresults,
                input_select=[0],
                grouping=[
                    "Epoch",
                    "Loss Type",
                    "Batch Size",
                    "Learning Rate",
                    "Weight Decay",
                ],
                features=["Loss Value"],
                task_tags="AE1_Training_Summary",
            )

            barplotresults = barplottask.map(
                input=unmapped(concatresults),
                input_select=unmapped([0]),
                grouping=unmapped(
                    ["Epoch", "Batch Size", "Learning Rate", "Weight Decay"]
                ),
                features=[["Training Loss"], ["Validation Loss"]],
                task_tags=unmapped("AE1_Training_Summary"),
            )

            simresults = simtask.map(
                input=unmapped(input_filtered),
                input_select=unmapped([0]),
                modelfile=select(aeresults, "Model File"),
                add_noise=unmapped(add_noise),
                sets=unmapped(sets),
                features=unmapped(train_features),
                grouping=unmapped(grouping),
                task_tags=combinations,
            )

            simresults_filtered = filtertask.map(
                input=simresults,
                input_select=unmapped([0]),
                task_tags=combinations,
            )

            heatmapresults = heatmaptask.map(
                input=simresults_filtered,
                input_select=unmapped([0]),
                grouping=unmapped([]),
                features=unmapped(train_features),
                corr_type=unmapped("spearman"),
                task_tags=combinations,
            )

            kderesults = kdetask.map(
                input=simresults_filtered,
                input_select=unmapped([0]),
                grouping=unmapped(["Treatment"]),
                features=unmapped(train_features),
                task_tags=combinations,
            )

            summaryresults2 = summarytask.map(
                input=simresults_filtered,
                input_select=unmapped([0]),
                grouping=unmapped(["Treatment"]),
                features=unmapped(train_features),
                singledf=unmapped(False),
                aggs=unmapped(["min", "max", "median", "count"]),
                task_tags=combinations,
            )

        return flow
