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

from collections import OrderedDict
from prefect import Task, Flow, Parameter, task, unmapped, flatten, tags
from prefect.tasks.core.collections import List
from prefect.tasks.core.constants import Constant
from prefect.tasks.files.operations import Glob
from prefect.executors import DaskExecutor
from prefect.executors.base import Executor
from prefect.engine.results import LocalResult

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
from flim.data.sortdata import Sort
from flim.analysis.aerun import RunAE
from flim.analysis.aetraining import AETraining, AETrainingConfigDlg
from flim.analysis.aeaugment import AEAugment
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
from flim.workflow.aetune import AEAugmentTuneConfigDlg
from flim.workflow.basicflow import AbsWorkFlow


@task
def list_to_dict(listofdict):
    """Fattens list of dict into dict."""
    result = {f"{k}-{i:04d}": v for i, d in enumerate(listofdict) for k, v in d.items()}
    return result


class AEFeatureConfigDialog(AEAugmentTuneConfigDlg):
    def __init__(
        self,
        parent,
        title,
        input={},
        data_choices={},
        train_on=None,
        run_on=None,
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
        self.data_choices = data_choices
        dc = list(data_choices.keys())
        if train_on not in dc:
            self.train_on = dc[0]
        else:
            self.train_on = train_on
        if run_on not in dc:
            self.run_on = dc[1] if len(dc) > 1 else self.train_on
        else:
            self.run_on = run_on

        super().__init__(
            parent,
            title,
            input=input,
            data_choices=data_choices,
            description=description,
            selectedgrouping=selectedgrouping,
            selectedfeatures=selectedfeatures,
            epoches=epoches,
            batch_size=batch_size,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            timeseries=timeseries,
            model=model,
            device=device,
            rescale=rescale,
            checkpoint_interval=checkpoint_interval,
            autosave=autosave,
            working_dir=working_dir,
        )

    def get_selectable_features(self):
        # training and running of model needs to occur on same features
        features = set(
            self.data_choices[self.train_on]
            .select_dtypes(include=["number"], exclude=["category"])
            .columns.values
        ).intersection(
            self.data_choices[self.run_on]
            .select_dtypes(include=["number"], exclude=["category"])
            .columns.values
        )
        logging.debug(f"selectable features={features}")
        return features

    def get_option_panels(self):
        fsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.train_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=self.train_on,
            choices=list(self.data_choices.keys()),
        )
        self.train_combobox.Bind(wx.EVT_COMBOBOX, self._update_input)
        self.run_combobox = wx.ComboBox(
            self.panel,
            wx.ID_ANY,
            style=wx.CB_READONLY,
            value=self.run_on,
            choices=list(self.data_choices.keys()),
        )
        self.run_combobox.Bind(wx.EVT_COMBOBOX, self._update_input)
        fsizer.Add(
            wx.StaticText(self.panel, label="Train Model on"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        fsizer.Add(self.train_combobox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        fsizer.Add(
            wx.StaticText(self.panel, label="Run Model on"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        fsizer.Add(self.run_combobox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(fsizer)
        for p in super().get_option_panels():
            sizer.Add(p)
        return [sizer]

    def _update_input(self, event):
        update = (
            self.train_on != self.train_combobox.GetValue()
            or self.run_on != self.run_combobox.GetValue()
        )
        if update:
            self.train_on = self.train_combobox.GetValue()
            self.run_on = self.run_combobox.GetValue()
            allfeatures = self.get_selectable_features()
            self.allfeatures = OrderedDict(
                (" ".join(c.split("\n")), c) for c in allfeatures
            )
            self._update_feature_cbs()

    def _get_selected(self):
        params = super()._get_selected()
        params["input"] = {
            "Train": self.data_choices[self.train_on],
            "Run": self.data_choices[self.run_on],
        }
        return params


@plugin(plugintype="Workflow")
class AEFeatureWorkflow(AbsWorkFlow):
    def __init__(self, name="FLIM Feature Analysis", **kwargs):
        super().__init__(name=name, **kwargs)
        # self.executor = None #DaskExecutor(address="tcp://172.18.75.87:8786")

    def get_required_features(self):
        return ["any"]
    
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
                "ctrl_group": "Ctrl",
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
                "model": "AE Dimensionality Reduction 2-10",
                "device": "cpu",
                "rescale": True,
                "checkpoint_interval": 2,  # 20
            }
        )
        return params

    def run_configuration_dialog(self, parent, data_choices={}):
        input = self.params["input"]
        if isinstance(input, dict) and len(input) > 0:
            train_on = list(input.keys())[0]
        else:
            train_on = ""
        if isinstance(input, dict) and len(input) > 1:
            run_on = list(input.keys())[1]
        else:
            run_on = train_on
        dlg = AEFeatureConfigDialog(
            parent,
            f"Configuration: {self.name}",
            input=input,
            data_choices=data_choices,
            train_on=train_on,
            run_on=run_on,
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

        data_train = list(self.input.values())[0]
        data_run = list(self.input.values())[1]
        logging.debug(f"Training on {data_train.shape}")
        logging.debug(f"Running on {data_run.shape}")
        sel_features = self.params["features"]
        all_features = [c for c in data_train.select_dtypes(include=np.number)]

        batch_sizes = [f"b{i}" for i in self.params["batch_size"]]
        rates = [f"l{i}" for i in self.params["learning_rate"]]
        decays = [f"d{i}" for i in self.params["weight_decay"]]
        last_epoch = self.params["epoches"]
        combinations = [
            utils.clean(f"{str(i)} e{last_epoch:04d}")
            for i in list(itertools.product(batch_sizes, rates, decays))
        ]
        logging.debug(f"combinations={combinations}")
        batch_sizes, rates, decays = utils.permutate(
            self.params["batch_size"],
            self.params["learning_rate"],
            self.params["weight_decay"],
        )

        # listtask = List()
        datatask = DataBucket(name="Input")
        globtask = Glob()
        filtertask = Filter()
        aetraintask = AETraining()
        aeruntask = RunAE()
        pcatask = PCAnalysis()
        stask = SummaryStats()  # , log_stdout=True)
        heatmaptask = Heatmap()
        concattask = Concatenator()
        pivottask = Pivot()
        seriestask = SeriesAnalyzer()
        bartask = BarPlot()
        kmeanstask = KMeansClustering()
        kdetask = KDE()
        mergetask = Merger()
        kstask = KSStats()
        sorttask = Sort()

        with Flow(
            f"{self.name}",
            executor=executor,
            result=result,
        ) as flow:
            working_dir = Parameter("working_dir", self.params["working_dir"])
            modelfile = f'AEFeature-{len(self.params["features"])}'

            timeseries = Parameter("timeseries", default=self.params["timeseries"])
            timeseries_vals = [
                v for v in data_train[self.params["timeseries"]].unique() if not "15" in v
            ]
            t_pairs = list(zip(timeseries_vals, timeseries_vals[1:]))
            ctrl_group = self.params["ctrl_group"]

            grouping = Parameter("grouping", default=self.params["grouping"])
            epoches = Parameter("epoches", default=self.params["epoches"])
            rates = Parameter("learning rate", default=rates)
            decays = Parameter("weight decay", default=decays)
            batch_sizes = Parameter("batch size", default=batch_sizes)
            features = Parameter(
                f'features: {len(self.params["features"])}',
                default=sel_features,
            )
            model = Parameter("model", default=self.params["model"])

            globresult = globtask(
                path="/Users/khs3z/FLIMAnalyzerResults/FLIM Feature Analysis/",
                pattern="AE*.model",
            )

            input_train = datatask(
                name="Input_Train",
                input=data_train,
                input_select=[0],
                task_tags="Input_Train",
            )

            input_train_filtered = filtertask(
                input=input_train,
                input_select=[0],
                task_tags="Input_Train_Filtered",
            )

            input_run = datatask(
                name="Input_Run",
                input=data_run,
                input_select=[0],
                task_tags="Input_Run",
            )

            input_run_filtered = filtertask(
                input=input_run,
                input_select=[0],
                task_tags="Input_Run_Filtered",
            )

            pcaresults = pcatask(
                input=input_run_filtered,
                input_select=[0],
                grouping=[],  # ['FOV', 'Cell'],
                features=features,
                task_tags="PCA",
            )

            aeresults = aetraintask.map(
                input=unmapped(input_train_filtered),
                input_select=unmapped([0]),
                grouping=unmapped(grouping),
                features=unmapped(features),
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

            aerunresults = aeruntask.map(
                input=unmapped(input_run_filtered),
                input_select=unmapped([0]),
                grouping=unmapped(grouping),
                features=unmapped(features),
                modelfile=select(
                    aeresults,
                    key_pattern="Model File",
                    value_pattern=f"{last_epoch:04d}",
                ),
                task_tags=combinations,
            )

            flirr_pca_f1_tags = ["FLIRR"] + ["PCA"] + [f"{c}" for c in combinations]
            logging.debug(f"flirr_pca_f1_tags={flirr_pca_f1_tags}")

            kderesults_F1 = kdetask.map(
                input=[input_run_filtered, pcaresults["Table: PCA Components"]]
                + select(aerunresults, "Table: Features"),
                input_select=unmapped([0]),
                grouping=unmapped(["Treatment"]),
                features=[["FLIRR"]] + [["PC 1"]] + len(combinations) * [["Feature 1"]],
                task_tags=flirr_pca_f1_tags,
            )

            summaryresults = stask.map(
                input=[input_run_filtered, pcaresults["Table: PCA Components"]]
                + select(aerunresults, "Table: Features"),
                input_select=unmapped([0]),
                grouping=unmapped(["Cell", "FOV", "Treatment"]),
                features=[["FLIRR"]] + [["PC 1"]] + len(combinations) * [["Feature 1"]],
                aggs=unmapped(["mean"]),
                singledf=unmapped(True),
                task_tags=flirr_pca_f1_tags,
            )

            pivotresult = pivottask.map(
                input=summaryresults,
                input_select=unmapped([0]),
                grouping=unmapped([timeseries]),
                features=[["FLIRR\nmean"]]
                + [["PC 1\nmean"]]
                + len(combinations) * [["Feature 1\nmean"]],
                task_tags=flirr_pca_f1_tags,
            )

            seriesresult = seriestask.map(
                input=pivotresult,
                input_select=unmapped([0]),
                features=[[f"FLIRR\nmean\n{t}" for t in timeseries_vals]]
                + [[f"PC 1\nmean\n{t}" for t in timeseries_vals]]
                + len(combinations)
                * [[f"Feature 1\nmean\n{t}" for t in timeseries_vals]],
                series_min=unmapped(False),
                series_max=unmapped(False),
                series_range=unmapped(False),
                series_mean=unmapped(False),
                series_median=unmapped(False),
                delta=unmapped(True),
                delta_min=unmapped(False),
                delta_max=unmapped(False),
                delta_sum=unmapped(False),
                delta_cum=unmapped(True),
                delta_norm=unmapped(True),
                merge_input=unmapped(False),
                task_tags=flirr_pca_f1_tags,
            )

            sortresult = sorttask.map(
                input=seriesresult,
                input_select=unmapped([0]),
                features=[
                    [
                        f"FLIRR\nmean\nnormalized delta {t[0]}:{t[1]}"
                        for t in t_pairs
                        if ctrl_group.lower() in t[0].lower()
                    ]
                ]
                + [
                    [
                        f"PC 1\nmean\nnormalized delta {t[0]}:{t[1]}"
                        for t in t_pairs
                        if ctrl_group.lower() in t[0].lower()
                    ]
                ]
                + len(combinations)
                * [
                    [
                        f"Feature 1\nmean\nnormalized delta {t[0]}:{t[1]}"
                        for t in t_pairs
                        if ctrl_group.lower() in t[0].lower()
                    ]
                ],
                ascending=unmapped(True),
                task_tags=flirr_pca_f1_tags,
            )

            barresult = bartask.map(
                input=sortresult,
                input_select=unmapped([0]),
                grouping=unmapped(["FOV", "Cell"]),
                features=[
                    [f"FLIRR\nmean\nnormalized delta {t[0]}:{t[1]}" for t in t_pairs]
                ]
                + [[f"PC 1\nmean\nnormalized delta {t[0]}:{t[1]}" for t in t_pairs]]
                + len(combinations)
                * [[f"Feature 1\nmean\nnormalized delta {t[0]}:{t[1]}" for t in t_pairs]],
                ordering=unmapped({}),
                orientation=unmapped("vertical"),  # 'horizontal'
                bar_type=unmapped("stacked"),  # 'single', '100% stacked'
                dropna=unmapped(True),
                error_bar=unmapped("+/-"),  # '+', 'None'
                error_type=unmapped("s.e.m"),
                task_tags=flirr_pca_f1_tags,
            )  # 'std'

            ksresults = kstask.map(
                input=[input_run_filtered, pcaresults["Table: PCA Components"]]
                + select(aerunresults, "Table: Features"),
                input_select=unmapped([0]),
                comparison=unmapped("Treatment"),
                grouping=unmapped(["Treatment", "FOV", "Cell"]),
                alpha=unmapped(0.05),
                features=[["FLIRR"]] + [["PC 1"]] + len(combinations) * [["Feature 1"]],
                prefix=["FLIRR:"] + ["PCA:"] + [f"AERun {c}:" for c in combinations],
                task_tags=flirr_pca_f1_tags,
            )

            concatresults = concattask(
                input=ksresults,  # list_to_dict(ksresults),
                input_select=None,
                type=False,
                numbers_only=False,
                task_tags="Summary",
            )

        return flow
