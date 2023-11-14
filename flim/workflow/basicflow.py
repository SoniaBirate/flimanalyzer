#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:35:56 2020

@author: khs3z
"""

import importlib
import inspect
import json
import logging
import os
import pkgutil
from abc import ABC, abstractmethod

import flim.resources
import graphviz
import math
import matplotlib.pyplot as plt
import networkx as nx
import networkx.classes.function
import networkx.drawing.nx_agraph
import networkx.drawing.nx_pydot
import numpy as np
import pandas as pd
import pydot
import wx

from importlib_resources import files
from prefect import Flow, Parameter, Task, task, unmapped
from prefect.executors import DaskExecutor
from prefect.executors.base import Executor
from prefect.tasks.core.collections import List
from prefect.tasks.core.constants import Constant

from flim.analysis.aerun import RunAE
from flim.analysis.aesimulate import AESimulate
from flim.analysis.aetraining import AETraining
from flim.analysis.barplots import BarPlot
from flim.analysis.heatmap import Heatmap
from flim.analysis.kde import KDE
from flim.analysis.kmeans import KMeansClustering
from flim.analysis.ksstats import KSStats
from flim.analysis.lineplots import LinePlot
from flim.analysis.pairplots import PairPlot
from flim.analysis.pca import PCAnalysis
from flim.analysis.relativechange import RelativeChange
from flim.analysis.scatterplots import ScatterPlot
from flim.analysis.seriesanalyzer import SeriesAnalyzer
from flim.analysis.summarystats import SummaryStats
from flim.core.graph import WorkflowGraph
from flim.data.concatdata import Concatenator
from flim.data.filterdata import Filter
from flim.data.mergedata import Merger
from flim.data.pivotdata import Pivot
from flim.data.unpivotdata import UnPivot
from flim.gui.dialogs import BasicAnalysisConfigDlg
from flim.plugin import ALL_FEATURES, AbstractPlugin, DataBucket, plugin
from flim.results import LocalResultClear


class AbsWorkFlow(AbstractPlugin):
    def __init__(self, name="Base Workflow", executor=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # self.name = "Abstract FLIM Workflow"
        # self.set_executor(executor)

    # def __repr__(self):
    #    return f"{'name: {self.name}'}"

    def __str__(self):
        return self.name

    def get_icon(self):
        source = files(flim.resources).joinpath("sim_tuning1.png")
        return wx.Bitmap(str(source))

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "grouping": self.get_required_categories(),
                "features": self.get_required_features(),
                "autosave": True,
                "executor": {
                    "class": "prefect.executors.local.LocalExecutor",
                    "args": {},
                },
                "register": False,
                "project": "FLIM",  # self.name,
            }
        )
        return params

    """
    def set_result(self):
        self.result = LocalResultClear(
            dir=f'{self.params["working_dir"]}',
            location="{flow_name}/"
            "{scheduled_start_time:%Y-%m-%d_%H-%M-%S}/{task_tags}/pickled/"
            "{task_full_name}",  # -{task_run_id}-{map_index}-
        )

    
    def set_executor(self, executor=None, **kwargs):
        if isinstance(executor, Executor):
            self.executor = executor
        else:    
            modulename = "<unresolved"
            classname = "<unresolved"
            if executor is None:
            	# use default, ignore kwargs
            	executor = self.params["executor"]["class"]
            	kwargs = self.params["executor"]["args"]
            elif isinstance(executor, dict):
            	kwargs = executor.get("args", {})
            	executor = executor.get("class", "prefect.executors.local.LocalExecutor")
            if isinstance(executor, str):
            	modulename, _, classname = executor.rpartition(".")
            	try:
            		module = importlib.import_module(modulename)
            		class_ = getattr(module, classname)
            		self.executor = class_(**kwargs)
            	except Exception as err:
            		logging.error(f"Error: {err}")
            		logging.error(
            			f"Error instantiating {modulename}.{classname} plugin tool."
            		)
            		self.executor = LocalExecutor()
            elif issubclass(self.executor, Executor):
            	modulename = clazz.__module__
            	classname = clazz.__name__
            	self.executor = executor
            logging.debug(f"Executor modulename={modulename}, classname={classname}")
            assert issubclass(self.executor.__class__, Executor)
    """

    def get_required_categories(self):
        return ["Treatment", "FOV", "Cell"]

    def get_required_features(self):
        return ["FLIRR", "trp t1"]

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

    @abstractmethod
    def construct_flow(self, executor, result):
        return Flow(name=self.name, executor=executor, result=result)

    def execute(self):
        # self.set_result()
        # self.set_executor(self.params["executor"])
        logging.info(f"Using {self.executor} for flow {self.name}")
        flow = self.construct_flow()
        if self.params["register"]:
            flow.register(
                project_name=self.params["project"],
                idempotency_key=flow.serialized_hash(),
            )  # prevent version bump when reregistering an unchanged flow

        flow.visualize()
        with open(f"{self.name}.json", "w") as fp:
            json.dump(flow.serialize(), fp, indent=4)

        results = {}
        state = flow.run()
        task_refs = flow.get_tasks()
        task_results = [state.result[tr]._result.value for tr in task_refs]
        output_results = [tr for tr in list(task_results) if not isinstance(tr, dict)]
        plugin_results = [
            tr for tr in list(task_results) if isinstance(tr, dict)
        ]  # state.result[tr]._result.value not in output_results]
        results = {
            f"{k}-{str(id(v))}": v
            for d in plugin_results
            if isinstance(d, dict)
            for k, v in d.items()
        }

        wg = WorkflowGraph(flow)
        fig, _ = wg.get_plot()
        results[f"{self.name} Graph"] = fig

        return results


def get_task_graph(task):
    g = nx.DiGraph()
    g.add_node(task.name)
    # g.add_node(tnode)
    for rname, robject in task.output_definition().items():
        print(f"edge: {task.name} -> {rname}")
        g.add_edge(task.name, rname)
    print(f"len(g):{len(g)}")
    return g


"""
@task
def results_to_tasks(task_in):
    datatask = DataBucket(None)
    results = task_in
    if len(task_in) > 1:
        print ("MAPPING", "|".join([t for t in task_in]))
        #task_list = [v for v in task_in.values()]
        #mapped = data_task.map(task_list)
        #out_tasks = {f'Out {i}':d for i,d in enumerate(mapped)}
        out_tasks = {k:datatask(name=k, data=task_in[k], input_select=[0]) for i, k in enumerate(task_in)}
    else:
        out_tasks = task_in
        #out_tasks = {k:datatask(name=k, data=results, input_select=[i]) for i, k in enumerate(task_in.output_definition())}
    return out_tasks
"""


def _get_input_label(self, data):
    if data == pd.DataFrame or isinstance(data, pd.DataFrame):
        return "Table"
    elif isinstance(data, matplotlib.figure.Figure):
        return "Plot"
    elif isinstance(data, str) and os.path.isabs(data):
        return "File"
    else:
        return "Data"


def results_to_tasks(task_in):
    # datatask = DataBucket(None)
    # print (f'task_in.output_definition()={task_in.output_definition()}')
    out_tasks = {}
    for i, output_name in enumerate(task_in.output_definition()):
        datatask = DataBucket()  # task_in, name=output_name, task_run_name="Hello")
        # out_tasks[output_name] = datatask(name=output_name, task_run_name="Hello", input=task_in, input_select=[i])
        dt_result = datatask(
            name=output_name, task_run_name="Hello", input=task_in, input_select=[i]
        )
        out_tasks[output_name] = dt_result
    # out_tasks = {k:datatask(name=k, data=task_in, input_select=[i]) for i, k in enumerate(task_in.output_definition())}
    # print (f'out_tasks={out_tasks}')
    return out_tasks


@task
def data_task(data):
    out = {"Out": data}
    return out


@task
def to_list(d):
    return [v for v in d.values()]


@plugin(plugintype="Workflow")
class BasicFLIRRWorkFlow(AbsWorkFlow):
    def __init__(self, name="Basic FLIRR Analysis", **kwargs):
        super().__init__(name=name, **kwargs)
        # self.executor = None #DaskExecutor(address="tcp://172.18.75.87:8786")

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update(
            {
                "grouping": ["Treatment", "FOV", "Cell"],
                "features": [
                    "FAD a1",
                    "FAD a1%",
                    "FAD a2",
                    "FAD a2%",
                    "FAD t1",
                    "FAD t2",
                    "FAD photons",
                    "NAD(P)H a1",
                    "NAD(P)H a1%",
                    "NAD(P)H a2",
                    "NAD(P)H a2%",
                    "NAD(P)H t1",
                    "NAD(P)H t2",
                    "NAD(P)H photons",
                ],
            }
        )
        return params

    def get_required_features(self):
        return ["any"]
    
    def get_icon(self):
        source = files(flim.resources).joinpath("flirr_test1.png")
        return wx.Bitmap(str(source))


    def construct_flow(self, executor, result):
        data = list(self.input.values())[0].copy()
        sel_grouping = self.params["grouping"]
        sel_features = self.params["features"]
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
        print(scatter_features)

        # listtask = List()
        datatask = DataBucket("Input")  # None, name='Input')
        stask = SummaryStats(
            None, features=[f"rel {f}" for f in sel_features], singledf=False
        )  # , log_stdout=True)
        scattertask = ScatterPlot(None, features=scatter_features)
        relchangetask = RelativeChange(None)
        lineplottask = LinePlot(None)
        pairplottask = PairPlot(None)
        pivottask = Pivot(None)
        barplottask = BarPlot(None)
        kdetask = KDE()
        # cats = list(data.select_dtypes('category').columns.values)
        with Flow(f"{self.name}", executor=executor, result=result) as flow:
            # input = Parameter('input', default=data)
            inputresult = datatask(
                name="Input", input=self.input, input_select=[0], task_tags="Input"
            )
            summary = stask(
                input=inputresult,
                input_select=[0],
                grouping=[sel_grouping[0]],
                features=sel_features,
                aggs=["count", "median", "mean", "std", "sem"],
                singledf=True,
                prefix="Stats-",
            )
            counts = stask(
                input=inputresult,
                input_select=[0],
                grouping=sel_grouping,
                features=[sel_features[0]],
                aggs=["count"],
                singledf=True,
                prefix="Counts-",
            )
            pairplots = pairplottask(
                input=inputresult,
                input_select=[0],
                grouping=[sel_grouping[0]],
                x_vars=[f for f in sel_features if "FAD" in f],
                y_vars=[f for f in sel_features if "FAD" in f],
                diag_kind="KDE",
                prefix="FAD-",
            )
            pairplots = pairplottask(
                input=inputresult,
                input_select=[0],
                grouping=[sel_grouping[0]],
                x_vars=[f for f in sel_features if "NAD" in f],
                y_vars=[f for f in sel_features if "NAD" in f],
                diag_kind="KDE",
                prefix="NADPH-",
            )
            reltable = relchangetask(
                input=inputresult,
                input_select=[0],
                grouping=sel_grouping[1:],
                features=sel_features,
                reference_group=sel_grouping[0],
                reference_value=data[sel_grouping[0]].unique()[0],
                method="mean",
            )
            srelresult = stask(
                input=reltable["Table: Relative Change"],
                input_select=[0],
                grouping=sel_grouping[::-1],
                features=[f"rel {f}" for f in sel_features],
                aggs=["mean"],
                singledf=True,
                prefix="Relative Stats-",
            )
            # pivotresult = results_to_tasks(pivottask(data=srelresult['Data: Summarize'], input_select=[0], grouping=['Treatment'],features=['rel FLIRR\nmean']))
            scatter = scattertask(
                input=inputresult,
                input_select=[0],
                grouping=[sel_grouping[0]],
                features=scatter_features,
            )
            lineplotresult = lineplottask.map(
                input=unmapped(reltable["Table: Relative Change"]),
                input_select=unmapped([0]),
                grouping=unmapped(
                    sel_grouping[0:2] if len(sel_grouping) > 1 else sel_grouping
                ),
                features=[["rel " + c] for c in sel_features],
            )

            # sresult = results_to_tasks(stask(data=input, input_select=[0], grouping=['Treatment', 'FOV', 'Cell'],features=allfeatures,aggs=['max','mean','median','count']))
            # listtask = to_list(scattertask(data=input, input_select=[0], grouping=['Treatment', 'FOV', 'Cell'],features=sel_features))
            # scatter = data_task.map(listtask)

            # reltable = datatask(name='Relative Change', data=relchresult)
            # lineplotresult = results_to_tasks(lineplottask(data=reltable['Table: Relative Change'], input_select=[0], grouping=['Treatment','FOV'], features=['rel '+c for c in sel_features]))

            # barplot = results_to_tasks(barplottask(data=input, input_select=[0], grouping=['Treatment', 'FOV', 'Cell'], features=['FLIRR']))
            kdeplot = kdetask.map(
                input=unmapped(inputresult),
                input_select=unmapped([0]),
                grouping=unmapped([sel_grouping[0]]),
                features=[[c] for c in sel_features],
            )
            # sresult2 = results_to_tasks(stask(data=sresult['Data: Summarize'], input_select=[0], grouping=['FOV'], features=['FLIRR\nmean'], aggs=['max']))
            # sresult3 = results_to_tasks(stask(data=pcaresult, input_select=[0], grouping=[], features=['Principal component 1'], aggs=['max']))
            # sresult4 = results_to_tasks(stask(data=pcaresult['PCA explained'], input_select=[0], grouping=['PCA component'], features=['explained var ratio'], aggs=['max']))
        return flow
