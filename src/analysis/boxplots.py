#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:11:37 2020

@author: khs3z
"""

import logging
import pandas as pd
from analysis.absanalyzer import AbstractAnalyzer
from gui.dialogs import BasicAnalysisConfigDlg
import wx
import matplotlib.pyplot as plt


class BoxPlot(AbstractAnalyzer):
    
    def __init__(self, data, categories, features, **kwargs):
        AbstractAnalyzer.__init__(self, data, grouping=categories, features=features, **kwargs)
        self.name = "Box Plot"
    
    def __repr__(self):
        return f"{'name': {self.name}}"
    
    def __str__(self):
        return self.name
    
    def get_required_categories(self):
        return []
    
    def get_required_features(self):
        return ['any']

    def run_configuration_dialog(self, parent):
        selgrouping = self.params['grouping']
        selfeatures = self.params['features']
        dlg = BasicAnalysisConfigDlg(parent, f'Configuration: {self.name}', self.data, selectedgrouping=selgrouping, selectedfeatures=selfeatures)
        if dlg.ShowModal() == wx.ID_OK:
            results = dlg.get_selected()
            self.params.update(results)
            return self.params
        else:	
            return None
        
    def execute(self):
        results = {}
        for feature in sorted(self.params['features']):
            logging.debug (f"\tcreating box plot for {feature}")
            fig,ax = self.grouped_boxplot(self.data, feature, categories=self.params['grouping'])
            results[f"Box Plot {feature}"] = (fig,ax)
        return results
    
    def grouped_boxplot(self, data, feature, title=None, categories=[], dropna=True, pivot_level=1, **kwargs):
        if data is None or not feature in data.columns.values:
            return None, None
    
        # plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots(constrained_layout=True)
        
        if categories is None:
            categories = []
        newkwargs = kwargs.copy()
        newkwargs.update({
                'column':feature,
                #'subplots': False,
                'ax':ax})
        if len(categories) > 0: 
            newkwargs.update({'by':categories})
        
        cols = [c for c in categories]
        cols.append(feature)
        if dropna:
            data = data[cols].dropna(how='any', subset=[feature])
        else:
            data = data[cols]
        #data.set_index(groups, inplace=True)
        #print (f"index.names={data.index.names}")
        fig.set_figheight(6)
        fig.set_figwidth(12)
        data.boxplot(**newkwargs)
        #grouped = data.groupby(level=list(range(len(groups))))
        #grouped.boxplot(ax=ax, subplots=False)
        
        miny = min(0,data[feature].min()) * 0.95
        maxy = max(0,data[feature].max()) * 1.05
        logging.debug (f'title={title}')
        ax.set_ylim(miny, maxy)
        if title is None:
            title = feature.replace('\n', ' ') #.encode('utf-8')
        if len(title) > 0:
            ax.set_title(title)
        # plt.rcParams.update({'figure.autolayout': False})
        
        self._add_picker(fig)

        return fig,ax   