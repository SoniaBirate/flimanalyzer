#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import os
import numpy as np
import numpy.random as random
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import wx
from wx.lib.masked import NumCtrl
from importlib_resources import files, as_file

from flim.plugin import AbstractPlugin
from flim.plugin import plugin
from flim.gui.dialogs import BasicAnalysisConfigDlg
import flim.analysis.ml.autoencoder as autoencoder
import flim.resources


class AESimConfigDlg(BasicAnalysisConfigDlg):

    def __init__(self, parent, title, input=None, selectedgrouping=['None'], selectedfeatures='All', modelfile='', device='cpu', sets=1, add_noise=True, noise=0.1):
        self.modelfile = modelfile
        self.device = device
        self.sets = sets
        self.add_noise = add_noise
        self.noise = noise
        BasicAnalysisConfigDlg.__init__(self, parent, title, input=input, selectedgrouping=selectedgrouping,
                                        selectedfeatures=selectedfeatures, optgridrows=0, optgridcols=1)

    def get_option_panels(self):
        self.modelfiletxt = wx.StaticText(self, label=self.modelfile)
        browsebutton = wx.Button(self, wx.ID_ANY, 'Choose...')
        browsebutton.Bind(wx.EVT_BUTTON, self.OnBrowse)

        timeseries_sizer = wx.BoxSizer(wx.HORIZONTAL)
        timeseries_sizer.Add(wx.StaticText(self, label="Load Model from File"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        timeseries_sizer.Add(self.modelfiletxt, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        timeseries_sizer.Add(browsebutton, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.device_combobox = wx.ComboBox(self, wx.ID_ANY, style=wx.CB_READONLY, value=self.device, choices=['cpu', 'cuda'])
        bottom_sizer.Add(wx.StaticText(self, label="Device"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        bottom_sizer.Add(self.device_combobox, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5)
        
        self.sets_spinner = wx.SpinCtrl(self,wx.ID_ANY,min=1,max=20,initial=self.sets)
        bottom_sizer.Add(wx.StaticText(self, label="Sets"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        bottom_sizer.Add(self.sets_spinner, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5)

        #self.noise_checkbox = wx.CheckBox(self.panel, wx.ID_ANY, label="Add noise")
        #self.noise_checkbox.SetValue(self.add_noise)
        #bottom_sizer.Add(self.noise_checkbox, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5)

        #self.noise_input = NumCtrl(self.panel,wx.ID_ANY, min=0.0, value=self.noise, fractionWidth=3)
        #bottom_sizer.Add(wx.StaticText(self.panel, label="Signal-to-Noise Ratio"), 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        #bottom_sizer.Add(self.noise_input, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5)

        return [timeseries_sizer, bottom_sizer]

    def OnBrowse(self, event):
        fpath = self.modelfiletxt.GetLabel()
        _,fname = os.path.split(fpath)
        with wx.FileDialog(self, 'Model File', style=wx.FD_OPEN) as fileDialog:
            fileDialog.SetPath(fpath)
            fileDialog.SetFilename(fname)
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            fname = fileDialog.GetPath()
            self.modelfiletxt.SetLabel(fname)

    def _get_selected(self):
        params = super()._get_selected()
        params['modelfile'] = self.modelfiletxt.GetLabel()
        params['device'] = self.device_combobox.GetValue()
        params['sets'] = self.sets_spinner.GetValue()
        #params['add_noise'] = self.noise_checkbox.GetValue()
        #params['noise'] = self.noise_input.GetValue()
        return params
                
        
@plugin(plugintype="Analysis")        
class AESimulate(AbstractPlugin):
    
    def __init__(self, name="Autoencoder: Simulate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.variables = self.params['features']
        self.modelfile = self.params['modelfile']
        self.device = self.params['device']
        self.sets = self.params['sets']
        self.add_noise = self.params['add_noise']
        self.noise = self.params['noise']

    def __repr__(self):
        return f"{'name': {self.name}}"

    def __str__(self):
        return self.name

    def get_icon(self):
        source = files(flim.resources).joinpath('aerun.png')
        return wx.Bitmap(str(source))
        
    def get_required_categories(self):
        return ["any"]

    def get_required_features(self):
        return ["any"]

    def get_default_parameters(self):
        params = super().get_default_parameters()
        params.update({
            'modelfile': '',
            'device': 'cpu',
            'sets': 1,
            'add_noise': True,
            'noise': 0.1, # 0.0 < noise < 1.0
        })
        return params

    def output_definition(self):
        return {'Table: Simulated': None}
        
    def run_configuration_dialog(self, parent, data_choices={}):
        dlg = AESimConfigDlg(parent, f'Configuration: {self.name}', 
                            input=self.input,
                            selectedgrouping=self.params['grouping'],
                            selectedfeatures=self.params['features'],
                            modelfile=self.params['modelfile'],
                            device=self.params['device'],
                            sets=self.params['sets'],
                            add_noise=self.params['add_noise'],
                            noise=self.params['noise'])
        if dlg.ShowModal() == wx.ID_OK:
            params = dlg.get_selected()
            self.params.update(params)
            return self.params
        else:	
            return None

    def execute(self):
        data = list(self.input.values())[0]
        cats = list(data.select_dtypes(['category']).columns.values)
        data_feat = data[self.params['features']]
        feat_cols = list(data_feat.columns)
        fc_lower = [x.lower() for x in feat_cols]

        FAD_feats = [feat_cols[r] for r in range(len(fc_lower)) 
            if ("fad" in fc_lower[r] and ("a1" in fc_lower[r] or "a2" in fc_lower[r]))]
        NADPH_feats = [feat_cols[r] for r in range(len(fc_lower)) 
            if (("nadph" in fc_lower[r] or "nad(p)h" in fc_lower[r]) and ("a1" in fc_lower[r] or "a2" in fc_lower[r]))]
        
        rng = random.default_rng()
        # load an AE model
        device = self.params['device']
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logging.info("CUDA selected, but no CUDA device available. Switching to CPU.")
        ae = torch.load(self.params['modelfile'], map_location=device)

        data_feat = data_feat.astype(float)
        my_imputer = SimpleImputer(strategy="constant",fill_value=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        
        sim_df = pd.DataFrame(columns=(cats+feat_cols))
        maxcell = np.amax(data['Cell'].astype(int).to_numpy())
        
        for simset in range(0, self.params['sets']):
            noise = rng.standard_normal(size=data_feat.shape)
            sdata_feat = data_feat.add(noise)
            raw_min = np.asarray(np.amin(sdata_feat, axis=0))
            raw_max = np.asarray(np.amax(sdata_feat, axis=0))

            sdata_feat = min_max_scaler.fit_transform(sdata_feat) # Normalization
            sdata_feat = my_imputer.fit_transform(sdata_feat)
            
            sdata_feat = torch.FloatTensor(sdata_feat)
            logging.debug(f'Sim set {simset+1}, original shape: {sdata_feat.shape}')
    
            data_input = Variable(sdata_feat)
            features, reconstructed = ae(data_input)
            logging.debug(f'Sim set {simset+1}, reconstructed shape: {reconstructed.shape}')
    
            features=torch.squeeze(features)
            logging.debug(f'Sim set {simset+1}, features shape: {features.shape}')
    
            criterion = nn.MSELoss()
            loss = criterion(reconstructed, sdata_feat)
            logging.debug(f'Sim set {simset+1}, loss: {loss.data}')
            
            recon_data = reconstructed.detach().numpy()
            raw_range = (raw_max-raw_min).reshape((1, -1))
            sim_data = recon_data*(raw_range) + raw_min
            
            temp = pd.DataFrame(columns=(cats+feat_cols))
            temp[cats] = data[cats]
            temp['Cell'] = temp['Cell'].astype(int) + maxcell*simset
            temp[feat_cols] = sim_data
            sim_df = pd.concat([sim_df, temp])

        FADtot = sim_df[FAD_feats[0]]+sim_df[FAD_feats[1]]
        FAD0 = sim_df[FAD_feats[0]]/FADtot*100
        FAD1 = sim_df[FAD_feats[1]]/FADtot*100
        NADPHtot = sim_df[NADPH_feats[0]]+sim_df[NADPH_feats[1]]
        NADPH0 = sim_df[NADPH_feats[0]]/NADPHtot*100
        NADPH1 = sim_df[NADPH_feats[1]]/NADPHtot*100
        calcdf = pd.DataFrame(({
            FAD_feats[0]+"[%]": FAD0,
            FAD_feats[1]+"[%]": FAD1,
            NADPH_feats[0]+"[%]": NADPH0,
            NADPH_feats[1]+"[%]": NADPH1,
        }))
        # concat and ensure unique index
        sim_df = pd.concat([sim_df, calcdf], axis=1).reset_index()
        outfeats = feat_cols+list(calcdf.columns.values)
        outfeats.sort() #ensure feature vectors will be applied correctly
        sim_df = sim_df[cats+outfeats]
        sim_df['Cell'] = sim_df['Cell'].astype(str).astype('category')
        
        return {'Table: Simulated': sim_df}
    