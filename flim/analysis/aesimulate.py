#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:30 2020

@author: khs3z
"""

import logging
import re
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
from joblib import load

NOISE_UNIT = ['linear', 'dB']


def to_db(value):
    return (10.0 * np.log10(value))


def db_to_linear(db_value):
    return 10.0**(db_value / 10.0)


class AESimConfigDlg(BasicAnalysisConfigDlg):
    def __init__(self,
                 parent,
                 title,
                 input=None,
                 selectedgrouping=['None'],
                 selectedfeatures='All',
                 modelfile='',
                 device='cpu',
                 sets=1,
                 add_noise=True,
                 snr_db=0.0,
                 snr_unit=NOISE_UNIT[-1]):
        self.modelfile = modelfile
        self.device = device
        self.sets = sets
        self.add_noise = add_noise
        self.snr_unit = snr_unit if snr_unit in NOISE_UNIT else NOISE_UNIT[-1]
        self.snr = snr_db if self.snr_unit == 'dB' else db_to_linear(snr_db)
        BasicAnalysisConfigDlg.__init__(self,
                                        parent,
                                        title,
                                        input=input,
                                        selectedgrouping=selectedgrouping,
                                        selectedfeatures=selectedfeatures,
                                        optgridrows=0,
                                        optgridcols=1)

    def get_option_panels(self):
        self.modelfiletxt = wx.StaticText(self.panel, label=self.modelfile)
        browsebutton = wx.Button(self.panel, wx.ID_ANY, 'Choose...')
        browsebutton.Bind(wx.EVT_BUTTON, self.OnBrowse)

        timeseries_sizer = wx.BoxSizer(wx.HORIZONTAL)
        timeseries_sizer.Add(wx.StaticText(self.panel, label="Load Model from File"), 0,
                             wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        timeseries_sizer.Add(self.modelfiletxt, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        timeseries_sizer.Add(browsebutton, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.device_combobox = wx.ComboBox(self.panel,
                                           wx.ID_ANY,
                                           style=wx.CB_READONLY,
                                           value=self.device,
                                           choices=['cpu', 'cuda'])
        bottom_sizer.Add(wx.StaticText(self.panel, label="Device"), 0,
                         wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        bottom_sizer.Add(self.device_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.sets_spinner = wx.SpinCtrl(self.panel, wx.ID_ANY, min=1, max=20, initial=self.sets)
        bottom_sizer.Add(wx.StaticText(self.panel, label="Sets"), 0,
                         wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        bottom_sizer.Add(self.sets_spinner, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.noise_checkbox = wx.CheckBox(self.panel, wx.ID_ANY, label="Add noise")
        self.noise_checkbox.SetValue(self.add_noise)
        self.noise_checkbox.Bind(wx.EVT_CHECKBOX, self._add_noise)
        bottom_sizer.Add(self.noise_checkbox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.noise_input = NumCtrl(self.panel, wx.ID_ANY, min=0.0, value=self.snr, fractionWidth=3)
        self.noise_input.SetMin(0.0)
        self.noise_label = wx.StaticText(self.panel, label="Signal-to-Noise Ratio")
        bottom_sizer.Add(self.noise_label, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        bottom_sizer.Add(self.noise_input, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.unit_combobox = wx.ComboBox(self.panel,
                                         wx.ID_ANY,
                                         style=wx.CB_READONLY,
                                         value=self.snr_unit,
                                         choices=NOISE_UNIT)
        self.unit_combobox.Bind(wx.EVT_COMBOBOX, self._update_unit)
        bottom_sizer.Add(self.unit_combobox, 0, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self._add_noise(None)
        return [timeseries_sizer, bottom_sizer]

    def _add_noise(self, event):
        if self.noise_checkbox.GetValue():
            self.noise_label.Enable()
            self.noise_input.Enable()
            self.unit_combobox.Enable()
        else:
            self.noise_label.Disable()
            self.noise_input.Disable()
            self.unit_combobox.Disable()

    def _update_unit(self, event):
        unit = self.unit_combobox.GetValue()
        if unit != self.snr_unit:
            noise = self.noise_input.GetValue()
            noise = to_db(noise) if unit == 'dB' else db_to_linear(noise)
            self.snr_unit = unit
            self.noise_input.SetValue(noise)

    def OnBrowse(self, event):
        fpath = self.modelfiletxt.GetLabel()
        _, fname = os.path.split(fpath)
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
        params['add_noise'] = self.noise_checkbox.GetValue()
        self.noise_input.Refresh()
        if self.unit_combobox.GetValue() == 'dB':
            params['snr_db'] = self.noise_input.GetValue()
        else:
            params['snr_db'] = to_db(self.noise_input.GetValue())
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
        self.snr_db = self.params['snr_db']

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
            'snr_db': 0.0,  # 0.0 < noise < 1.0
        })
        return params

    def output_definition(self):
        return {'Table: Simulated': None}

    def run_configuration_dialog(self, parent, data_choices={}):
        dlg = AESimConfigDlg(parent,
                             f'Configuration: {self.name}',
                             input=self.input,
                             selectedgrouping=self.params['grouping'],
                             selectedfeatures=self.params['features'],
                             modelfile=self.params['modelfile'],
                             device=self.params['device'],
                             sets=self.params['sets'],
                             add_noise=self.params['add_noise'],
                             snr_db=self.params['snr_db'])
        if dlg.ShowModal() == wx.ID_OK:
            params = dlg.get_selected()
            self.params.update(params)
            return self.params
        else:
            return None

    def _model_noise(self, signal, snr_db):
        logging.debug(f'Adding {snr_db} dB white noise.')
        # Calculate signal power and convert its mean to dB
        signal_power = signal**2
        signal_avg_power = np.mean(signal_power)
        signal_avg_db = 10 * np.log10(signal_avg_power)
        # Calculate power of noise
        noise_avg_db = signal_avg_db - snr_db
        noise_avg_power = 10**(noise_avg_db / 10)

        std_noise = np.sqrt(noise_avg_power)
        mean_noise = np.zeros(std_noise.shape)
        return mean_noise, std_noise

    def execute(self):
        data = list(self.input.values())[0]
        cats = list(data.select_dtypes(['category']).columns.values)
        data_feat = data[self.params['features']]
        grouping = self.params['grouping']
        feat_cols = list(data_feat.columns)
        fc_lower = [x.lower() for x in feat_cols]

        signals = set([re.findall(r'\S+', col)[0] for col in feat_cols])
        amplitudes = {
            k: [
                col for col in feat_cols
                if len(re.findall(rf'^{re.escape(k)}\sa\d$', col, re.IGNORECASE)) > 0
            ]
            for k in signals
        }
        logging.debug(f'amplitudes={amplitudes}')

        # load an AE model
        device = self.params['device']
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            logging.info("CUDA selected, but no CUDA device available. Switching to CPU.")
        sim_df = pd.DataFrame(columns=(cats + feat_cols))
        noise_df = pd.DataFrame(columns=(cats + feat_cols))

        mean_noise, std_noise = self._model_noise(data_feat, self.params['snr_db'])
        #ae = torch.load(self.params['modelfile'], map_location = device)
        ae_pipeline = load(self.params['modelfile'])

        data_feat = data_feat.to_numpy(dtype=np.float32)

        for simset in range(0, self.params['sets']):
            if self.params['add_noise']:
                noise = np.random.normal(mean_noise, scale=std_noise, size=data_feat.shape)
                sdata_feat = data_feat.add(noise)
            else:
                sdata_feat = data_feat

            features, reconstructed = ae_pipeline.transform(sdata_feat)

            sdata_feat = torch.FloatTensor(sdata_feat)
            logging.debug(f'Sim set {simset+1}, original shape: {sdata_feat.shape}')

            #features, reconstructed = ae(data_input)
            logging.debug(f'Reconstructed shape: {reconstructed.shape}')

            features = torch.squeeze(features)
            logging.debug(f'Features shape: {features.shape}')

            criterion = nn.MSELoss()
            loss = criterion(reconstructed, sdata_feat)
            logging.debug(f'Sim set {simset+1}, loss: {loss.data}')

            recon_data = reconstructed.detach().numpy()
            scaler = ae_pipeline.named_steps['minmaxscaler']
            sim_data = scaler.inverse_transform(recon_data)
            temp = pd.DataFrame(columns=(cats + feat_cols))
            temp[cats] = data[cats]
            temp[feat_cols] = sim_data
            sim_df = pd.concat([sim_df, temp])

        calcdf = pd.DataFrame()
        for k, amps in amplitudes.items():
            total_col = f'{k} total'
            total_amp = sim_df[amps].sum(axis=1)
            for a in amps:
                calc_col = [
                    c for c in data.columns.values if (a in c and '%' in c and '/' not in c)
                ]
                calc_col = calc_col[0] if len(calc_col) > 0 else f'{a}%'
                logging.debug(f'Calculating {calc_col}.')
                calcdf[calc_col] = sim_df[a] / total_amp * 100.0

        # concat and ensure unique index
        sim_df = pd.concat([sim_df, calcdf], axis=1).reset_index()
        outfeats = feat_cols + list(calcdf.columns.values)
        outfeats.sort()  #ensure feature vectors will be applied correctly
        sim_df = sim_df[cats + outfeats]
        sim_df[grouping[-1]] = sim_df[grouping[-1]].astype(str).astype('category')

        return {
            'Table: Simulated': sim_df,
            'Table: Calculated': calcdf
        }  #, 'Table: Noise': noise_df}
