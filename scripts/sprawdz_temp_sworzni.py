# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:12:15 2021

@author: OW

skrypt analizuje temperatury sworzni w kolejnych dniach



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

module_path = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import funkcje as f
import importlib
importlib.reload(f)


from os import listdir
from os.path import isfile, join

from scipy.signal import find_peaks

vehicle_id = '20269'

path_to_data = r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\data_smieciarki\{}'.format(vehicle_id)

data_files = [filename for filename in os.listdir(path_to_data) if filename.endswith(".csv")]

#file_name = 'all_days_summary.csv'
    
df = pd.DataFrame()



window_filtr_zapelenienie = 1 * 60 # sekundy
windom_minuty = 15    # szukanie peakow w oknie czasowym X minutowym
window = windom_minuty * 60



for plik in data_files:
    print('\n\n',plik)
        

     
    df_temp = pd.read_csv(path_to_data + '\\' + plik, usecols = ['Data_godzina', 'temperatura_zewn','temperatura_IN12','temperatura_IN14', 'przebieg_km'])
    df_temp['Data_godzina'] = pd.to_datetime(df_temp['Data_godzina'])
    

    fig,axs = plt.subplots(3,1,figsize = (15,8))
    axs[0].plot(df_temp['Data_godzina'], df_temp['temperatura_zewn'], label = 'ambient temp', c='r')
    axs[0].plot(df_temp['Data_godzina'], df_temp['temperatura_IN12'], label = 'temperature PIN 1', c='b')
    axs[0].plot(df_temp['Data_godzina'], df_temp['temperatura_IN14'], label = 'temperature PIN 2', c='g')
 
    axs[1].plot(df_temp['Data_godzina'], df_temp['temperatura_zewn']- df_temp['temperatura_IN12'], label = 'delta T PIN 1 ', c='b')
    axs[1].plot(df_temp['Data_godzina'], df_temp['temperatura_zewn'] - df_temp['temperatura_IN14'], label = 'delta T PIN 2', c='g')
    
 
    axs[2].plot(df_temp['przebieg_km'], df_temp['temperatura_zewn'], label = 'ambient temp', c='r')
    axs[2].plot(df_temp['przebieg_km'], df_temp['temperatura_IN12'], label = 'temperature PIN 1', c='b')
    axs[2].plot(df_temp['przebieg_km'], df_temp['temperatura_IN14'], label = 'temperature PIN 2', c='g')
    
    for ax in axs:
        ax.legend()
        
    df=pd.concat((df,df_temp))
    
    
    
    
    
