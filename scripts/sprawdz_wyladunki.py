# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:12:15 2021

@author: OW

skrypt analizuje zapelnienie skrzyni w kolejnych dniach



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
        

     
    df_temp = pd.read_csv(path_to_data + '\\' + plik, usecols = ['Data_godzina', 'polozenie_sciany_mm','zapelnienie_skrzyni_procent','motogodziny_total','wysuwanie_sciany','odwlok_otwieranie'])
    df_temp['Data_godzina'] = pd.to_datetime(df_temp['Data_godzina'])
    df_temp.loc[df_temp['zapelnienie_skrzyni_procent']<0, 'zapelnienie_skrzyni_procent'] = 0
        
    
    # przefiltrowany sygnal zapelnienia skrzyni
    df_temp['zapelnienie_skrzyni_procent'] = df_temp['zapelnienie_skrzyni_procent'].rolling(window = window_filtr_zapelenienie).mean().ffill().bfill()
    
    peaks_wyladunek, peaks_wyladunek_properites = find_peaks(-df_temp['zapelnienie_skrzyni_procent'].diff(), height=0.1, distance = window )
    peaks_sciana, peaks_sciana_properites = find_peaks(df_temp['wysuwanie_sciany'], height=1, distance = window, )
    peaks_odwlok, peaks_odwlok_properites = find_peaks(df_temp['odwlok_otwieranie'], height=1, distance = window, )
    

    fig,axs = plt.subplots(3,1,figsize = (15,8))
    axs[0].plot(df_temp['Data_godzina'], df_temp['zapelnienie_skrzyni_procent'], label = 'procent zapelnienia skrzyni', c='g')
    axs[0].plot(df_temp.loc[peaks_wyladunek, 'Data_godzina'], df_temp.loc[peaks_wyladunek, 'zapelnienie_skrzyni_procent'], label = 'procent zapelnienia skrzyni PEAK', c='r', marker='o', linewidth=0)
    
    axs[1].plot(df_temp['Data_godzina'], df_temp['wysuwanie_sciany'], label='sygnal wysuwania sciany',c='k')
    axs[1].plot(df_temp.loc[peaks_sciana, 'Data_godzina'], df_temp.loc[peaks_sciana, 'wysuwanie_sciany'], label = 'sygnal wysuwania sciany PEAK', c='r', marker='o', linewidth=0)
    
    axs[2].plot(df_temp['Data_godzina'], df_temp['odwlok_otwieranie'], label = 'sygnal otwierania odwloka',c='b')
    axs[2].plot(df_temp.loc[peaks_odwlok, 'Data_godzina'], df_temp.loc[peaks_odwlok, 'odwlok_otwieranie'], label = 'sygnal otwierania odwloka PEAK', c='r', marker='o', linewidth=0)
    axs[0].set_title(plik)
    for ax in axs:
        ax.legend()
        
    df=pd.concat((df,df_temp))
    
    
    
    
    
