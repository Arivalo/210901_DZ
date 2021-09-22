# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:56:31 2021

@author: OW

tworzy pojedyncze pliki podsumowan dniowych z pojadu na podstawie plikow z danymi z calego dnia z lokalizacji

...\data_smieciarki\20269


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

vehicle_id = '20269'

path_to_vehicle_data =r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\data_smieciarki' + '\\' + vehicle_id
path_to_save_podsumowanie = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki\data\{}\podsumowanie_dnia'.format(vehicle_id)




data_files = [filename for filename in os.listdir(path_to_vehicle_data) if (filename.startswith("data_") & filename.endswith("csv"))]

podsumowanie_files = [filename for filename in os.listdir(path_to_save_podsumowanie) if filename.startswith("summary_")]

# Oblicz podsumowanie dnia dla kazdego dnia i zapisz osobny plik xlsx


def tabela_statystyk_dnia(df):
    dane_dnia = pd.DataFrame()

    # motogodziny
    dane_dnia['Motohours total'] = [df['motogodziny_total'].max()]
    dane_dnia['Motohours idle'] = [df['motogodziny_jalowy'].max()]
    dane_dnia['Motohours 900rpm stop'] = [df['motogodziny_900rpm_zabudowa'].max()]
    dane_dnia['Motohours driving'] = [df['motogodziny_jazda'].max()]
    dane_dnia['Motohours >26t'] = [df['motogodziny_przeladowana'].max()]

    # przebieg
    dane_dnia['Distance [km]'] = [df['przebieg_km'].max()]
    dane_dnia['Distance >26t [km]'] = [df['przebieg_km_przeladowana'].max()]

    # paliwo
    dane_dnia['Fuel consumed [dm3]'] = [df['Fuel_consumption'].max()]
    if df['przebieg_km'].max() > 0:
        dane_dnia['Fuel consumption per distance [dm3/100km]'] = [df['Fuel_consumption'].max()/df['przebieg_km'].max()*100]
    else:
        dane_dnia['Fuel consumption per distance [dm3/100km]'] = 0
    if df['motogodziny_total'].max() > 0:
        dane_dnia['Hourly fuel consumption [dm3/h]'] = [df['Fuel_consumption'].max()/df['motogodziny_total'].max()]
    else:
        dane_dnia['Hourly fuel consumption [dm3/h]'] = 0 

    # energia hydrauliczna
    dane_dnia['Hydraulic energy [kJ]'] = [df['hydraulic_energy'].max()]
    dane_dnia['Compaction hydraulic energy [kJ]'] = [df['energia_hydr_zageszczania'].max()]
    
    # Nacisk na osie
    #temp_df[temp_df['RPM'] > 800]['Nacisk_total']
    dane_dnia['Max overload >26t [t]'] = max(0, df[df['RPM'].astype(int) > 800]['Nacisk_total'].max()/1000-26)

    dane_dnia = dane_dnia.T.rename(columns={0:"Selected day"})
    #dane_dnia.columns = dane_dnia.iloc[0]
    #dane_dnia = dane_dnia.iloc[1:]
    
    return dane_dnia.round(1)#.astype(str)


for plik in data_files:
    print('\n\n',plik)
    
    dzien = plik.split('_')[1]
    file_name = 'summary_' + dzien + '.csv'
    
    if ~(file_name in podsumowanie_files):
        
        df = pd.read_csv(path_to_vehicle_data + '\\' + plik, index_col = 0)
        
        
        summary = tabela_statystyk_dnia(df).T
        summary['dzien'] = dzien
        summary.to_csv(path_to_save_podsumowanie + '\\' + file_name)
        
    else:
        print(file_name, ' alreday exists')
    
    
    
    
    
    
    
    #df['Data_godzina'] = pd.to_datetime(df['Data_godzina'])