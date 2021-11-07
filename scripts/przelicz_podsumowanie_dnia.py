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
    dane_dnia['Hydraulic energy [GJ]'] = [df['hydraulic_energy'].max()/1000]
    dane_dnia['Compaction hydraulic energy [GJ]'] = [df['energia_hydr_zageszczania'].max()/1000]
    
    # przepompowany olej
    dane_dnia['Hydraulic oil [m3]'] = [df['ilosc_przepompowanego_oleju'].max()]
    dane_dnia['Hydraulic oil >120 bar [m3]'] = [df['ilosc_przepompowanego_oleju_120bar'].max()]
    
    # Nacisk na osie
    #temp_df[temp_df['RPM'] > 800]['Nacisk_total']
    dane_dnia['Max overload >26t [t]'] = max(0, df[df['RPM'].astype(int) > 800]['Nacisk_total'].max()/1000-26)
    
    # Masa smieci
    dane_dnia['Waste mass [t]'] = [df['Masa_smieci'].max()/1000]
    
    # Tonokilometry
    dane_dnia['Waste mass x kilometers [t*km]'] = [df['Tonokilometry_masa_smieci'].max()]
    dane_dnia['Vehicle overload x kilometers [t*km]'] = [df['Tonokilometry_przeladowane'].max()]    
    
    # Zapelnienie skrzyni (w momencie maks nacisku na osie)
    dane_dnia['Body capacity used [%]'] = [df.loc[df['Nacisk_total'].argmax(), 'zapelnienie_skrzyni_procent']]

    # energia na tone smieci
    dane_dnia['Hydraulic energy per 1 t of waste [GJ/t]'] = np.round(df['hydraulic_energy'].max() / (df['Masa_smieci'].max()/1000)/1000,2)
    dane_dnia['Compaction hydraulic energy per 1 t of waste [GJ/t]'] = np.round(df['energia_hydr_zageszczania'].max() / (df['Masa_smieci'].max()/1000)/1000,2)
    
    # srednia moc ukladu hydraulicznego w czasie pracy zabudowy (rpm=900)
    dane_dnia['Average power of hydraulic system during body operation [kW]'] = df.loc[(df['RPM']>800) & (df['predkosc_kol']<2),'hydrualic_power'].mean()
    
    # liczba cykli prasy
    dane_dnia['Total # of press cycles'] = df['cykle_zageszczania'].max()
    dane_dnia['Total # of press cycles - low forces'] = df.loc[df['cykle_lekkie'] == 1,'cykle_zageszczania'].nunique()
    dane_dnia['Total # of press cycles - medium forces'] = df.loc[df['cykle_srednie'] == 1,'cykle_zageszczania'].nunique()
    dane_dnia['Total # of press cycles - high forces'] = df.loc[df['cykle_ciezkie'] == 1,'cykle_zageszczania'].nunique()


    dane_dnia = dane_dnia.T.rename(columns={0:"Selected day"})
    #dane_dnia.columns = dane_dnia.iloc[0]
    #dane_dnia = dane_dnia.iloc[1:]
    
    # Parametry akumulowane
    # tbc
    
    
    
    
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