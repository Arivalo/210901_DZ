# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:23:10 2021

@author: d07321ow
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


# %% PARAMETRY

okno_czasowe = 60 # min
vehicle_id = '20269'

path_to_vehicle_data =r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\data_smieciarki' + '\\' + vehicle_id
path_to_save_okna = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki\data\{}\podsumowanie_dnia'.format(vehicle_id)

# tbc

onlyfiles = [f for f in listdir(path_to_vehicle_data) if isfile(join(path_to_vehicle_data, f))]

data_files = [f for f in onlyfiles if f.startswith('data_')]



# %% Wczytaj dane dniowe
df_loaded = pd.DataFrame()

for plik in data_files:
    
    df_temp = pd.read_csv(r'{}\{}'.format(path_to_vehicle_data, plik), index_col = 0)

    df_loaded = pd.concat((df_loaded,df_temp))
    print(plik, '  --  loaded')
    
df_loaded['Data_godzina'] = pd.to_datetime(df_loaded['Data_godzina'])
df_loaded=df_loaded.reset_index(drop=True)


df = df_loaded.copy()


df['polozenie_sciany_mm'].plot()
plt.axhline(4250)

df.loc[df['polozenie_sciany_mm']>4250, 'polozenie_sciany_mm'] = 4250

# %% Aplikuj okno czasowe


df['Data_godzina_rounded'] = df['Data_godzina'].dt.round(freq = '{}min'.format(okno_czasowe))


# %% Przyrosty w oknie czasowym

df_deltas = df.groupby('Data_godzina_rounded').last() - df.groupby('Data_godzina_rounded').first()


# %% Definicja kolumn z przyrostami

kolumny_deltas = ['temperatura_IN12', 'temperatura_IN14',
                  'temperatura_zewn',
                  'przebieg_km',
                  'motogodziny_jalowy',
                  'motogodziny_900rpm_zabudowa',
                  'motogodziny_jazda',
                  
                  'polozenie_sciany_mm',
                  'hydraulic_energy', 
                  'energia_hydr_zageszczania',
                  'Masa_smieci', 
                  'cykle_zageszczania_100',
        'cykle_zageszczania_150',
        'cykle_zageszczania_200',
                  
                  
                 ]

df_deltas = df_deltas[kolumny_deltas]


# %% Wspolczynniki wzg okna czasowego

df_deltas_ratios = (df_deltas / okno_czasowe *60).round(3)


# %% Filtruj max przesuniecie sciany
max_przesuniecie_w_oknie_czasowym = 2000 # mm

index_ = (df_deltas_ratios['polozenie_sciany_mm'].abs()<max_przesuniecie_w_oknie_czasowym) & (df_deltas_ratios['polozenie_sciany_mm'] < 100)

df_deltas_ratios = df_deltas_ratios[index_]


df_deltas_ratios['temp_bezw_IN12_first'] = df.groupby('Data_godzina_rounded').first()['temperatura_IN12'][index_]

df_deltas_ratios['temp_bezw_IN14_first'] = df.groupby('Data_godzina_rounded').first()['temperatura_IN14'][index_]


df_deltas_ratios[df_deltas_ratios==-np.inf] = 0
df_deltas_ratios[df_deltas_ratios==np.inf] = 0

df_deltas_ratios['polozenie_sciany_mm'].plot()

# %% ZAPISZ DANE

df_deltas_ratios.to_excel(r'{}\df_deltas_ratios_{}_{}_okno_{}min.xlsx'.format(path_to_save_okna, data_files[0].split('.')[0].split('_')[1], data_files[-1].split('.')[0].split('_')[1], okno_czasowe))


print('SAVED TO {}'.format(path_to_save_okna))



plt.plot(df_deltas_ratios['polozenie_sciany_mm'])




