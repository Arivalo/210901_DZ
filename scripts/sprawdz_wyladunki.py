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


nr_przejazdu = 1
for plik in data_files:
    print('\n\n',plik)
    
    

     
    df_temp = pd.read_csv(path_to_data + '\\' + plik, usecols = ['Data_godzina', 'polozenie_sciany_mm','zapelnienie_skrzyni_procent','motogodziny_total','wysuwanie_sciany','odwlok_otwieranie','przebieg_km','Masa_smieci','cisnienie_bar'])
    df_temp['Data_godzina'] = pd.to_datetime(df_temp['Data_godzina'])
    df_temp.loc[df_temp['zapelnienie_skrzyni_procent']<0, 'zapelnienie_skrzyni_procent'] = 0
        
    
    # przefiltrowany sygnal zapelnienia skrzyni
    df_temp['zapelnienie_skrzyni_procent'] = df_temp['zapelnienie_skrzyni_procent'].rolling(window = window_filtr_zapelenienie).mean().ffill().bfill()
    
    peaks_wyladunek, peaks_wyladunek_properites = find_peaks(-df_temp['zapelnienie_skrzyni_procent'].diff(), height=0.1, distance = window )
    peaks_sciana, peaks_sciana_properites = find_peaks(df_temp['wysuwanie_sciany'], height=1, distance = window, )
    peaks_odwlok, peaks_odwlok_properites = find_peaks(df_temp['odwlok_otwieranie'], height=1, distance = window, )
    
    df_temp['nr_przejazdu_z_zaladunkiem'] = np.nan
    
    for index_wyladunku in peaks_wyladunek[::-1]:
        
        df_temp.loc[df_temp.index < index_wyladunku-60, 'nr_przejazdu_z_zaladunkiem'] = nr_przejazdu
        
        nr_przejazdu += 1
    
    
    
    # fig,axs = plt.subplots(3,1,figsize = (15,8))
    # axs[0].plot(df_temp['Data_godzina'], df_temp['zapelnienie_skrzyni_procent'], label = 'procent zapelnienia skrzyni', c='g')
    # axs[0].plot(df_temp.loc[peaks_wyladunek, 'Data_godzina'], df_temp.loc[peaks_wyladunek, 'zapelnienie_skrzyni_procent'], label = 'procent zapelnienia skrzyni PEAK', c='r', marker='o', linewidth=0)
    
    # axs[1].plot(df_temp['Data_godzina'], df_temp['wysuwanie_sciany'], label='sygnal wysuwania sciany',c='k')
    # axs[1].plot(df_temp.loc[peaks_sciana, 'Data_godzina'], df_temp.loc[peaks_sciana, 'wysuwanie_sciany'], label = 'sygnal wysuwania sciany PEAK', c='r', marker='o', linewidth=0)
    
    # axs[2].plot(df_temp['Data_godzina'], df_temp['odwlok_otwieranie'], label = 'sygnal otwierania odwloka',c='b')
    # axs[2].plot(df_temp.loc[peaks_odwlok, 'Data_godzina'], df_temp.loc[peaks_odwlok, 'odwlok_otwieranie'], label = 'sygnal otwierania odwloka PEAK', c='r', marker='o', linewidth=0)
    # axs[0].set_title(plik)
    # for ax in axs:
    #     ax.legend()
        
    df=pd.concat((df,df_temp))
    
    

df['nr_przejazdu_z_zaladunkiem'].unique()

df['Day'] = df['Data_godzina'].dt.date
df['Czas'] = df['Data_godzina'].dt.time

df['Day'].nunique()

import seaborn as sns
#sns.relplot(data = df, x = 'Czas', y = 'zapelnienie_skrzyni_procent', hue='Day')


fig,ax=plt.subplots(figsize = (8,5))        
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    ax.plot(df_temp['Czas'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Time')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100]) 
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\rekomendowanie_przesuniecia_sciany\zapelnienie_vs_time.png') 
    
    
    

fig,ax=plt.subplots(figsize = (8,5))        
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    df_temp['Seconds'] = (df_temp['Data_godzina']  - df_temp['Data_godzina'].min()) / np.timedelta64(1, 's')
    df_temp['Relative_time'] = df_temp['Seconds'] / df_temp['Seconds'].max() * 100

    ax.plot(df_temp['Relative_time'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Relative time to discharge [%]')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100]) 
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\rekomendowanie_przesuniecia_sciany\zapelnienie_vs_rel_time.png') 
    
   
    
    

fig,ax=plt.subplots(figsize = (8,5))       
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    
    df_temp['Relative_distance'] = (df_temp['przebieg_km']- df_temp['przebieg_km'].min()) / (df_temp['przebieg_km'].max() - df_temp['przebieg_km'].min()) * 100
    

    ax.plot(df_temp['Relative_distance'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Relative distance to discharge [%]')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100]) 
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\rekomendowanie_przesuniecia_sciany\zapelnienie_vs_rel_dist.png') 
    


fig,ax=plt.subplots(figsize = (8,5))        
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    
    df_temp['Relative_distance'] = (df_temp['przebieg_km']- df_temp['przebieg_km'].min()) / (df_temp['przebieg_km'].max() - df_temp['przebieg_km'].min()) * 100
    
    df_temp['Masa_smieci']=df_temp['Masa_smieci'].rolling(601).mean()

    ax.plot(df_temp['Relative_distance'], df_temp['Masa_smieci'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Relative distance to discharge [%]')   
ax.set_ylabel('Waste mass [kg]')    
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\rekomendowanie_przesuniecia_sciany\masa_vs_rel_dist.png')
#ax.set_ylim([0,100]) 



fig,ax=plt.subplots(figsize = (8,5))        
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    
    df_temp['Seconds'] = (df_temp['Data_godzina']  - df_temp['Data_godzina'].min()) / np.timedelta64(1, 's')
    df_temp['Relative_time'] = (df_temp['Seconds']-df_temp['Seconds'].min()) / (df_temp['Seconds'].max()-df_temp['Seconds'].min()) * 100
    
    df_temp['Masa_smieci']=df_temp['Masa_smieci'].rolling(601).mean().ffill().bfill()

    ax.plot(df_temp['Relative_time'], df_temp['Masa_smieci'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Relative time to discharge [%]')   
ax.set_ylabel('Waste mass [kg]')    
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\rekomendowanie_przesuniecia_sciany\masa_vs_rel_time.png')
#ax.set_ylim([0,100]) 







fig,ax=plt.subplots(figsize = (8,5))       
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())-25):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    
    df_temp['Seconds'] = (df_temp['Data_godzina']  - df_temp['Data_godzina'].min()) / np.timedelta64(1, 's')
    df_temp['Relative_time'] = (df_temp['Seconds']-df_temp['Seconds'].min()) / (df_temp['Seconds'].max()-df_temp['Seconds'].min()) * 100
    
    df_temp['Masa_smieci']=df_temp['Masa_smieci'].rolling(601).mean().ffill().bfill()

    ax.plot(df_temp['Czas'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Time')
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100])
plt.tight_layout()




fig,ax=plt.subplots(figsize = (8,5))    
pd.plotting.register_matplotlib_converters()
for nr_przejazdu in range(int(df['nr_przejazdu_z_zaladunkiem'].max())-25):
    
    df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu+1]
    
    
    df_temp['Masa_smieci']=df_temp['Masa_smieci'].rolling(601).mean().ffill().bfill()

    ax.plot(df_temp['przebieg_km']-df_temp['przebieg_km'].min(), df_temp['zapelnienie_skrzyni_procent'],
            alpha=0.5,
            linewidth=3,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Distance to discharge [lm]')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100])
plt.tight_layout()





# Analiza pojedynczego dnia: zapelenienie vs cisnienie w prasie

nr_przejazdu = 12
df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu]


fig,axs=plt.subplots(3,1,figsize = (12,8))  
ax=axs[0]
ax.plot(df_temp['Czas'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Time')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100])
plt.tight_layout()

ax=axs[1]
ax.plot(df_temp['Czas'], df_temp['Masa_smieci'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))

ax.set_xlabel('Time')   
ax.set_ylabel('Waste mass kg')
ax.set_ylim([0,20000])
plt.tight_layout()


ax=axs[2]
ax.plot(df_temp['Czas'], df_temp['cisnienie_bar'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))

ax.set_xlabel('Time')   
ax.set_ylabel('Pressure [bar]')
ax.set_ylim([0,230])
plt.tight_layout()





# Analiza pojedynczego dnia: zapelenienie vs cisnienie w prasie

nr_przejazdu = 12
df_temp = df[df['nr_przejazdu_z_zaladunkiem']==nr_przejazdu]


fig,axs=plt.subplots(3,1,figsize = (12,8))  
ax=axs[0]
ax.plot(df_temp['Czas'], df_temp['zapelnienie_skrzyni_procent'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))


#ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05))
ax.set_xlabel('Time')   
ax.set_ylabel('Body capacity used [%]')    
ax.set_ylim([0,100])
plt.tight_layout()

ax=axs[1]
ax.plot(df_temp['Czas'], df_temp['Masa_smieci'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))

ax.set_xlabel('Time')   
ax.set_ylabel('Waste mass kg')
ax.set_ylim([0,20000])
plt.tight_layout()


ax=axs[2]
ax.plot(df_temp['Czas'], df_temp['cisnienie_bar'],
            alpha=1,
            linewidth=1,
            label = str(df_temp.reset_index().loc[0, 'Day']))

ax.set_xlabel('Time')   
ax.set_ylabel('Pressure [bar]')
ax.set_ylim([0,230])
plt.tight_layout()





