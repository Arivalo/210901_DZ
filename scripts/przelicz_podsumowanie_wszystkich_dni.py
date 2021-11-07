# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:33:26 2021

@author: OW

skrypt tworzy plik z tabela zawierajaca dane z podsumowan pojedynczych dni



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

path_to_save_podsumowanie = r'C:\Users\d07321ow\Google Drive\GUT\PM_v1\smieciarki\data\{}\podsumowanie_dnia'.format(vehicle_id)

podsumowanie_files = [filename for filename in os.listdir(path_to_save_podsumowanie) if filename.startswith("summary_")]

file_name = 'all_days_summary.csv'
    
summary = pd.DataFrame()

for plik in podsumowanie_files:
    print('\n\n',plik)
        

     
    df_temp = pd.read_csv(path_to_save_podsumowanie + '\\' + plik)
    
    summary = pd.concat((summary, df_temp))
    
    summary['Waste mass [t] cumulative'] = summary['Waste mass [t]'].cumsum()
    
    summary['Total # of press cycles cumulative'] = summary['Total # of press cycles'].cumsum()
    summary['Total # of press cycles low forces cumulative'] = summary['Total # of press cycles - low forces'].cumsum()
    summary['Total # of press cycles medium forces cumulative'] = summary['Total # of press cycles - medium forces'].cumsum()
    summary['Total # of press cycles high forces cumulative'] = summary['Total # of press cycles - high forces'].cumsum()
    
    

summary = summary.reset_index(drop=True)
summary.to_csv(path_to_save_podsumowanie + '\\' + file_name)
summary.to_excel(path_to_save_podsumowanie + '\\' + file_name.split('.')[0] + '.xlsx')
