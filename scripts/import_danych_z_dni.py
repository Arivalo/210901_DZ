# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:31:18 2021

@author: OW


Skrypt do sciagania danych z danego dnia z servera xtrack 2139 i zapisu przeliczonych danych dniowych do folderu na Google Drive
Tworzy duże pliki z odczytami co 1 s.



"""
import requests
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

####################################################################################################################################################################################
####################################################################################################################################################################################



# Funkcje do pobrania danych
def utworz_url(nr_mon, od_k, do_k, godz_od, godz_do):

    url_string = f"https://2139.xtrack.com/rest/api/source-sets/archive-events?from={od_kiedy}T{od_kiedy_godzina}:00Z&"\
                 f"to={do_kiedy}T{do_kiedy_godzina}:00Z&monitoredId={nr_monitorowany}&limit=5000000"

    return url_string
def pobierz_dane(url, haslo='', login='wysocki;2139'):
    '''
    Pobiera dane z serwera Xtrack wedlug danego linku "url" i wyznacza kolumny z danymi GPS
    
    Wartosci wejsciowe:
    
    url <String> - pelny link z zapytaniem
    haslo <String> - haslo do autentykacji BASIC
    login <String> - (default: "wysocki;2139") login do autentykacji BASIC
    
    Wyjscie:
    
    <pandas.DataFrame> - otrzymany z zapytania DataFrame z danymi
    
    '''

    r = requests.get(url,auth=(login, haslo))

    j = r.json()
    
    try:
        df = pd.DataFrame.from_dict(j['entities'])
    except KeyError:
        print(j)
        print(".\n.\n.\nBlad wczytywania")

    if not df.empty:
        try:
            df['longtitude'] = [x['coordinates']['x'] for x in df['_meta']]
            df['latitude'] = [y['coordinates']['y'] for y in df['_meta']]
            df.pop('_meta')   
        except KeyError:
            print(f'Problem z url: {url}')

    
    return df
def pobierz_i_wstepnie_przygotuj_dane(haslo, nr_mon, od_k, do_k, godz_od, godz_do):
    '''
    Pobiera i przygotowuje dane - wypelnia NaN poprzednimi wartosciami, reszte zeruje
    '''
    df = pobierz_dane(utworz_url(nr_mon, od_k, do_k, godz_od, godz_do), haslo=haslo)
    df['updatedAt'] = pd.to_datetime(df['updatedAt'])
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    #df.dropna(inplace=True)
    return df


####################################################################################################################################################################################
####################################################################################################################################################################################



# CONSTANTS
path_to_save_csv = r'C:\Users\d07321ow\Google Drive\GUT\Zoeller\pomiary_na_smieciarce\data_smieciarki\20269'

nr_monitorowany = 22 # 20269 śmieciarka
#nr_monitorowany = 23 # DEMO SUEZ GPU 7RM1


od_kiedy_godzina = "05:00" # format HH-mm string
do_kiedy_godzina = "20:00" # format HH-mm string



dni = [#'2021-09-01',
       '2021-09-02',
       '2021-09-09',
       '2021-09-10',
       '2021-09-17',
       '2021-09-20',
       '2021-09-22',
       ]

dni = [#'2021-08-16',
#        '2021-08-17',
#        '2021-08-18',
#        '2021-08-19',
#        '2021-08-20',
#        '2021-08-21',
#        '2021-08-23',
#        '2021-08-24',
#        '2021-08-25',
#        '2021-08-26',
       '2021-08-27',
       '2021-08-31',
       ]

# Pobierz dane
    
    
for dzien in dni:
    print('\n\n', dzien)
    od_kiedy = dzien # format YYYY-MM-dd string
    do_kiedy = dzien # format YYYY-MM-dd string
    
    ####################################################################################################################################################################################
    ####################################################################################################################################################################################
    
    

    df = pobierz_i_wstepnie_przygotuj_dane(nr_mon=nr_monitorowany, od_k=od_kiedy, do_k=do_kiedy, godz_od=od_kiedy_godzina, godz_do=do_kiedy_godzina, haslo='oskwys')
    
    
    df.dropna(subset = ['updatedAt'],axis=0)
    
    # przelicz dane wg funkcji
    df = f.przygotuj_dane(df)
    
    
    # usun znacznik strefy czasowej
    df['Data_godzina'] = df['Data_godzina'].dt.tz_localize(None)
    
    
    
    # zapisz do pliku xlsx
    df.round(2).to_csv(path_to_save_csv+ '\data_{}_{}.csv'.format(od_kiedy, do_kiedy))
    print('\nZapisano dzien {}    do {}'.format(dzien, path_to_save_csv))    




























