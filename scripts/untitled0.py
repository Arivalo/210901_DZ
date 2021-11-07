# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:54:11 2021

@author: d07321ow
"""
import requests


def pobierz_dane(url, haslo='oskwys', login='wysocki;2139', retry=5):
    '''
    Pobiera dane z serwera Xtrack wedlug danego linku "url" i wyznacza kolumny z danymi GPS
    
    Wartosci wejsciowe:
    
    url <String> - pelny link z zapytaniem
    haslo <String> - haslo do autentykacji BASIC
    login <String> - (default: "wysocki;2139") login do autentykacji BASIC
    retry <number< - (default: 5) liczba powtorzen prob pobierania w przypadku bledu
    
    Wyjscie:
    
    <pandas.DataFrame> - otrzymany z zapytania DataFrame z danymi
    
    '''

    i = 0

    while i < retry:
        r = requests.get(url,auth=(login, haslo))

        try:
            j = r.json()
            break
        except:
            i += 1
            print(f"Try no.{i} failed")

    if i == retry:
        print(f"Failed to fetch data for: {url}")
        return pd.DataFrame()
        
            
    return df


url = 'https://2139.xtrack.com/rest/api/monitored/movies/270c291d0c293aa3/frames/10?monitoredId=616ceb4326d308a091824866&paramName=MOV4&context=ARCHIVE&maxWidth=400&maxHeight=400'
url ='https://2139.xtrack.com/rest/api/monitored/movies/270c26fc132504c9/frames/9?monitoredId=616ceb4326d308a091824866&paramName=MOV1&context=ARCHIVE&maxWidth=400&maxHeight=400'

url ='https://2139.xtrack.com/aaa'


haslo='oskwys'
login='wysocki;2139'

a = requests.get(url,auth=(login, haslo))
print(a)
