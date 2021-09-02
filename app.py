import streamlit as st
import requests
import pandas as pd
import datetime as dt
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image
from funkcje import *
import matplotlib.dates as mdates


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def download_data(url, haslo=st.secrets['password'], login=st.secrets['username'], retry=5):

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
        
    df = pd.DataFrame.from_dict(j['entities'])
    if not df.empty:
        try:
            df['longtitude'] = [x['coordinates']['x'] for x in df['_meta']]
            df['latitude'] = [y['coordinates']['y'] for y in df['_meta']]
            df.pop('_meta')   
            
        except KeyError:
            print(f'Url error: {url}')
            
        df.ffill(inplace=True)
        df['updatedAt'] = pd.to_datetime(df['updatedAt']).dt.tz_localize(None)         
            
    return przygotuj_dane(df)

def utworz_url(data_od, data_do):
    str_base = st.secrets['url']
    data_do_parted = str(data_do).split("-")
    data_jutro = dt.date(int(data_do_parted[0]), int(data_do_parted[1]), int(data_do_parted[2])) + dt.timedelta(days=1)
    str_out = f"{str_base}?from={data_od}T02:00:00Z&to={data_jutro}T02:00:00Z&monitoredId=22&limit=10000000"
    return str_out
    
def get_table_download_link(df, nazwa_pliku):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{nazwa_pliku}.csv">Download stats table</a>'
    
    return href
    
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
    dane_dnia['Fuel consumption per distance [dm3/100km]'] = [df['Fuel_consumption'].max()/df['przebieg_km'].max()*100]
    dane_dnia['Hourly fuel consumption [dm3/h]'] = [df['Fuel_consumption'].max()/df['motogodziny_total'].max()]

    # energia hydrauliczna
    dane_dnia['Hydraulic energy [kJ]'] = [df['hydraulic_energy'].max()]
    dane_dnia['Compaction hydraulic energy [kJ]'] = [df['energia_hydr_zageszczania'].max()]

    dane_dnia = dane_dnia.T.rename(columns={0:"Selected day"})
    #dane_dnia.columns = dane_dnia.iloc[0]
    #dane_dnia = dane_dnia.iloc[1:]
    
    return dane_dnia.round(1).astype(str)
    
def stworz_tabele_statystyk(df, data):
    # usuwanie kolumn z samymi 0
    df = df.loc[:, (df != 0).any(axis=0)]

    data_parted = str(data).split("-")
    data = dt.date(int(data_parted[0]), int(data_parted[1]), int(data_parted[2]))
    dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7) if str(data + dt.timedelta(days=d-data.weekday())) in df.columns]
    
    df_out = pd.DataFrame()
    
    try:
        df_out['Day'] = df[str(data)]
    except KeyError:
        df_out['Day'] = 0.0
    
    df_out['Week'] = df[dni_tydzien].median(axis=1)
    
    df_out['Month'] = df.median(axis=1)
    
    return df_out.fillna(0).round(1).astype(str)
    
######################################################################################################################

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black;'>Dashboard GPU7RM1</h1>", unsafe_allow_html=True)

c1, c2, c3 = st.columns((1,2,1))


# lokacje plikow
im_sketch = "res/sketch_lowres.png"
im_central = "res/drawing.PNG"
im_logo = "res/logo_xt.png"
im_logo2 = "res/logo_zoeller.png"
im_table = "res/tabelka.PNG"
table_stats = "data/tabela_sierpien.csv"

# wczytanie plikow
sketch = Image.open(im_sketch).convert('RGB')
central_sketch = Image.open(im_central).convert('RGB')
xt_logo = Image.open(im_logo).convert('RGB')
zoe_logo = Image.open(im_logo2).convert('RGB')
tabela_info = Image.open(im_table).convert('RGB')
df_stats = pd.read_csv(table_stats, index_col=0)


### HEADER ###

## c1
c1.title("WIP")
c1.image(sketch, use_column_width=True)
 
c1.header("Day choice:")

data_od = c1.date_input("", value=dt.date(2021,8,3), min_value=dt.date(2021,8,1), max_value=dt.date.today())
c1.write("------------------")
#data_do = c1.date_input("To:", min_value=data_od)

## c2

#c2.title("Dashboard Dashboard GPU7RM1")
c2.image(central_sketch, use_column_width=True)



## c3
#c3.title("WIP")
c3.image(xt_logo, use_column_width=True)
c3.image(zoe_logo, use_column_width=True)
c3.image(tabela_info, use_column_width=True)


### MAIN ###

url = utworz_url(data_od, data_od)

    
if str(data_od) in df_stats.columns:
    dane_z_dnia = df_stats[str(data_od)]
else:

    try:
    
        df = download_data(url)
    
        if not df.empty:
            dane_z_dnia = tabela_statystyk_dnia(df)
            
    except KeyError:
        dane_z_dnia = df_stats[df_stats.columns[0]].copy().rename({df_stats.columns[0]:"Selected day"})
        dane_z_dnia["Selected day"] = 0

c2.write(f"Statistics from {data_od} (aggregates as median, only working days):")
#c2.dataframe(dane_z_dnia, height=500)


#c3.table(dane_z_dnia)

tabela_statystyk = stworz_tabele_statystyk(df_stats, data_od)

c2.table(tabela_statystyk)
c1.markdown(get_table_download_link(tabela_statystyk, f'GPU7RM1_stats_{data_od}'), unsafe_allow_html=True)

# c2.table(tabela_statystyk)
#c3.table(tabela_statystyk)
#c2.dataframe(tabela_statystyk, height=500)

## WYKRESY ##

try:
    x = df.max()
except NameError:
    try:
        df = download_data(url)
    except KeyError:
        df = pd.DataFrame()


cols = st.columns((1,1,1))

# fig = px.line(df, x='Data_godzina', y='Nacisk_total', title="Truck weight during the day",
            # labels={'Data_godzina':"Time", "Nacisk_total":"Total weight"})

if not df.empty:
    ## WYKRESY DOLNE ##
    xfmt = mdates.DateFormatter('%H:%M')

    
    # MOTOGODZINY
    fig_p0, ax_0 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['motogodziny_total'], ls='--', lw=3, c='red', label='mth total [h]')
    plt.fill_between(df['Data_godzina'], 0, df['motogodziny_jazda'], color="green", label='mth driving [h]')
    plt.fill_between(df['Data_godzina'], df['motogodziny_jazda'], df['motogodziny_jazda']+df['motogodziny_900rpm_zabudowa'], color="orange", label='mth 900rpm stop [h]')
    plt.fill_between(df['Data_godzina'], df['motogodziny_jazda']+df['motogodziny_900rpm_zabudowa'], df['motogodziny_jazda']+df['motogodziny_900rpm_zabudowa']+df['motogodziny_jalowy'], color="blue", label='mth idle [h]')
    
    plt.ylabel("Motohours", fontsize=24)
    plt.title("Daily data:", fontsize=24)
    ax_0.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    
    # MOTOGODZINY PRZELADOWANY
    fig_p1, ax_1 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['motogodziny_total'], lw=3, label='mth total [h]')
    plt.plot(df['Data_godzina'], df['motogodziny_przeladowana'], lw=3, label='mth overload [h]')
    plt.ylabel("Motohours overloaded", fontsize=24)
    ax_1.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    
    # PRZEBIEG + PRZELADOWANY
    fig_p2, ax_2 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['przebieg_km'], lw=3, label='Total distance [km]')
    plt.plot(df['Data_godzina'], df['przebieg_km_przeladowana'], lw=3, label='Distance with overload >26t [km]')
    plt.ylabel("Distance", fontsize=24)
    ax_2.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.tight_layout()
    plt.legend()

    #c2.plotly_chart(fig)

    # wykresy
    cols[0].write(fig_p0)
    cols[0].write(fig_p1)
    cols[0].write(fig_p2)
    
    # temp
    fig_q0 = plt.figure(figsize=(8,5))
    plt.title("Weekly data (PLACEHOLDER)", fontsize=24)
    plt.bar([1,2,3], [2,3,1])
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    cols[1].write(fig_q0)
    
    fig_r0 = plt.figure(figsize=(8,5))
    plt.title("Normal distribution (PLACEHOLDER)", fontsize=24)
    plt.bar([1,2,3, 4, 5], [1,2,4,2,1])
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    cols[2].write(fig_r0)
    
    
    #cols[2].plotly_chart(fig)

    

