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


@st.cache(suppress_st_warning=True)
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
    csv = df.to_csv(header=False)
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
    dane_dnia['Przebieg [km]'] = [df['przebieg_km'].max()]
    dane_dnia['Przebieg >26t [km]'] = [df['przebieg_km_przeladowana'].max()]

    # paliwo
    dane_dnia['Fuel consumption [dm3]'] = [df['Fuel_consumption'].max()]
    dane_dnia['Fuel consumption per distance [dm3/100km]'] = [df['Fuel_consumption'].max()/df['przebieg_km'].max()*100]
    dane_dnia['Hourly fuel consumption [dm3/h]'] = [df['Fuel_consumption'].max()/df['motogodziny_total'].max()]

    # energia hydrauliczna
    dane_dnia['Hydraulic energy [kJ]'] = [df['hydraulic_energy'].max()]
    dane_dnia['Compaction hydraulic energy [kJ]'] = [df['energia_hydr_zageszczania'].max()]

    dane_dnia = dane_dnia.T.rename(columns={0:"Selected day"})
    #dane_dnia.columns = dane_dnia.iloc[0]
    #dane_dnia = dane_dnia.iloc[1:]
    
    return dane_dnia.round(1).astype(str)
    
######################################################################################################################

st.set_page_config(layout="wide")

c1, c2, c3 = st.columns((1,2,1))

im_sketch = "res/sketch_lowres.png"
im_central = "res/drawing.PNG"
im_logo = "res/logo_xt.png"
im_logo2 = "res/logo_zoeller.png"
im_table = "res/tabelka.png"

sketch = Image.open(im_sketch).convert('RGB')
central_sketch = Image.open(im_central).convert('RGB')
xt_logo = Image.open(im_logo).convert('RGB')
zoe_logo = Image.open(im_logo2).convert('RGB')
tabela_info = Image.open(im_table).convert('RGB')

### HEADER ###

## c1
c1.title("WIP")
c1.image(sketch, use_column_width=True)
 
c1.header("Day choice:")

data_od = c1.date_input("", value=dt.date(2021,8,3), min_value=dt.date(2021,8,1), max_value=dt.date.today())
c1.write("------------------")
#data_do = c1.date_input("To:", min_value=data_od)

## c2
c2.markdown("<h1 style='text-align: center; color: black;'>Dashboard GPU7RM1</h1>", unsafe_allow_html=True)
#c2.title("Dashboard Dashboard GPU7RM1")
c2.image(central_sketch, use_column_width=True)



## c3
#c3.title("WIP")
c3.image(xt_logo, use_column_width=True)
c3.image(zoe_logo, use_column_width=True)
c3.image(tabela_info, use_column_width=True)


### MAIN ###

url = utworz_url(data_od, data_od)

try:

    df = download_data(url)

    dane_z_dnia = tabela_statystyk_dnia(df)
    
    c2.write(f"Statistics from {data_od}:")
    #c2.dataframe(dane_z_dnia, height=500)
    
   
    c3.table(dane_z_dnia)
    c1.markdown(get_table_download_link(dane_z_dnia, f'GPU7RM1_stats_{data_od}'), unsafe_allow_html=True)
    
    
    ## WYKRESY ##

    
    cols = st.columns((1,1,1))

    fig = px.line(df, x='Data_godzina', y='Nacisk_total', title="Truck weight during the day",
                labels={'Data_godzina':"Time", "Nacisk_total":"Total weight"})
    
    
    ## WYKRESY DOLNE ##
    xfmt = mdates.DateFormatter('%H:%M')
    
    fig_p0, ax_0 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['Fuel_consumption'])
    plt.title("Fuel consumption", fontsize=24)
    ax_0.xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    fig_p1, ax_1 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['predkosc_osi'])
    plt.title("Velocity", fontsize=24)
    ax_1.xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    fig_p2, ax_2 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['motogodziny_total'])
    plt.title("Motohours during the day", fontsize=24)
    ax_2.xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    c2.plotly_chart(fig)

    cols[0].write(fig_p0)
    cols[1].write(fig_p1)
    cols[2].write(fig_p2)
    #cols[2].plotly_chart(fig)
    
    
except KeyError:
    
    st.write("No data for selected date")
    

