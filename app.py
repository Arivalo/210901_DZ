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

## TO DO ##
# tabela
# - oddzielenia tematyczne w tabeli - mth, dist, itp
# - % udzialy np. mth idle w total
# inne
# - wskaźnik obciążenia - przeciążenie over 26t
# - wymuszanie jasnego motywu
# - wykresy dla przeciążeń - dodanie nowego parametru


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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def wczytaj_statystyki_csv(loc):

    df = pd.read_csv(loc, index_col=0)
    
    return df
    

def stworz_tabele_statystyk(df, data):
    # usuwanie kolumn z samymi 0
    df = df.loc[:, (df != 0).any(axis=0)]

    data_parted = str(data).split("-")
    data = dt.date(int(data_parted[0]), int(data_parted[1]), int(data_parted[2]))
    dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7) if str(data + dt.timedelta(days=d-data.weekday())) in df.columns]
    
    df_out = pd.DataFrame()
    
    try:
        df_out['selected day'] = df[str(data)]
    except KeyError:
        df_out['selected day'] = 0.0
    
    df_out['avg. week'] = df[dni_tydzien].mean(axis=1)
    
    df_out['avg. month'] = df.mean(axis=1)
    
    df_out['total'] = df.sum(axis=1)
    
    df_out = df_out.fillna(0).round(1)
    
    # usuwanie statystyk gdzie total nie ma sensu
    df_out['total'].T['Fuel consumption per distance [dm3/100km]'] = "-"
    df_out['total'].T['Hourly fuel consumption [dm3/h]'] = "-"
    
    ## %%
    # daily
    mth_total = df_out['selected day'].T['Motohours total']
    distance_total = df_out['selected day'].T['Distance [km]']
    hyd_en_total = df_out['selected day'].T['Hydraulic energy [kJ]']
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1]
    
    percentage_daily = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['selected day'].values, dividers)]
    
    df_out["%"] = percentage_daily
    
    # weekly
    mth_total = df_out['avg. week'].T['Motohours total']
    distance_total = df_out['avg. week'].T['Distance [km]']
    hyd_en_total = df_out['avg. week'].T['Hydraulic energy [kJ]']
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1]
    
    percentage_weekly = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['avg. week'].values, dividers)]
    
    df_out["% "] = percentage_weekly
    
    # monthly
    mth_total = df_out['avg. month'].T['Motohours total']
    distance_total = df_out['avg. month'].T['Distance [km]']
    hyd_en_total = df_out['avg. month'].T['Hydraulic energy [kJ]']
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1]
    
    percentage_monthly = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['avg. month'].values, dividers)]
    
    df_out[" %"] = percentage_monthly
    
    # fake statystyki
    df_out.loc['Average of rpm engine per day'] = "-"
    df_out.loc['Leakage cylinder detection'] = "-"
    df_out.loc['Time with oil over 60°C [h]'] = "-"
    
    df_out = df_out[['selected day', '%', 'avg. week', '% ', 'avg. month', ' %', 'total']]
    
    return df_out.astype(str)
    
    
def wykres_z_tygodnia(df, data, lista_kolumn, lista_etykiet, title=""):
    
    fig, ax = plt.subplots(1, figsize=(8,5))
    
    colors = ['green', 'orange', 'blue', 'yellow', 'magenta', 'purple', 'cyan']
    
    dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7)]
    
    previous = [0 for x in range(7)]
    
    ax.bar(data, df[str(data)].T["Motohours total"].max()+0.18, width=1, color="yellow", label="Selected day",
            alpha=0.6)
    
    for kolumna, label, color in zip(lista_kolumn, lista_etykiet, colors):
    
        temp_y = df[dni_tydzien].T[kolumna]
        ax.bar(dni_tydzien,  temp_y, width=0.67,  label=label, bottom=previous, color=color)
        previous = [x+y for x,y in zip(previous, temp_y)]
          
    ax.legend()
    plt.title(title, fontsize=24)
    plt.grid()
    plt.tight_layout()
    
    return fig
    
def wykres_z_tygodnia2(df, data, lista_kolumn, lista_etykiet, title=""):
    
    fig, ax = plt.subplots(1, figsize=(8,5))
    
    dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7)]
    
    ax.bar(data, df[str(data)].T[lista_kolumn[0]].max()*1.1, width=1, color="yellow", label="Selected day",
            alpha=0.6)
    
    for kolumna, label in zip(lista_kolumn, lista_etykiet):
    
        temp_y = df[dni_tydzien].T[kolumna]
        ax.bar(dni_tydzien,  temp_y, width=0.67,  label=label)
          
    ax.legend()
    plt.title(title, fontsize=24)
    plt.grid()
    plt.tight_layout()
    
    return fig
    
    
def tabela_statystyk_wyswietl(df):

    df = df.reset_index().rename(columns={'index':""})
    
    df[""] = [f"<b>{val}</b>" for val in df[""]]
    
    df["selected day"] = [f"{val} <sub>{percent}</sub>" if percent!="-"  else val for val, percent in zip(df["selected day"], df["%"])]
    
    df["avg. week"] = [f"{val} <sub>{percent}</sub>" if percent!="-"  else val for val, percent in zip(df["avg. week"], df["% "])]
    
    df["avg. month"] = [f"{val} <sub>{percent}</sub>" if percent!="-"  else val for val, percent in zip(df["avg. month"], df[" %"])]
    
    df = df[["", "selected day", "avg. week", "avg. month", "total"]]
    
    fig = go.Figure(data=[go.Table(
    columnwidth = [200, 100, 100, 100, 90],
    header=dict(values=list([f"<b>{col}</b>" for col in df.columns]),
                fill_color='gray',
                line_color='darkslategray',
                align=["left", 'center', 'center', 'center', 'center', 'center', 'center', 'center', ],
                font=dict(color='white', size=16),),
    cells=dict(values=[df[col] for col in df.columns],
               align=["left", 'center', 'center', 'center', 'center', 'center', 'center', 'center', ],
               line_color='darkslategray',
               fill_color=[["seashell", "seashell", "seashell", "seashell", "seashell",
               "honeydew", "honeydew",
               "mintcream", "mintcream", "mintcream",
               "lemonchiffon", "lemonchiffon",
               "mistyrose",
               "floralwhite", "floralwhite", "floralwhite", ]*5],
               font=dict(size=14),))
    ])
    
    
    fig.update_layout(height=900)
    
    return fig

    
    
    
######################################################################################################################

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black;'>FQN9002</h1>", unsafe_allow_html=True)




# lokacje plikow
im_sketch = "res/sketch_lowres.png"
im_central = "res/drawing.PNG"
im_logo2 = "res/logo_faun.png"
im_logo = "res/logo_zoeller_new.png"
im_table = "res/tabelka.PNG"
table_stats = "data/tabela_sierpien.csv"

# wczytanie plikow
sketch = Image.open(im_sketch).convert('RGB')
central_sketch = Image.open(im_central).convert('RGB')
xt_logo = Image.open(im_logo).convert('RGB')
zoe_logo = Image.open(im_logo2).convert('RGB')
tabela_info = Image.open(im_table).convert('RGB')
df_stats = wczytaj_statystyki_csv(table_stats)


### HEADER ###

c1, c2, c3 = st.columns((1,3,1))

## c1
c1.title("WIP")
c1.image(sketch, use_column_width=True)
 
c1.header("Select day:")

data_od = c1.date_input("", value=dt.date(2021,8,3), min_value=dt.date(2021,8,1), max_value=dt.date.today(), help="Choose day you want to analyze")
c1.write("------------------")
#data_do = c1.date_input("To:", min_value=data_od)

## c2

#c2.title("Dashboard Dashboard GPU7RM1")
c2.image(central_sketch, use_column_width=True)



## c3
#c3.title("WIP")
c3.image(xt_logo, use_column_width=True)
c3.image(zoe_logo, use_column_width=True)
c2.image(tabela_info, use_column_width=True)


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

c3.write(f"Statistics from {data_od}")
#c2.dataframe(dane_z_dnia, height=500)


#c3.table(dane_z_dnia)

tabela_statystyk = stworz_tabele_statystyk(df_stats, data_od)

stat_fig = tabela_statystyk_wyswietl(tabela_statystyk)

c2.plotly_chart(stat_fig, use_container_width=True)

#c2.table(tabela_statystyk)

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

    ## DAILY
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
    
    ## WEEKLY
    # MOTOGODZINY
    fig_q0 = wykres_z_tygodnia(df_stats, data_od, ["Motohours driving", "Motohours 900rpm stop", "Motohours idle"], ["Motohours driving", "Motohours 900rpm stop", "Motohours idle"], title="Weekly data")
    #plt.legend()
    cols[1].write(fig_q0)
    
    # MOTOGODZINY PRZELADOWANY
    fig_q1 = wykres_z_tygodnia2(df_stats, data_od, ["Motohours total", "Motohours >26t"], ["Motohours total", "Motohours >26t"])
    #plt.legend()
    cols[1].write(fig_q1)
    
    # DYSTANS
    fig_q2 = wykres_z_tygodnia2(df_stats, data_od, ["Distance [km]", "Distance >26t [km]"], ["Distance [km]", "Distance >26t [km]"])
    #plt.legend()
    cols[1].write(fig_q2)
    
    ## NORMAL DISTRIBUTION
    # MOTOGODZINY
    fig_r0 = plt.figure(figsize=(8,5))
    plt.title("Distribution", fontsize=24)
    mth_data = df_stats.T['Motohours total'].copy().values
    #n, bins, patches = plt.hist(mth_data, bins='auto', orientation='horizontal', edgecolor='black')
    
    bins = int(np.ceil(max(mth_data)))
    hist, bin_edges = np.histogram(mth_data, bins=[x for x in range(bins)])
    
    plt.barh([-0.5+x for x in range(1,bins)], hist, edgecolor='black', height=1)
    
    mth_today = dane_z_dnia.T["Motohours total"]
    plt.barh(np.ceil(mth_today)-0.5, hist[int(np.ceil(mth_today))-1], height=1, edgecolor='black', label="Selected day")
    
    plt.grid()
    plt.xlabel("Amount of days")
    plt.tight_layout()
    plt.legend()
    cols[2].write(fig_r0)
    
    
    #cols[2].plotly_chart(fig)

# TYMCZASOWY WYKRES OBCIĄŻENIA OSI
    fig_s0, ax_s0 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['Nacisk_total'], lw=3, label='mth total [h]')
    plt.ylabel("Load", fontsize=24)
    ax_s0.xaxis.set_major_formatter(xfmt)
    plt.axhline(26000, c='r', ls='--')
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    
    cols[0].write(fig_s0)
    
# WEEKLY OBCIĄŻENIA

    fig_s1 = wykres_z_tygodnia2(df_stats, data_od, ["Max overload >26t [t]"], ["Max overload >26t [t]"])
    #plt.legend()
    plt.ylabel("Overload [t]")
    plt.tight_layout()
    cols[1].write(fig_s1)

    
    
#st.write(df.columns)

## TABELKA PM

column, _ = st.columns((1, 1))

column.header("Diagnostics")

tabela_pm = go.Figure(data=[go.Table(header=dict(values=[' ', 'diagnosis'], font=dict(color='black', size=20), height=36),
                 cells=dict(values=[["PP joint", "CP sliding blocks", "CP rollers", "Hydraulic leakages", "Gresing system"], ["OK", "OK", "OK", "OK", "OK"]], 
                 fill=dict(color=['paleturquoise', 'lime']), 
                 font_size=16,
                 height=30
                 ))
                     ])

column.plotly_chart(tabela_pm)

