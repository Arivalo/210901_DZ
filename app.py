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
from matplotlib.ticker import MaxNLocator
import seaborn as sns

## TODO
# WYKRESY
# olej - olej total/>120bar
# tonokilometrowe x2
# body cap


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
        df = df.fillna(0)
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
    

    dane_dnia = dane_dnia.T.rename(columns={0:"Selected day"})
    #dane_dnia.columns = dane_dnia.iloc[0]
    #dane_dnia = dane_dnia.iloc[1:]
    
    # Parametry akumulowane
    # tbc
    
    
    
    
    return dane_dnia.round(1).astype(str)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def wczytaj_statystyki_csv(loc):

    df = pd.read_csv(loc, index_col=0)
    
    df = df[df.columns[1:]].set_index("dzien", drop=True).T
    
    return df
    

def stworz_tabele_statystyk(df, data):
    # usuwanie kolumn z samymi 0
    df = df.loc[:, (df != 0).any(axis=0)]

    data_parted = str(data).split("-")
    data = dt.date(int(data_parted[0]), int(data_parted[1]), int(data_parted[2]))
    dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7) if str(data + dt.timedelta(days=d-data.weekday())) in df.columns]
    dni_miesiac = [dzien for dzien in df.columns if dzien[:7]==str(data)[:7]]
    
    df_out = pd.DataFrame()
    
    try:
        df_out['selected day'] = df[str(data)]
    except KeyError:
        df_out['selected day'] = 0.0
    
    df_out['avg. week'] = df[dni_tydzien].mean(axis=1)
    
    df_out['avg. month'] = df[dni_miesiac].mean(axis=1)
    
    df_out['total'] = df.sum(axis=1)
    
    df_out = df_out.fillna(0).astype("float64").round(1)
    
    # usuwanie statystyk gdzie total nie ma sensu
    df_out['total'].T['Fuel consumption per distance [dm3/100km]'] = "-"
    df_out['total'].T['Hourly fuel consumption [dm3/h]'] = "-"
    df_out['total'].T['Max overload >26t [t]'] = "-"
    df_out['total'].T['Body capacity used [%]'] = "-"
    df_out['total'].T['Hydraulic energy per 1 t of waste [GJ/t]'] = "-"
    df_out['total'].T['Compaction hydraulic energy per 1 t of waste [GJ/t]'] = "-"
    df_out['total'].T['Average power of hydraulic system during body operation [kW]'] = "-"
    
    ## %%
    # daily
    mth_total = df_out['selected day'].T['Motohours total']
    distance_total = df_out['selected day'].T['Distance [km]']
    hyd_en_total = df_out['selected day'].T['Hydraulic energy [GJ]']
    
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,]
    
    #print(dividers)
    percentage_daily = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['selected day'].values, dividers)]
    
    df_out["%"] = percentage_daily
    
    # weekly
    mth_total = df_out['avg. week'].T['Motohours total']
    distance_total = df_out['avg. week'].T['Distance [km]']
    hyd_en_total = df_out['avg. week'].T['Hydraulic energy [GJ]']
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,]
    
    percentage_weekly = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['avg. week'].values, dividers)]
    
    df_out["% "] = percentage_weekly
    
    # monthly
    mth_total = df_out['avg. month'].T['Motohours total']
    distance_total = df_out['avg. month'].T['Distance [km]']
    hyd_en_total = df_out['avg. month'].T['Hydraulic energy [GJ]']
    
    dividers = [mth_total, mth_total, mth_total, mth_total, mth_total, distance_total, distance_total, -1, -1, -1, hyd_en_total, hyd_en_total, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,]
    
    percentage_monthly = [str(round(x/y*100, 1))+"%" if y>0 else "-" for x,y in zip(df_out['avg. month'].values, dividers)]
    
    df_out[" %"] = percentage_monthly
    
    # fake statystyki
    # df_out.loc['Average of rpm engine per day'] = "-"
    # df_out.loc['Leakage cylinder detection'] = "-"
    # df_out.loc['Time with oil over 60°C [h]'] = "-"
    
    df_out = df_out[['selected day', '%', 'avg. week', '% ', 'avg. month', ' %', 'total']]
    
    return df_out.astype(str)
    

#@st.cache(suppress_st_warning=True)
def wykres_z_tygodnia(df, data, lista_kolumn, lista_etykiet, title="", zakres_dni=None):
    
    fig, ax = plt.subplots(1, figsize=(8,5))
    
    colors = ['green', 'orange', 'blue', 'yellow', 'magenta', 'purple', 'cyan']
    
    if zakres_dni is None:
        dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7)]
    else:
        dni_tydzien = [date for date in pd.date_range(zakres_dni[0], zakres_dni[1]).astype(str) if date in df.columns]
    
    previous = [0 for x in range(len(dni_tydzien))]
    
    for dzien in dni_tydzien:
        if dzien not in df.columns:
            df[dzien] = 0
    
    if str(data) in dni_tydzien:
        ax.bar(data, df[str(data)].T["Motohours total"].max()*1.1, width=1, color="yellow", label="Selected day",
            alpha=0.6)
    
    for kolumna, label, color in zip(lista_kolumn, lista_etykiet, colors):
    
        temp_y = df[dni_tydzien].T[kolumna]
        ax.bar(pd.to_datetime(dni_tydzien),  temp_y, width=0.67,  label=label, bottom=previous, color=color)
        previous = [x+y for x,y in zip(previous, temp_y)]
          
    ax.legend()
    plt.title(title, fontsize=24)
    plt.grid()
    plt.tight_layout()
    
    return fig

#@st.cache(suppress_st_warning=True)    
def wykres_z_tygodnia2(df, data, lista_kolumn, lista_etykiet, title="", zakres_dni=None):
    
    fig, ax = plt.subplots(1, figsize=(8,5))
    
    if zakres_dni is None:
        dni_tydzien = [str(data + dt.timedelta(days=d-data.weekday())) for d in range(7)]
    else:
        dni_tydzien = [date for date in pd.date_range(zakres_dni[0], zakres_dni[1]).astype(str) if date in df.columns]
    
    for dzien in dni_tydzien:
        if dzien not in df.columns:
            df[dzien] = 0
    
    if str(data) in dni_tydzien:
        ax.bar(data, df[str(data)].T[lista_kolumn[0]].max()*1.1, width=1, color="yellow", label="Selected day", alpha=0.6)
    
    for kolumna, label in zip(lista_kolumn, lista_etykiet):
    
        temp_y = df[dni_tydzien].T[kolumna]
        ax.bar(pd.to_datetime(dni_tydzien),  temp_y, width=0.67,  label=label)
          
    ax.legend()
    plt.title(title, fontsize=24)
    plt.grid()
    plt.tight_layout()
    
    return fig
    

@st.cache(suppress_st_warning=True)   
def tabela_statystyk_wyswietl(df):

    def my_value(number):
        '''
        example
        -------
        input: 5000000
        output: "5,000,000"
        '''
        return "{:,}".format(number)
        
    def h_to_time(h):
        '''
        example
        -------
        input: 6.4
        output: "6:24"
        '''
        try:
            HH = int(h)
            mm = int((h-HH)*60)
        except ValueError:
            print(h)
            return "-"
            
        if mm < 10:
            mm = f"0{mm}"
        
        return f"{HH}:{mm}"
        
    
    data_cols = ["selected day", "avg. week", "avg. month", "total"]
    mth_cols = ["Motohours total", "Motohours idle", "Motohours 900rpm stop", "Motohours driving",
                "Motohours >26t"]
    hyd_cols = ["Hydraulic energy [GJ]", "Compaction hydraulic energy [GJ]"]
    
    for col in data_cols:
        df[col].T[mth_cols] = df[col].T[mth_cols].astype("float64").apply(h_to_time)
        #df[col].T[hyd_cols] = (df[col].T[hyd_cols].astype("float64")/1000).round(1)
        
    #df = df.rename(index={hyd_cols[0]:"Hydraulic energy [GJ]", hyd_cols[1]:"Compaction hydraulic energy [GJ]"})

    df = df.reset_index().rename(columns={'index':""})
    
    df[""] = [f"<b>{val}</b>" for val in df[""]]
    df["total"].iloc[5:] = ["{:,}".format(float(number)).replace(","," ") if number !="-" else "-" for number in df["total"].iloc[5:]]
    
    
    
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
               "mistyrose", "mistyrose",
               "floralwhite", "floralwhite", "floralwhite", "floralwhite","floralwhite",
               "aliceblue",  "aliceblue",]*5],
               font=dict(size=14),))
    ])
    
    
    fig.update_layout(height=1200)
    
    return fig

#@st.cache(suppress_st_warning=True)
def wykres_dystrybucja(df, dane_z_dnia, kolumna):

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    data = df.T[kolumna].copy().values
    
    bins = int(np.ceil(max(data)))+1
    hist, bin_edges = np.histogram(data, bins=[x for x in range(bins)])
    
    plt.barh([-0.5+x for x in range(1,bins)], hist, edgecolor='black', height=1)
    
    data_today = dane_z_dnia.T[kolumna]
    
    if np.ceil(data_today) == 0:
        data_today += 0.1
    
    plt.barh(np.ceil(data_today)-0.5, hist[int(min(np.ceil(data_today)-1, len(hist)-1))], height=1, edgecolor='black', label="Selected day", color='yellow')
    
    plt.grid()
    plt.xlabel("Amount of days")
    plt.tight_layout()
    plt.legend()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig
    
#@st.cache(suppress_st_warning=True)  
def wykres_dystrybucja2(df, dane_z_dnia, kolumna):

    fig, ax = plt.subplots(1, figsize=(8,5))
    
    data = df.T[kolumna].copy().values
    
    bins = int(np.ceil(max(data))/20)
    hist, bin_edges = np.histogram(data, bins=[x*20 for x in range(bins)])
    
    plt.barh([(-0.5+x)*20 for x in range(1,bins)], hist, edgecolor='black', height=20)
    
    data_today = dane_z_dnia.T[kolumna]
    
    if np.ceil(data_today) == 0:
        data_today += 0.1
    
    plt.barh((np.ceil(data_today/20)*20-10), hist[int(np.floor(data_today)/20)], height=20, edgecolor='black', label="Selected day", color='yellow')
    
    plt.grid()
    plt.xlabel("Amount of days")
    plt.tight_layout()
    plt.legend()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig

#@st.cache(suppress_st_warning=True) 
def wykres_dystrybucja_v2(df, stat, today_stats=None, mean=True, bin_w=1, fs=(8,5), xlabel=None):
    
    fig, ax = plt.subplots(1, figsize=fs)
    sns.set_palette("bright")
    
    # filtrowanie '0' motogodzin
    df_hist = df.T
    df_hist = df_hist[df_hist["Motohours total"] > 0]
    
    sns.histplot(data=df_hist, x=stat, color="gray", stat='count', line_kws={"color":"red"}, binwidth=bin_w, binrange=(0, df_hist[stat].max()), shrink=0.95)
    sns.histplot(data=df_hist, x=stat, fill=False, color="red", stat='count', kde=True, line_kws={"lw":2}, binwidth=bin_w, binrange=(0, df_hist[stat].max()), shrink=0.95)
    sns.histplot(data=df_hist, x=stat, fill=False, color="k", stat='count', binwidth=bin_w, binrange=(0, df_hist[stat].max()), shrink=0.95)
    
    plt.ylabel("amount of days")
    
    plt.plot(0,0, color="red", label="Distribution line")
    
    if mean:
        plt.axvline(df_hist[stat].mean(), label="Daily avg.", color='blue', lw=2.5, ls='--')
        
    if today_stats is not None:
        #print(today_stats.T[stat].astype("float64"))
        try:
            plt.axvline(today_stats.T[stat].astype("float64"), label="Selected day", color='yellow', lw=2.5)
        except ValueError:
            plt.axvline(today_stats.T[stat].astype("float64").values[0], label="Selected day", color='yellow', lw=2.5)
        
    if xlabel is not None:
        plt.xlabel(xlabel)
        
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    plt.xlim((0, np.ceil(df_hist[stat].max()/bin_w)*bin_w))
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig
    
    
    
######################################################################################################################

st.set_page_config(layout="wide", page_title='Dashboard')

st.markdown("<h1 style='text-align: center; color: black;'>FQN9002</h1>", unsafe_allow_html=True)




# lokacje plikow
im_sketch = "res/sketch_lowres.png"
im_central = "res/drawing.PNG"
im_logo2 = "res/logo_faun.png"
im_logo = "res/logo_zoeller_new.png"
im_table = "res/tabelka.PNG"
table_stats = "data/20269/podsumowanie_dnia/all_days_summary.csv"

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
 

#data_do = c1.date_input("To:", min_value=data_od)

## c2

#c2.title("Dashboard Dashboard GPU7RM1")
c2.image(central_sketch, use_column_width=True)

c2.image(tabela_info, use_column_width=True)


## c3
#c3.title("WIP")
c3.image(xt_logo, use_column_width=True)
c3.image(zoe_logo, use_column_width=True)



c1, c2, c3 = st.columns((1,5,1))

## c1
c1.header("Select day:")

data_od = c1.date_input("", value=dt.date.today()-dt.timedelta(days=7), min_value=dt.date(2021,8,16), max_value=dt.date.today(), help="Choose day you want to analyze")
c1.write("------------------")


### MAIN ###

url = utworz_url(data_od, data_od)

    
if str(data_od) in df_stats.columns:
    dane_z_dnia = df_stats[str(data_od)]
else:

    try:
    
        df = download_data(url)
    
        if not df.empty:
            dane_z_dnia = tabela_statystyk_dnia(df)
            df_stats[str(data_od)] = dane_z_dnia["Selected day"].astype("float64")
            
    except KeyError:
        dane_z_dnia = df_stats[df_stats.columns[0]].copy().rename({df_stats.columns[0]:"Selected day"})
        dane_z_dnia["Selected day"] = 0
        c1.write("No data for selected day")
        df_stats[str(data_od)] = 0
    
        
    

#c3.write(f"Statistics from {data_od}")
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


#cols = st.columns((1,1,1))

# fig = px.line(df, x='Data_godzina', y='Nacisk_total', title="Truck weight during the day",
            # labels={'Data_godzina':"Time", "Nacisk_total":"Total weight"})

if not df.empty:
    #st.write(df.columns)
    ## WYKRESY DOLNE ##
    xfmt = mdates.DateFormatter('%H:%M')

    
    ## wykresy rozwijane
    
    exp0 = st.expander("Motohours")
    
    cols = exp0.columns((2,3,2))
    
    zakres_dni0 = cols[1].slider("Range of days", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    cols = exp0.columns((1,1,1))
    
    ## MOTOGODZINY ##
    
    # DAILY
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
    
    # WEEKLY
    
    fig_q0 = wykres_z_tygodnia(df_stats, data_od, ["Motohours driving", "Motohours 900rpm stop", "Motohours idle"], ["Motohours driving", "Motohours 900rpm stop", "Motohours idle"], title="", zakres_dni=zakres_dni0)
    #plt.legend()
    
    # DYSTRYBUCJA
    fig_r0 = wykres_dystrybucja_v2(df_stats, "Motohours total", today_stats=dane_z_dnia)
    plt.title("Distribution", fontsize=24)
    plt.tight_layout()
    
    cols[0].write(fig_p0)
    cols[1].write(fig_q0)
    cols[2].write(fig_r0)
    
    ## MOTOGODZINY PRZEŁADOWANE ##
    
    exp1 = st.expander("Motohours overloaded")
    
    cols = exp1.columns((2,3,2))
    
    zakres_dni1 = cols[1].slider("Range of days ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p1, ax_1 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['motogodziny_total'], lw=3, label='mth total [h]')
    plt.plot(df['Data_godzina'], df['motogodziny_przeladowana'], lw=3, label='mth overload [h]')
    plt.ylabel("Motohours overloaded", fontsize=24)
    ax_1.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    
    fig_q1 = wykres_z_tygodnia2(df_stats, data_od, ["Motohours total", "Motohours >26t"], ["Motohours total", "Motohours >26t"], zakres_dni=zakres_dni1)
    
    fig_r1 = wykres_dystrybucja_v2(df_stats, "Motohours >26t", today_stats=dane_z_dnia)
    
    cols = exp1.columns((1,1,1))
    cols[0].write(fig_p1)
    cols[1].write(fig_q1)
    cols[2].write(fig_r1)
    
    
    ## PRZEBIEG DYSTANS ##
    
    exp2 = st.expander("Distance")
    
    cols = exp2.columns((2,3,2))
    
    zakres_dni2 = cols[1].slider("Range of days  ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p2, ax_2 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['przebieg_km'], lw=3, label='Total distance [km]')
    plt.plot(df['Data_godzina'], df['przebieg_km_przeladowana'], lw=3, label='Distance with overload >26t [km]')
    plt.ylabel("Distance", fontsize=24)
    ax_2.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    
    fig_q2 = wykres_z_tygodnia2(df_stats, data_od, ["Distance [km]", "Distance >26t [km]"], ["Distance [km]", "Distance >26t [km]"], zakres_dni=zakres_dni2)
    
    fig_r2 = wykres_dystrybucja_v2(df_stats, "Distance [km]", today_stats=dane_z_dnia, bin_w=20)
    
    cols = exp2.columns((1,1,1))
    cols[0].write(fig_p2)
    cols[1].write(fig_q2)
    cols[2].write(fig_r2)
    
    ## OBCIĄŻENIE ##
    
    exp3 = st.expander("Load")
    
    cols = exp3.columns((2,3,2))
    
    zakres_dni3 = cols[1].slider("Range of days   ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_s0, ax_s0 = plt.subplots(1, figsize=(8,5))
    plt.plot(df[df['RPM'] > 800]['Data_godzina'], df[df['RPM'] > 800]['Nacisk_total'], lw=3, label='mth total [h]')
    plt.ylabel("Load", fontsize=24)
    ax_s0.xaxis.set_major_formatter(xfmt)
    plt.axhline(26000, c='r', ls='--')
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    
    fig_s1 = wykres_z_tygodnia2(df_stats, data_od, ["Max overload >26t [t]"], ["Max overload >26t [t]"], zakres_dni=zakres_dni3)
    #plt.legend()
    plt.ylabel("Overload [t]")
    plt.tight_layout()
    
    fig_s2 = wykres_dystrybucja_v2(df_stats,"Max overload >26t [t]", today_stats=dane_z_dnia)   
    
    cols = exp3.columns((1,1,1))
    cols[0].write(fig_s0)
    cols[1].write(fig_s1)
    cols[2].write(fig_s2)
    
# WYKRESY NOWE

    # OLEJ
    exp4 = st.expander("Hydraulic oil")
    
    cols = exp4.columns((2,3,2))
    
    zakres_dni4 = cols[1].slider("Range of days    ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p4, ax_p4 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['ilosc_przepompowanego_oleju'], lw=3, label='hydraulic oil pumped [m3]')
    plt.plot(df['Data_godzina'], df['ilosc_przepompowanego_oleju_120bar'], lw=3, label='hydraulic oil pumped at >120 bar [m3]')
    plt.ylabel("Oil pumped", fontsize=24)
    ax_p4.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    fig_q4 = wykres_z_tygodnia2(df_stats, data_od, ['Hydraulic oil [m3]', 'Hydraulic oil >120 bar [m3]'], ["hydraulic oil pumped [m3]", "hydraulic oil pumped at >120 bar [m3]"], zakres_dni=zakres_dni4)
    plt.legend()
    plt.ylabel("Oil pumped [m3]")
    plt.tight_layout()
    
    fig_r4 = wykres_dystrybucja_v2(df_stats,"Hydraulic oil [m3]", today_stats=dane_z_dnia, bin_w=2)
    
    cols = exp4.columns((1,1,1))
    cols[0].write(fig_p4)
    cols[1].write(fig_q4)
    cols[2].write(fig_r4)
    
    
    # TONOKILOMETRY
    exp5 = st.expander("Distance x load")
    
    cols = exp5.columns((2,3,2))
    
    zakres_dni5 = cols[1].slider("Range of days     ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p5, ax_p5 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['Tonokilometry_masa_smieci'], lw=3, label='Waste mass x kilometers [t*km]')
    plt.plot(df['Data_godzina'], df['Tonokilometry_przeladowane'], lw=3, label='Vehicle overload x kilometers [t*km]')
    plt.ylabel("Mass x kilometers", fontsize=24)
    ax_p5.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    fig_q5 = wykres_z_tygodnia2(df_stats, data_od, ['Waste mass x kilometers [t*km]', 'Vehicle overload x kilometers [t*km]'], ['Waste mass x kilometers [t*km]', 'Vehicle overload x kilometers [t*km]'], zakres_dni=zakres_dni5)
    plt.legend()
    plt.ylabel("Overload [t]")
    plt.tight_layout()
    
    fig_r5 = wykres_dystrybucja_v2(df_stats,'Waste mass x kilometers [t*km]', today_stats=dane_z_dnia, bin_w=50)
    
    cols = exp5.columns((1,1,1))
    cols[0].write(fig_p5)
    cols[1].write(fig_q5)
    cols[2].write(fig_r5)
    
    
    ## KOPIA  DF POMOCNICZA
    df2 = df.copy()[df['RPM'] > 800]
    
    # BODY CAPACITY
    exp6 = st.expander("Body capacity")
    
    cols = exp6.columns((2,3,2))
    
    zakres_dni6 = cols[1].slider("Range of days      ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p6, ax_p6 = plt.subplots(1, figsize=(8,5))
    plt.plot(df2['Data_godzina'], df2['zapelnienie_skrzyni_procent'], lw=3, label='Body capacity used [%]')
    plt.ylabel("Capacity used [%]", fontsize=24)
    ax_p6.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.ylim(0, 101)
    #plt.legend()
    plt.tight_layout()
    
    fig_q6 = wykres_z_tygodnia2(df_stats, data_od, ['Body capacity used [%]'], ['Body capacity used [%]'], zakres_dni=zakres_dni6)
    plt.legend()
    plt.ylabel("Body capacity used [%]")
    plt.tight_layout()
    
    fig_r6 = wykres_dystrybucja_v2(df_stats, 'Body capacity used [%]', today_stats=dane_z_dnia, bin_w=10)
    
    cols = exp6.columns((1,1,1))
    cols[0].write(fig_p6)
    cols[1].write(fig_q6)
    cols[2].write(fig_r6)
    
    # HYDRAULIC ENERGY PER WASTE
    exp7 = st.expander("Hydraulic energy per ton")
    
    cols = exp7.columns((2,3,2))
    
    zakres_dni7 = cols[1].slider("Range of days       ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_p7, ax_p7 = plt.subplots(1, figsize=(8,5))
    plt.plot(df2['Data_godzina'], df2['hydraulic_energy'] / (df2['Masa_smieci']/1000)/1000, lw=3, label='Hydraulic energy per ton of waste')
    plt.plot(df2['Data_godzina'], df2['energia_hydr_zageszczania'] / (df2['Masa_smieci']/1000)/1000, lw=3, label='Compation hydraulic energy per ton of waste')
    plt.ylabel("Hydraulic energy per ton of waste [GJ/t]", fontsize=14)
    ax_p7.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    fig_q7 = wykres_z_tygodnia2(df_stats, data_od, ['Hydraulic energy per 1 t of waste [GJ/t]', 'Compaction hydraulic energy per 1 t of waste [GJ/t]'], ['Hydraulic energy per 1 t of waste [GJ/t]', 'Compaction hydraulic energy per 1 t of waste [GJ/t]'], zakres_dni=zakres_dni7)
    plt.legend()
    plt.ylabel("Hydraulic energy per ton of waste [GJ/t]")
    plt.tight_layout()
    
    fig_r7 = wykres_dystrybucja_v2(df_stats, 'Hydraulic energy per 1 t of waste [GJ/t]', today_stats=dane_z_dnia, bin_w=1)
    
    cols = exp7.columns((1,1,1))
    cols[0].write(fig_p7)
    cols[1].write(fig_q7)
    cols[2].write(fig_r7)
    
    exp8 = st.expander("Average power of hydraulic system during body operation")
    
    cols = exp8.columns((2,3,2))
    
    zakres_dni8 = cols[1].slider("Range of days        ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    fig_q8 = wykres_z_tygodnia2(df_stats, data_od, ['Average power of hydraulic system during body operation [kW]'], ['Average power of hydraulic system during body operation [kW]'], zakres_dni=zakres_dni8)
    plt.legend()
    plt.ylabel("Average power of hydraulic system during body operation [kW]")
    plt.tight_layout()
    
    fig_r8 = wykres_dystrybucja_v2(df_stats, 'Average power of hydraulic system during body operation [kW]', today_stats=dane_z_dnia, bin_w=1)
    
    cols = exp8.columns((1,1))
    cols[0].write(fig_q8)
    cols[1].write(fig_r8)
    
    # PP JOINT
    
    exp9 = st.expander("PP joint")
    
    cols = exp9.columns((2,3,2))
    
    zakres_dni9 = cols[1].slider("Range of days         ", min_value=dt.date(2021,8,16), max_value=dt.date.today(), value=(dt.date.today()-dt.timedelta(days=7+dt.date.today().weekday()), dt.date.today()-dt.timedelta(days=dt.date.today().weekday()+1)))
    
    # cykle do zrobienia
    fig_p9_1, ax_p9_1 = plt.subplots(1, figsize=(8,5))
    plt.tight_layout()
    
    # masa odpadow
    df_stats_2 = df_stats.T.cumsum().T
    fig_p9_2, ax_p9_2 = plt.subplots(1, figsize=(8,5))
    #fig_p9_2 = wykres_z_tygodnia2(df_stats_2, data_od, ['Masa_smieci'], ['Waste weight [kg]'], zakres_dni=zakres_dni9)
    plt.tight_layout()
    
    
    # temperatury
    fig_q9_1, ax_q9_1 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df["temperatura_IN12"], label="temperature PIN 1")
    plt.plot(df['Data_godzina'], df["temperatura_IN14"], label="temperature PIN 2")
    plt.plot(df['Data_godzina'], df["temperatura_zewn"], label="ambient temperature")
    
    plt.ylabel("Temperature [°C]")
    
    ax_q9_1.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    
    # delta temp.
    fig_q9_2, ax_q9_2 = plt.subplots(1, figsize=(8,5))
    plt.plot(df['Data_godzina'], df['temperatura_zewn']- df['temperatura_IN12'], label = 'delta T PIN 1 ', c='b')
    plt.plot(df['Data_godzina'], df['temperatura_zewn'] - df['temperatura_IN14'], label = 'delta T PIN 2', c='g')
    
    plt.ylabel("Temperature [°C]")
    
    ax_q9_2.xaxis.set_major_formatter(xfmt)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    # rysowanie
    cols = exp9.columns((1,1))
    cols[0].write(fig_p9_1)
    cols[0].write(fig_p9_2)
    cols[1].write(fig_q9_1)
    cols[1].write(fig_q9_2)
    
    #cols[0].write(df)
    

    
    
#st.write(df.columns)

## TABELKA PM

_, column, _ = st.columns((1, 1, 1))

column.markdown("<h1 style='text-align: center; color: black;'>Diagnostics</h1>", unsafe_allow_html=True)

tabela_pm = go.Figure(data=[go.Table(header=dict(values=[' ', 'diagnosis'], font=dict(color='black', size=20), height=36),
                 cells=dict(values=[["PP joint", "CP sliding blocks", "CP rollers", "Hydraulic leakages", "Gresing system"], ["OK", "OK", "OK", "OK", "OK"]], 
                 fill=dict(color=['paleturquoise', 'lime']), 
                 font_size=16,
                 height=30
                 ))
                     ])

column.plotly_chart(tabela_pm, use_container_width=True)

#st.write(df.columns)
