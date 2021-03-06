import requests
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans


def przelicz_polozenie_sciany(series, series_rpm):
    '''
    wartości na CANie są od 0 do 4095. 0 oznacza 0mm, 4095 oznacza 10000mm    
    
    0 - 4095
    0 - 10000
    
    
    '''
    
    
    polozenie_mm = series * 10000 / 4095

    # filtr - ustawia jako nan wszystkie wiersze gdzie rpm<800 poniewaz sygnal z czasu gdy zabudowa jest wlaczona jest bardziej wiarygodny
    polozenie_mm[series_rpm<800] = np.nan

    polozenie_mm[polozenie_mm < 10] = np.nan

    polozenie_mm = polozenie_mm.fillna(method='ffill')

    return polozenie_mm
    
def przelicz_przebieg(series_predkosc_osi):
    '''
    Przelicza przebieg w danej sekundzie oraz laczny przebieg, na podstawie predkosci w km/h
    
    
    Input:
        series_predkosc_osi - Series z wartosciamy predkosci osi w km/h
    
    Returns:
        distance_m - przebieg w metrach w danej sekundzie
        total_distance_km - laczny przebieg od poczatku pracy
    '''
    
    distance_m = np.round(series_predkosc_osi *1000/3600 * 1, 2) # m
    distance_km = distance_m/1000 # km

    total_distance_km = distance_km.cumsum().round(3)

    return distance_m, total_distance_km

def przelicz_przebieg_z_naciskiem_X(series_predkosc_osi, nacisk_na_osie_series, nacisk_na_osie_max = 26000):
    '''
        Przelicza przebieg w danej sekundzie oraz laczny przebieg, na podstawie predkosci w km/h
        dla momentu gdy naciski na osie sa mniejsze niz nacisk_na_osie_max zeruje predkosc do obliczen

        Input:
            series_predkosc_osi - Series z wartosciamy predkosci osi w km/h
            nacisk_na_osie_series - Series z lacznym naciskiem na wszystkie osie [kg]
            nacisk_na_osie_max - ponizej tej wartosci predkosc zostanie ustawiona na zero do obliczen [kg]
        Returns:
            distance_m - przebieg w metrach w danej sekundzie
            total_distance_km - laczny przebieg od poczatku pracy
        '''
    index_ = nacisk_na_osie_series < nacisk_na_osie_max

    series_predkosc_osi_temp = series_predkosc_osi.copy()
    series_predkosc_osi_temp[index_] = 0

    distance_m = np.round(series_predkosc_osi_temp * 1000 / 3600 * 1, 2)  # m
    distance_km = distance_m / 1000  # km

    total_distance_km = distance_km.cumsum().round(3)

    return distance_m, total_distance_km

def przelicz_ilosc_oleju(series_cisnienie_bar, series_rpm, cisnienie_min=0, r_pto = 1.19, pump_displacement = 69):
    '''
        oil = oil flow


        Input:
            series_cisnienie - Series z cisnieniem hydraulicznym [bar]
            series_rpm - Series z predkoscia obrotowa silnika [rpm]
            r_pto - przelozenie przystawki odbioru mocy
            pump_displacement - pojemnosc jednostkowa pompy w cm3/rev

        Returns:
            oil_flow - Series przeplyw chwilowy oleju w dm3/min
            oil_flow_cumulative - Series laczny przeplyw oleju w m3
        '''


    p = series_cisnienie_bar.values

    # wyzeruj wiersze gdzie cisnienie jest mniejsze niz cisnienie_min
    index_ = p > cisnienie_min
    series_rpm_temp = series_rpm.copy()
    series_rpm_temp[~index_] = 0

    pump_displacement_dm3 = pump_displacement / 1000 # dm3 per 1 rev

    oil_flow = series_rpm_temp * r_pto * pump_displacement_dm3 # dm3 / min
    oil_flow_m3s = oil_flow / 60 / 1000 # m3 / s

    oil_flow_cumulative = oil_flow_m3s.cumsum().round(2) # m3

    oil_flow = oil_flow.round(2)

    return oil_flow, oil_flow_cumulative


    
def przelicz_energie_hydr(series_cisnienie, series_rpm, r_pto = 1.19, pump_displacement = 69):
    '''
    hydraulic power = oil flow * pressure
    
    
    Input:
        series_cisnienie - Series z cisnieniem hydraulicznym [bar]
        series_rpm - Series z predkoscia obrotowa silnika [rpm]
        r_pto - przelozenie przystawki odbioru mocy
        pump_displacement - pojemnosc jednostkowa pompy w cm3/rev
    
    Returns:
        hydraulic_power - power generated in the hydraulic system at 1 s [kW]
        hydraulic_energy - cumulative sum of hydraulic_power in kJ
    '''
    
    pump_displacement_dm3 = pump_displacement / 1000 # dm3 per 1 rev

    oil_flow = series_rpm * r_pto * pump_displacement_dm3 # dm3 / min
    oil_flow = oil_flow * 10**-3 / 60 # m3 / s

    p = series_cisnienie.values
    p = p * 10**5 # Pa

    hydraulic_power = (p * oil_flow / 10**3).round(3) # kN * m /s = kW
    
    hydraulic_energy = hydraulic_power.cumsum().round(0) # kJ
    
    return hydraulic_power, hydraulic_energy
    
def przelicz_energie_z_paliwa(series_fuel_consumption, density = 0.82, efficiency = 0.2, heat_value = 43000):
    '''
    
    Input:
        series_fuel_consumption - fuel consumed from the start, cumulative sum [Liters]
        density - diesel density in kg/dm3
        efficiency - assumed average efficiency of combustion engine [-]
        heat_value  - heat value of diesel fuel in kJ/kg
    
    Returns:
        energy_from_fuel - series with cumulative values of energy from fuel in kJ
    '''
    

    # 1 litr of dieses is 
    energy_in_liter = density * heat_value # kJ/dm3

    energy_from_fuel = series_fuel_consumption * energy_in_liter * efficiency # kJ

    return energy_from_fuel
    
def przelicz_polozenie_srodka_masy_pustego_pojazdu(N1, N2, L):
    '''
    Input:
        N1 - nacisk na os 1
        N2 - nacisk na os 2
        L - rozstaw osi [mm]
    
    Return:
        a - polozenie srodka ciezkosci pojazdu od osi 1 [mm]
        
    '''
    
    a = L * N2 / (N1+N2) # mm
    
    
    return a

def przelicz_naciski_i_masy(N, x1, Wp = 15060, a = 2200, b = 560, c=4227, L = 4000):
    '''
    17/08/2021 - WCIAZ DO WERYFIKACJI SYGNAL Z CZUJNIKA LASEROWEGO !!!



    Dane:
    N - laczny nacisk na osie 2 i 3 [kg], parametr axle_weight
    x1 - polozenie sciany wypychajacej [mm]
    Wp - waga pojazdu pustego [kg]
    a - polozenie srodka ciezkosci pojazdu pustego od osi 1 [mm]
    b - odleglosc miedzy osia 1 a poczatkiem skrzyni zaladowczej [mm]
    c - max polozenie sciany  [mm]
    L - rozstaw osi [mm], w przypadku podwozia 3 osiowego to jest odleglosc od osi 1 do polowy rozstawu miedzy 2 i 3.
    
    Oblicza:
    Ls - polozenie srodka ciezkosci odpadow/smieci (Ms) wzg osi 1
    Ms - masa smieci [kg]
    N_total - laczna masa pojazdu aktualna  [kg]
    N1 - aktualny nacisk na os 1 [kg]
    ratio_N1N2 - stosunek nacisku na osie N1 do N2. wg przepisow nie moze byc mniejszy niz 20%; wartosc w %
    
    
    '''

    # obliczenia z pominieciem mas czlonkow zalogi smieciarki
    Ls = b + x1 + (c-x1)/2 # mm
    
    Ms = np.round((N * L  - Wp*a) / Ls, 0) # kg

    N_total = np.round(Wp + Ms, 0 )# kg

    N1 = N_total - N # kg

    Ls[Ls<0] = 0
    Ms[Ms<0] = 0
    N_total[N_total<0] = 0.01
    N1[N1<0] = 0

    ratio_N1N2 = np.round(N1 / N_total, 3) * 100 # [%]

    return Ls, Ms, N_total, N1, ratio_N1N2


def przelicz_energie_zuzyta_na_zageszczanie(series_moc_hydrauliczna, series_predkosc_osi, series_rpm, min_rpm=850):
    '''
    energia hydrauliczna liczona w czasie postoju gdy predkosc obrotowa silnika > 850 rpm
    
    Input:
    series_moc_hydrauliczna
    series_predkosc_osi
    series_rpm
    
    Returns:
    
    energia_zageszczania - Series z energia zageszczania od poczatku pracy (cumulative sum) [kJ]
    
    
    '''
    index_ = (series_predkosc_osi == 0) & (series_rpm  > min_rpm)
    
    energia_zageszczania_temp = series_moc_hydrauliczna.copy()
    
    energia_zageszczania_temp[~index_] = 0 
    
    energia_zageszczania = energia_zageszczania_temp.cumsum()
        
    
    
    return energia_zageszczania

def przelicz_gestosc_odpadow(x1, Ms, c= 4227,  A = 5022500):
    
    '''
    17/08/2021 - WCIAZ DO WERYFIKACJI SYGNAL Z CZUJNIKA LASEROWEGO !!!


    Dane:
    
    A - pole powierzchnie przekroju skrzyni [mm2]
    x1 - polozenie sciany wypychajacej [mm]
    c - max wysuniecie sciany podczas jazdy [mm], max wskazanie czujnika laserowego
    Ms - masa smieci w skrzyni [kg]
    
    
    '''
    A_m2 = A / 1000000 # m
    V_smieci = A_m2 * (c - x1) / 1000 # m3
    V_smieci[V_smieci<1] = 1000
    
    gestosc = np.round( Ms / V_smieci, 1) # kg/m3
    
    return gestosc

def przelicz_cisnienie_bar(series_cisnienie):
    
    
    cisnienie_bar = series_cisnienie / 10
    
    return cisnienie_bar



def przelicz_czujnik_temperatury(df_Series, czujnik='Febi Bilstein 28334'):
    """
    przelicza temperature z ohm na st.C


    Charakterystyka dla czujnika Febi Bilstein 28334:

    temperatura_C = -34.27 * ln(odczyt_w_ohm) + 281.81

    Febi Bilstein 28334 to czujnik temperatury cieczy montowany w łozyskach tramwaju (27/10/2020).

    Parameters
    ----------
    df_Series: pandas Series
    Kolumna z df zawierajaca wartosci w [ohm] odczytane z webx

    Returns
    -------
    temperatura: np array
    Wektor zawierajacy wartosci temperatury w st. C

    """
    if czujnik == 'Febi Bilstein 28334':
        temperatura = (-34.27 * np.log(df_Series.values) + 281.81).round(1)
    else:
        temperatura = df_Series.values * np.nan

    return temperatura


def przelicz_temperature(df, kolumny_z_czujnikami, czujnik='Febi Bilstein 28334'):
    for kolumna_czujnik in kolumny_z_czujnikami:
        df[kolumna_czujnik] = przelicz_czujnik_temperatury(df[kolumna_czujnik], czujnik)

    return df


def przelicz_delta_temperatur(series_temp_pin ,series_temp_zewn):

    """

    :param series_temp_pin:
    :param series_temp_zewn:
    :return:

    delta - Series z roznica temp miedzy temp w sworzniu a temperatura otoczenia (z CAN)
    """

    delta_temp = series_temp_pin - series_temp_zewn

    return delta_temp



def przelicz_akcelerometr(df_Series, czujnik='IFM VTV122'):
    """
    Przeliczenie wartości czujnika na wartosc mierzona

    Czujnik:        IFM VTV122
    Rodzaj pomiaru: wartosc RMS
    Wyjście:        4-20    mA
    Zakres pomiaru: 0-25    RMS

    Parametry
    ---------
    df_Series: pandas.Series
    Kolumna z danymi wejsciowymi

    Zwraca
    ------
    rms: np.array
    Przeliczona kolumna wartosci

    Przyklad
    -----
    df['Wartosc_przeliczona'] = przelicz_akcelerometr(df['Wartosc_surowa'], czujnik='IFM VTV122')

    """
    if czujnik == 'IFM VTV122':
        rms = ((df_Series.values-4000)/16000*25).round(1)
    else:
        rms = df_Series.values * np.nan

    return rms

def przelicz_akcelerometry(df, kolumny_z_czujnikami, czujnik='IFM VTV122'):
    for kolumna_czujnik in kolumny_z_czujnikami:
        df[kolumna_czujnik] = przelicz_akcelerometr(df[kolumna_czujnik], czujnik)

    return df


def dane_bypass_wysuwanie_sciany(df_Series, kolumny=['wysuwanie_sciany', 'bypass_1']):
    '''
    Przerabia dane liczbowe na stany okreslajace poszczegolne ruchy suwaka i zgarniaka

    Parametry
    ---------
    df_Series: pandas.Series
    Kolumna zawierajaca okreslona ramke danych (203)

    kolumny: lista
    Kolumna z nazwami kolumn w kolejnosci bitow
    default: ['wysuwanie_sciany', 'bypass_1']

    Zwraca
    ------
    pandas.DataFrame
    Tabela z danymi o stanach 0 i 1

    Przyklad
    --------
    dane_bypass_sciana =  dane_bypass_wysuwanie_sciany(df["bypass1_1"])
    '''
    # 80 - wysuwanie sciany; 64 - sciana stoi; 2 - bypass zalaczony; 0 - bypass wylaczony
    # upraszczam do 16 - wysuwanie sciany (roznica miedzy tymi stanami, sama 16 nie wystepuje)
    dane_binarne = df_Series.astype('int').apply(lambda x: [int(x) for x in bin(x%32)[2:].zfill(4)[0]] + [int(x) for x in bin(x%32)[2:].zfill(4)[-2]])

    return pd.DataFrame.from_records(dane_binarne, columns=kolumny)


def oblicz_minimalna_gestosc_pelnej_skrzyni(Ms, c, A):
    '''
    Oblicza minimalna gestosc smieci przyjmujac ze smieci zajmuja cala objetosc skrzyni przy masie smieci Ms
    
    Minimalna gestosc smieci odpowiada minimalnej energii zuzytej na zageszczanie odpadow. 
    
    Dane:
    Ms- masa smieci [kg]
    c - dlugosc skrzyni [mm]
    A - pole przekroju poprzecznego skrzyni [mm2]
    
    Oblicza:
    gestosc_min - minimalna gestosc smieci
    
    
    '''
    
    V_skrzyni = c * A / 10**9 # m3
    gestosc_min = Ms / V_skrzyni # kg/m3
    
    return gestosc_min

def oblicz_procent_zapelnienia_skrzyni(x1, polozenie_sciany_max = 4227):
    '''
    Oblicz w ilu procentach zapelniona jest skrzynia na podstawie polozenia sciany wypychajacej w stosunku do dlugosci skrzyni
    
    Dane:
    x1 - polozenie sciany wypychajacej [mm]
    polozenie_sciany_max - max odczyt z czujnika laserowego podczas jazdy [mm], nie mylic z max odczyt w ogole, poniewaz przy wyladunku sciana wyjezdza poza skrzynie
    
    Oblicza:
    procent - w %
    
    '''
    
    procent = ((polozenie_sciany_max - x1) / polozenie_sciany_max) * 100 # [%]
    
    return procent

def oblicz_energie_na_tone(masa_smieci_series, energia_hydrauliczna_series):
    '''

    :param masa_smieci_series:  # w kg
    :param energia_hydrauliczna_series: w kJ
    :return:

    energia_na_tone - energia w przeliczeniu na 1 tone zebranych smieci w kJ/t
    '''

    masa_t = masa_smieci_series/1000 # masa smieci w tonach

    energia_na_tone = energia_hydrauliczna_series / masa_t # kJ/t

    #200*1000 /

    #energia_na_tone =
    energia_na_tone = energia_na_tone.round(1)


    return energia_na_tone


def oblicz_tonokilometry(series_masa_smieci, series_distance_m, min_masa_smieci_kg=0):
    """

    :param series_masa_smieci: masa smieci w kg
    :param series_distance_m: przebieg chwilowy w m
    :param min_masa_smieci_kg:

    :return:
    tonokilomerty - Series
    """



    # wyzeruj wiersze gdzie masa pojazdu jest mniejsza niz min_masa_smieci_kg
    series_masa_smieci[series_masa_smieci < min_masa_smieci_kg] = 0



    ms_t = series_masa_smieci / 1000  # masa smieci w tonach
    dist_km = series_distance_m / 1000  # chwilowy przebieg w km

    tonokilomerty_temp = (ms_t * dist_km).round(1)

    tonokilomerty = tonokilomerty_temp.cumsum()

    return tonokilomerty


def oblicz_tonokilometry_przeladowane(series_nacisk_total, series_distance_m, nacisk_min=26000):
    """

    :param series_nacisk_total: masa calkowita pojazdu, czyli naciski na osie lacznie
    :param series_distance_m: przebieg chwilowy w m
    :param nacisk_min: masa pojazdu uznawana za mase dopuszczalna = 26 t


    :return:
    tonokilomerty - Series
    """

    # wyzeruj wiersze gdzie masa pojazdu jest mniejsza niz min_masa_smieci_kg
    series_nacisk_total = series_nacisk_total-nacisk_min
    series_nacisk_total[series_nacisk_total < 0] = 0



    masa_t = series_nacisk_total / 1000  # naciks na osie w tonach
    dist_km = series_distance_m / 1000  # chwilowy przebieg w km

    tonokilomerty_temp = (masa_t * dist_km).round(1)

    tonokilomerty = tonokilomerty_temp.cumsum()

    return tonokilomerty


def oblicz_motogodziny(series_predkosc_kol, series_rpm, rpm_min=500, rpm_max=650, predkosc_max=0, predkosc_min=0, tylko_postoj=True):
    '''

    :param series_predkosc_kol:
    :param series_rpm:
    :param rpm_min:
    :param rpm_max:
    :param predkosc_max:
    :param predkosc_min:
    :param tylko_postoj:
    :return:
    '''


    if tylko_postoj:
        index_ = (series_predkosc_kol <= predkosc_max) & ((series_rpm < rpm_max) & (series_rpm > rpm_min))
    else:
        index_ = (series_predkosc_kol >= predkosc_min) & ((series_rpm < rpm_max) & (series_rpm > rpm_min))

    sekundy = series_rpm.copy()

    sekundy[~index_] = 0
    sekundy[index_] = 1

    sekundy = sekundy.cumsum()

    motogodziny = np.round(sekundy / 3600, 2)

    return motogodziny


def oblicz_motogodziny_z_naciskiem(series_nacisk_total, nacisk_min=26000):
    '''

    :param series_nacisk_total:
    :param nacik_min:
    :return:
    '''
    index_ = (series_nacisk_total >= nacisk_min)

    sekundy = series_nacisk_total.copy()

    sekundy[~index_] = 0
    sekundy[index_] = 1

    sekundy = sekundy.cumsum()

    motogodziny = np.round(sekundy / 3600, 2)

    return motogodziny


def oblicz_zuzycie_paliwa(zuzyte_paliwo_total, przebieg, motogodziny):
    '''
    Oblicza godzinowe_zuzycie_paliwa i przebiegowe_zuzycie_paliwa


    :param zuzyte_paliwo_total: float w dm3
    :param przebieg: float w km
    :param motogodziny: float w h
    :return:
    godzinowe_zuzycie_paliwa - float
    przebiegowe_zuzycie_paliwa - float
    '''

    godzinowe_zuzycie_paliwa = np.round(zuzyte_paliwo_total / motogodziny, 1)  # dm3/h
    przebiegowe_zuzycie_paliwa = np.round(zuzyte_paliwo_total / przebieg * 100, 1)  # dm/100km

    return godzinowe_zuzycie_paliwa, przebiegowe_zuzycie_paliwa

def znajdz_cykle_zageszczania2(control_signal, dt_series, offset_start=-2, offset_end=12):
    '''
    Na podstawie sygnalu sterujacego znajduje krawedz typu "rising" (kolejna wartosc jest wieksza od poprzedniej),
    nastepnie wyznacza okresy czasu wg. offset_start i offset_end i tworzy cykle czasu z ktorych zwraca serię
    z nr cykli odpowiadającemu danemu wierszowi, dla pozostałych danych nr cyklu rowny -1

    Input
    -----
    control_signal, <list type>
    dane sterowania do znajdywania cykli, najlepiej w formie 0 i 1 gdzie przeskok 0->1 bedzie wykrywany jako cykl

    dt_series, <pd.Series>
    dane data czas wg. ktorych zostana podzielone cykle

    offset_start, <number> (default: -2)
    przesuniecie poczatku cyklu w sekundach

    offset_end, <number> (default: 12)
    przesuniecie konca cyklu w sekundach

    Output
    ------
    series_out, <pd.Series>
    podzielona na cykle seria danych
    '''

    series_out = control_signal.copy()
    series_out.loc[:] = -1

    for i, id in enumerate(control_signal[control_signal.diff() > 0].index):
        time_now = dt_series[id]
        time_start = time_now + pd.Timedelta(seconds=offset_start)
        time_end = time_now + pd.Timedelta(seconds=offset_end)

        series_out.loc[dt_series[dt_series > time_start][dt_series < time_end].index] = i


    return series_out


def znajdz_cykle_zageszczania(series_cisnienie, cisnienie_min=120):
    '''
    Znajduje cykle zageszczania smieni na podstawie pikow cisnienia w przebiegu cisnienia. Dla uproszczenia wszystkie cisnienia mniejsze niz cisnienie_min zostaly wyzerowane.


    :param series_cisnienie:
    :param cisnienie_min:
    :return:

    cykl_prasowania - Series
    '''
    series_cisnienie_temp = series_cisnienie.copy()


    series_cisnienie_temp[series_cisnienie_temp < cisnienie_min] = 0

    piki = find_peaks(series_cisnienie_temp)[0]

    cykle_zageszczania = series_cisnienie_temp*0
    cykle_zageszczania[piki] = 1

    return cykle_zageszczania


def przelicz_cykle_zageszczania_w_dniu(cykl_prasowania_series):


    cykle_cumsum = cykl_prasowania_series.cumsum()

    return cykle_cumsum


def grupowanie_przejazdow2(df, kolumna_podzialu='cykl', kolumna_danych='cisnienie', kolumna_czas='Data_godzina',
                           metric="dtw", ilosc_grup=8,
                           load=False, load_loc=None, save=False, save_loc=".", savename="TSKM"):
    '''
    Grupuje przejazdy na podstawie kolumny podzialu i kolumny danych uzywajac klasyfikatora Kmeans

    *************
    In:

     df <pandas.DataFrame> - dane
     kolumna_podzialu <String> - (default: 'cykl') kolumna wg ktorej zostana podzielone przejazdy
     kolumna_danych <String> - (default: 'cisnienie') kolumna z danymi ktore zostana podane do klasyfikatora
     metric <String> - (default:"dtw") kernel KMeans, mozliwe opcje {“euclidean”, “dtw”, “softdtw”} wiecej w dokumentacji
                         tslearn.clustering.TimeSeriesKMeans

     ilosc_grup <int> - (default: 8) ilosc grup wg ktorych bedzie grupowal klasyfikator

     okno_czas <int> - (default: 50) badany czas dla kazdego przejazdu w sekundach

     load <boolean> - (default: False) decyduje o tym czy klasyfikator bedzie wczytany z lokalizacji load_loc czy trenowany na podstawie danych
     load_loc <string> - (default: None) lokalizacja klasyfikatora z ktorej bedzie wczytany jesli load = True

     save <boolean> - (default: False) decyduje o tym klasyfikator zostanie zapisany
     save_loc <string> - (default: ".") lokalizacja zapisanego klasyfikatora jesli save = True
     savename <string> - (default: "TSKM") nazwa zapisanego klasyfikatora

    Out:

     <pandas.DataFrame> - Dataframe z danymi i kolumna "grupa"
     <tslearn.clustering.TimeSeriesKMeans> - KMeans ktory byl wykorzystany do klasyfikacji, pozwala np. na wyciagniecie centrow grup

    '''

    X_data = []
    y_keys = []
    km = TimeSeriesKMeans(n_clusters=ilosc_grup, verbose=False, metric=metric)

    df_temp = df[df[kolumna_podzialu] >= 0]

    for i in df_temp[kolumna_podzialu].unique():
        if len(df[df[kolumna_podzialu] == i][kolumna_danych].values) > 1:
            temp = df[df[kolumna_podzialu] == i]
            dt_time = temp[kolumna_czas].dt.to_pydatetime()
            temp_dt = [(x - dt_time[0]).total_seconds() for x in dt_time]
            y_keys.append(i)
            X_data.append(list(zip(temp_dt,
                                   temp[kolumna_danych].values)))

    X_data = to_time_series_dataset(X_data)

    if load and load_loc is not None:
        km = TimeSeriesKMeans.from_json(load_loc)
        y_pred = km.predict(X_data)

    else:
        y_pred = km.fit_predict(X_data)

    if save and save_loc is not None:
        km.to_json(save_loc + "/" + savename + ".json")

    slownik_mapowania = {}

    for nr_prz, klasa in zip(y_keys, y_pred):
        slownik_mapowania[nr_prz] = klasa

    series = df[kolumna_podzialu].map(slownik_mapowania)

    return series

def przypisz_ocene_cyklu(series_nr_clustra_cyklu):


    cykle_lekkie_obciazenie = series_nr_clustra_cyklu*0
    cykle_srednie_obciazenie = series_nr_clustra_cyklu * 0
    cykle_duze_obciazenie = series_nr_clustra_cyklu * 0

    nr_clastow_lekkie_obc = [4]
    nr_clastow_srednie_obc = [0,5]
    nr_clastow_duze_obc = [1,2,3,6,7]



    cykle_lekkie_obciazenie.loc[series_nr_clustra_cyklu.isin(nr_clastow_lekkie_obc)] = 1

    cykle_srednie_obciazenie.loc[series_nr_clustra_cyklu.isin(nr_clastow_srednie_obc)] = 1

    cykle_duze_obciazenie.loc[series_nr_clustra_cyklu.isin(nr_clastow_duze_obc)] = 1

    return cykle_lekkie_obciazenie, cykle_srednie_obciazenie, cykle_duze_obciazenie



def dane_ruch_zgarniaka_i_suwaka(df_Series, kolumny=['podnoszenie_suwaka', 'opuszczanie_suwaka', 'zamykanie_zgarniaka', 'otwieranie_zgarniaka']):
    '''
    Przerabia dane liczbowe na stany okreslajace poszczegolne ruchy suwaka i zgarniaka

    Parametry
    ---------
    df_Series: pandas.Series
    Kolumna zawierajaca okreslona ramke danych (203)

    kolumny: lista
    Kolumna z nazwami kolumn w kolejnosci bitow
    default: ['podnoszenie_suwaka', 'opuszczanie_suwaka', 'zamykanie_zgarniaka', 'otwieranie_zgarniaka']

    Zwraca
    ------
    pandas.DataFrame
    Tabela z danymi o stanach 0 i 1

    Przyklad
    --------
    dane_suwak_zgarniak =  dane_ruch_zgarniaka_i_suwaka(df['ruch_zgarniaka_i_suwaka'])
    '''
    # zamiana liczb na listy wartosci binarnych np. 64+7 -> [0, 1, 1, 1] gdzie 64 usuwane przez modulo 16
    dane_binarne = df_Series.astype('int').apply(lambda x: [int(x) for x in bin(x%16)[2:].zfill(4)])

    return pd.DataFrame.from_records(dane_binarne, columns=kolumny)


def przygotuj_dane(df):

    ## PARAMETRY POJAZDU
    
    #    Wp - waga pojazdu pustego [kg]
    #    a - polozenie srodka ciezkosci pojazdu pustego od osi 1 [mm]
    #    b - odleglosc miedzy osia 1 a poczatkiem skrzyni zaladowczej [mm]
    #    c - max wysuniecie sciany podczas jazdy [mm], max wskazanie czujnika laserowego
    #    L - rozstaw osi [mm]
    #    A - pole powierzchnie przekroju skrzyni [mm2]
        
        
    Wp = 15070
    a = 4504 * (6435+4030)/15070
    b = 560
    c = 4227
    L = 4504
    A = 5022500

    mapa_kolumn = {
        'updatedAt':'Data_godzina',
        'FMSVW':'Nacisk_osie_23',
        'VEL':'predkosc_osi',
        'FMSFL':'Fuel_level',
        'FMSLFC':'Fuel_consumption',
        'FMSWBSPEED':'predkosc_kol',
        'FMSAIRTEMP':'temperatura_zewn',
        'FMSRPM':'RPM',
        'XT_XCAN_U16_000':'polozenie_sciany',
        'XT_XCAN_U16_001':'cisnienie',
        'XT_XCAN_U8_000':'bypass1_1',
        'XT_XCAN_U8_001':'odwlok_otwieranie',
        'XT_XCAN_U8_002':'bypass1_2',
        'XT_XCAN_U8_003':'wsuwanie_sciany_niska_predkosc',
        'XT_XCAN_U8_004':'wsuwanie_sciany_wysoka_predkosc',
        'XT_XCAN_U8_005':'przetwornik_cisnienia',
        'XT_XCAN_U8_006':'ruch_zgarniaka_i_suwaka',
        'XT_XCAN_U8_007':'pozycja_suwaka_i_odwloka',
        'XT_XCAN_U8_008':'prawa_start_automatyczny',
        'XT_XCAN_U8_009':'prawa_otworz_zgarniak',
        'XT_XCAN_U8_010':'prawa_przycisk_suwaka',
        'XT_XCAN_U8_011':'system_bezpieczenstwa_1',
        'XT_XCAN_U8_012':'system_bezpieczenstwa_2',
        # 'XT_XCAN_U8_013':'blank1',
        # 'XT_XCAN_U8_014':'blank2',
        # 'XT_XCAN_U8_015':'blank3',
        'XT_XCAN_U8_016':'alert_1_1',
        'XT_XCAN_U8_017':'alert_1_2',
        'XT_XCAN_U8_018':'alert_1_3',
        'XT_XCAN_U8_019':'alert_1_4',
        'XT_XCAN_U8_020':'alert_1_5',
        'XT_XCAN_U8_021':'alert_1_6',
        'XT_XCAN_U8_022':'alert_1_7',
        'XT_XCAN_U8_023':'alert_1_8',
        'XT_XCAN_U8_024':'alert_2_1',
        'XT_XCAN_U8_025':'alert_2_2',
        'XT_XCAN_U8_026':'alert_2_3',
        'XT_XCAN_U8_027':'alert_2_4',
        'XT_XCAN_U8_028':'alert_2_5',
        'XT_XCAN_U8_029':'alert_2_6',
        'XT_XCAN_U8_030':'alert_2_7',
        'XT_XCAN_U8_031':'alert_2_8',
        'XT_UCAN_U8_000':'downloadID',
        'XT_UCAN_U8_001':'wersja_software',
        'XT_UCAN_I16_000':'rms_IN00',
        'XT_UCAN_I16_001':'prad_IN01',
        'XT_UCAN_I16_002':'rms_IN02',
        'XT_UCAN_I16_003':'prad_IN03',
        'XT_UCAN_I16_004':'napiecie_IN04',
        'XT_UCAN_I16_005':'napiecie_IN05',
        'XT_UCAN_I16_006':'napiecie_IN06',
        'XT_UCAN_I16_007':'napiecie_IN07',
        'XT_UCAN_I16_008':'rezystancja_IN08',
        'XT_UCAN_I16_009':'rezystancja_IN10',
        'XT_UCAN_I16_010':'temperatura_IN12',
        'XT_UCAN_I16_011':'temperatura_IN14',
        'XT_UCAN_I16_020':'rms_X_1',
        'XT_UCAN_I16_021':'rms_Y_1',
        'XT_UCAN_I16_022':'rms_Z_1',
        'XT_UCAN_I16_024':'rms_X_2',
        'XT_UCAN_I16_025':'rms_Y_2',
        'XT_UCAN_I16_026':'rms_Z_2',
    }

    # Usun duplikaty
    df = df.drop_duplicates(['updatedAt'])

    # Zmien nazwy kolumn
    df = df[list(mapa_kolumn.keys())].rename(columns=mapa_kolumn)#.ffill().bfill()

    # Popraw indeks tak ze kazdy wiersz odpowiada 1 sekundzie. nany wypelnione metoda 'ffill'
    df['sekundy'] = (df['Data_godzina'].values - df.iloc[[0]]['Data_godzina'].values[0])
    df['sekundy'] = pd.to_timedelta(df['sekundy']).dt.total_seconds().astype('int')
    new_index = pd.DataFrame()
    x = df['sekundy'].iloc[-1]
    new_index['sekundy'] = np.arange(0, x)
    df = df.reset_index().merge(new_index, on='sekundy', how='right').fillna(method='ffill')
    df['Data_godzina'] = df['Data_godzina'][0] + pd.to_timedelta(new_index['sekundy'], unit='s')
    df = df.dropna(subset=['Data_godzina'])

    # Cisnienie
    df['cisnienie_bar'] = przelicz_cisnienie_bar(df['cisnienie'])

    # Temperatura sworznia
    kolumny_z_czujnikami = ['temperatura_IN12', 'temperatura_IN14']
    przelicz_temperature(df, kolumny_z_czujnikami, czujnik='Febi Bilstein 28334')
    df['delta_temp_PIN_1'] = przelicz_delta_temperatur(df['temperatura_IN12'], df['temperatura_zewn'])
    df['delta_temp_PIN_2'] = przelicz_delta_temperatur(df['temperatura_IN14'], df['temperatura_zewn'])

    # Czujnik RMS
    kolumny_z_akcelerometrami = ['rms_IN00', 'rms_IN02']
    df = przelicz_akcelerometry(df, kolumny_z_akcelerometrami, czujnik='IFM VTV122')

    # Sterowanie
    # Bypass
    df[['wysuwanie_sciany','bypass_1']] = dane_bypass_wysuwanie_sciany(df['bypass1_1'], kolumny=['wysuwanie_sciany', 'bypass_1'])

    df[['podnoszenie_suwaka', 'opuszczanie_suwaka', 'zamykanie_zgarniaka', 'otwieranie_zgarniaka']] = dane_ruch_zgarniaka_i_suwaka(df['ruch_zgarniaka_i_suwaka'], kolumny=['podnoszenie_suwaka', 'opuszczanie_suwaka', 'zamykanie_zgarniaka', 'otwieranie_zgarniaka'])

    # Przebieg
    df['distance_m'], df['przebieg_km'] = przelicz_przebieg(df['predkosc_kol'])

    # Motogodziny
    df['motogodziny_jalowy'] = oblicz_motogodziny(df['predkosc_kol'], df['RPM'], rpm_min=500, rpm_max=650,
                                                  predkosc_max=0, predkosc_min=0, tylko_postoj=True)
    df['motogodziny_900rpm_zabudowa'] = oblicz_motogodziny(df['predkosc_kol'], df['RPM'], rpm_min=850, rpm_max=950,
                                                           predkosc_max=0, predkosc_min=0, tylko_postoj=True)
    df['motogodziny_jazda'] = oblicz_motogodziny(df['predkosc_kol'], df['RPM'], rpm_min=500, rpm_max=2500,
                                                  predkosc_max=1000, predkosc_min=1, tylko_postoj=False)
    df['motogodziny_total'] = oblicz_motogodziny(df['predkosc_kol'], df['RPM'], rpm_min=500, rpm_max=2500,
                                                 predkosc_max=1000, predkosc_min=0, tylko_postoj=False)


    # Paliwo
    df['Fuel_consumption'] = df['Fuel_consumption'] - df['Fuel_consumption'][0]

    # Energia
    df['hydrualic_power'], df['hydraulic_energy'] = przelicz_energie_hydr(df['cisnienie_bar'], df['RPM'])
    df['energia_hydr_zageszczania'] = przelicz_energie_zuzyta_na_zageszczanie(df['hydrualic_power'], df['predkosc_kol'], df['RPM'], min_rpm=850)
    df['energia_z_paliwa'] = przelicz_energie_z_paliwa(df['Fuel_consumption'], density = 0.82, efficiency = 0.2, heat_value = 43000)

    # Zapelnienie skrzyni
    df['polozenie_sciany_mm'] = przelicz_polozenie_sciany(df['polozenie_sciany'], df['RPM'])
    df['zapelnienie_skrzyni_procent'] = oblicz_procent_zapelnienia_skrzyni(df['polozenie_sciany_mm'], c)

    # Masa i gestosc smieci
    df.loc[(df['predkosc_kol'] >1) & (df['RPM'] < 800), 'Nacisk_osie_23'] = np.nan
    df['Nacisk_osie_23'] = df['Nacisk_osie_23'].fillna(method='ffill').fillna(method='bfill')
    df['srodek_ciezkosci_smieci'], df['Masa_smieci'] , df['Nacisk_total'], df['Nacisk_os_1'], df['ratio_N1N2'] = przelicz_naciski_i_masy(N = df['Nacisk_osie_23'], x1 = df['polozenie_sciany_mm'], Wp = Wp, a = a, b = b, c=c, L = L)
    df['distance_m_przeladowana'], df['przebieg_km_przeladowana'] = przelicz_przebieg_z_naciskiem_X(df['predkosc_kol'], df['Nacisk_total'], nacisk_na_osie_max = 26000 )
    df['gestosc_smieci_min'] = oblicz_minimalna_gestosc_pelnej_skrzyni(df['Masa_smieci'], c, A)
    df['gestosc_smieci'] = przelicz_gestosc_odpadow(df['polozenie_sciany_mm'], df['Masa_smieci'], c= c,  A = A)
    df['motogodziny_przeladowana'] = oblicz_motogodziny_z_naciskiem(df['Nacisk_total'], nacisk_min=26000)
    df['distance_m_przeladowana'], df['przebieg_km_przeladowana'] = przelicz_przebieg_z_naciskiem_X(df['predkosc_kol'], df['Nacisk_total'], nacisk_na_osie_max = 26000)

    # Tonokilometry
    df['Tonokilometry_masa_smieci'] = oblicz_tonokilometry(df['Masa_smieci'], df['distance_m'], min_masa_smieci_kg=0)
    df['Tonokilometry_przeladowane'] = oblicz_tonokilometry_przeladowane(df['Nacisk_total'], df['distance_m'], nacisk_min=26000)

    # Cykle robocze

    df['cykle_zageszczania'] = znajdz_cykle_zageszczania2(df['podnoszenie_suwaka'], df['Data_godzina'], offset_start=-2, offset_end=12)
    df['cykle_zageszczania_wg_zamykanie_zgarniaka'] = znajdz_cykle_zageszczania2(df['zamykanie_zgarniaka'], df['Data_godzina'], offset_start=-2,
                                                          offset_end=12)
    df['cykle_zageszczania_wg_opuszczanie_suwaka'] = znajdz_cykle_zageszczania2(df['opuszczanie_suwaka'], df['Data_godzina'], offset_start=-2,
                                                          offset_end=12)

    df['cluster_cyklu_prasowania'] = grupowanie_przejazdow2(df, kolumna_podzialu='cykle_zageszczania', kolumna_danych='cisnienie', kolumna_czas='Data_godzina',
                           metric="dtw", ilosc_grup=8,
                           load=True, load_loc='data/misc/KMeans_cykle_21-09do24-09.json')


    df['cykl_zageszczania_100'] = znajdz_cykle_zageszczania(df['cisnienie_bar'], cisnienie_min=100)
    df['cykle_zageszczania_100'] = przelicz_cykle_zageszczania_w_dniu(df['cykl_zageszczania_100'])

    df['cykl_zageszczania_150'] = znajdz_cykle_zageszczania(df['cisnienie_bar'], cisnienie_min=150)
    df['cykle_zageszczania_150'] = przelicz_cykle_zageszczania_w_dniu(df['cykl_zageszczania_150'])

    df['cykl_zageszczania_200'] = znajdz_cykle_zageszczania(df['cisnienie_bar'], cisnienie_min=200)
    df['cykle_zageszczania_200'] = przelicz_cykle_zageszczania_w_dniu(df['cykl_zageszczania_200'])

    df['cykle_lekkie'], df['cykle_srednie'], df['cykle_ciezkie'] = przypisz_ocene_cyklu(df['cluster_cyklu_prasowania'])


    # Ilosc przepompowanego oleju
    df['wydatek_oleju_chwilowy'], df['ilosc_przepompowanego_oleju'] = przelicz_ilosc_oleju(df['cisnienie_bar'], df['RPM'], cisnienie_min=0, r_pto = 1.19, pump_displacement = 69)
    _ ,  df['ilosc_przepompowanego_oleju_120bar'] = przelicz_ilosc_oleju(df['cisnienie_bar'], df['RPM'], cisnienie_min=120, r_pto=1.19, pump_displacement=69)

    # Energia na tone
    df['energia_hydr_na_tone_smieci'] = oblicz_energie_na_tone(df['Masa_smieci'], df['hydraulic_energy'], )
    df['energia_hydr_zageszczania_na_tone_smieci'] = oblicz_energie_na_tone(df['Masa_smieci'], df['energia_hydr_zageszczania'])


    return df




