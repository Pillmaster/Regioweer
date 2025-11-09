import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone 
import numpy as np
import time
from math import radians, sin, cos, sqrt, atan2, degrees
import plotly.express as px
import altair as alt

# --- Globale Configuratie ---
BASE_API_URL = "http://api.temperatur.nu/tnu_1.17.php"
REFRESH_INTERVAL_SECONDS = 300 # 5 minuten (300 seconden)
REFRESH_COORDS_TTL = 3600 * 24 * 7 # 1 week

# UW LOCATIE
UW_LAT = 63.0243695749981
UW_LON = 17.03321183785952
# Locatie voor Voorspelling (Graninge, overgenomen uit grafiek5.py)
FORECAST_LAT = 63.024625
FORECAST_LON = 17.035304

# CENTRALE STATIONSCONFIGURATIE
STATIONS = {
    "Graninge/Sjön": "graninge_sjon",    
    "Åkroken": "akroken",
    "Graninge": "graninge_tv",     
    "Ärtrik": "artrik",
    "Grillom": "grillom",
    "Fångsjöbacken": "fangsjobacken",
    "Långsele": "langsele",
    "Sollefteå/Trästa": "trasta"
    
}

# Mapping van de gebruiksvriendelijke labels naar de API-parameters
TIDSSPANNE_KEUZES = {
    'Huidige Kalenderdag (00:00 - nu)': '1day', 
    'Laatste 24 uur': '1day',
    'Laatste week': '1week',
    'Laatste maand': '1month'
}
TIDSSPANNE_KEYS_ORDERED = list(TIDSSPANNE_KEUZES.keys())

# --- FUNCTIES VOOR GEOGRAFISCHE BEREKENINGEN (Ongewijzigd) ---

def bepaal_windrichting(hoek):
    """Vertaalt een hoek in graden (0-360) naar een windrichting."""
    richtingen = ["N", "NNO", "NO", "ONO", "O", "OZO", "ZO", "ZZO", "Z", "ZZW", "ZW", "WZW", "W", "WNW", "NW", "NNW"]
    index = round(hoek / (360. / len(richtingen))) % len(richtingen)
    return richtingen[index]

def bereken_afstand_en_richting(lat1, lon1, lat2, lon2):
    """
    Berekent de afstand (Haversine formule) en de windrichting (initial bearing).
    Output in km and Dutch wind direction.
    """
    R = 6371 # Straal van de aarde in kilometers
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    afstand_km = R * c

    y = sin(dlon) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    bearing_rad = atan2(y, x)
    bearing_deg = (degrees(bearing_rad) + 360) % 360
    richting_nl = bepaal_windrichting(bearing_deg)

    # Retourneer de volledige, opgemaakte string voor het label
    return f"{afstand_km:.1f} km 🧭 {richting_nl}"


# --- FUNCTIES VOOR API-AANROEPEN (Ongewijzigd) ---

@st.cache_data(ttl=REFRESH_COORDS_TTL)
def haal_station_coordinaten_op(station_codes):
    """Haalt de coördinaten voor de stations op via de 'coordinates' API parameter."""
    p_param = ",".join(station_codes)
    params = {'p': p_param, 'cli': 'streamlit_coords_app', 'coordinates': '' }
    coordinaten_dict = {}
    try:
        response = requests.get(BASE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('stations'):
            for station_data in data['stations']:
                station_code_from_api = station_data.get('id') 
                station_name = next((name for name, code in STATIONS.items() if code == station_code_from_api), None)
                if station_name and 'lat' in station_data and 'lon' in station_data:
                    coordinaten_dict[station_name] = (
                        float(station_data['lat'].replace(',', '.')), 
                        float(station_data['lon'].replace(',', '.'))
                    )
        return coordinaten_dict
    except Exception as e:
        return {}


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def haal_actuele_data_op(station_id):
    """Haalt de meest recente temperatuur en de lastUpdate op voor één station."""
    params = {'p': station_id, 'cli': 'streamlit_current_app', 'verbose': ''} 
    try:
        response = requests.get(BASE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('stations'):
            station_data = data['stations'][0]
            temp_data = {
                'temp': station_data.get('temp'),
                'lastUpdate': station_data.get('lastUpdate'),
            }
            if temp_data['temp'] is not None and temp_data['lastUpdate']:
                return temp_data, "OK"
        return None, f"Fout: Actuele data ontbreekt in de JSON-respons voor {station_id}."
    except requests.exceptions.RequestException as e:
        return None, f"Verbindingsfout (Actueel): {e}"
    except Exception as e:
        return None, f"Onverwachte fout (Actueel): {e}"


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def haal_historische_data_op(station_id, span_value):
    """Haalt historische ruwe meetpunten op voor één station en tijdspanne."""
    params = {'p': station_id, 'cli': 'streamlit_history_app', 'data': '', 'span': span_value }
    try:
        response = requests.get(BASE_API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('stations') and data['stations'][0].get('data'):
            raw_data = data['stations'][0]['data']
            if isinstance(raw_data, list) and len(raw_data) > 0:
                df = pd.DataFrame(raw_data)
                df.columns = ['Timestamp', 'Temperature']
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
                return df.set_index('Timestamp'), "OK" 
            else:
                return pd.DataFrame(), f"De API leverde geen geldige historische meetpunten voor {station_id} in deze periode."
        return pd.DataFrame(), f"Fout: Historische data ontbreekt in de JSON-respons voor {station_id}."
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Verbindingsfout (Historisch): {e}"
    except Exception as e:
        return pd.DataFrame(), f"Onverwachte fout (Historisch): {e}"

# --- FUNCTIES VOOR KORTE TERMIJN VOORSPELLINGEN (Plotly) ---

@st.cache_data(ttl=3600)
def get_smhi_forecast(lat, lon):
    """Haalt temperatuurvoorspelling op van SMHI (pmp3g/latest) en filtert op 1-uurs interval."""
    url = f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{lon:.4f}/lat/{lat:.4f}/data.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        forecast_data = []
        for item in data.get('timeSeries', []):
            temp_param = next((param['values'][0] for param in item['parameters'] if param['name'] == 't'), None)
            if temp_param is not None:
                forecast_data.append({
                    'Tijd (UTC)': pd.to_datetime(item['validTime']), 
                    'Temperatuur (°C)': temp_param,
                    'Bron': 'SMHI'
                })

        df = pd.DataFrame(forecast_data)
        if not df.empty:
            df = df.sort_values('Tijd (UTC)').reset_index(drop=True)
            time_diffs = df['Tijd (UTC)'].diff().dt.total_seconds().fillna(0)
            first_large_jump_index = time_diffs[time_diffs > 3600].index.min()
            
            if pd.notna(first_large_jump_index):
                df = df.iloc[:first_large_jump_index]
            
        return df, None

    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Fout bij SMHI API: {e}"


@st.cache_data(ttl=3600)
def get_yr_forecast(lat, lon):
    """Haalt temperatuurvoorspelling op van YR.no (MET Locationforecast 2.0) en filtert op 1-uurs interval."""
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat:.4f}&lon={lon:.4f}"
    headers = {
        'User-Agent': 'MijnStreamlitApp/1.0.0 (https://github.com/Pillmaster)' 
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        forecast_data = []
        for item in data.get('properties', {}).get('timeseries', []):
            temp_data = item.get('data', {}).get('instant', {}).get('details', {}).get('air_temperature')
            
            if temp_data is not None:
                forecast_data.append({
                    'Tijd (UTC)': pd.to_datetime(item['time']), 
                    'Temperatuur (°C)': temp_data,
                    'Bron': 'YR.no'
                })

        df = pd.DataFrame(forecast_data)
        if not df.empty:
            df = df.sort_values('Tijd (UTC)').reset_index(drop=True)
            time_diffs = df['Tijd (UTC)'].diff().dt.total_seconds().fillna(0)
            first_large_jump_index = time_diffs[time_diffs > 3600].index.min()
            
            if pd.notna(first_large_jump_index):
                df = df.iloc[:first_large_jump_index]

        return df, None

    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Fout bij YR.no API: {e}"
        
# --- NIEUWE FUNCTIES VOOR LANGE TERMIJN VOORSPELLINGEN (Altair) ---
# Mapping voor neerslag categorieën (pcat)
PRECIP_CATEGORIES = {
    0: 'Geen Neerslag', 1: 'Sneeuw', 2: 'Sneeuw/Regen', 3: 'Regen', 
    4: 'Motregen', 5: 'IJzel', 6: 'Vriezende Regen'
}

@st.cache_data(ttl=3600)
def fetch_long_term_smhi_data(lat, lon):
    """Haalt uitgebreide weersvoorspelling op van de SMHI API en verwerkt deze."""
    # Gebruik een lichte afronding voor de API URL
    url = (
        f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/"
        f"lon/{lon:.4f}/lat/{lat:.4f}/data.json"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
    except requests.exceptions.RequestException as e:
        return None, None, f"Fout bij SMHI (Lange Termijn) API: {e}"

    approved_time_utc = data.get('approvedTime')
    approved_time_formatted = None
    if approved_time_utc:
        try:
            # Converteer UTC-tijd naar de lokale tijdzone (Stockholm)
            dt_utc = datetime.fromisoformat(approved_time_utc.replace('Z', '+00:00'))
            # stuur het door met UTC in de naam, de grafiek zal het verwerken
            approved_time_formatted = dt_utc.astimezone(timezone(timedelta(hours=1))).strftime("%A %d %B %Y om %H:%M uur (CET/CEST)")
        except ValueError:
            approved_time_formatted = None

    forecast_data = []

    for entry in data.get('timeSeries', []):
        time_utc = entry['validTime']
        
        params = {p['name']: p['values'][0] for p in entry['parameters']}
        
        temperature = params.get('t')
        wind_speed = params.get('ws')
        wind_direction = params.get('wd')
        precip_category = params.get('pcat')
        precip_mean = params.get('pmean')

        if temperature is not None and wind_speed is not None:
            forecast_data.append({
                # Deze kolom is nu een tijdzone-aware datetime-object (UTC)
                'Datum/Tijd (UTC)': datetime.fromisoformat(time_utc.replace('Z', '+00:00')), 
                'Temperatuur (°C)': temperature,
                'Wind Snelheid (m/s)': wind_speed,
                'Wind Richting (gr)': wind_direction,
                'Neerslag (mm/h)': precip_mean,
                'Neerslag Type': PRECIP_CATEGORIES.get(precip_category, 'Onbekend'),
                'Temp > 0': temperature > 0 
            })

    if not forecast_data:
        return None, approved_time_formatted, "Geen bruikbare voorspellingsgegevens gevonden."
        
    df = pd.DataFrame(forecast_data)
    
    # Filter de data voor de komende ~10 dagen
    end_time = datetime.now(timezone.utc) + timedelta(days=10)
    df = df[df['Datum/Tijd (UTC)'] <= end_time].copy()
    
    # FIX: Verwijder tz_localize, omdat de kolom al tz-aware is (UTC)
    df['Datum/Tijd'] = df['Datum/Tijd (UTC)'].dt.tz_convert('Europe/Stockholm')


    return df, approved_time_formatted, "OK"


def create_temp_chart(df):
    """
    Grafiek voor temperatuur met geïnterpoleerde punten op de 0°C-as.
    """
    df_temp = df.copy()

    # --- Stap 1: Interpoleren van 0°C kruispunten (overgenomen van weergraf4.py) ---
    df_temp['sign'] = df_temp['Temperatuur (°C)'].apply(lambda x: 1 if x > 0 else -1)
    df_temp['sign_change'] = (df_temp['sign'] != df_temp['sign'].shift(-1)) & (df_temp['sign'].shift(-1).notna())

    zero_crossings = []
    
    for index, row in df_temp[df_temp['sign_change']].iterrows():
        t1 = row['Temperatuur (°C)']
        time1 = row['Datum/Tijd'] # Lokale tijd
        
        if index + 1 in df_temp.index:
            t2 = df_temp.loc[index + 1, 'Temperatuur (°C)']
            time2 = df_temp.loc[index + 1, 'Datum/Tijd'] # Lokale tijd
            
            fraction = abs(t1) / (abs(t1) + abs(t2))
            time_zero = time1 + (time2 - time1) * fraction
            
            zero_crossings.append({
                'Datum/Tijd': time_zero,
                'Temperatuur (°C)': 0.0,
                # Alleen de benodigde kolommen toevoegen, de gesplitste kolommen worden later gemaakt
            })

    if zero_crossings:
        # Belangrijk: De index moet de 'Datum/Tijd' kolom bevatten om samen te voegen
        df_crossings = pd.DataFrame(zero_crossings)
        df_temp = pd.concat([df_temp.drop(columns=['sign', 'sign_change', 'Datum/Tijd (UTC)']), df_crossings], ignore_index=True)
        df_temp = df_temp.sort_values(by='Datum/Tijd').reset_index(drop=True)
        
    # --- Stap 2: Splitsen voor Altair plotten (met NaN/None) ---

    df_temp['Temp_Rood'] = df_temp.apply(
        lambda row: row['Temperatuur (°C)'] if row['Temperatuur (°C)'] > 0 else (0.0 if row['Temperatuur (°C)'] == 0.0 else None),
        axis=1
    )
    
    df_temp['Temp_Blauw'] = df_temp.apply(
        lambda row: row['Temperatuur (°C)'] if row['Temperatuur (°C)'] <= 0 else (0.0 if row['Temperatuur (°C)'] == 0.0 else None), 
        axis=1
    )
    
    # --- Stap 3: Altair plotten ---

    base = alt.Chart(df_temp).encode(
        x=alt.X('Datum/Tijd', axis=alt.Axis(title='Tijd (Lokaal: CET/CEST)', format="%a %d %H:%M")),
        tooltip=[
            alt.Tooltip('Datum/Tijd', title='Tijd', format="%a %d-%m %H:%M"),
            alt.Tooltip('Temperatuur (°C)', format='.1f')
        ],
    ).properties(
        title='Temperatuur Voorspelling (Rood > 0°C, Blauw ≤ 0°C)'
    )

    chart_red = base.mark_line(point={'filled': True, 'size': 60}).encode(
        y=alt.Y('Temp_Rood', title='Temperatuur (°C)'), 
        color=alt.value('red')
    )

    chart_blue = base.mark_line(point={'filled': True, 'size': 60}).encode(
        y=alt.Y('Temp_Blauw', title='Temperatuur (°C)'), 
        color=alt.value('blue')
    )

    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', 
        size=2        
    ).encode(
        y='y'
    )
    
    chart = (chart_red + chart_blue + zero_line).interactive()

    return chart


def create_precipitation_chart(df):
    """Grafiek voor neerslag."""
    precip_df = df[df['Neerslag (mm/h)'] > 0.0].copy()
    
    if precip_df.empty:
         return alt.Chart(pd.DataFrame({'Tijd': [''], 'Neerslag (mm/h)': [0]})).mark_text(
            align='center', baseline='middle', fontSize=18, text='Geen Neerslag Voorspeld'
        ).encode()

    chart = alt.Chart(precip_df).mark_bar(color='blue').encode(
        x=alt.X('Datum/Tijd', axis=alt.Axis(title='Tijd (Lokaal: CET/CEST)', format="%a %d %H:%M")),
        y=alt.Y('Neerslag (mm/h)', title='Neerslag (mm/h)'),
        color=alt.Color('Neerslag Type', title='Type'),
        tooltip=[
            alt.Tooltip('Datum/Tijd', title='Tijd', format="%a %d-%m %H:%M"),
            alt.Tooltip('Neerslag (mm/h)', format='.2f'),
            alt.Tooltip('Neerslag Type')
        ]
    ).properties(title='Neerslag Voorspelling (mm/h)').interactive()
    return chart

def create_wind_chart(df):
    """
    Grafiek voor wind Snelheid (Lijn) en Richting (Pijlen op een vaste, negatieve positie).
    """
    df_chart = df.copy()
    # Voeg een constante kolom toe voor de vaste Y-positie van de pijlen (op -0.5 m/s)
    df_chart['Pijl_Positie'] = -0.5 
    
    base = alt.Chart(df_chart).encode(
        x=alt.X('Datum/Tijd', axis=alt.Axis(title='Tijd (Lokaal: CET/CEST)', format="%a %d %H:%M")),
        y=alt.Y('Wind Snelheid (m/s)', title='Wind Snelheid (m/s)', scale=alt.Scale(zero=False))
    )
    
    wind_line = base.mark_line(point={'filled': True, 'size': 50}, color='gray').encode(
        tooltip=[
            alt.Tooltip('Datum/Tijd', title='Tijd', format="%a %d-%m %H:%M"),
            alt.Tooltip('Wind Snelheid (m/s)', format='.1f'),
            alt.Tooltip('Wind Richting (gr)', format='.0f')
        ]
    )
    
    wind_arrows = base.mark_point(
        size=300,            
        shape="arrow",       
        color='black'
    ).encode(
        y=alt.Y('Pijl_Positie'), 
        angle=alt.Angle('Wind Richting (gr)', title='Richting'),
        tooltip=[
            alt.Tooltip('Datum/Tijd', title='Tijd', format="%a %d-%m %H:%M"),
            alt.Tooltip('Wind Snelheid (m/s)', format='.1f'), 
            alt.Tooltip('Wind Richting (gr)', format='.0f')
        ]
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', 
        size=2        
    ).encode(
        y='y'
    )
    
    chart = (wind_line + wind_arrows + zero_line).properties(
        title='Wind Snelheid (Lijn) en Richting (Pijlen)'
    ).interactive()

    return chart

# --- Dashboard UI Logica ---

st.set_page_config(
    page_title="Multi-Station Weer Dashboard",
    layout="wide"
)

st.title("❄️ Multi-Station Weer Dashboard")
# Update de timestamp om aan te geven wanneer de pagina voor het laatst is geladen
st.caption(f"Data van Trafikverket VViS via Temperatur.nu. **Dashboard laatst ververst:** **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    
# STAP 1: HAAL COÖRDINATEN OP VIA API
station_coordinaten_van_api = haal_station_coordinaten_op(list(STATIONS.values()))

# STAP 2: BEREKEN CONTEXT EN SLA AFSTANDSINFO OP
station_options_only_name = list(STATIONS.keys())
afstand_info_per_station = {}
for naam in STATIONS.keys():
    lat_st, lon_st = station_coordinaten_van_api.get(naam, (0, 0))
    
    if (lat_st, lon_st) != (0, 0):
        # Afstand en richting worden hier al opgemaakt als string (bijv. "14.1 km 🧭 ZO")
        afstand_info_per_station[naam] = bereken_afstand_en_richting(UW_LAT, UW_LON, lat_st, lon_st)
    else:
        afstand_info_per_station[naam] = "Onbekend"
        
# --- Linker Kolom (Sidebar) voor Variabelen/Keuzes ---
with st.sidebar:
    st.header("Instellingen")

    # VUL DE SIDEBAR met compacte stationsnamen (om afkappen te voorkomen)
    default_selection = [naam for naam in station_options_only_name][:3]

    gekozen_stations_namen = st.multiselect(
        "Kies één of meer weerstations:",
        options=station_options_only_name,
        default=default_selection
    )
    
    # 2. TIJDSPANNE KEUZE 
    keuze_label_tijd = st.selectbox(
        "Kies de periode voor de grafiek en statistieken:",
        options=TIDSSPANNE_KEYS_ORDERED,
        index=0
    )
    
    # NIEUWE CHECKBOX VOOR KORTE TERMIJN VOORSPELLING (PLOTLY)
    forecast_enabled = st.checkbox("Toon Korte Termijn Voorspelling (SMHI / YR.no)", value=False) 
    
    # NIEUWE CHECKBOX VOOR LANGE TERMIJN VOORSPELLING (ALTAIR)
    long_term_forecast_enabled = st.checkbox("Toon Uitgebreide SMHI-voorspelling (7-10 dagen)", value=False) # <<< NIEUW

    st.divider()
    st.markdown("ℹ️ Dashboard toont data van Trafikverket VViS en temperatur.nu.")
    st.markdown(f"Aantal beschikbare stations: **{len(STATIONS)}** (Vaste selectie)")
    st.markdown(f"Uw locatie (gesch.) gebruikt voor afstandsbepaling: **{UW_LAT:.3f} N, {UW_LON:.3f} E**")


# Vertaal de gekozen label naar de API-parameter
tijdspanne_api = TIDSSPANNE_KEUZES[keuze_label_tijd] 

# Rechter Kolom (Hoofdscherm) voor Data
st.header(f"Vergelijking: {', '.join(gekozen_stations_namen)}")

# --- Ophalen, Filteren en Plotten (Ongewijzigd) ---
if not gekozen_stations_namen:
    st.warning("Selecteer minstens één weerstation in het linkermenu om data te tonen.")
else:
    # DataFrames en lijsten voor verzameling
    alle_historie_df = pd.DataFrame()
    actuele_data = {}
    historische_stats_lijst = [] 
    foutmeldingen = []
    calculated_daily_extremes = {}
    
    # Itereren over geselecteerde stations
    for naam in gekozen_stations_namen:
        station_id = STATIONS.get(naam)
        
        # 1. Haal actuele data op 
        temp_data, status_actueel = haal_actuele_data_op(station_id)
        actuele_data[naam] = temp_data if status_actueel == "OK" else {"temp": None, "lastUpdate": None}
        
        # 2. Haal historische data op
        historie_df, historie_status = haal_historische_data_op(station_id, tijdspanne_api)
        
        if historie_status == "OK" and not historie_df.empty:
            
            temp_df = historie_df.copy()
            
            # --- Lokale Filtering voor 'Huidige Kalenderdag' ---
            if keuze_label_tijd == 'Huidige Kalenderdag (00:00 - nu)':
                vandaag_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                temp_df = temp_df[temp_df.index >= vandaag_start]
            
            # Sla de statistieken van de GEFILTERDE data op
            if not temp_df.empty:
                 
                max_temp = temp_df['Temperature'].max()
                min_temp = temp_df['Temperature'].min()
                mean_temp = temp_df['Temperature'].mean()

                max_tijdstip = temp_df['Temperature'].idxmax()
                min_tijdstip = temp_df['Temperature'].idxmin()

                # VUL DE DICTIONARY VOOR DE ACTUELE STATUS (alleen voor dagspecifieke spans)
                if keuze_label_tijd in ['Laatste 24 uur', 'Huidige Kalenderdag (00:00 - nu)']:
                    calculated_daily_extremes[naam] = {
                        'min': f"{min_temp:.1f}",
                        'max': f"{max_temp:.1f}"
                    }

                # HISTORISCHE STATISTIEKEN VOOR DE TABEL
                historische_stats_lijst.append({
                    'Station': naam,
                    'Max Temp (°C)': f"{max_temp:.1f}",
                    'Tijdstip Max': max_tijdstip.strftime('%Y-%m-%d %H:%M'),
                    'Min Temp (°C)': f"{min_temp:.1f}",
                    'Tijdstip Min': min_tijdstip.strftime('%Y-%m-%d %H:%M'),
                    'Gemiddelde Temp (°C)': f"{mean_temp:.1f}",
                })
                
            temp_df = temp_df.rename(columns={'Temperature': naam})
            
            if alle_historie_df.empty:
                alle_historie_df = temp_df
            else:
                alle_historie_df = alle_historie_df.join(temp_df, how='outer')
        
        elif historie_status != "OK":
             foutmeldingen.append(f"❌ {naam}: {historie_status}")
    
    
    # --- UI Layout: Actuele Status, Statistieken (Tabel), Grafiek ---
    
    # Deel 1: Actuele Status (Met berekende Extremen en Afstandsinfo)
    st.subheader("Actuele Temperatuur en Berekende Extremen")
    
    cols_metrics = st.columns(len(gekozen_stations_namen))
    
    for i, naam in enumerate(gekozen_stations_namen):
        data = actuele_data[naam]
        
        # Haal afstand/richting info op (bijv. "14.1 km 🧭 ZO")
        afstand_info = afstand_info_per_station.get(naam, "Afstand onbekend")
        
        with cols_metrics[i]:
            if data and data.get("temp") is not None:
                temperatuur = float(data["temp"])
                
                try:
                    dt_obj = datetime.strptime(data["lastUpdate"], '%Y-%m-%d %H:%M:%S')
                    tijd_format = dt_obj.strftime('%H:%M')
                except:
                    tijd_format = "Tijd onbekend"

                # HAAL BEREKENDE EXTREMEN OP
                extremes = calculated_daily_extremes.get(naam)
                if extremes:
                    min_vandaag_str = extremes['min']
                    max_vandaag_str = extremes['max']
                else:
                    min_vandaag_str = 'N/A'
                    max_vandaag_str = 'N/A'
                
                # 🚀 DE DEFINITIEVE LOCATIE: Afstand/richting direct in het label
                st.metric(
                    label=f"🌡️ {naam} ({afstand_info})", 
                    value=f"{temperatuur:.1f}°C",
                )
                
                # De caption is nu kort en bevat alleen de tijd en min/max
                st.caption(f"_{tijd_format}_ | Min: **{min_vandaag_str}°C** | Max: **{max_vandaag_str}°C**") 
            else:
                st.warning(f"{naam}: Geen actuele data.")

    st.divider()

    # Deel 2: Periode Overzicht (Statistieken in Tabel)
    if historische_stats_lijst:
        st.subheader(f"📊 Statistieken Overzicht: {keuze_label_tijd}")
        stats_df = pd.DataFrame(historische_stats_lijst)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.divider()
    
    # Deel 3: Historische Grafiek
    st.subheader(f"📈 Historisch Temperatuurverloop")

    if not alle_historie_df.empty:
        
        # Herformateer naar 'long' format voor Plotly Express
        plot_df = alle_historie_df.reset_index().melt(
            id_vars='Timestamp', 
            value_vars=alle_historie_df.columns, 
            var_name='Station', 
            value_name='Temperatuur (°C)'
        ).dropna(subset=['Temperatuur (°C)'])

        fig_historie = px.line(
            plot_df, 
            x='Timestamp', 
            y='Temperatuur (°C)', 
            color='Station', 
            title=f"Temperatuurverloop over {keuze_label_tijd}",
            markers=False, 
            hover_data={'Timestamp': "|%Y-%m-%d %H:%M", 'Temperatuur (°C)': ':.1f'}
        )
        
        fig_historie.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="0°C")

        fig_historie.update_xaxes(
            tickformat="%d-%m %H:%M", 
            showgrid=True,
            gridcolor='#eeeeee',
            title='Tijd (Lokaal: CET/CEST)' 
        )
        
        fig_historie.update_layout(
            yaxis_title="Temperatuur (°C)",
            margin=dict(b=70) 
        )

        st.plotly_chart(fig_historie, use_container_width=True) 

    else:
        st.warning("Er is geen geldige historische data beschikbaar voor de geselecteerde stations en periode.")
        
    for melding in foutmeldingen:
         st.error(melding)
    
    st.divider()

    # --- Deel 4: Korte Termijn Voorspellingsgrafiek (Plotly) ---
    if forecast_enabled:
        
        st.header("Korte Termijn Voorspelling (1-uurs interval)")

        # 1. Haal data op
        df_smhi, err_smhi = get_smhi_forecast(FORECAST_LAT, FORECAST_LON)
        df_yr, err_yr = get_yr_forecast(FORECAST_LAT, FORECAST_LON)

        # 2. Toon foutmeldingen
        if err_smhi:
            st.error(f"SMHI Korte Termijn Voorspellingsfout: {err_smhi}")
        if err_yr:
            st.error(f"YR.no Korte Termijn Voorspellingsfout: {err_yr}")

        # 3. Combineer DataFrames
        all_dfs = [df for df in [df_smhi, df_yr] if not df.empty]

        if all_dfs:
            combined_df = pd.concat(all_dfs).reset_index(drop=True)
            
            # --- START CORRECTIE: UTC NAAR LOKALE TIJD (EUROPE/STOCKHOLM) ---
            # De Tijd (UTC) kolom is al tz-aware, dus we gebruiken direct tz_convert
            combined_df['Tijd (Lokaal)'] = combined_df['Tijd (UTC)'].dt.tz_convert('Europe/Stockholm')
            combined_time_column = 'Tijd (Lokaal)'
            # --- EINDE CORRECTIE ---

            duration_seconds = (combined_df['Tijd (UTC)'].max() - combined_df['Tijd (UTC)'].min()).total_seconds()
            duration_hours = round(duration_seconds / 3600)
            
            st.subheader("Voorspelde Temperatuur (°C)") 

            # 4. Plaatsing beschrijving aangepast
            st.markdown(f"Toont de **voorspelde temperatuur** voor de komende **{duration_hours} uur** (1-uurs interval) op **{FORECAST_LAT:.4f} N, {FORECAST_LON:.4f} E** (nabij Graninge). **Tijden in lokale zone (CET/CEST).**")

            # 5. Plotten met Plotly Express 
            fig_voorspelling = px.line(
                combined_df, 
                x=combined_time_column, 
                y='Temperatuur (°C)', 
                color='Bron', 
                title=None,
                markers=True,
                hover_data={
                    combined_time_column: "|%Y-%m-%d %H:%M", 
                    'Temperatuur (°C)': ':.1f',
                    'Bron': True
                }
            )

            # 5a. Optimalisatie van de X-as
            fig_voorspelling.update_xaxes(
                tickformat="%H:%M", 
                dtick=4 * 60 * 60 * 1000, 
                showgrid=True,
                gridcolor='#eeeeee',
                title='Tijd (Lokaal: CET/CEST)'
            )
            
            fig_voorspelling.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="0°C")


            # 5b. Toevoegen van de verticale dagwissellijnen en datumannotaties
            midnight_data = combined_df[combined_df[combined_time_column].dt.hour == 0].drop_duplicates(subset=[combined_time_column]) 

            shapes = []
            annotations = []

            min_y = combined_df['Temperatuur (°C)'].min() if not combined_df.empty else -5 
            max_y = combined_df['Temperatuur (°C)'].max() if not combined_df.empty else 5

            for index, row in midnight_data.iterrows():
                day_start_time = row[combined_time_column] 
                date_label = day_start_time.strftime('%d-%m')
                
                # 1. Verticale stippellijn
                shapes.append(
                    dict(
                        type="line",
                        xref="x", yref="y",
                        x0=day_start_time, y0=min_y - 1,
                        x1=day_start_time, y1=max_y + 1,
                        line=dict(color="LightGrey", width=1, dash="dot")
                    )
                )
                
                # 2. Datum label als annotatie
                annotations.append(
                    dict(
                        x=day_start_time, 
                        y=min_y - 2, 
                        xref="x", yref="y",
                        text=f'<b>{date_label}</b>', 
                        showarrow=False,
                        font=dict(size=12, color="#333333"),
                        xanchor='center', 
                        yanchor='top'
                    )
                )

            # 5c. Update de layout
            fig_voorspelling.update_layout(
                shapes=shapes,
                annotations=annotations,
                yaxis_title="Temperatuur (°C)",
                margin=dict(b=70) 
            )

            # 5d. Tonen van de grafiek
            st.plotly_chart(fig_voorspelling, use_container_width=True)

            # 6. Ruwe data
            with st.expander("Toon Ruwe Data (Korte Termijn Voorspelling)"):
                st.dataframe(combined_df.sort_values(combined_time_column).head(100), use_container_width=True)
            
        else:
            st.warning("Kon geen korte termijn voorspellingsgegevens ophalen van beide API's.")

    st.divider()
    
    # --- Deel 5: LANGE TERMIJN VOORSPELLINGSGRAFIEK (Altair, overgenomen van weergraf4.py) ---
    if long_term_forecast_enabled: # <<< NIEUWE CHECK
        
        st.header("🗓️ Uitgebreide SMHI-voorspelling (7-10 dagen)")
        
        df_long_term, approved_time, status_msg = fetch_long_term_smhi_data(FORECAST_LAT, FORECAST_LON)

        if status_msg != "OK":
            st.error(status_msg)
        
        if df_long_term is not None and not df_long_term.empty:
            
            caption_text = f"Data via SMHI Open Data API voor **{FORECAST_LAT:.4f} N, {FORECAST_LON:.4f} E**."
            if approved_time:
                caption_text += f" Generatie SMHI-model: **{approved_time}**."
            st.caption(caption_text)
            
            st.markdown("---")

            # 1. Temperatuur Grafiek
            st.subheader("1. Temperatuur Voorspelling")
            st.altair_chart(create_temp_chart(df_long_term), use_container_width=True)
            
            st.markdown("---")

            # 2. Neerslag Grafiek
            st.subheader("2. Neerslag Voorspelling")
            st.altair_chart(create_precipitation_chart(df_long_term), use_container_width=True)
            
            st.markdown("---")

            # 3. Wind Grafiek
            st.subheader("3. Wind Snelheid en Richting")
            st.altair_chart(create_wind_chart(df_long_term), use_container_width=True)
            
            st.markdown("---")
            
            with st.expander("Toon Ruwe Data (Lange Termijn Voorspelling)"):
                # Toon de lokale tijd kolom in de ruwe data voor consistentie
                st.dataframe(df_long_term.head(50), use_container_width=True)
                
        elif df_long_term is not None and df_long_term.empty:
            st.warning("Er zijn geen recente voorspellingen beschikbaar voor de opgegeven periode.")
            

# --- Automatisch verversen d.m.v. Streamlit Rerun ---
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()
