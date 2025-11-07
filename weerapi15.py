import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
from math import radians, sin, cos, sqrt, atan2, degrees
import plotly.express as px

# --- Globale Configuratie ---
BASE_API_URL = "http://api.temperatur.nu/tnu_1.17.php"
REFRESH_INTERVAL_SECONDS = 300 # 5 minuten (300 seconden)
REFRESH_COORDS_TTL = 3600 * 24 * 7 # 1 week

# UW LOCATIE
UW_LAT = 63.0243695749981
UW_LON = 17.03321183785952
# Locatie voor Voorspelling (Graningen, overgenomen uit grafiek5.py)
FORECAST_LAT = 63.024625
FORECAST_LON = 17.035304

# CENTRALE STATIONSCONFIGURATIE
STATIONS = {
    "Åkroken": "akroken",
    "Fångsjöbacken": "fangsjobacken",
    "Graninge": "graninge_tv", 
    "Långsele": "langsele",
    "Ärtrik": "artrik",
    "Sollefteå/Trästa": "trasta",
    "Grillom": "grillom"
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

# --- FUNCTIES VOOR VOORSPELLINGEN (Ongewijzigd) ---

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
    
    # NIEUWE CHECKBOX VOOR VOORSPELLING (STANDAARD UIT)
    forecast_enabled = st.checkbox("Toon Weersvoorspelling (SMHI / YR.no)", value=False) # <- Aangepast naar False

    st.divider()
    st.markdown("ℹ️ Dashboard toont data van Trafikverket VViS.")
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
            title='Tijdstip (UTC)'
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

    # --- Deel 4: Voorspellingsgrafiek (Ingepakt met checkbox) ---
    if forecast_enabled:
        
        # 1. Titel is nu "Voorspelde Temperatuur (°C)" met subtext eronder
        
        # 2. Haal data op
        df_smhi, err_smhi = get_smhi_forecast(FORECAST_LAT, FORECAST_LON)
        df_yr, err_yr = get_yr_forecast(FORECAST_LAT, FORECAST_LON)

        # 3. Toon foutmeldingen
        if err_smhi:
            st.error(f"SMHI Voorspellingsfout: {err_smhi}")
        if err_yr:
            st.error(f"YR.no Voorspellingsfout: {err_yr}")

        # 4. Combineer DataFrames
        all_dfs = [df for df in [df_smhi, df_yr] if not df.empty]

        if all_dfs:
            combined_df = pd.concat(all_dfs).reset_index(drop=True)
            
            duration_seconds = (combined_df['Tijd (UTC)'].max() - combined_df['Tijd (UTC)'].min()).total_seconds()
            duration_hours = round(duration_seconds / 3600)
            
            st.subheader("Voorspelde Temperatuur (°C)") # <- Hoofdtitel

            # 2. Plaatsing beschrijving aangepast: NU onder de subheader
            st.markdown(f"Toont de **voorspelde temperatuur** voor de komende **{duration_hours} uur** (1-uurs interval) op **{FORECAST_LAT:.4f} N, {FORECAST_LON:.4f} E** (nabij Graningen).")

            # 5. Plotten met Plotly Express 
            fig_voorspelling = px.line(
                combined_df, 
                x='Tijd (UTC)', 
                y='Temperatuur (°C)', 
                color='Bron', 
                title=None,
                markers=True,
                hover_data={
                    'Tijd (UTC)': "|%Y-%m-%d %H:%M",
                    'Temperatuur (°C)': ':.1f',
                    'Bron': True
                }
            )

            # 5a. Optimalisatie van de X-as voor uren
            fig_voorspelling.update_xaxes(
                tickformat="%H:%M", 
                dtick=4 * 60 * 60 * 1000, # Toon labels om de 4 uur (in milliseconden)
                showgrid=True,
                gridcolor='#eeeeee',
                title='Tijd (UTC)'
            )
            
            fig_voorspelling.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="0°C")


            # 5b. Toevoegen van de verticale dagwissellijnen en datumannotaties
            midnight_data = combined_df[combined_df['Tijd (UTC)'].dt.hour == 0].drop_duplicates(subset=['Tijd (UTC)'])

            shapes = []
            annotations = []

            min_y = combined_df['Temperatuur (°C)'].min() if not combined_df.empty else -5 
            max_y = combined_df['Temperatuur (°C)'].max() if not combined_df.empty else 5

            for index, row in midnight_data.iterrows():
                day_start_time = row['Tijd (UTC)']
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
                
                # 2. Datum label als annotatie (onder de grafieklijn)
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

            # 6. Ruwe Historische Data en Voorspellingsdata onder de tweede grafiek (Ongewijzigd t.o.v. vorige versie)
            with st.expander("Toon Ruwe Data (Historisch & Voorspelling)"):
                st.markdown(f"#### Historische Data ({keuze_label_tijd})")
                if not alle_historie_df.empty:
                    st.dataframe(alle_historie_df.tail(20), use_container_width=True)
                else:
                    st.info("Geen historische data beschikbaar om te tonen.")
                    
                st.markdown(f"#### Voorspellingsdata (SMHI & YR.no)")
                st.dataframe(combined_df.sort_values('Tijd (UTC)').head(100), use_container_width=True)
            
        else:
            st.warning("Kon geen voorspellingsgegevens ophalen van beide API's. Controleer de foutmeldingen hierboven.")


         
# --- Automatisch verversen d.m.v. Streamlit Rerun ---
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()
