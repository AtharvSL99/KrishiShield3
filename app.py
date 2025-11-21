import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta, date
import os
import xgboost as xgb
import requests
from io import StringIO

# --- Configuration ---
PRIMARY_DATA_FILE = 'semifinal.csv'
FALLBACK_DATA_FILE = 'test.csv'
MODEL_FILE = 'Onion.pkl'
SCALER_FILE = 'scaler.pkl'
LAG_WINDOW = 4
DEFAULT_COMMODITY = 'Onion'
PRICE_COLUMN = 'Modal_Price'

# --- 1. Data Loading & Setup ---

@st.cache_data
def load_base_data():
    """Loads raw data to get Market list."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    if not os.path.exists(file_to_load):
        st.error(f"Base data file ({file_to_load}) not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '0':
            df = df.drop(columns=[df.columns[0]])
            if len(df.columns) > 0 and (df.columns[0].startswith('Unnamed') or df.columns[0] == '0'):
                df = df.drop(columns=[df.columns[0]])
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def get_market_coordinates():
    """Extracts market coordinates."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        if 'Market Name' in df.columns:
            df['Market Name'] = df['Market Name'].astype(str).str.strip()
        
        if 'latitude_x' in df.columns and 'longitude_x' in df.columns:
            coords = df[['Market Name', 'latitude_x', 'longitude_x']].drop_duplicates().set_index('Market Name')
            return coords.to_dict('index')
        return {}
    except:
        return {}

@st.cache_data
def get_seasonal_baseline(commodity, market_name, target_month):
    """
    Calculates the historical average price with hierarchical fallback:
    1. Specific Market + Specific Month (Best)
    2. Specific Market + All Year (Good)
    3. All Markets + Specific Month (Okay - captures seasonality)
    4. Global Average (Fallback)
    """
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        
        # Clean up market names for matching
        if 'Market Name' in df.columns:
            df['Market Name'] = df['Market Name'].astype(str).str.strip().str.lower()
        target_market = str(market_name).strip().lower()
        
        # Standardize Date
        date_col = 'Price Date' if 'Price Date' in df.columns else 'Date'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Filter by Commodity
            df_com = df[df['Commodity'] == commodity]
            
            # Level 1: Market + Month
            df_market = df_com[df_com['Market Name'] == target_market]
            df_market_month = df_market[df_market[date_col].dt.month == target_month]
            
            if not df_market_month.empty:
                return df_market_month['Modal_Price'].mean()
            
            # Level 2: Market Only (All Year)
            if not df_market.empty:
                return df_market['Modal_Price'].mean()
            
            # Level 3: Month Only (All Markets - Seasonality)
            df_month = df_com[df_com[date_col].dt.month == target_month]
            if not df_month.empty:
                return df_month['Modal_Price'].mean()
            
        # Level 4: Global Average
        return df[df['Commodity'] == commodity]['Modal_Price'].mean()
    except:
        return 2500.0 # Safe fallback

@st.cache_resource
def load_artifacts():
    """Loads model/scaler, forcing CPU."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        
        if isinstance(model, xgb.XGBRegressor):
            model.set_params(device='cpu', tree_method='hist')
            try:
                model.get_booster().set_param({'device': 'cpu', 'tree_method': 'hist'})
            except:
                pass
            
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

# --- 2. Hybrid Data Fetching (CSV + API) ---

def load_historical_weather_from_csv(market_name):
    """Loads weather history directly from the static CSV file."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        
        # Filter Market
        df = df[df['Market Name'].astype(str).str.strip() == market_name.strip()]
        
        # Date Index
        date_col = 'Price Date' if 'Price Date' in df.columns else 'Date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Select Weather Cols
        w_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                  'precipitation_sum', 'precipitation_hours', 'wind_speed_10m_max', 'weather_code']
        
        # Return only existing columns
        existing = [c for c in w_cols if c in df.columns]
        return df[existing]
    except:
        return pd.DataFrame()

def fetch_api_weather(lat, lon, start_date, end_date):
    """Fetches weather from Open-Meteo for gaps not covered by CSV."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,precipitation_hours,wind_speed_10m_max",
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        daily = pd.DataFrame(data.get('daily', {}))
        if not daily.empty:
            daily['time'] = pd.to_datetime(daily['time'])
            daily = daily.set_index('time')
        return daily
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

def get_hybrid_weather_data(market_name, lat, lon, target_date, lookback_days=45):
    """
    Intelligent Data Fetcher:
    1. Tries to get data from CSV (semifinal.csv) first.
    2. If CSV doesn't cover the target date (e.g., it's too new), uses API.
    """
    end_date = pd.Timestamp(target_date)
    start_date = end_date - timedelta(days=lookback_days)
    
    # 1. Get CSV Data
    df_csv = load_historical_weather_from_csv(market_name)
    
    # Check coverage
    csv_covers_window = False
    if not df_csv.empty:
        csv_max_date = df_csv.index.max()
        csv_min_date = df_csv.index.min()
        
        # If our requested window is entirely inside the CSV's range
        if csv_min_date <= start_date and csv_max_date >= end_date:
            # Use CSV slice
            return df_csv[(df_csv.index >= start_date) & (df_csv.index <= end_date)], "Historical CSV"
        
        # Partial coverage or gap?
        if end_date > csv_max_date:
            # We need data from API for recent dates
            pass 

    # 2. Use API (Fallback or for New Dates)
    st.caption(f"Date {target_date} outside CSV range. Switching to Live API.")
    df_api = fetch_api_weather(lat, lon, start_date, end_date)
    return df_api, "Open-Meteo API"

# --- 3. Processing Logic ---

def process_weather_risk(df_weather, baseline_price):
    """Aggregates weather data and injects seasonal baseline price."""
    # Add Baseline Price
    df_weather['Modal_Price'] = baseline_price
    
    # Weekly Aggregation
    agg_rules = {
        'temperature_2m_max': 'max',
        'temperature_2m_min': 'min',
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'sum',
        'precipitation_hours': 'sum',
        'wind_speed_10m_max': 'mean',
        'weather_code': lambda x: x.mode()[0] if not x.empty else 0,
        'Modal_Price': 'mean' 
    }
    
    # Fill missing cols
    for col in agg_rules:
        if col not in df_weather.columns:
            df_weather[col] = 0 if col != 'Modal_Price' else baseline_price

    df_weekly = df_weather.resample('W').agg(agg_rules)
    
    # Create Lags
    recent_weeks = df_weekly.tail(LAG_WINDOW).iloc[::-1] 
    
    simulated_data = []
    for i in range(len(recent_weeks)):
        row = recent_weeks.iloc[i]
        simulated_data.append({
            'Week_Ending_Date': recent_weeks.index[i].strftime('%Y-%m-%d'),
            f'{PRICE_COLUMN}_Lag1': row['Modal_Price'],
            'temperature_2m_max': row['temperature_2m_max'],
            'precipitation_sum': row['precipitation_sum'],
            'weather_code': row['weather_code'],
            'temperature_2m_mean': row['temperature_2m_mean'],
            'temperature_2m_min': row['temperature_2m_min'],
            'wind_speed_10m_max': row['wind_speed_10m_max'],
            'precipitation_hours': row['precipitation_hours'],
        })
        
    # Pad
    while len(simulated_data) < LAG_WINDOW:
        simulated_data.append(simulated_data[-1] if simulated_data else {})
        
    return simulated_data

def prepare_input_features(simulated_data, model_features, baseline_price):
    """Aligns user input with model features."""
    X_pred = pd.DataFrame(0.0, index=[0], columns=model_features)
    
    for lag in range(1, LAG_WINDOW + 1):
        data = simulated_data[lag-1]
        for col in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
                   'wind_speed_10m_max', 'precipitation_sum', 'precipitation_hours', 'weather_code']:
            if col in data:
                X_pred.loc[0, f'{col}_Lag{lag}'] = data[col]
        X_pred.loc[0, f'{PRICE_COLUMN}_Lag{lag}'] = baseline_price

    if simulated_data:
        proxy = simulated_data[0]
        for col in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
                   'wind_speed_10m_max', 'precipitation_sum', 'precipitation_hours', 'weather_code']:
            if col in X_pred.columns and col in proxy:
                X_pred.loc[0, col] = proxy[col]
            
    return X_pred

# --- Main App ---

def main():
    st.set_page_config(page_title="Crop Risk Calculator", layout="wide")
    st.title(f"🌾 {DEFAULT_COMMODITY} Risk Calculator")
    st.markdown("Assess **Planting Risk** based on weather trends.")
    
    df_base = load_base_data()
    model, scaler = load_artifacts()
    market_coords = get_market_coordinates() 
    
    if df_base.empty or model is None:
        st.error("System Error: Missing Data or Model.")
        st.stop()
        
    # --- Sidebar ---
    st.sidebar.header("Location & Time")
    markets = sorted(df_base['Market Name'].unique())
    market = st.sidebar.selectbox("Select Market", markets)
    
    min_date = date(2020, 1, 1) 
    max_date = date(2026, 12, 31)
    prediction_date = st.sidebar.date_input("Assessment Date", date.today(), min_value=min_date, max_value=max_date)
    
    # --- Updated Baseline Call ---
    target_month_name = prediction_date.strftime('%B')
    # Now passing 'market' to get localized baseline
    baseline_price = get_seasonal_baseline(DEFAULT_COMMODITY, market, prediction_date.month)
    
    st.sidebar.divider()
    st.sidebar.markdown(f"**Baseline ({market}, {target_month_name}):**")
    st.sidebar.markdown(f"# ₹{baseline_price:,.0f}")
    st.sidebar.caption("Average historical price for this specific market and month.")
    
    st.divider()
    
    if st.button("Calculate Risk Factor", type="primary"):
        if market not in market_coords:
            clean_market = market.strip()
            found_key = next((k for k in market_coords.keys() if str(k).strip() == clean_market), None)
            if found_key:
                lat, lon = market_coords[found_key]['latitude_x'], market_coords[found_key]['longitude_x']
            else:
                st.error("Coordinates not found.")
                st.stop()
        else:
            lat, lon = market_coords[market]['latitude_x'], market_coords[market]['longitude_x']

        with st.spinner("Fetching weather data..."):
            # Hybrid Fetch: Decides between CSV and API
            df_weather, source = get_hybrid_weather_data(market, lat, lon, prediction_date)
            
            if not df_weather.empty:
                # Process
                simulated_data = process_weather_risk(df_weather, baseline_price)
                
                # Predict
                feat_names = model.get_booster().feature_names
                X_input = prepare_input_features(simulated_data, feat_names, baseline_price)
                X_scaled = scaler.transform(X_input)
                predicted_price = model.predict(X_scaled)[0]
                
                # --- Risk Metrics ---
                deviation = predicted_price - baseline_price
                pct_change = (deviation / baseline_price) * 100
                
                # Risk Score: Amplifies deviations. +20% price = High Risk.
                risk_score = min(max((pct_change + 10) * 2.5, 0), 100)
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Projected Price Impact", f"₹ {predicted_price:,.0f}", 
                           delta=f"{pct_change:.1f}% vs Baseline", delta_color="inverse")
                
                # Visuals
                if risk_score > 60:
                    risk_label = "HIGH RISK (Adverse Weather)"
                    risk_color = "red"
                elif risk_score > 30:
                    risk_label = "MODERATE RISK"
                    risk_color = "orange"
                else:
                    risk_label = "LOW RISK (Favorable Weather)"
                    risk_color = "green"
                    
                col2.markdown(f"### Risk Level")
                col2.markdown(f"<h2 style='color:{risk_color}'>{risk_label}</h2>", unsafe_allow_html=True)
                col2.progress(int(risk_score) / 100)
                
                col3.info(f"""
                **Data Source:** {source}
                
                **Analysis:**
                The model analyzed weather patterns for the 4 weeks ending on **{prediction_date}**.
                Comparing against the historical **{market}** average in **{target_month_name}** (₹{baseline_price:,.0f}).
                """)
                
                with st.expander("View Weather Data Used"):
                    st.dataframe(pd.DataFrame(simulated_data))
            else:
                st.error("Could not fetch weather data (Check CSV dates or API connection).")

if __name__ == "__main__":
    main()