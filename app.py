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
def get_baseline_price(commodity):
    """Calculates the historical average price to use as a constant baseline."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        # Filter by commodity
        df = df[df['Commodity'] == commodity]
        # Return mean Modal Price
        return df['Modal_Price'].mean()
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

# --- 2. Real-Time Weather API ---

def fetch_live_weather(lat, lon, target_date, lookback_days=40):
    """Fetches daily weather history from Open-Meteo."""
    end_date_str = target_date.strftime('%Y-%m-%d')
    start_date_str = (target_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,precipitation_hours,wind_speed_10m_max",
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        daily_data = data.get('daily', {})
        df_weather = pd.DataFrame(daily_data)
        if not df_weather.empty:
            df_weather['time'] = pd.to_datetime(df_weather['time'])
            df_weather = df_weather.set_index('time')
            df_weather.index.name = 'Date'
        
        return df_weather
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

# --- 3. Processing Logic (Risk Calculation) ---

def process_weather_risk(df_weather, baseline_price):
    """
    Aggregates weather data and injects CONSTANT baseline price.
    """
    # 1. Add Constant Price Column
    df_weather['Modal_Price'] = baseline_price
    
    # 2. Weekly Aggregation
    agg_rules = {
        'temperature_2m_max': 'max',
        'temperature_2m_min': 'min',
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'sum',
        'precipitation_hours': 'sum',
        'wind_speed_10m_max': 'mean',
        'weather_code': lambda x: x.mode()[0] if not x.empty else 0,
        'Modal_Price': 'mean' # Will remain constant
    }
    
    # Ensure columns exist
    for col in agg_rules:
        if col not in df_weather.columns:
            df_weather[col] = 0 if col != 'Modal_Price' else baseline_price

    df_weekly = df_weather.resample('W').agg(agg_rules)
    
    # 3. Create Lags (Get Last LAG_WINDOW weeks)
    recent_weeks = df_weekly.tail(LAG_WINDOW).iloc[::-1] 
    
    simulated_data = []
    for i in range(len(recent_weeks)):
        row = recent_weeks.iloc[i]
        simulated_data.append({
            'Week_Ending_Date': recent_weeks.index[i].strftime('%Y-%m-%d'),
            f'{PRICE_COLUMN}_Lag1': row['Modal_Price'], # Constant Baseline
            'temperature_2m_max': row['temperature_2m_max'],
            'precipitation_sum': row['precipitation_sum'],
            'weather_code': row['weather_code'],
            'temperature_2m_mean': row['temperature_2m_mean'],
            'temperature_2m_min': row['temperature_2m_min'],
            'wind_speed_10m_max': row['wind_speed_10m_max'],
            'precipitation_hours': row['precipitation_hours'],
        })
        
    # Pad if needed
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
        # Always use baseline for lags
        X_pred.loc[0, f'{PRICE_COLUMN}_Lag{lag}'] = baseline_price

    # Current Week Proxy
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
    
    # Calculate Baseline
    baseline_price = get_baseline_price(DEFAULT_COMMODITY)
    st.sidebar.markdown(f"**Historical Average Price:** ₹{baseline_price:,.0f}")
    
    min_date = date(2020, 1, 1) 
    max_date = date(2026, 12, 31)
    prediction_date = st.sidebar.date_input("Assessment Date", date.today(), min_value=min_date, max_value=max_date)
    
    st.divider()
    
    if st.button("Calculate Risk Factor", type="primary"):
        if market not in market_coords:
            # Fuzzy match logic
            clean_market = market.strip()
            found_key = next((k for k in market_coords.keys() if str(k).strip() == clean_market), None)
            if found_key:
                lat, lon = market_coords[found_key]['latitude_x'], market_coords[found_key]['longitude_x']
            else:
                st.error("Coordinates not found.")
                st.stop()
        else:
            lat, lon = market_coords[market]['latitude_x'], market_coords[market]['longitude_x']

        with st.spinner("Analyzing weather patterns..."):
            # Fetch ONLY weather
            df_weather = fetch_live_weather(lat, lon, prediction_date)
            
            if not df_weather.empty:
                # Process with CONSTANT baseline price
                simulated_data = process_weather_risk(df_weather, baseline_price)
                
                # Predict
                feat_names = model.get_booster().feature_names
                X_input = prepare_input_features(simulated_data, feat_names, baseline_price)
                X_scaled = scaler.transform(X_input)
                predicted_price = model.predict(X_scaled)[0]
                
                # --- RISK CALCULATION ---
                # Logic: If model predicts Price >> Baseline, it implies weather is forcing price up (Supply Constraint/Bad Weather)
                deviation = predicted_price - baseline_price
                pct_change = (deviation / baseline_price) * 100
                
                # Risk Score (0 to 100)
                # 0% change = Risk 20 (Normal)
                # +50% change = Risk 90 (High)
                # -20% change = Risk 10 (Low/Bumper)
                risk_score = min(max((pct_change + 20) * 1.5, 0), 100)
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Projected Price Impact", f"₹ {predicted_price:,.0f}", delta=f"{pct_change:.1f}% vs Avg")
                
                # Visual Risk Display
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
                **Logic:** The model predicts price based **only** on recent weather, assuming a starting price of ₹{baseline_price:,.0f}.
                
                Higher predicted prices indicate weather patterns historically associated with crop damage or supply shortages.
                """)
                
                with st.expander("See Weather Data Used"):
                    st.dataframe(pd.DataFrame(simulated_data))
            else:
                st.error("Could not fetch weather data.")

if __name__ == "__main__":
    main()