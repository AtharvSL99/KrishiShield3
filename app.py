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
def get_seasonal_averages(commodity, market_name, target_month):
    """Calculates historical averages for filling gaps."""
    file_to_load = PRIMARY_DATA_FILE if os.path.exists(PRIMARY_DATA_FILE) else FALLBACK_DATA_FILE
    defaults = {
        'Modal_Price': 2500.0, 'temperature_2m_max': 30.0, 'temperature_2m_min': 20.0,
        'temperature_2m_mean': 25.0, 'precipitation_sum': 0.0, 'precipitation_hours': 0.0,
        'wind_speed_10m_max': 10.0, 'weather_code': 0.0
    }
    try:
        df = pd.read_csv(file_to_load, index_col=False)
        df.columns = df.columns.str.strip()
        target_market = str(market_name).strip().lower()
        date_col = 'Price Date' if 'Price Date' in df.columns else 'Date'
        
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df_filtered = df[
                (df['Commodity'] == commodity) & 
                (df['Market Name'].str.lower().str.strip() == target_market) & 
                (df[date_col].dt.month == target_month)
            ]
            if df_filtered.empty:
                df_filtered = df[(df['Commodity'] == commodity) & (df[date_col].dt.month == target_month)]

            if not df_filtered.empty:
                means = df_filtered.mean(numeric_only=True)
                for col in defaults.keys():
                    if col in means: defaults[col] = means[col]
        return defaults
    except:
        return defaults

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
            except: pass
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

# --- 2. Unified API (History + Forecast) ---

def fetch_unified_weather(lat, lon):
    """
    Fetches a continuous block of weather data:
    - Past 7 days (Observed History)
    - Next 16 days (Forecast)
    Total: ~23 days of continuous data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,precipitation_hours,wind_speed_10m_max",
        "timezone": "auto",
        "past_days": 7,      # Week 0 (Observed)
        "forecast_days": 16  # Weeks 1-2 (Forecast)
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        daily_data = data.get('daily', {})
        df = pd.DataFrame(daily_data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df.index.name = 'Date'
        return df
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

# --- 3. Scenario Builder ---

def build_hybrid_scenario(df_api, historical_avgs):
    """
    Stitches the timeline:
    1. API Data (Observed + Forecast) -> Approx 3 weeks
    2. Historical Averages (Projection) -> Fills Week 4 to complete the lag window
    """
    # 1. Start with API data
    df_scenario = df_api.copy()
    
    # 2. We need a total of roughly 28-30 days to form 4 weekly lags
    # API gives ~23 days. We need to project the rest.
    last_api_date = df_scenario.index.max()
    start_date = df_scenario.index.min()
    
    # Determine how many more days needed to reach 4 weeks (28 days)
    current_duration = (last_api_date - start_date).days
    days_needed = max(0, 28 - current_duration)
    
    # 3. Append Historical Projection
    if days_needed > 0:
        projection_rows = []
        current_date = last_api_date + timedelta(days=1)
        for _ in range(days_needed + 1): # +1 buffer
            row = historical_avgs.copy()
            row.pop('Modal_Price', None) # Don't put price in weather columns
            row['Date'] = current_date
            projection_rows.append(row)
            current_date += timedelta(days=1)
            
        df_proj = pd.DataFrame(projection_rows).set_index('Date')
        df_scenario = pd.concat([df_scenario, df_proj])
    
    # 4. Add Constant Baseline Price
    df_scenario['Modal_Price'] = historical_avgs['Modal_Price']
    
    return df_scenario

def aggregate_to_weekly_lags(df_daily):
    """Aggregates daily data into exactly 4 weekly lags."""
    agg_rules = {
        'temperature_2m_max': 'max', 'temperature_2m_min': 'min', 'temperature_2m_mean': 'mean',
        'precipitation_sum': 'sum', 'precipitation_hours': 'sum', 'wind_speed_10m_max': 'mean',
        'weather_code': lambda x: x.mode()[0] if not x.empty else 0,
        'Modal_Price': 'mean' 
    }
    
    # Ensure cols exist
    for col in agg_rules:
        if col not in df_daily.columns: df_daily[col] = 0
            
    # Resample '7D' starting from the first day
    df_weekly = df_daily.resample('7D', origin='start').agg(agg_rules)
    
    # Take first 4 weeks
    df_weekly = df_weekly.iloc[:LAG_WINDOW]
    
    # If short, pad with last known
    while len(df_weekly) < LAG_WINDOW:
        last_row = df_weekly.iloc[[-1]].copy()
        last_row.index = last_row.index + timedelta(days=7)
        df_weekly = pd.concat([df_weekly, last_row])
        
    # Reverse for Model Input (Row 0 = Most Recent/Future-most Week)
    # For the model:
    # Lag 1 = The week closest to the target date (The "Future")
    # Lag 4 = The week furthest away (The "Past/Observed")
    
    # The resampling is chronological:
    # Row 0: Week 0 (Past/Observed)
    # Row 1: Week 1 (Forecast)
    # Row 2: Week 2 (Forecast)
    # Row 3: Week 3 (Projection)
    
    # We need to REVERSE this so:
    # Lag 1 <- Row 3 (Projection/Future)
    # Lag 2 <- Row 2
    # Lag 3 <- Row 1
    # Lag 4 <- Row 0 (Past)
    
    recent_weeks = df_weekly.iloc[::-1]
    
    simulated_data = []
    sources = ["Projection (Hist. Avg)", "Forecast (API)", "Forecast (API)", "Observed (API)"]
    
    for i in range(len(recent_weeks)):
        row = recent_weeks.iloc[i]
        # Guard bounds for source label
        src_label = sources[i] if i < len(sources) else "Projection"
        
        simulated_data.append({
            'Week_Start': recent_weeks.index[i].strftime('%Y-%m-%d'),
            'Data_Source': src_label,
            f'{PRICE_COLUMN}_Lag1': row['Modal_Price'],
            'temperature_2m_max': row['temperature_2m_max'],
            'precipitation_sum': row['precipitation_sum'],
            'weather_code': row['weather_code'],
            'temperature_2m_mean': row['temperature_2m_mean'],
            'temperature_2m_min': row['temperature_2m_min'],
            'wind_speed_10m_max': row['wind_speed_10m_max'],
            'precipitation_hours': row['precipitation_hours'],
        })
        
    return simulated_data, df_weekly

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
    st.title(f"🌾 {DEFAULT_COMMODITY} Future Risk Calculator")
    st.markdown("""
    **Rolling Horizon Analysis:** Combines **Observed** (Past Week) + **Forecast** (Next 2 Weeks) + **Projection** (Week 4) 
    to estimate price risk ~1 month from today.
    """)
    
    df_base = load_base_data()
    model, scaler = load_artifacts()
    market_coords = get_market_coordinates() 
    
    if df_base.empty or model is None:
        st.error("System Error: Missing Data or Model.")
        st.stop()
        
    # --- Sidebar ---
    st.sidebar.header("Location")
    markets = sorted(df_base['Market Name'].unique())
    market = st.sidebar.selectbox("Select Market", markets)
    
    today = date.today()
    
    # Baseline Calculation
    target_month_name = (today + timedelta(days=30)).strftime('%B') # Next month
    historical_avgs = get_seasonal_averages(DEFAULT_COMMODITY, market, today.month)
    baseline_price = historical_avgs['Modal_Price']
    
    st.sidebar.divider()
    st.sidebar.markdown(f"**Historical Baseline:**")
    st.sidebar.markdown(f"# ₹{baseline_price:,.0f}")
    st.sidebar.caption(f"Average for {market} around this time of year.")
    
    st.divider()
    
    if st.button("Analyze Future Risk", type="primary"):
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

        with st.spinner("Fetching Live Observations & Forecasts..."):
            
            # 1. Unified Fetch (Past 7 days + Next 16 days)
            df_api = fetch_unified_weather(lat, lon)
            
            if not df_api.empty:
                # 2. Build Hybrid Scenario
                df_scenario = build_hybrid_scenario(df_api, historical_avgs)
                
                # 3. Aggregate (Weekly)
                simulated_data, df_weekly_view = aggregate_to_weekly_lags(df_scenario)
                
                # 4. Predict
                feat_names = model.get_booster().feature_names
                X_input = prepare_input_features(simulated_data, feat_names, baseline_price)
                X_scaled = scaler.transform(X_input)
                predicted_price = model.predict(X_scaled)[0]
                
                # --- Risk Metrics ---
                deviation = predicted_price - baseline_price
                pct_change = (deviation / baseline_price) * 100
                
                # Risk Logic
                risk_score = min(max((pct_change + 10) * 2.5, 0), 100)
                
                col1, col2, col3 = st.columns(3)
                
                target_date_approx = today + timedelta(weeks=4)
                
                col1.metric(f"Projected Price ({target_date_approx.strftime('%b %d')})", 
                           f"₹ {predicted_price:,.0f}", 
                           delta=f"{pct_change:.1f}% vs Norm", delta_color="inverse")
                
                if risk_score > 60:
                    risk_label, risk_color = "HIGH RISK", "red"
                    risk_desc = "Weather mix (Observed + Forecast) indicates adverse conditions likely to spike prices."
                elif risk_score > 30:
                    risk_label, risk_color = "MODERATE RISK", "orange"
                    risk_desc = "Conditions are slightly deviating from ideal, creating moderate price pressure."
                else:
                    risk_label, risk_color = "LOW RISK", "green"
                    risk_desc = "Favorable weather conditions (Observed & Forecasted) suggest stable or lower prices."
                    
                col2.markdown(f"### Risk Level")
                col2.markdown(f"<h2 style='color:{risk_color}'>{risk_label}</h2>", unsafe_allow_html=True)
                col2.progress(int(risk_score) / 100)
                
                col3.info(risk_desc)
                
                # Visualization table
                st.subheader("Data Timeline Analysis")
                
                # Clean up for display
                display_cols = ['Data_Source', 'Week_Start', 'temperature_2m_max', 'precipitation_sum', 'weather_code']
                display_df = pd.DataFrame(simulated_data)[display_cols]
                
                # Rename for clarity
                display_df.columns = ['Source', 'Week Start', 'Max Temp (°C)', 'Rain (mm)', 'WMO Code']
                
                # Map codes to text
                def code_to_text(c):
                    if c == 0: return "Clear"
                    if c < 3: return "Cloudy"
                    if c < 60: return "Drizzle"
                    if c < 80: return "Rain"
                    return "Storm"
                display_df['Condition'] = display_df['WMO Code'].apply(code_to_text)
                
                st.dataframe(display_df, use_container_width=True)
                
            else:
                st.error("Could not fetch weather data.")

if __name__ == "__main__":
    main()