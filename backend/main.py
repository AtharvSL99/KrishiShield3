from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import requests
import os
import json
import random
import math
from datetime import datetime, timedelta, date
from io import StringIO

app = FastAPI()

# --- 1. CONFIGURATION & CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRIMARY_DATA_FILE = 'semifinal.csv'
ALERTS_FILE = 'alert_history.json'
RISK_CACHE_FILE = 'risk_state_cache.json'
CURRENT_PETROL = 106.31
CURRENT_DIESEL = 94.27

# Agmarknet Config
AGMARKNET_API_KEY = "579b464db66ec23bdd00000125b231d6079f44e54c664acfce4c26dc"
AGMARKNET_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"

# --- 2. DATA LOADING ---
if os.path.exists(PRIMARY_DATA_FILE):
    DF_BASE = pd.read_csv(PRIMARY_DATA_FILE, index_col=False)
    DF_BASE.columns = DF_BASE.columns.str.strip()
    if 'Market Name' in DF_BASE.columns:
        DF_BASE['Market Name'] = DF_BASE['Market Name'].astype(str).str.strip()
    if 'Commodity' in DF_BASE.columns:
        DF_BASE['Commodity'] = DF_BASE['Commodity'].astype(str).str.strip()
    
    if 'latitude_x' in DF_BASE.columns:
        MARKET_COORDS = DF_BASE[['Market Name', 'latitude_x', 'longitude_x']].drop_duplicates(subset=['Market Name']).set_index('Market Name').to_dict('index')
    else:
        MARKET_COORDS = {}
else:
    DF_BASE = pd.DataFrame()
    MARKET_COORDS = {}

# --- 3. HELPER FUNCTIONS ---

def load_model_and_scaler(crop):
    try:
        if 'onion' in crop.lower() and os.path.exists("onion_xgboost_model_diff.json"):
            print(f"--- Loading NEW JSON Diff Model for {crop} ---")
            model = xgb.XGBRegressor()
            model.load_model("onion_xgboost_model_diff.json")
            model.set_params(device='cpu', n_jobs=1)
            return model, None 
            
        elif os.path.exists(f"{crop}.pkl"):
            with open(f"{crop}.pkl", 'rb') as f:
                model = pickle.load(f)
                if isinstance(model, xgb.XGBRegressor):
                    model.set_params(device='cpu', n_jobs=1)
        
        scaler_path = f"{crop.lower()}_scaler.pkl"
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
        return model, scaler
    except Exception as e:
        print(f"Error loading model for {crop}: {e}")
        return None, None

def fetch_agmarknet_data(crop, market):
    """ Fetches REAL Data from Agmarknet API """
    print(f"--- Fetching Agmarknet Data for {crop} in {market} ---")
    today = date.today()
    date_from = (today - timedelta(weeks=10)).strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    url = (
        f"https://api.data.gov.in/resource/{AGMARKNET_RESOURCE_ID}"
        f"?api-key={AGMARKNET_API_KEY}"
        f"&format=csv"
        f"&filters[State]=Maharashtra"
        f"&filters[Commodity]={crop}"
        f"&range[Arrival_Date][gte]={date_from}"
        f"&range[Arrival_Date][lte]={date_to}"
        f"&limit=5000"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        if df.empty: return None

        df.columns = df.columns.str.strip()
        
        # Filter by Market
        market_col = 'Market' if 'Market' in df.columns else 'District'
        if market_col:
            mask = df[market_col].astype(str).str.contains(market, case=False, na=False)
            market_df = df[mask].copy()
        else:
            market_df = df.copy()

        if market_df.empty: market_df = df # Fallback
            
        market_df['Arrival_Date'] = pd.to_datetime(market_df['Arrival_Date'], dayfirst=True)
        market_df = market_df.sort_values('Arrival_Date')
        market_df.set_index('Arrival_Date', inplace=True)
        
        daily_series = market_df['Modal_Price'].resample('D').mean().ffill()
        if daily_series.empty: return None

        current_price = daily_series.iloc[-1]
        
        stats = {}
        stats['Modal_Price'] = current_price
        stats['Market_Mean_Price'] = daily_series.mean()
        
        stats['Lag_1'] = daily_series.shift(7).iloc[-1] if len(daily_series) > 7 else current_price
        stats['Lag_2'] = daily_series.shift(14).iloc[-1] if len(daily_series) > 14 else current_price
        stats['Lag_3'] = daily_series.shift(21).iloc[-1] if len(daily_series) > 21 else current_price
        stats['Lag_4'] = daily_series.shift(28).iloc[-1] if len(daily_series) > 28 else current_price
        
        rolling = daily_series.rolling(window=28, min_periods=1)
        stats['Rolling_Mean_Price_4w'] = rolling.mean().iloc[-1]
        stats['Rolling_Std_Price_4w'] = rolling.std().iloc[-1]
        if pd.isna(stats['Rolling_Std_Price_4w']): stats['Rolling_Std_Price_4w'] = current_price * 0.05

        print(f"Agmarknet Success: Latest Price: ₹{current_price}")
        return stats

    except Exception as e:
        print(f"Agmarknet API Failed: {e}")
        return None

def fetch_extended_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,wind_speed_10m_max",
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm",
            "timezone": "auto", "past_days": 21, "forecast_days": 7
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        hourly = pd.DataFrame(data.get('hourly', {}))
        if hourly.empty: return pd.DataFrame(), "Weather Unavailable"

        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly.set_index('time', inplace=True)
        hourly = hourly.fillna(0)

        agg_dict = {
            'temperature_2m': ['mean'], 
            'relative_humidity_2m': ['max', 'min', 'mean'],
            'surface_pressure': ['max', 'min', 'mean'],
            'soil_moisture_9_to_27cm': ['max', 'min', 'mean'],
            'soil_moisture_27_to_81cm': ['max', 'min', 'mean']
        }
        daily_calc = hourly.resample('D').agg(agg_dict)
        daily_calc.columns = [f"{col[0]}_{col[1]}" for col in daily_calc.columns]

        rename_map = {
            'soil_moisture_9_to_27cm_max': 'soil_moisture_7_to_28cm_max',
            'soil_moisture_9_to_27cm_min': 'soil_moisture_7_to_28cm_min',
            'soil_moisture_9_to_27cm_mean': 'soil_moisture_7_to_28cm_mean',
            'soil_moisture_27_to_81cm_max': 'soil_moisture_28_to_100cm_max',
            'soil_moisture_27_to_81cm_min': 'soil_moisture_28_to_100cm_min',
            'soil_moisture_27_to_81cm_mean': 'soil_moisture_28_to_100cm_mean'
        }
        daily_calc.rename(columns=rename_map, inplace=True)

        daily_direct = pd.DataFrame(data.get('daily', {}))
        daily_direct['Date'] = pd.to_datetime(daily_direct['time'])
        daily_direct.set_index('Date', inplace=True)
        daily_direct.drop(columns=['time'], inplace=True)

        final_df = daily_calc.join(daily_direct, how='inner')
        final_df.index.name = 'Date'
        
        return final_df.fillna(0), "Live Weather API"

    except Exception as e:
        print(f"Weather Fetch Error: {e}")
        return pd.DataFrame(), "Weather Unavailable"

def prepare_input_vector(model, weather_df, price_stats):
    required_features = model.get_booster().feature_names
    today = pd.Timestamp(date.today()).normalize()
    if today not in weather_df.index:
        if not weather_df.empty: current_idx = weather_df.index[-1]
        else: return pd.DataFrame() 
    else:
        current_idx = today

    input_dict = {}
    
    # 1. Base Prices
    modal_price = price_stats.get('Modal_Price', 2500)
    input_dict['Modal_Price'] = modal_price
    input_dict['Min_Price'] = modal_price * 0.85
    input_dict['Max_Price'] = modal_price * 1.15
    input_dict['Petrol_Price'] = CURRENT_PETROL
    input_dict['Diesel_Price'] = CURRENT_DIESEL

    # 2. Weather
    temp_max = weather_df.loc[current_idx, 'temperature_2m_max'] if 'temperature_2m_max' in weather_df.columns else 0
    humid_max = weather_df.loc[current_idx, 'relative_humidity_2m_max'] if 'relative_humidity_2m_max' in weather_df.columns else 0
    rain_sum = weather_df.loc[current_idx, 'precipitation_sum'] if 'precipitation_sum' in weather_df.columns else 0

    for col in weather_df.columns:
        if col in required_features:
            input_dict[col] = weather_df.loc[current_idx, col]

    # 3. Lags
    for lag in range(1, 5):
        suffix = f"_Lag_{lag}"
        lag_val = price_stats.get(f'Lag_{lag}', modal_price)
        
        input_dict[f'Modal_Price{suffix}'] = lag_val
        input_dict[f'Min_Price{suffix}'] = lag_val * 0.85
        input_dict[f'Max_Price{suffix}'] = lag_val * 1.15
        input_dict[f'Petrol_Price{suffix}'] = CURRENT_PETROL
        input_dict[f'Diesel_Price{suffix}'] = CURRENT_DIESEL
        input_dict[f'weather_code{suffix}'] = 0
        
        lag_weather_date = current_idx - timedelta(days=lag)
        if lag_weather_date in weather_df.index:
             for col in weather_df.columns:
                feat_name = f"{col}{suffix}"
                if feat_name in required_features:
                    input_dict[feat_name] = weather_df.loc[lag_weather_date, col]

    # 4. Engineered Features
    month = today.month
    input_dict['Month'] = month
    input_dict['Rain_x_Month'] = rain_sum * month
    input_dict['Humidity_x_Temp'] = humid_max * temp_max
    input_dict['Petrol_x_Price'] = CURRENT_PETROL * modal_price
    input_dict['Diesel_x_Price'] = CURRENT_DIESEL * modal_price
    
    input_dict['Market_Mean_Price'] = price_stats.get('Market_Mean_Price', modal_price)
    input_dict['Rolling_Mean_Price_4w'] = price_stats.get('Rolling_Mean_Price_4w', modal_price)
    input_dict['Rolling_Std_Price_4w'] = price_stats.get('Rolling_Std_Price_4w', modal_price * 0.05)

    input_df = pd.DataFrame(0.0, index=[0], columns=required_features)
    for col, val in input_dict.items():
        if col in input_df.columns:
            if pd.isna(val): val = 0.0
            input_df[col] = val
            
    return input_df

def generate_price_trend(baseline, projected):
    trend = []
    today = date.today()
    offsets = [-30, -21, -14, -7, 0, 7] 
    
    if math.isnan(projected): projected = baseline
    
    for days in offsets:
        dt = today + timedelta(days=days)
        date_str = dt.strftime("%b %d")
        if days < 0:
            variation = random.uniform(-50, 50)
            price = int(baseline + variation)
        elif days == 0:
            price = int(baseline)
        else:
            price = int(projected)
        trend.append({"date": date_str, "price": price})
    return trend

# --- 4. API ENDPOINTS ---

class AnalysisRequest(BaseModel):
    crop: str
    market: str

@app.get("/alerts")
def get_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'r') as f: return json.load(f)
    return []

@app.get("/markets/{crop}")
def get_markets(crop: str):
    if DF_BASE.empty: return {"markets": []}
    if 'Commodity' in DF_BASE.columns:
        markets = sorted(DF_BASE[DF_BASE['Commodity'] == crop]['Market Name'].unique().tolist())
    else:
        markets = sorted(DF_BASE['Market Name'].unique().tolist())
    return {"markets": markets}

@app.post("/analyze")
def analyze_risk(req: AnalysisRequest):
    if req.market not in MARKET_COORDS:
        raise HTTPException(status_code=404, detail="Market coordinates not found.")
    
    lat = MARKET_COORDS[req.market]['latitude_x']
    lon = MARKET_COORDS[req.market]['longitude_x']
    
    model, scaler = load_model_and_scaler(req.crop)
    if not model:
        raise HTTPException(status_code=500, detail=f"Model for {req.crop} not found.")

    weather_df, source = fetch_extended_weather(lat, lon)
    
    # FETCH REAL DATA
    price_stats = fetch_agmarknet_data(req.crop, req.market)
    if not price_stats:
        print("Agmarknet failed/empty. Falling back to default.")
        price_stats = {'Modal_Price': 2500.0} 
    
    baseline_price = price_stats.get('Modal_Price', 2500)
    
    if weather_df.empty:
         raise HTTPException(status_code=503, detail="Weather data fetch failed.")

    

    try:
        input_data = prepare_input_vector(model, weather_df, price_stats)
        
        if scaler:
            input_scaled = scaler.transform(input_data)
            prediction = float(model.predict(input_scaled)[0])
        else:
            prediction = float(model.predict(input_data)[0])
        
        # Final Price Logic
        if 'onion' in req.crop.strip().lower():
             print(f"\n[DEBUG] {req.market} | Baseline: {baseline_price} | Diff: {prediction}")
             predicted_price = baseline_price + prediction
             print(f"[DEBUG] Final: {predicted_price}\n")
        else:
             predicted_price = prediction

        if math.isnan(predicted_price) or math.isinf(predicted_price):
            predicted_price = float(baseline_price)
            
        # Percentage Change
        pct_change = ((predicted_price - baseline_price) / baseline_price) * 100
        
        # --- NEW RISK LOGIC (INVERTED) ---
        # Price RISE (+) = High Risk (Supply Stress)
        # Price DROP (-) = Low Risk (Good Supply)
        
        if pct_change > 0:
            # RISING PRICES -> HIGH RISK
            # Scale: +20% rise => Risk Score 50
            risk_score = min(pct_change * 2.5, 100)
            
            if risk_score > 60:
                risk_level = "HIGH RISK"
                msg = f"Price surge of {pct_change:.1f}% predicted! Crop shortage likely."
                adv_color = "error"
                adv_title = "Critical Supply Alert"
                adv_steps = ["Secure stock immediately", "Check for pest outbreaks"]
            elif risk_score > 20:
                risk_level = "MODERATE RISK"
                msg = f"Price rising by {pct_change:.1f}%. Moderate stress."
                adv_color = "warning"
                adv_title = "Inflationary Trend"
                adv_steps = ["Monitor daily rates", "Prepare for volatility"]
            else:
                risk_level = "LOW RISK"
                msg = "Slight price increase. Stable market."
                adv_color = "success"
                adv_title = "Stable Conditions"
                adv_steps = ["Standard operations"]
        else:
            # DROPPING PRICES -> LOW RISK (for crop stability)
            risk_score = 10 # Base low score
            risk_level = "LOW RISK"
            msg = f"Price dropping by {abs(pct_change):.1f}%. Good supply expected."
            adv_color = "success"
            adv_title = "Favorable Harvest"
            adv_steps = ["Plan for storage", "Expect good availability"]

        timeline_data = []
        if not weather_df.empty:
            today = pd.Timestamp(date.today())
            start_date = today
            end_date = today + timedelta(days=6) 
            mask = (weather_df.index >= start_date) & (weather_df.index <= end_date)
            timeline_slice = weather_df.loc[mask]
            timeline_data = timeline_slice.reset_index().replace({np.nan: None}).to_dict('records')

        price_trend = generate_price_trend(baseline_price, predicted_price)
        target_date = (date.today() + timedelta(days=7)).strftime("%b %d")

        return {
            "projected_price": int(predicted_price),
            "baseline_price": int(baseline_price),
            "target_date": target_date,
            "historical_norm": int(baseline_price),
            "risk_level": risk_level,
            "risk_score": int(risk_score),
            "price_change": round(pct_change, 2),
            "weather_source": source,
            "message": msg,
            "advisory": { "title": adv_title, "body": msg, "steps": adv_steps, "color": adv_color },
            "timeline": timeline_data,
            "price_trend": price_trend
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

def monitor_risks():
    pass 

scheduler = BackgroundScheduler()
scheduler.add_job(monitor_risks, 'interval', hours=24) 
scheduler.start()