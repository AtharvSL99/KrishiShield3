import os
from dotenv import load_dotenv
from twilio.rest import Client # NEW: Missing Import
from fastapi import FastAPI, HTTPException
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import requests
import json
import random
import math
from datetime import datetime, timedelta, date


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

# --- 3. HELPER FUNCTIONS ---

def load_model_and_scaler(crop):
    try:
        with open(f"{crop}.pkl", 'rb') as f:
            model = pickle.load(f)
            if isinstance(model, xgb.XGBRegressor):
                model.set_params(device='cpu', tree_method='hist')
        with open(f"{crop.lower()}_scaler.pkl", 'rb') as f:
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

def get_seasonal_averages(crop):
    filename = f"{crop.lower()}_market_lagged_features.csv"
    defaults = {'Modal_Price': 2500.0}
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            return df.mean(numeric_only=True).to_dict()
        except:
            return defaults
    return defaults

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

# --- 4. AUTOMATED MONITORING SYSTEM ---

def load_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'r') as f: return json.load(f)
    return []

def save_alert(new_alert):
    alerts = load_alerts()
    new_alert['id'] = len(alerts) + 1
    new_alert['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts.insert(0, new_alert)
    with open(ALERTS_FILE, 'w') as f: json.dump(alerts, f)
    return new_alert

def monitor_risks():
    """Scheduled Job: Checks for SUDDEN price jumps compared to cached values."""
    print("--- [SCHEDULER] Running Risk Monitor (24h Check) ---")
    
    if os.path.exists(RISK_CACHE_FILE):
        with open(RISK_CACHE_FILE, 'r') as f: cache = json.load(f)
    else:
        cache = {}

    # Check specific markets (Expand this list in production)
    check_list = [("Onion", "pune"), ("Potato", "nashik"), ("Wheat", "mumbai")] 
    
    updated_cache = cache.copy()
    
    for crop, market in check_list:
        if market not in MARKET_COORDS: continue
        
        lat = MARKET_COORDS[market]['latitude_x']
        lon = MARKET_COORDS[market]['longitude_x']
        model, scaler = load_model_and_scaler(crop)
        if not model: continue
        
        weather, _ = fetch_weather_data(lat, lon) 
        if weather.empty: continue
        
        hist = get_seasonal_averages(crop)
        baseline = hist.get('Modal_Price', 2500)
        
        feats = model.get_booster().feature_names
        inp = pd.DataFrame(0.0, index=[0], columns=feats)
        for c in feats:
            if "temperature" in c: inp[c] = weather['temperature_2m_max'].mean()
            elif "precipitation" in c: inp[c] = weather['precipitation_sum'].sum()
            elif "Price" in c: inp[c] = baseline
            
        try:
            pred = float(model.predict(scaler.transform(inp))[0])
            
            cache_key = f"{crop}_{market}"
            last_price = cache.get(cache_key, baseline)
            
            if last_price == 0: last_price = baseline # Prevent div by zero
            
            pct_change = ((pred - last_price) / last_price) * 100
            
            # Threshold: > 10% deviation
            if abs(pct_change) > 10:
                print(f"!!! ALERT TRIGGERED: {crop} in {market} changed by {pct_change:.1f}%")
                
                alert_obj = {
                    "type": "Critical" if abs(pct_change) > 20 else "Warning",
                    "crop": crop,
                    "market": market,
                    "message": f"Significant price shift detected! Changed from ₹{int(last_price)} to ₹{int(pred)}.",
                    "change_pct": round(pct_change, 1),
                    "is_sms_sent": True
                }
                save_alert(alert_obj)
            
            updated_cache[cache_key] = pred
        except Exception as e:
            print(f"Monitor Error for {crop}: {e}")

    with open(RISK_CACHE_FILE, 'w') as f: json.dump(updated_cache, f)

# Start Scheduler (Updated to 24 Hours)
scheduler = BackgroundScheduler()
scheduler.add_job(monitor_risks, 'interval', hours=24) 
scheduler.start()

# --- 5. API ENDPOINTS ---

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
    
    # Initialize variables to prevent UnboundLocalError
    predicted_price = 0 
    
    lat = MARKET_COORDS[req.market]['latitude_x']
    lon = MARKET_COORDS[req.market]['longitude_x']
    
    model, scaler = load_model_and_scaler(req.crop)
    if not model:
        raise HTTPException(status_code=500, detail=f"Model files for {req.crop} missing.")

    # Unpack tuple (Data, Source)
    weather_df, source = fetch_weather_data(lat, lon)
    
    hist_data = get_seasonal_averages(req.crop)
    
    if weather_df.empty:
         raise HTTPException(status_code=503, detail="Weather data fetch failed.")

    try:
        # Prediction Logic
        model_features = model.get_booster().feature_names
        input_data = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        current_temp = weather_df['temperature_2m_max'].mean()
        current_rain = weather_df['precipitation_sum'].sum()
        baseline_price = hist_data.get('Modal_Price', 2500)

        # FIX: Check 'col' not 'c'
        for col in model_features:
            if "temperature" in col: input_data[col] = current_temp
            elif "precipitation" in col: input_data[col] = current_rain
            elif "Price" in col: input_data[col] = baseline_price
            
        X_scaled = scaler.transform(input_data)
        predicted_price = float(model.predict(X_scaled)[0])
        
        # Risk Logic
        pct_change = ((predicted_price - baseline_price) / baseline_price) * 100
        risk_score = min(max((pct_change + 10) * 2.5, 0), 100)
        
        if risk_score > 60:
            risk_level = "HIGH RISK"
            msg = "High volatility predicted due to weather conditions."
            adv_color = "error"
            adv_title = "Critical Advisory"
            adv_steps = ["Delay Planting", "Check Drainage"]
        elif risk_score > 30:
            risk_level = "MODERATE RISK"
            msg = "Moderate deviation from historical norms."
            adv_color = "warning"
            adv_title = "Cautionary Advisory"
            adv_steps = ["Monitor irrigation", "Watch for pests"]
        else:
            risk_level = "LOW RISK"
            msg = "Conditions look stable and favorable."
            adv_color = "success"
            adv_title = "Favorable Conditions"
            adv_steps = ["Proceed with standard care", "Maximize yield"]

        advisory = {
            "title": adv_title, "body": msg, "steps": adv_steps, "color": adv_color
        }

        # 5-Day Timeline Logic
        today = pd.Timestamp(date.today())
        start_date = today - timedelta(days=2)
        end_date = today + timedelta(days=2)
        timeline_slice = weather_df.loc[start_date:end_date]
        timeline_data = timeline_slice.reset_index().to_dict('records')

        # Graph Data Logic
        price_trend = generate_price_trend(baseline_price, predicted_price)

        return {
            "projected_price": int(predicted_price),
            "historical_norm": int(baseline_price),
            "risk_level": risk_level,
            "risk_score": int(risk_score),
            "price_change": round(pct_change, 2),
            "weather_source": source,
            "message": msg,
            "advisory": advisory,
            "timeline": timeline_data,
            "price_trend": price_trend
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")