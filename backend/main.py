
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from datetime import datetime, timedelta
from cachetools import cached, TTLCache

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

DATA_DIR = "data"

# In-memory cache for 15 minutes
data_cache = TTLCache(maxsize=100, ttl=900)

# Load mock data with caching
@cached(data_cache)
def load_weather_data(location_id):
    file_path = os.path.join(DATA_DIR, f'weather_data_{location_id}.csv')
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Weather data for {location_id} not found.")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@cached(data_cache)
def load_market_data(location_id, commodity):
    file_path = os.path.join(DATA_DIR, f'market_data_{location_id}_{commodity}.csv')
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Market data for {location_id}, {commodity} not found.")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@app.get("/locations")
async def get_locations():
    files = os.listdir(DATA_DIR)
    locations = set()
    for f in files:
        if f.startswith('weather_data_') and f.endswith('.csv'):
            parts = f.replace('.csv', '').split('_')
            if len(parts) == 3:
                locations.add(parts[2])
    return list(locations)

@app.get("/commodities/{location_id}")
async def get_commodities(location_id: str):
    files = os.listdir(DATA_DIR)
    commodities = set()
    for f in files:
        if f.startswith(f'market_data_{location_id}_') and f.endswith('.csv'):
            parts = f.replace('.csv', '').split('_')
            if len(parts) == 4:
                commodities.add(parts[3])
    return list(commodities)

@app.get("/")
async def read_root():
    return {"message": "Welcome to KrishiShield Backend API"}

@app.get("/weather/{location_id}")
async def get_weather_data(location_id: str):
    try:
        df = load_weather_data(location_id)
        return df.to_dict(orient='records')
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{location_id}/{commodity}")
async def get_market_data(location_id: str, commodity: str):
    try:
        df = load_market_data(location_id, commodity)
        return df.to_dict(orient='records')
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Risk Engine (placeholder for now)
@app.get("/risk/{location_id}/{commodity}")
async def get_risk_assessment(location_id: str, commodity: str):
    try:
        weather_df = load_weather_data(location_id)
        market_df = load_market_data(location_id, commodity)
    except HTTPException as e:
        raise e

    # --- Constants for Risk Rules ---
    CROP_HEAT_THRESHOLD = 35
    CROP_MIN_RAINFALL_14_DAYS = 10
    FLOOD_RAINFALL_3_DAYS = 50
    PEST_HUMIDITY_THRESHOLD = 80
    PEST_TEMP_MIN = 20
    PEST_TEMP_MAX = 30
    PRICE_CRASH_THRESHOLD_7_DAYS = -0.10
    VOLATILITY_THRESHOLD_14_DAYS = 0.2

    weather_score = 0
    pest_score = 0
    market_score = 0
    
    advisories = []

    # --- Weather Risk ---
    if not weather_df.empty:
        # Heat Stress (using last 3 days as forecast)
        last_3_days_weather = weather_df.tail(3)
        if not last_3_days_weather.empty and last_3_days_weather['temp_max'].mean() > CROP_HEAT_THRESHOLD:
            weather_score = max(weather_score, 70)
            advisories.append("Heat Stress: High temperatures expected. Ensure adequate irrigation.")

        # Drought Risk
        last_14_days_weather = weather_df.tail(14)
        if not last_14_days_weather.empty and last_14_days_weather['rainfall_mm'].sum() < CROP_MIN_RAINFALL_14_DAYS:
            weather_score = max(weather_score, 60)
            advisories.append("Drought Risk: Low rainfall detected. Monitor soil moisture and irrigate if necessary.")

        # Flood Risk
        if not last_3_days_weather.empty and last_3_days_weather['rainfall_mm'].sum() > FLOOD_RAINFALL_3_DAYS:
            weather_score = max(weather_score, 85)
            advisories.append("Flood Risk: Heavy rainfall detected. Ensure proper drainage to avoid waterlogging.")

    # --- Pest Risk ---
    if not weather_df.empty:
        last_5_days_weather = weather_df.tail(5)
        avg_humidity = last_5_days_weather['humidity'].mean()
        avg_temp = (last_5_days_weather['temp_max'].mean() + last_5_days_weather['temp_min'].mean()) / 2
        if avg_humidity > PEST_HUMIDITY_THRESHOLD and PEST_TEMP_MIN < avg_temp < PEST_TEMP_MAX:
            pest_score = 75
            advisories.append("Pest Risk: Conditions are favorable for pest infestation. Inspect crops and consider preventive measures.")

    # --- Market Risk ---
    if not market_df.empty and len(market_df) > 7:
        # Price Crash
        price_7_days_ago = market_df['mandi_price'].iloc[-8]
        latest_price = market_df['mandi_price'].iloc[-1]
        price_change_pct = (latest_price - price_7_days_ago) / price_7_days_ago
        if price_change_pct <= PRICE_CRASH_THRESHOLD_7_DAYS:
            market_score = max(market_score, 80)
            advisories.append("Market Risk: Significant price drop detected. Consider delaying sale if storage is available.")

        # High Volatility
        last_14_days_market = market_df.tail(14)
        if len(last_14_days_market) > 1:
            volatility = last_14_days_market['mandi_price'].std() / last_14_days_market['mandi_price'].mean()
            if volatility > VOLATILITY_THRESHOLD_14_DAYS:
                market_score = max(market_score, 65)
                advisories.append("Market Volatility: Prices are fluctuating significantly. Stay informed on market trends.")

    # --- Final Risk Score and Advisory ---
    final_risk_score = max(weather_score, pest_score, market_score)
    
    risk_type = "None"
    if final_risk_score > 0:
        if final_risk_score == weather_score:
            risk_type = "Weather"
        elif final_risk_score == pest_score:
            risk_type = "Pest"
        else:
            risk_type = "Market"

    final_advisory = " ".join(advisories) if advisories else "All good for now."

    return {
        "location_id": location_id,
        "commodity": commodity,
        "risk_score": final_risk_score,
        "risk_type": risk_type,
        "advisory": final_advisory,
        "date": datetime.now().isoformat()
    }
