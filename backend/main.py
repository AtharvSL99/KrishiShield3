
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

DATA_DIR = "backend/data"

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
    # This is a placeholder. Actual risk logic will be implemented here.
    # For now, let's return a dummy risk assessment.
    
    # Load some data to simulate risk calculation
    try:
        weather_df = load_weather_data(location_id)
        market_df = load_market_data(location_id, commodity)
    except HTTPException as e:
        raise e

    # Simple dummy risk logic
    risk_score = 0
    risk_type = "None"
    advisory = "All good for now."

    # Example: if max temp is high, suggest heat risk
    if not weather_df.empty and weather_df['temp_max'].iloc[-1] > 35:
        risk_score += 30
        risk_type = "Heat Stress"
        advisory = "High temperatures expected. Ensure adequate irrigation."
    
    # Example: if market price dropped significantly
    if not market_df.empty:
        latest_price = market_df['mandi_price'].iloc[-1]
        previous_price = market_df['mandi_price'].iloc[-2] if len(market_df) > 1 else latest_price
        if (previous_price - latest_price) / previous_price > 0.10: # 10% drop
            risk_score += 40
            risk_type = "Market Price Drop"
            advisory = "Significant price drop detected. Consider delaying sale if storage is available."

    risk_score = min(risk_score, 100) # Cap score at 100

    return {
        "location_id": location_id,
        "commodity": commodity,
        "risk_score": risk_score,
        "risk_type": risk_type,
        "advisory": advisory,
        "date": datetime.now().isoformat()
    }
