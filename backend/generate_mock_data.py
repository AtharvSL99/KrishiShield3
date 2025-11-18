
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_weather_data(start_date, end_date, location_id, output_dir='data'):
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    temp_max = np.random.uniform(25, 40, len(dates)).round(1)
    temp_min = np.random.uniform(15, 25, len(dates)).round(1)
    rainfall_mm = np.random.uniform(0, 50, len(dates)).round(1)
    humidity = np.random.uniform(60, 95, len(dates)).round(1)

    df = pd.DataFrame({
        'date': dates,
        'location_id': location_id,
        'temp_max': temp_max,
        'temp_min': temp_min,
        'rainfall_mm': rainfall_mm,
        'humidity': humidity
    })
    
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'weather_data_{location_id}.csv')
    df.to_csv(file_path, index=False)
    print(f"Generated weather data: {file_path}")

def generate_market_data(start_date, end_date, location_id, commodity, output_dir='data'):
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    mandi_price = np.random.uniform(1000, 5000, len(dates)).round(2)

    df = pd.DataFrame({
        'date': dates,
        'location_id': location_id,
        'commodity': commodity,
        'mandi_price': mandi_price
    })
    
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'market_data_{location_id}_{commodity}.csv')
    df.to_csv(file_path, index=False)
    print(f"Generated market data: {file_path}")

if __name__ == "__main__":
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)
    location_id = 'LOC001'
    commodity = 'Wheat'

    generate_weather_data(start_date, end_date, location_id, output_dir='backend/data')
    generate_market_data(start_date, end_date, location_id, commodity, output_dir='backend/data')
