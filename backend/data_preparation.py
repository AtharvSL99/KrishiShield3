import pandas as pd
import numpy as np

# --- Configuration ---
# Define the target commodity for this time series analysis.
TARGET_COMMODITY = 'Wheat'
# Define the number of preceding weeks to use as features
LAG_WINDOW = 4 

def load_data(file_path):
    """Loads the CSV file into a DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=False)
    # Drop the first two unnamed/redundant index columns if they exist
    df = df.iloc[:, 2:]
    return df

def drop_redundant_columns(df):
    """
    Drops columns that are constant, near-constant, or provide redundant
    or non-numerical/non-essential information for this time series model.
    """
    columns_to_drop = [
        'STATE', 'Variety', 'Grade', 'id', 'city', 'std-code', 
        'state', 'gst-state-code', 'iso_3166-2', 'latitude_x', 
        'longitude_x', 'altitude', 'date', 'latitude_y', 'longitude_y', 
        'daylight_duration', 'sunshine_duration',
    ]
    columns_to_drop.extend(['population', 'rank'])
    
    columns_to_drop = list(set(columns_to_drop) & set(df.columns))
    if 'Market Name' in columns_to_drop:
        columns_to_drop.remove('Market Name')
        
    print(f"Dropping {len(columns_to_drop)} columns.")
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    return df_cleaned

def preprocess_and_aggregate_weekly(df_cleaned, commodity=TARGET_COMMODITY):
    """
    Filters data, converts to datetime, and aggregates to weekly average.
    """
    df_filtered = df_cleaned[df_cleaned['Commodity'] == commodity].copy()
    print(f"\nFiltered data for commodity: {commodity}. Records: {len(df_filtered)}")
    
    if df_filtered.empty:
        print(f"No data found for {commodity}.")
        return None

    df_filtered['Price Date'] = pd.to_datetime(df_filtered['Price Date'])
    df_filtered = df_filtered.sort_values(by=['Market Name', 'Price Date'])
    df_filtered = df_filtered.drop(columns=['Commodity', 'District Name'], errors='ignore')

    # Aggregation Rules
    agg_rules = {
        'Min_Price': 'mean',
        'Max_Price': 'mean',
        'Modal_Price': 'mean',
        'temperature_2m_mean': 'mean',
        'temperature_2m_max': 'max',
        'temperature_2m_min': 'min',
        'wind_speed_10m_max': 'mean',
        'precipitation_sum': 'sum', 
        'precipitation_hours': 'sum',
        # We aggregate weather_code by Mode (most common weather that week)
        'weather_code': lambda x: x.mode()[0] if not x.empty else np.nan
    }
    agg_rules = {col: func for col, func in agg_rules.items() if col in df_filtered.columns}

    def resample_market(market_group):
        market_group = market_group.set_index('Price Date').sort_index()
        return market_group.resample('W').agg(agg_rules).fillna(method='ffill')

    df_weekly = df_filtered.groupby('Market Name').apply(resample_market)
    
    print(f"Data aggregated weekly for {commodity}.")
    return df_weekly

def create_lagged_features(df_weekly, lag_window=LAG_WINDOW):
    """
    Creates lagged features for weather columns and Modal_Price.
    UPDATED: Now includes weather_code as a direct numerical lag (no OHE).
    """
    print(f"\nCreating lagged features (Lag 1 up to Lag {lag_window})...")
    
    # Added 'weather_code' to this list to implement the lag
    features_to_lag = [
        'Modal_Price',
        'temperature_2m_mean', 
        'temperature_2m_max', 
        'temperature_2m_min', 
        'wind_speed_10m_max', 
        'precipitation_sum',
        'precipitation_hours',
        'weather_code'  # <-- NOW INCLUDED AS STANDARD LAG
    ]

    features_to_lag = [f for f in features_to_lag if f in df_weekly.columns]
    df_lagged = df_weekly.copy()
    
    for feature in features_to_lag:
        for lag in range(1, lag_window + 1):
            new_col_name = f'{feature}_Lag{lag}'
            df_lagged[new_col_name] = df_lagged.groupby(level=0)[feature].shift(lag)

    df_lagged = df_lagged.dropna()

    print(f"Finished creating lagged features. Final Shape: {df_lagged.shape}")
    print("Features include simple weather_code lags (e.g., weather_code_Lag1).")
    
    return df_lagged

if __name__ == "__main__":
    file_path = 'semifinal.csv' 
    df = load_data(file_path)
    df_cleaned = drop_redundant_columns(df)
    df_time_series = preprocess_and_aggregate_weekly(df_cleaned, commodity=TARGET_COMMODITY)

    if df_time_series is not None:
        df_final_features = create_lagged_features(df_time_series, lag_window=LAG_WINDOW)
        output_file = f'{TARGET_COMMODITY.lower()}_market_lagged_features.csv'
        df_final_features.to_csv(output_file)
        print(f"\n--- SUCCESS ---")
        print(f"Saved to: {output_file}")
        print("Please re-run model_training.py now.")