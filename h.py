import pandas as pd
from io import StringIO
import requests
import sys

# --- 1. Setup ---
# FIX 1: REMOVED ".csv" from the end of this URL. 
# The ID "9ef84268..." is correct for "Current Daily Price (Mandi)".
API = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

params = {
    # Your key (if this fails, try the backup key below)
    "api-key": "579b464db66ec23bdd00000125b231d6079f44e54c664acfce4c26dc",
    # Backup public key found in documentation if yours hits a limit:
    # "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
    "filters[state.keyword]": "MAHARASHTRA",
    "format": "csv"  # This forces the CSV format
}

print("Fetching data...")
try:
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
except Exception as e:
    print(f"Network Error: {e}")
    sys.exit(1)

# --- 2. Error Handling ---
if "invalid_resource_id" in r.text:
    print("Error: Invalid Resource ID. The government might have rotated the dataset ID.")
    sys.exit(1)
    
if r.text.strip().startswith("<?xml") or "<error>" in r.text:
    print("Error: API returned XML error message:")
    print(r.text)
    sys.exit(1)

# --- 3. Load and Clean ---
print("Parsing CSV...")
try:
    df = pd.read_csv(StringIO(r.text))
except pd.errors.EmptyDataError:
    print("Error: The API returned an empty CSV. Check your filters.")
    sys.exit(1)

# CLEAN HEADERS: Convert "Modal_x0020_Price" or "Modal Price" -> "modal_price"
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('_x0020_', '_')

print(f"Columns found: {list(df.columns)}")

# --- 4. Process Data ---
# 1. Find Date Column
possible_dates = [c for c in df.columns if 'date' in c]
if not possible_dates:
    print("Error: No 'date' column found.")
    sys.exit(1)

date_col = possible_dates[0]
print(f"Using '{date_col}' as date column.")

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')

# 2. Filter Date Range
start = '2025-10-01'
end   = '2025-10-31'
mask = (df[date_col] >= start) & (df[date_col] <= end)
df_oct = df.loc[mask].copy()

if df_oct.empty:
    print(f"Warning: No data found between {start} and {end}.")
    print("Showing first 5 rows of raw data to check available dates:")
    print(df.head())
    sys.exit(0)

# 3. Filter Commodities
# Ensure 'commodity' column exists
if 'commodity' in df_oct.columns:
    wanted = ['ONION', 'WHEAT', 'POTATO', 'TOMATO', 'RICE']
    # Filter case-insensitive
    df_oct = df_oct[df_oct['commodity'].str.upper().isin(wanted)]
    
    if df_oct.empty:
         print("Warning: Data found for dates, but not for selected commodities (Onion, Wheat, etc).")
         sys.exit(0)

    # Grouping
    # Ensure price columns exist
    price_col = 'modal_price' if 'modal_price' in df_oct.columns else 'max_price'
    
    grouped = df_oct.groupby(['commodity', 'market', date_col]).agg({
        price_col: 'first',
        'min_price': 'min',
        'max_price': 'max'
    }).reset_index()

    grouped.to_csv('maha_mandi_prices_oct2025.csv', index=False)
    print("Success! Saved: maha_mandi_prices_oct2025.csv")
    print(grouped.head())

else:
    print("Error: 'commodity' column not found.")
    print(df.columns)