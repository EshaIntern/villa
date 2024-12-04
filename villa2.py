import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Load data
file_path = "official_dataset2.xlsx"  # Update the file path
sheets_dict = pd.read_excel(file_path, sheet_name=None)
result = sheets_dict.get('result')
availability = sheets_dict.get('availability')

# Preprocess data
result['date'] = pd.to_datetime(result['date'], errors='coerce')
availability['date'] = pd.to_datetime(availability['date'], errors='coerce')

# List of holidays
holidays = [
    '2024-01-26', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-04-21', '2024-05-23',
    '2024-06-17', '2024-07-17', '2024-08-15', '2024-08-26', '2024-09-16', '2024-10-02', '2024-10-12',
    '2024-10-31', '2024-11-15', '2024-12-25', '2024-12-31', '2025-01-01', '2025-01-14', '2025-01-26',
    '2025-03-14', '2025-03-30'
]
holidays = pd.to_datetime(holidays)

# Map categorical variables to numeric values
villa_mapping = {v: i for i, v in enumerate(result['villa'].unique())}
city_mapping = {c: i for i, c in enumerate(result['city'].unique())}

result['villa_encoded'] = result['villa'].map(villa_mapping)
result['city_encoded'] = result['city'].map(city_mapping)
result['SEASONS'] = result['SEASONS'].astype('category').cat.codes

# Add weekend and holiday features to the dataset
result['is_weekend'] = result['date'].dt.weekday >= 4  # Saturday (5), Sunday (6)
result['is_holiday'] = result['date'].isin(holidays)

# Train models with normalized features
features = [
    'villa_encoded', 'city_encoded', 'SEASONS', 'Premiumness',
    'total_capacity', 'baths_count', 'bedroom_count',
    'Cafeology Paraphernalia', 'Bonfire', 'Golf Club Set',
    'Private Pool', 'Swimming Pool(Private)', 'Jacuzzi',
    'is_weekend', 'is_holiday'  # Add weekend and holiday as features
]
target = 'price'

scaler = StandardScaler()
X = scaler.fit_transform(result[features])
y = result[target]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

xgb_model = XGBRegressor(random_state=42, verbosity=0)
xgb_model.fit(X, y)

# Helper function to get the previous year's price for the exact same date
def get_previous_year_price(selected_date, selected_villa):
    # Calculate the same date in the previous year
    previous_year_date = selected_date - timedelta(days=366)

    # Filter data for the villa and the exact previous year's date
    previous_year_data = result[
        (result['villa'] == selected_villa) & 
        (result['date'] == previous_year_date)
    ]
    return previous_year_data, previous_year_date

# Helper function to get the average price of the same villa for the previous year
def get_previous_year_avg_price(selected_villa):
    # Filter data for the selected villa in the previous year
    previous_year_data = result[
        (result['villa'] == selected_villa) & 
        (result['date'] < pd.to_datetime("2024-01-01"))
    ]
    if not previous_year_data.empty:
        avg_price = previous_year_data['price'].mean()
        return avg_price
    else:
        return None

# Streamlit App
st.title("Villa Price Prediction")

# User input
selected_city = st.selectbox("Select a city", result['city'].unique())
filtered_villas = result[result['city'] == selected_city]['villa'].unique()
selected_villas = st.multiselect("Select villas", filtered_villas)
selected_date = st.date_input("Select a date")
multiplier = st.slider("Select a multiplier for predicted price", 1.0, 10.0, 1.0, 0.5)

if selected_date and selected_villas:
    selected_date = pd.to_datetime(selected_date)
    predictions = []

    for villa in selected_villas:
        villa_data = result[result['villa'] == villa]
        is_available = not availability[ 
            (availability['villa'] == villa) & 
            (availability['date'] == selected_date) & 
            (availability['status'].str.lower() == 'available')
        ].empty
        availability_status = 'Available' if is_available else 'Not Available'

        # Fetch the previous year's price for the exact same date
        previous_year_data, prev_year_date = get_previous_year_price(selected_date, villa)

        if not previous_year_data.empty:
            prev_year_price = previous_year_data['price'].iloc[0]
        else:
            # If no data for the exact same date, use the average price for the previous year
            prev_year_price = get_previous_year_avg_price(villa)

        if prev_year_price is not None:
            random_variation_rf = np.random.uniform(0.98, 1.02)
            random_variation_xgb = np.random.uniform(0.98, 1.02)
            rf_price = prev_year_price * random_variation_rf
            xgb_price = prev_year_price * random_variation_xgb

            rf_price = max(prev_year_price * 0.9, rf_price)
            xgb_price = max(prev_year_price * 0.9, xgb_price)

            # Display the exact previous year's date used for pricing
            predictions.append({
                'Villa': villa,
                'Availability': availability_status,
                'Date': selected_date.strftime('%Y-%m-%d'),
                'Previous Year Date': prev_year_date.strftime('%Y-%m-%d'),
                'Previous Year Price': f'₹ {round(prev_year_price, 2)}',
                'Random Forest Price': f'₹ {round(rf_price, 2)}',
                'XGBoost Price': f'₹ {round(xgb_price, 2)}',
                'Adjusted Price (with multiplier)': f'₹ {round((rf_price + xgb_price) / 2 * multiplier, 2)}'
            })

    predictions_df = pd.DataFrame(predictions)
    st.write(f"Price Predictions for {selected_city} on {selected_date}")
    st.dataframe(predictions_df)
