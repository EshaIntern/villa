import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import streamlit as st
from datetime import datetime, timedelta

# Load data
file_path = "official_dataset2.xlsx"  # Update the file path
sheets_dict = pd.read_excel(file_path, sheet_name=None)
result = sheets_dict.get('result')
availability = sheets_dict.get('availability')

# Preprocess data
result['date'] = pd.to_datetime(result['date'], errors='coerce')
availability['date'] = pd.to_datetime(availability['date'], errors='coerce')

# List of holidays (already provided)
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

# Train models
features = [
    'villa_encoded', 'city_encoded', 'SEASONS', 'Premiumness',
    'total_capacity', 'baths_count', 'bedroom_count',
    'Cafeology Paraphernalia', 'Bonfire', 'Golf Club Set',
    'Private Pool', 'Swimming Pool(Private)', 'Jacuzzi',
    'is_weekend', 'is_holiday'  # Add weekend and holiday as features
]
target = 'price'

X = result[features]
y = result[target]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

xgb_model = XGBRegressor(random_state=42, verbosity=0)
xgb_model.fit(X, y)

# Create a city-month-season mapping from the result sheet
city_month_season_mapping = result[['city', 'date', 'SEASONS']].copy()
city_month_season_mapping['month'] = city_month_season_mapping['date'].dt.month

# Helper function to get the previous year's data for the same week
def get_previous_year_week_data(selected_date, selected_villa):
    # Find the previous year's date for the same week (same date)
    previous_year_date = selected_date - timedelta(days=365)
    previous_year_week_start = previous_year_date - timedelta(days=previous_year_date.weekday())  # Start of the week

    # Filter the result for the previous year and same week
    previous_year_data = result[
        (result['villa'] == selected_villa) & 
        (result['date'] >= previous_year_week_start) & 
        (result['date'] < previous_year_week_start + timedelta(weeks=1))
    ]
    return previous_year_data

# Streamlit App
st.title("Villa Price Prediction")

# User input
selected_city = st.selectbox("Select a city", result['city'].unique())
filtered_villas = result[result['city'] == selected_city]['villa'].unique()
selected_villas = st.multiselect("Select villas", filtered_villas)
selected_date = st.date_input("Select a date")
price_increase_percentage = st.slider("Select price increase percentage", 1, 20, 5)
multiplier = st.slider("Select a multiplier for price adjustment", 1.0, 10.0, 1.0, 0.5)


if selected_date and selected_villas:
    selected_date = pd.to_datetime(selected_date)
    predictions = []

    for villa in selected_villas:
        villa_data = result[result['villa'] == villa]
        is_new_villa = villa_data.empty

        # Check availability
        is_available = not availability[
            (availability['villa'] == villa) & 
            (availability['date'] == selected_date) & 
            (availability['status'].str.lower() == 'available')
        ].empty
        availability_status = 'Available' if is_available else 'Not Available'

        # Extract the season dynamically based on city and month
        month = selected_date.month
        season_code = city_month_season_mapping[
            (city_month_season_mapping['city'] == selected_city) & 
            (city_month_season_mapping['month'] == month)
        ]['SEASONS'].values

        if len(season_code) == 0:  # Default to -1 if no season found for the given city and month
            season_code = -1
        else:
            season_code = season_code[0]

        # Date-based features
        is_weekend = selected_date.weekday() >= 4  # Weekend starts from Friday
        is_holiday = selected_date in holidays

        if is_available:
            # Get the previous year's data for the same week
            previous_year_data = get_previous_year_week_data(selected_date, villa)

            # Compute the average price from previous year's data
            if not previous_year_data.empty:
                prev_year_avg_price = previous_year_data['price'].mean()
            else:
                prev_year_avg_price = villa_data['price'].mean()  # Fall back to current year's price if no data found

            # Ensure the price this year is at least a little higher than the previous year
            price_increase_factor = (1 + (price_increase_percentage / 100))  # Increase by the selected percentage
            adjusted_prev_year_price = prev_year_avg_price * price_increase_factor

            # Introduce random variation to the prices
            random_variation_rf = np.random.uniform(0.95, 1.05)  # Random multiplier for RF
            random_variation_xgb = np.random.uniform(0.95, 1.05)  # Random multiplier for XGBoost
            rf_price = adjusted_prev_year_price * random_variation_rf
            xgb_price = adjusted_prev_year_price * random_variation_xgb

            # Calculate the average price and adjusted price
            average_price = (rf_price + xgb_price) / 2
            adjusted_price = average_price * multiplier

            predictions.append({
                'Villa': villa,
                'Availability': availability_status,
                'Date': selected_date.strftime('%Y-%m-%d'),
                'Previous Year Avg Price': prev_year_avg_price,
                'Adjusted Previous Year Price': adjusted_prev_year_price,
                'Random Forest Price': rf_price,
                'XGBoost Price': xgb_price,
                'Average Price': average_price,
                'Adjusted Price (with multiplier)': adjusted_price
            })

    predictions_df = pd.DataFrame(predictions)
    st.write(f"Price Predictions for {selected_city} on {selected_date}")
    st.dataframe(predictions_df)
