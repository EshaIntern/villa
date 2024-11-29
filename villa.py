import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime

# Holiday list with occasion, date, and day
holidays = {
    "January 26": 1.2,  # Republic Day
    "March 25": 1.2,    # Holi
    "March 29": 1.2,    # Good Friday
    "April 11": 1.2,    # Id-ul-Fitr
    "April 17": 1.2,    # Ram Navmi
    "April 21": 1.2,    # Mahavir Jayanti
    "May 23": 1.2,      # Buddha Purnima
    "June 17": 1.2,     # Id-ul-Zuha (Bakrid)
    "July 17": 1.2,     # Muharram
    "August 15": 1.2,   # Independence Day / Parsi New Year's Day / Nauraj
    "August 26": 1.2,   # Janamashtami (Vaishnva)
    "September 16": 1.2,  # Milad-un-Nabi or Id-e-Milad
    "October 2": 1.2,    # Mahatma Gandhi’s Birthday
    "October 12": 1.2,   # Dussehra
    "October 31": 1.2,   # Diwali
    "November 15": 1.2,  # Guru Nanak’s Birthday
    "December 25": 1.2,  # Christmas
}

# File path or URL
file_path = "official_dataset2.xlsx"  # Replace with your file path or URL

try:
    # Load the file
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    result = sheets_dict.get('result')
    availability = sheets_dict.get('availability')

    # Preprocessing Data
    def preprocess_data(result):
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result['day'] = result['date'].dt.day
        result['month'] = result['date'].dt.month
        result['year'] = result['date'].dt.year
        result['day_of_week'] = result['date'].dt.dayofweek  # 0 = Monday, ..., 6 = Sunday

        # Encoding categorical variables
        result['city_encoded'] = result['city'].astype('category').cat.codes
        result['villa_encoded'] = result['villa'].astype('category').cat.codes
        result['season_encoded'] = result['SEASONS'].astype('category').cat.codes

        # Features and target
        features = ['city_encoded', 'villa_encoded', 'month', 'day', 'day_of_week', 'total_capacity', 
                    'baths_count', 'bedroom_count', 'Premiumness', 'season_encoded']
        target = 'price'

        X = result[features]
        y = result[target]

        return X, y, features

    # Train models
    def train_models(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)

        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)

        return rf_model, xgb_model, scaler

    # Adjust price based on seasonality
    def adjust_price_for_season(price, season):
        season_multipliers = {'Superpeak': 1.5, 'Peak': 1.2, 'Mid': 1.1, 'Low': 1.0}
        return price * season_multipliers.get(season, 1.0)

    # Adjust price for premiumness
    def adjust_price_for_premiumness(price, premiumness):
        premiumness_multipliers = {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.5}
        return price * premiumness_multipliers.get(premiumness, 1.0)

    # Check if the date is a holiday
    def is_holiday(selected_date):
        # Format the selected date as 'Month Day' (e.g., 'October 31')
        holiday_date_str = selected_date.strftime('%B %d')

        # Debugging: Print the formatted date to ensure it's correct
        print(f"Formatted Date: {holiday_date_str}")

        # Return the holiday multiplier if the date is found, otherwise return 1.0
        holiday_multiplier = holidays.get(holiday_date_str, 1.0)

        # Debugging: Print the multiplier
        print(f"Holiday Multiplier: {holiday_multiplier}")

        return holiday_multiplier

    # Adjust price for weekends
    def adjust_price_for_weekend(price, selected_date):
        if selected_date.weekday() == 5 or selected_date.weekday() == 6:  # Saturday (5) or Sunday (6)
            return price * 1.2  # 20% increase for weekends
        return price

    # In the 'predict_prices_for_date' function, add the weekend adjustment
    def predict_prices_for_date(selected_date, selected_city, selected_villa=None):
        selected_date = pd.to_datetime(selected_date)
        month = selected_date.month
        day = selected_date.day
        day_of_week = selected_date.dayofweek

        # Check if the date is a holiday
        holiday_multiplier = is_holiday(selected_date)

        # Filter available villas for the selected date (status = 'available')
        available_villas = availability[(availability['date'] == selected_date.strftime('%Y-%m-%d')) & 
                                        (availability['status'] == 'available')]

        if available_villas.empty:
            st.warning("No villas available for the selected date!")
            return []

        # Filter by city
        available_villas = available_villas[available_villas['villa'].isin(result[result['city'] == selected_city]['villa'].unique())]

        if selected_villa:
            available_villas = available_villas[available_villas['villa'] == selected_villa]

        if available_villas.empty:
            st.warning("No villas available for the selected city and/or villa on this date!")
            return []

        predictions = []
        for _, row in available_villas.iterrows():
            villa = row['villa']
            city = selected_city  # City is already filtered by the UI selection

            # Get villa details from result sheet
            villa_details = result[result['villa'] == villa]
            if villa_details.empty:
                continue

            selected_season = villa_details[villa_details['month'] == month]['SEASONS'].mode()
            selected_season = selected_season[0] if not selected_season.empty else 'Low'
            premiumness = villa_details['Premiumness'].values[0]

            # Prepare input data
            input_data = {
                'city_encoded': [villa_details['city_encoded'].iloc[0]],
                'villa_encoded': [villa_details['villa_encoded'].iloc[0]],
                'month': [month],
                'day': [day],
                'day_of_week': [day_of_week],
                'total_capacity': [villa_details['total_capacity'].iloc[0]],
                'baths_count': [villa_details['baths_count'].iloc[0]],
                'bedroom_count': [villa_details['bedroom_count'].iloc[0]],
                'Premiumness': [premiumness],
                'season_encoded': [villa_details[villa_details['SEASONS'] == selected_season]['season_encoded'].iloc[0]]
            }

            input_df = pd.DataFrame(input_data)
            scaled_input = scaler.transform(input_df)

            # Predict using both models
            rf_price = rf_model.predict(scaled_input)[0]
            xgb_price = xgb_model.predict(scaled_input)[0]

            # Adjust predictions for seasonality, premiumness, holiday, and weekend
            rf_price = adjust_price_for_season(rf_price, selected_season)
            xgb_price = adjust_price_for_season(xgb_price, selected_season)

            rf_price = adjust_price_for_premiumness(rf_price, premiumness)
            xgb_price = adjust_price_for_premiumness(xgb_price, premiumness)

            # Apply holiday price adjustment
            rf_price *= holiday_multiplier
            xgb_price *= holiday_multiplier

            # Apply weekend price adjustment
            rf_price = adjust_price_for_weekend(rf_price, selected_date)
            xgb_price = adjust_price_for_weekend(xgb_price, selected_date)

            # Calculate final average price
            avg_price = np.mean([rf_price, xgb_price])

            predictions.append({
                'villa': villa,
                'city': city,
                'predicted_price': avg_price
            })

        return predictions

    # UI input for selecting date, city, and villa
    selected_date = st.date_input("Select Date", min_value=datetime.today())
    selected_city = st.selectbox("Select City", options=result['city'].unique())
    selected_villa = st.selectbox("Select Villa (Optional)", options=["All"] + result['villa'].unique().tolist())

    # When button is pressed, make predictions
    if st.button("Predict Price"):
        predictions = predict_prices_for_date(selected_date, selected_city, selected_villa if selected_villa != "All" else None)

        if predictions:
            st.write("Predicted Prices:")
            for pred in predictions:
                st.write(f"Villa: {pred['villa']}, City: {pred['city']}, Predicted Price: ₹{pred['predicted_price']:.2f}")
        else:
            st.write("No predictions available for the selected inputs.")
    
except Exception as e:
    st.error(f"Error: {e}")
