import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import geopy.distance
import requests  # For public transport API

# Load dataset
DATA_PATH = "C:/Users/Joshika K/jsshack/aqi_bangalore_traffic.csv"
MODEL_PATH = "aqi_forecast_model.h5"

# Load the trained model
try:
    model = load_model(
        MODEL_PATH, custom_objects={"mse": tf.keras.losses.MeanSquaredError}
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Set model to None if loading fails
    st.error(f"Error loading model: {e}")
    if model is None:
        st.stop()

# Load the data
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# --- Utility Functions ---
def get_aqi_level(aqi):
    if 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Moderate"
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi <= 200:
        return "Unhealthy"
    elif 201 <= aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_recommendations(aqi_level, profile=None):
    recommendations = []
    if aqi_level == "Good":
        recommendations.append("Enjoy outdoor activities!")
    elif aqi_level == "Moderate":
        recommendations.append("Acceptable air quality. Enjoy outdoor activities.")
    elif aqi_level == "Unhealthy for Sensitive Groups":
        recommendations.append(
            "Sensitive groups (children, elderly, people with respiratory issues) should limit prolonged outdoor exertion."
        )
    elif aqi_level == "Unhealthy":
        recommendations.append(
            "Everyone may experience health effects; sensitive groups may experience more serious effects."
        )
        recommendations.append("Consider limiting outdoor activities.")
    elif aqi_level == "Very Unhealthy":
        recommendations.append("Significant health risk. Avoid prolonged outdoor activities.")
    else:  # Hazardous
        recommendations.append("Health emergency! Everyone should avoid outdoor activities.")

    # Personalized Recommendations
    if profile:  # If user profile is available
        if (
            profile.get("respiratory_condition") == "yes"
            and aqi_level
            in ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        ):
            recommendations.append(
                "Individuals with respiratory conditions should stay indoors and use an air purifier if available."
            )
        if (
            profile.get("age_group") == "child"
            and aqi_level
            in ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        ):
            recommendations.append("Children should avoid outdoor play.")
    return recommendations

def suggest_transportation(traffic_volume, aqi_level):
    if (
        traffic_volume > 70
        and aqi_level in ["Unhealthy", "Very Unhealthy", "Hazardous"]
    ):
        return "Consider using public transport, cycling, or walking to reduce traffic congestion and pollution."
    elif traffic_volume > 70:
        return "Traffic is heavy. Consider alternative routes or transportation."
    else:
        return "Traffic is relatively light."

def forecast_aqi(area_name, forecast_days=7, seq_length=10):
    if area_name not in df["area_name"].unique():
        return None, None

    # Filter data for the given area
    area_data = df[df["area_name"] == area_name].sort_values(by="date")

    # Extract AQI values
    aqi_data = area_data[["overall_aqi"]].values

    # Normalize AQI values
    scaler = MinMaxScaler()
    aqi_scaled = scaler.fit_transform(aqi_data)

    # Get the latest date in dataset
    latest_dataset_date = area_data["date"].max()
    today = pd.Timestamp.today().normalize()
    start_date = max(today, latest_dataset_date)

    # Prepare input sequence for prediction
    forecast_input = aqi_scaled[-seq_length:]

    predictions = []
    for _ in range(forecast_days):
        forecast_input = forecast_input.reshape(1, seq_length, 1)
        pred = model.predict(forecast_input, verbose=0)
        predictions.append(pred[0][0])
        forecast_input = np.append(forecast_input[0][1:], pred).reshape(seq_length, 1)

    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Generate future dates
    future_dates = [start_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Create DataFrame for AQI forecast
    aqi_forecast_df = pd.DataFrame(
        {"Date": future_dates, "Predicted_AQI": predictions.flatten()}
    )

    # Get latest pollutant levels
    latest_data = area_data.iloc[-1]
    current_values = {
        "PM2.5": latest_data["pm2.5"],
        "PM10": latest_data["pm10"],
        "CO": latest_data["co"],
        "SO2": latest_data["so2"],
        "NO2": latest_data["no2"],
        "NH3": latest_data["nh3"],
        "Traffic Volume": latest_data["traffic_volume"],
        "Environmental Impact": latest_data["environmental_impact"],
    }

    # Add aqi interpretation to data frame
    aqi_forecast_df["AQI_Level"] = aqi_forecast_df["Predicted_AQI"].apply(
        get_aqi_level
    )

    return current_values, aqi_forecast_df

# --- Session State Management ---
if "user_accounts" not in st.session_state:
    st.session_state["user_accounts"] = {}
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "preferred_locations" not in st.session_state:
    st.session_state["preferred_locations"] = {}
if "report_submitted" not in st.session_state:
    st.session_state["report_submitted"] = False
if "reports" not in st.session_state:
    st.session_state["reports"] = []
if "alerts" not in st.session_state:
    st.session_state["alerts"] = {}
if "trees_planted" not in st.session_state:
    st.session_state["trees_planted"] = 0

# --- User Authentication ---
def create_account():
    with st.form("create_account_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        submitted = st.form_submit_button("Create Account")

        if submitted:
            if new_username in st.session_state["user_accounts"]:
                st.error("Username already exists. Please choose a different username.")
            else:
                st.session_state["user_accounts"][new_username] = new_password
                st.success("Account created successfully! Please log in.")

def login():
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in st.session_state["user_accounts"] and st.session_state["user_accounts"][username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid username or password.")

# --- Sidebar ---
st.sidebar.title("Navigation")
menu = ["Login", "Create Account", "AQI Forecast", "Community Forum", "Business Recommendations", "Take Action"]
choice = st.sidebar.selectbox("Menu", menu)

# Login and Account Creation
if choice == "Login":
    if not st.session_state["logged_in"]:
        login()
    else:
        st.write(f"Logged in as {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.success("Logged out successfully.")
            st.experimental_rerun()  # Rerun to update UI
elif choice == "Create Account":
    create_account()

# Mock Green Spaces Data (Replace with a real data source)
green_spaces_data = {
    "Cubbon Park": {"latitude": 12.9768, "longitude": 77.5987, "rating": 4.5},
    "Lal Bagh": {"latitude": 12.9514, "longitude": 77.5908, "rating": 4.2},
    "Sankey Tank": {"latitude": 13.0083, "longitude": 77.5702, "rating": 4.0},
    # Add more green spaces
}

# Function to find nearest green spaces
def find_nearest_green_spaces(user_lat, user_lon, n=3):
    distances = []
    for name, data in green_spaces_data.items():
        green_lat, green_lon = data["latitude"], data["longitude"]
        distance = geopy.distance.geodesic((user_lat, user_lon), (green_lat, green_lon)).km
        distances.append((name, distance, data))

    nearest_spaces = sorted(distances, key=lambda x: x[1])[:n]
    return nearest_spaces

# --- Main UI ---
if choice == "AQI Forecast":
    st.title("🌍 AQI Forecasting for Bangalore Traffic")
    st.write("This app predicts AQI for the next 7 days using an LSTM model.")

    if st.session_state["logged_in"]:
        # --- User Profile (Simple Example) ---
        st.sidebar.header("User Profile")
        age_group = st.sidebar.selectbox("Age Group", ["Adult", "Child", "Elderly"])
        respiratory_condition = st.sidebar.radio("Respiratory Condition?", ["No", "Yes"])
        user_profile = {
            "age_group": age_group.lower(),
            "respiratory_condition": respiratory_condition.lower(),
        }

        # User input for area selection
        area_name = st.text_input("Enter the area name:", "Indiranagar")

        # --- Traffic Volume Simulation (Placeholder) ---
        # In a real application, you'd integrate with a traffic API
        # or use real-time traffic data. For this example, we'll just
        # simulate a relationship.
        # Forecast AQI when button is clicked
        if st.button("Predict AQI"):
            current_values, forecast_df = forecast_aqi(area_name)
            if current_values:
                # Display current AQI values
                st.subheader("📌 Current Environmental Data")
                for key, value in current_values.items():
                    st.write(f"**{key}**: {value}")

                # Display forecast data
                st.subheader("📈 AQI Forecast for Next 7 Days")
                st.dataframe(forecast_df)

                # Generate transportation suggestion based on traffic volume and AQI
                traffic_suggestion = suggest_transportation(
                    current_values["Traffic Volume"], forecast_df["AQI_Level"].iloc[0]
                )  # Use first forecast AQI
                st.write(f"**Traffic Suggestion:** {traffic_suggestion}")

                # --- Personalized Recommendations ---
                st.subheader("💡 Personalized Recommendations")
                # Determine the *worst* AQI level from the forecast
                worst_aqi_level = forecast_df["AQI_Level"].value_counts().index[0]
                recommendations = get_recommendations(worst_aqi_level, user_profile)
                for recommendation in recommendations:
                    st.write(f"- {recommendation}")

                # Plot AQI forecast
                st.subheader("📊 AQI Trend")
                fig, ax = plt.subplots()
                ax.plot(
                    forecast_df["Date"],
                    forecast_df["Predicted_AQI"],
                    marker="o",
                    linestyle="-",
                    color="b",
                    label="Predicted AQI",
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("AQI")
                ax.set_title(f"AQI Forecast for {area_name} (Next 7 Days)")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Provide CSV download link
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=csv,
                    file_name="AQI_Forecast.csv",
                    mime="text/csv",
                )

                # --- Alert System ---
                st.subheader("🔔 Alert System")
                location_key = area_name.lower()
                if location_key not in st.session_state["alerts"]:
                    st.session_state["alerts"][location_key] = False

                alert_enabled = st.checkbox(
                    f"Enable AQI Alert for {area_name}",
                    key=f"alert_{location_key}",
                    value=st.session_state["alerts"][location_key],
                )

                st.session_state["alerts"][location_key] = alert_enabled

                if alert_enabled:
                    # Check if AQI level triggers alert
                    if worst_aqi_level in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
                        st.warning(
                            f"Alert: AQI is forecast to be {worst_aqi_level} in {area_name}! Consider limiting outdoor activities."
                        )
                else:
                    st.info(f"Alerts disabled for {area_name}.")

                # Find the coordinates for area_name
                area_data = df[df["area_name"] == area_name].sort_values(by="date")
                if not area_data.empty and "latitude" in area_data.columns and "longitude" in area_data.columns:  # Check if latitude/longitude exist
                    user_lat = area_data["latitude"].iloc[0]
                    user_lon = area_data["longitude"].iloc[0]

                    # Find nearest green spaces
                    nearest_green_spaces = find_nearest_green_spaces(user_lat, user_lon)

                    st.subheader("🌳 Nearest Green Spaces")
                    for name, distance, data in nearest_green_spaces:
                        st.write(f"- **{name}**: {distance:.2f} km away, Rating: {data['rating']}")

                    # --- Green Spaces Mapping ---
                    
        # --- Report an Air Quality Issue ---
        st.subheader("Report an Air Quality Issue")

        report_type = st.selectbox("Type of Issue", ["Construction Dust", "Illegal Burning", "Industrial Emissions", "Other"])
        report_location = st.text_input("Location of Issue")
        report_description = st.text_area("Description of Issue")

        if st.button("Submit Report"):
            report = {
                "user": st.session_state['username'],
                "type": report_type,
                "location": report_location,
                "description": report_description
            }
            st.session_state["reports"].append(report)
            st.success("Report submitted successfully!")
            st.session_state["report_submitted"] = True
            st.balloons()

# --- Community Forum ---
if choice == "Community Forum":
    st.subheader("Community Forum")
    if st.session_state["logged_in"]:
        st.write("Share your concerns and ideas about air quality!")
        # List of reported issues
        st.subheader("Reported Issues")
        if st.session_state["reports"]:
            for i, report in enumerate(st.session_state["reports"]):
                st.write(f"**Report {i+1}**")
                st.write(f"- **User:** {report['user']}")
                st.write(f"- **Type:** {report['type']}")
                st.write(f"- **Location:** {report['location']}")
                st.write(f"- **Description:** {report['description']}")
                st.write("---")
        else:
            st.write("No issues reported yet.")
        # Add functionality for users to post and view threads here
    else:
        st.write("Please log in to participate in the forum.")

# Business Recommendations Page
if choice == "Business Recommendations":
    st.subheader("Recommendations for Businesses")
    st.write("Here are some recommendations for businesses to improve air quality:")

    st.write("- **Install air purifiers:** Use high-efficiency particulate air (HEPA) filters to remove pollutants from the air.")
    st.write("- **Use green building materials:** Use materials that emit fewer volatile organic compounds (VOCs).")
    st.write("- **Reduce energy consumption:** Optimize energy usage to reduce emissions from power generation.")
    st.write("- **Promote sustainable transportation:** Encourage employees to use public transport, cycle, or walk to work.")
    st.write("- **Implement green landscaping:** Plant trees and vegetation to absorb pollutants and improve air quality.")
    st.write("- **Monitor air quality:** Regularly monitor indoor and outdoor air quality to identify and address potential problems.")
# --- Take Action Page ---
if choice == "Take Action":
    st.subheader("Take Action for Cleaner Air")

    # --- Carbon Footprint Calculator ---
    st.subheader("👣 Carbon Footprint Calculator")

    with st.form("carbon_footprint_form"):
        transportation_type = st.selectbox("Transportation Type", ["Car", "Public Transport", "Motorcycle"])
        transportation_distance = st.number_input("Distance traveled per week (km):", value=0)
        energy_consumption = st.number_input("Monthly electricity bill (INR):", value=0)
        meat_consumption = st.selectbox("Meat Consumption", ["High", "Medium", "Low", "Vegetarian", "Vegan"])
        submitted = st.form_submit_button("Calculate Carbon Footprint")

        if submitted:
            # Carbon footprint calculation factors (simplified)
            transport_factors = {"Car": 0.25, "Public Transport": 0.1, "Motorcycle": 0.15}
            meat_factors = {"High": 0.3, "Medium": 0.2, "Low": 0.1, "Vegetarian": 0.05, "Vegan": 0.01}

            # Calculate carbon footprint
            carbon_footprint = (
                transportation_distance * transport_factors[transportation_type]
                + energy_consumption * 0.05
                + meat_factors[meat_consumption] * 10
            )

            st.write(f"**Estimated weekly carbon footprint:** {carbon_footprint:.2f} kg CO2e")

            st.write("💡 Tips to reduce your footprint:")
            st.write("- Use public transport, cycle, or walk when possible")
            st.write("- Reduce energy consumption by using energy-efficient appliances")
            st.write("- Reduce meat consumption")

    # --- Plant a Tree Campaign ---
    st.subheader("🌱 Plant a Tree to Offset Your Carbon Footprint")
    st.write("Partnering with a local organization to plant trees in Bangalore.")
    st.write("- Donate Now : [https://bjsm.org.in/donations/donate-to-tree-plantation-in-india/]")
    donation_amount = st.number_input("Enter donation amount (INR):", value=0)
    trees_per_donation = 0.1  # Example: 1 tree per INR 10 donation

    if st.button("Donate to Plant Trees"):
        # Replace with actual donation processing logic
        trees_to_plant = int(donation_amount * trees_per_donation)
        st.session_state["trees_planted"] += trees_to_plant
        st.success(f"Thank you for your donation! {trees_to_plant} trees will be planted on your behalf.")

    st.write(f"Total trees planted through this app: {st.session_state['trees_planted']}")

    # --- Promote Eco-Friendly Products ---
    st.subheader("🛒 Eco-Friendly Products")
    st.write("Support sustainable businesses and reduce your environmental impact with these eco-friendly products:")
    st.write("- Eco-friendly products: [https://www.amazon.in/eco-friendly-products/s?k=eco+friendly+products]")
    st.write("- EarthHero [https://earthhero.com/?srsltid=AfmBOoqHYB8YprOjjzd62DW44HiOuGpsDHWb3lVnYG1HIB4B3q4GlQJU]")
    st.write("- BrownLiving: [https://brownliving.in/?srsltid=AfmBOopLQEprFGgoTpzFiS1Un1nKPENlY9JUcWf3SFcBnwD5LW-0gBdk]")
    st.write("- GreenFeels: [https://greenfeels.in/pages/shop-by-brand?srsltid=AfmBOopMVl_JBKuysHctVTfHBgrEbDXd9EVyTGKH_03tZsqu8VgyS3zN]")
else:
    st.write("Please log in to use the AQI Forecasting app.")