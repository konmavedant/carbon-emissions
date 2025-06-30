import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from emissions_calculator import EmissionsCalculator
from ml_models import MLModels
from utils import format_number, create_emissions_chart

# Page configuration
st.set_page_config(
    page_title="Carbon Emission Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_models():
    """Load ML models and emissions calculator"""
    calc = EmissionsCalculator()
    ml_models = MLModels()
    return calc, ml_models

def main():
    st.title("🌱 Carbon Emission Calculator")
    st.markdown("Calculate your carbon footprint using GHG Protocol standards with AI-powered insights")
    
    # Load models
    calc, ml_models = load_models()
    
    # Sidebar inputs
    st.sidebar.header("📊 Input Your Data")
    
    # Scope 1 Emissions
    st.sidebar.subheader("🚗 Scope 1: Direct Emissions")
    
    # Fuel consumption
    fuel_type = st.sidebar.selectbox(
        "Fuel Type",
        ["Gasoline", "Diesel", "Natural Gas", "Propane", "Coal"]
    )
    fuel_quantity = st.sidebar.number_input(
        "Fuel Quantity (liters/cubic meters)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )
    
    # Vehicle data
    vehicle_km = st.sidebar.number_input(
        "Vehicle Distance (km/month)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )
    vehicle_type = st.sidebar.selectbox(
        "Vehicle Type",
        ["Car - Gasoline", "Car - Diesel", "Motorcycle", "Truck", "Van"]
    )
    
    # Scope 2 Emissions
    st.sidebar.subheader("⚡ Scope 2: Indirect Emissions")
    
    electricity_kwh = st.sidebar.number_input(
        "Electricity Consumption (kWh/month)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )
    
    country = st.sidebar.selectbox(
        "Country/Region",
        ["United States", "United Kingdom", "Germany", "France", "Canada", 
         "Australia", "Japan", "China", "India", "Brazil"]
    )
    
    # Business activities
    st.sidebar.subheader("💼 Business Activities")
    
    business_travel_km = st.sidebar.number_input(
        "Business Travel (km/month)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )
    
    waste_kg = st.sidebar.number_input(
        "Waste Generated (kg/month)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )
    
    # Historical data for predictions
    st.sidebar.subheader("📈 Historical Data (Optional)")
    months_history = st.sidebar.slider(
        "Months of historical data",
        min_value=3,
        max_value=12,
        value=6
    )
    
    # Calculate button
    if st.sidebar.button("🧮 Calculate Emissions", type="primary"):
        try:
            # Calculate emissions
            scope1_emissions = calc.calculate_scope1_emissions(
                fuel_type, fuel_quantity, vehicle_type, vehicle_km
            )
            
            scope2_emissions = calc.calculate_scope2_emissions(
                electricity_kwh, country
            )
            
            business_emissions = calc.calculate_business_emissions(
                business_travel_km, waste_kg
            )
            
            total_emissions = scope1_emissions + scope2_emissions + business_emissions
            
            # Store results in session state
            st.session_state.emissions_calculated = True
            st.session_state.scope1 = scope1_emissions
            st.session_state.scope2 = scope2_emissions
            st.session_state.business = business_emissions
            st.session_state.total = total_emissions
            
            # Generate historical data for ML models
            historical_data = generate_historical_data(total_emissions, months_history)
            st.session_state.historical_data = historical_data
            
            # ML predictions and clustering
            user_profile = {
                'electricity_kwh': electricity_kwh,
                'fuel_quantity': fuel_quantity,
                'vehicle_km': vehicle_km,
                'business_travel_km': business_travel_km,
                'waste_kg': waste_kg,
                'total_emissions': total_emissions
            }
            
            # Get predictions
            prediction = ml_models.predict_future_emissions(historical_data)
            cluster = ml_models.get_user_cluster(user_profile)
            recommendations = ml_models.generate_recommendations(user_profile, cluster)
            
            st.session_state.prediction = prediction
            st.session_state.cluster = cluster
            st.session_state.recommendations = recommendations
            
        except Exception as e:
            st.error(f"Error calculating emissions: {str(e)}")
    
    # Display results
    if hasattr(st.session_state, 'emissions_calculated') and st.session_state.emissions_calculated:
        display_results()

def generate_historical_data(current_emissions, months):
    """Generate synthetic historical data for ML models"""
    base_date = datetime.now() - timedelta(days=30 * months)
    dates = [base_date + timedelta(days=30 * i) for i in range(months)]
    
    # Generate variations around current emissions
    np.random.seed(42)  # For reproducible results
    emissions = []
    base_emission = current_emissions * 0.8  # Start lower
    
    for i in range(months):
        # Add trend and noise
        trend = i * 0.02 * base_emission  # Slight upward trend
        noise = np.random.normal(0, 0.1 * base_emission)
        month_emission = max(0, base_emission + trend + noise)
        emissions.append(month_emission)
    
    return pd.DataFrame({
        'date': dates,
        'emissions': emissions
    })

def display_results():
    """Display calculation results and ML insights"""
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔥 Scope 1 Emissions",
            f"{format_number(st.session_state.scope1)} kg CO₂e"
        )
    
    with col2:
        st.metric(
            "⚡ Scope 2 Emissions",
            f"{format_number(st.session_state.scope2)} kg CO₂e"
        )
    
    with col3:
        st.metric(
            "💼 Business Emissions",
            f"{format_number(st.session_state.business)} kg CO₂e"
        )
    
    with col4:
        st.metric(
            "🌍 Total Emissions",
            f"{format_number(st.session_state.total)} kg CO₂e"
        )
    
    # Charts and predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Emissions Breakdown")
        
        # Pie chart
        labels = ['Scope 1', 'Scope 2', 'Business Activities']
        values = [st.session_state.scope1, st.session_state.scope2, st.session_state.business]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Carbon Emissions by Category')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Emissions Forecast")
        
        # Historical vs predicted chart
        fig = create_emissions_chart(st.session_state.historical_data, st.session_state.prediction)
        st.pyplot(fig)
    
    # ML Insights
    st.subheader("🤖 AI Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Your Emission Profile")
        cluster_colors = {
            'Low Emitter': '🟢',
            'Medium Emitter': '🟡',
            'High Emitter': '🔴'
        }
        cluster_emoji = cluster_colors.get(st.session_state.cluster, '⚪')
        st.write(f"**Classification:** {cluster_emoji} {st.session_state.cluster}")
        
        st.write(f"**Next Month Prediction:** {format_number(st.session_state.prediction)} kg CO₂e")
        
        # Comparison with average
        avg_emissions = 500  # Average monthly emissions
        if st.session_state.total > avg_emissions:
            st.write(f"📈 {format_number(st.session_state.total - avg_emissions)} kg CO₂e above average")
        else:
            st.write(f"📉 {format_number(avg_emissions - st.session_state.total)} kg CO₂e below average")
    
    with col2:
        st.subheader("💡 Personalized Recommendations")
        st.write(st.session_state.recommendations)
    
    # Action buttons
    st.subheader("🚀 Take Action")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <a href="https://carbonica-ledger.netlify.app/" target="_blank">
                <button style="
                    background-color: #2E8B57;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    width: 100%;
                ">
                    🌱 Register Project on Carbon Offset Registry
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <a href="https://karbon-compass.netlify.app/" target="_blank">
                <button style="
                    background-color: #4682B4;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    width: 100%;
                ">
                    📊 Add Project to Carbon Dashboard
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
    
    # Environmental impact context
    st.subheader("🌍 Environmental Impact Context")
    
    # Convert to trees equivalent
    trees_equivalent = st.session_state.total / 21.8  # Average tree absorbs 21.8 kg CO2/year
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🌳 Trees Needed (Annual)",
            f"{format_number(trees_equivalent)} trees"
        )
    
    with col2:
        # Car equivalent
        car_km_equivalent = st.session_state.total / 0.12  # Average car emits 0.12 kg CO2/km
        st.metric(
            "🚗 Car Distance Equivalent",
            f"{format_number(car_km_equivalent)} km"
        )
    
    with col3:
        # Home energy equivalent
        home_days_equivalent = st.session_state.total / 16.4  # Average home emits 16.4 kg CO2/day
        st.metric(
            "🏠 Home Energy Equivalent",
            f"{format_number(home_days_equivalent)} days"
        )

if __name__ == "__main__":
    main()
