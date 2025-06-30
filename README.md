# Carbon Emission Calculator

A comprehensive web application for calculating carbon emissions using GHG Protocol standards, with AI-powered predictions and personalized recommendations.

## Features

### 🌍 GHG Protocol Calculations
- **Scope 1 Emissions**: Direct emissions from fuel consumption and vehicles
- **Scope 2 Emissions**: Indirect emissions from electricity consumption
- **Business Activities**: Emissions from business travel and waste generation
- Country-specific emission factors for accurate calculations

### 🤖 AI/ML Features
- **Emission Prediction**: Forecast future emissions using machine learning
- **User Clustering**: Classify users into Low/Medium/High emitter categories
- **Personalized Recommendations**: LLM-powered suggestions for emission reduction

### 📊 Interactive Dashboard
- Real-time emission calculations
- Historical vs. predicted emissions visualization
- Comprehensive emission breakdown charts
- Environmental impact context (trees, car equivalents)

### 🔗 External Integrations
- Carbon Offset Registry integration
- Carbon Dashboard project tracking
- Export capabilities for detailed reporting

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Development

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib transformers matplotlib torch
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

### Deployment

The application is configured for easy deployment on platforms like Replit, Heroku, or Streamlit Cloud.

**Streamlit Cloud Deployment:**
1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. The app will automatically deploy with the included configuration

## Usage Guide

### Step 1: Input Your Data
Use the sidebar to enter your monthly consumption data:
- **Fuel consumption** (liters/cubic meters)
- **Electricity usage** (kWh)
- **Vehicle travel** (kilometers)
- **Business activities** (travel and waste)

### Step 2: Calculate Emissions
Click the "Calculate Emissions" button to:
- Generate GHG Protocol-compliant calculations
- Analyze your emission profile
- Generate AI-powered predictions
- Receive personalized recommendations

### Step 3: View Results
The results dashboard displays:
- **Emission breakdown** by scope and category
- **Predictions** for future emissions
- **Classification** into emission categories
- **Recommendations** for reduction strategies
- **Environmental impact** context

### Step 4: Take Action
Use the provided links to:
- Register offset projects
- Track progress on carbon dashboard
- Implement recommended actions

## Technical Architecture

### Core Components

1. **emissions_calculator.py**: GHG Protocol calculations
2. **ml_models.py**: AI/ML models for prediction and clustering
3. **utils.py**: Utility functions for formatting and visualization
4. **app.py**: Streamlit web interface

### Data Sources

- **Emission Factors**: Based on latest GHG Protocol guidelines
- **Grid Factors**: Country-specific electricity emission factors
- **Vehicle Factors**: Average emission factors by vehicle type

### ML Models

- **Prediction Model**: Random Forest Regressor for emission forecasting
- **Clustering Model**: K-Means clustering for user classification
- **Recommendation Engine**: LLM-powered personalized suggestions

## API Reference

### EmissionsCalculator Class

```python
from emissions_calculator import EmissionsCalculator

calc = EmissionsCalculator()

# Calculate Scope 1 emissions
scope1 = calc.calculate_scope1_emissions(
    fuel_type="Gasoline",
    fuel_quantity=100.0,
    vehicle_type="Car - Gasoline",
    vehicle_km=800.0
)

# Calculate Scope 2 emissions
scope2 = calc.calculate_scope2_emissions(
    electricity_kwh=300.0,
    country="United States"
)
