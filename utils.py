"""
Utility functions for the Carbon Emissions Calculator.
Shared helper functions for formatting, validation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import re

def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format numbers for display with appropriate decimal places and thousands separators.
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted number string
    """
    if value == 0:
        return "0"
    
    if abs(value) < 1:
        return f"{value:.{decimal_places}f}"
    
    if abs(value) < 1000:
        return f"{value:,.{decimal_places}f}"
    
    # For larger numbers, use abbreviated notation
    magnitude = 0
    while abs(value) >= 1000:
        magnitude += 1
        value /= 1000.0
    
    suffixes = ['', 'K', 'M', 'B', 'T']
    suffix = suffixes[min(magnitude, len(suffixes) - 1)]
    
    return f"{value:,.{decimal_places}f}{suffix}"

def validate_positive_number(value: Union[int, float], field_name: str) -> Optional[str]:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Error message if invalid, None if valid
    """
    if not isinstance(value, (int, float)):
        return f"{field_name} must be a number"
    
    if value < 0:
        return f"{field_name} cannot be negative"
    
    if np.isnan(value) or np.isinf(value):
        return f"{field_name} must be a valid number"
    
    return None

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 100.0 if new_value > 0 else 0.0
    
    return ((new_value - old_value) / old_value) * 100

def create_emissions_chart(historical_data: pd.DataFrame, prediction: float) -> plt.Figure:
    """
    Create a chart showing historical emissions and future prediction.
    
    Args:
        historical_data: DataFrame with historical emissions data
        prediction: Predicted future emissions
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_data['date'], historical_data['emissions'], 
            'o-', color='#2E8B57', linewidth=2, markersize=6, label='Historical Emissions')
    
    # Add prediction point
    future_date = historical_data['date'].iloc[-1] + timedelta(days=30)
    ax.plot(future_date, prediction, 'o', color='#FF6B6B', 
            markersize=10, label=f'Predicted: {format_number(prediction)} kg CO₂e')
    
    # Add trend line
    if len(historical_data) > 1:
        # Calculate trend
        dates_numeric = mdates.date2num(historical_data['date'])
        coeffs = np.polyfit(dates_numeric, historical_data['emissions'], 1)
        trend_line = np.poly1d(coeffs)
        
        # Extend trend line to prediction
        extended_dates = np.linspace(dates_numeric[0], mdates.date2num(future_date), 100)
        ax.plot(mdates.num2date(extended_dates), trend_line(extended_dates), 
                '--', color='#FFB347', alpha=0.7, label='Trend')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Emissions (kg CO₂e)', fontsize=12)
    ax.set_title('Carbon Emissions Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def create_emissions_breakdown_chart(emissions_data: Dict[str, float]) -> plt.Figure:
    """
    Create a pie chart showing emissions breakdown by category.
    
    Args:
        emissions_data: Dictionary with emission categories and values
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filter out zero values
    filtered_data = {k: v for k, v in emissions_data.items() if v > 0}
    
    if not filtered_data:
        ax.text(0.5, 0.5, 'No emissions data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=16)
        return fig
    
    labels = list(filtered_data.keys())
    values = list(filtered_data.values())
    
    # Color scheme
    colors = ['#2E8B57', '#4682B4', '#FFB347', '#FF6B6B', '#9370DB']
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors[:len(labels)])
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Carbon Emissions Breakdown', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def generate_emission_report(user_profile: Dict, calculations: Dict, 
                           recommendations: str) -> str:
    """
    Generate a comprehensive emission report.
    
    Args:
        user_profile: User's emission profile
        calculations: Calculated emissions data
        recommendations: Personalized recommendations
        
    Returns:
        Formatted report string
    """
    report = f"""
# Carbon Emissions Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Emission Summary
- **Total Emissions**: {format_number(calculations['total'])} kg CO₂e per month
- **Scope 1 (Direct)**: {format_number(calculations['scope1'])} kg CO₂e ({calculations['scope1_percentage']}%)
- **Scope 2 (Indirect)**: {format_number(calculations['scope2'])} kg CO₂e ({calculations['scope2_percentage']}%)
- **Business Activities**: {format_number(calculations['business'])} kg CO₂e ({calculations['business_percentage']}%)

## Input Data
- **Electricity Consumption**: {format_number(user_profile.get('electricity_kwh', 0))} kWh/month
- **Fuel Consumption**: {format_number(user_profile.get('fuel_quantity', 0))} liters/month
- **Vehicle Travel**: {format_number(user_profile.get('vehicle_km', 0))} km/month
- **Business Travel**: {format_number(user_profile.get('business_travel_km', 0))} km/month
- **Waste Generated**: {format_number(user_profile.get('waste_kg', 0))} kg/month

## Environmental Impact
- **Trees Needed (Annual)**: {format_number(calculations['total'] * 12 / 21.8)} trees
- **Car Distance Equivalent**: {format_number(calculations['total'] / 0.12)} km
- **Home Energy Equivalent**: {format_number(calculations['total'] / 16.4)} days

## Recommendations
{recommendations}

## Next Steps
1. Implement the recommended actions
2. Track your progress monthly
3. Consider carbon offset projects for remaining emissions
4. Share your experience with others

---
*This report is based on GHG Protocol standards and current emission factors.*
"""
    
    return report

def save_data_to_csv(data: pd.DataFrame, filename: str) -> bool:
    """
    Save data to CSV file.
    
    Args:
        data: DataFrame to save
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data.to_csv(filename, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return False

def load_data_from_csv(filename: str) -> Optional[pd.DataFrame]:
    """
    Load data from CSV file.
    
    Args:
        filename: Input filename
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None

def calculate_emission_trends(historical_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate emission trends and statistics.
    
    Args:
        historical_data: DataFrame with historical emissions
        
    Returns:
        Dictionary with trend statistics
    """
    if len(historical_data) < 2:
        return {"trend": 0, "average": 0, "std_dev": 0}
    
    emissions = historical_data['emissions'].values
    
    # Calculate linear trend
    x = np.arange(len(emissions))
    coeffs = np.polyfit(x, emissions, 1)
    trend = coeffs[0]  # Slope
    
    # Calculate statistics
    average = np.mean(emissions)
    std_dev = np.std(emissions)
    
    return {
        "trend": round(trend, 2),
        "average": round(average, 2),
        "std_dev": round(std_dev, 2),
        "min": round(np.min(emissions), 2),
        "max": round(np.max(emissions), 2),
        "latest": round(emissions[-1], 2)
    }

def get_emission_benchmark(total_emissions: float) -> Dict[str, str]:
    """
    Get emission benchmarks and comparisons.
    
    Args:
        total_emissions: Total monthly emissions in kg CO₂e
        
    Returns:
        Dictionary with benchmark information
    """
    # Annual emissions
    annual_emissions = total_emissions * 12
    
    # Global average per capita: ~4,800 kg CO₂e/year
    global_average = 4800
    
    # Calculate comparisons
    vs_global = ((annual_emissions - global_average) / global_average) * 100
    
    # Classification
    if annual_emissions < 2000:
        classification = "Very Low"
        icon = "🟢"
    elif annual_emissions < 4000:
        classification = "Low"
        icon = "🟡"
    elif annual_emissions < 8000:
        classification = "Average"
        icon = "🟠"
    elif annual_emissions < 15000:
        classification = "High"
        icon = "🔴"
    else:
        classification = "Very High"
        icon = "🔴"
    
    return {
        "classification": classification,
        "icon": icon,
        "annual_emissions": round(annual_emissions, 2),
        "vs_global_average": round(vs_global, 1),
        "global_average": global_average
    }

def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def validate_date_range(start_date: datetime, end_date: datetime) -> Optional[str]:
    """
    Validate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Error message if invalid, None if valid
    """
    if start_date >= end_date:
        return "Start date must be before end date"
    
    if end_date > datetime.now():
        return "End date cannot be in the future"
    
    if (end_date - start_date).days > 365 * 5:
        return "Date range cannot exceed 5 years"
    
    return None
