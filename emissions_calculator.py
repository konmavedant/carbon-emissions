"""
Carbon Emissions Calculator based on GHG Protocol standards.
Implements Scope 1, Scope 2, and basic business emissions calculations.
"""

import pandas as pd
from typing import Dict, Optional

class EmissionsCalculator:
    """
    Calculate carbon emissions following GHG Protocol methodology.
    """
    
    def __init__(self):
        """Initialize calculator with emission factors."""
        self.fuel_emission_factors = self._load_fuel_emission_factors()
        self.electricity_emission_factors = self._load_electricity_emission_factors()
        self.vehicle_emission_factors = self._load_vehicle_emission_factors()
        self.business_emission_factors = self._load_business_emission_factors()
    
    def _load_fuel_emission_factors(self) -> Dict[str, float]:
        """
        Load fuel emission factors (kg CO2e per liter/cubic meter).
        Based on GHG Protocol guidelines and IPCC factors.
        """
        return {
            "Gasoline": 2.31,  # kg CO2e per liter
            "Diesel": 2.68,    # kg CO2e per liter
            "Natural Gas": 1.93,  # kg CO2e per cubic meter
            "Propane": 1.51,   # kg CO2e per liter
            "Coal": 2.42       # kg CO2e per kg
        }
    
    def _load_electricity_emission_factors(self) -> Dict[str, float]:
        """
        Load electricity emission factors (kg CO2e per kWh) by country.
        Based on latest grid emission factors from various national sources.
        """
        return {
            "United States": 0.386,
            "United Kingdom": 0.233,
            "Germany": 0.401,
            "France": 0.057,
            "Canada": 0.150,
            "Australia": 0.634,
            "Japan": 0.462,
            "China": 0.681,
            "India": 0.708,
            "Brazil": 0.074
        }
    
    def _load_vehicle_emission_factors(self) -> Dict[str, float]:
        """
        Load vehicle emission factors (kg CO2e per km).
        Based on average vehicle efficiency and fuel consumption.
        """
        return {
            "Car - Gasoline": 0.12,
            "Car - Diesel": 0.11,
            "Motorcycle": 0.08,
            "Truck": 0.35,
            "Van": 0.18
        }
    
    def _load_business_emission_factors(self) -> Dict[str, float]:
        """
        Load business activity emission factors.
        """
        return {
            "business_travel": 0.15,  # kg CO2e per km (average flight/car)
            "waste": 0.5,  # kg CO2e per kg of waste
            "paper": 1.8,  # kg CO2e per kg of paper
            "office_equipment": 0.3  # kg CO2e per unit
        }
    
    def calculate_scope1_emissions(self, fuel_type: str, fuel_quantity: float, 
                                 vehicle_type: str, vehicle_km: float) -> float:
        """
        Calculate Scope 1 (direct) emissions from fuel consumption and vehicles.
        
        Args:
            fuel_type: Type of fuel used
            fuel_quantity: Quantity of fuel consumed
            vehicle_type: Type of vehicle
            vehicle_km: Distance traveled in kilometers
            
        Returns:
            Total Scope 1 emissions in kg CO2e
        """
        fuel_emissions = 0.0
        vehicle_emissions = 0.0
        
        # Fuel emissions
        if fuel_type in self.fuel_emission_factors and fuel_quantity > 0:
            fuel_emissions = fuel_quantity * self.fuel_emission_factors[fuel_type]
        
        # Vehicle emissions
        if vehicle_type in self.vehicle_emission_factors and vehicle_km > 0:
            vehicle_emissions = vehicle_km * self.vehicle_emission_factors[vehicle_type]
        
        total_scope1 = fuel_emissions + vehicle_emissions
        
        return round(total_scope1, 2)
    
    def calculate_scope2_emissions(self, electricity_kwh: float, country: str) -> float:
        """
        Calculate Scope 2 (indirect) emissions from electricity consumption.
        
        Args:
            electricity_kwh: Electricity consumption in kWh
            country: Country/region for grid emission factor
            
        Returns:
            Total Scope 2 emissions in kg CO2e
        """
        if country not in self.electricity_emission_factors:
            raise ValueError(f"Emission factor not available for {country}")
        
        if electricity_kwh <= 0:
            return 0.0
        
        emission_factor = self.electricity_emission_factors[country]
        scope2_emissions = electricity_kwh * emission_factor
        
        return round(scope2_emissions, 2)
    
    def calculate_business_emissions(self, business_travel_km: float, 
                                   waste_kg: float) -> float:
        """
        Calculate basic business activity emissions.
        
        Args:
            business_travel_km: Business travel distance in km
            waste_kg: Waste generated in kg
            
        Returns:
            Total business emissions in kg CO2e
        """
        travel_emissions = 0.0
        waste_emissions = 0.0
        
        if business_travel_km > 0:
            travel_emissions = business_travel_km * self.business_emission_factors["business_travel"]
        
        if waste_kg > 0:
            waste_emissions = waste_kg * self.business_emission_factors["waste"]
        
        total_business = travel_emissions + waste_emissions
        
        return round(total_business, 2)
    
    def calculate_total_emissions(self, scope1: float, scope2: float, 
                                business: float) -> Dict[str, float]:
        """
        Calculate total emissions and provide breakdown.
        
        Args:
            scope1: Scope 1 emissions
            scope2: Scope 2 emissions
            business: Business emissions
            
        Returns:
            Dictionary with emission breakdown
        """
        total = scope1 + scope2 + business
        
        return {
            "scope1": scope1,
            "scope2": scope2,
            "business": business,
            "total": round(total, 2),
            "scope1_percentage": round((scope1 / total * 100) if total > 0 else 0, 1),
            "scope2_percentage": round((scope2 / total * 100) if total > 0 else 0, 1),
            "business_percentage": round((business / total * 100) if total > 0 else 0, 1)
        }
    
    def get_emission_factor(self, category: str, item: str) -> Optional[float]:
        """
        Get specific emission factor for a category and item.
        
        Args:
            category: Category of emission factor
            item: Specific item within category
            
        Returns:
            Emission factor or None if not found
        """
        factor_maps = {
            "fuel": self.fuel_emission_factors,
            "electricity": self.electricity_emission_factors,
            "vehicle": self.vehicle_emission_factors,
            "business": self.business_emission_factors
        }
        
        if category in factor_maps:
            return factor_maps[category].get(item)
        
        return None
    
    def get_available_options(self, category: str) -> list:
        """
        Get available options for a specific category.
        
        Args:
            category: Category to get options for
            
        Returns:
            List of available options
        """
        options_map = {
            "fuel": list(self.fuel_emission_factors.keys()),
            "electricity": list(self.electricity_emission_factors.keys()),
            "vehicle": list(self.vehicle_emission_factors.keys()),
            "business": list(self.business_emission_factors.keys())
        }
        
        return options_map.get(category, [])
    
    def validate_inputs(self, **kwargs) -> Dict[str, str]:
        """
        Validate calculation inputs.
        
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
        # Check for negative values
        for key, value in kwargs.items():
            if isinstance(value, (int, float)) and value < 0:
                errors[key] = f"{key} cannot be negative"
        
        # Check for valid fuel types
        if "fuel_type" in kwargs:
            fuel_type = kwargs["fuel_type"]
            if fuel_type and fuel_type not in self.fuel_emission_factors:
                errors["fuel_type"] = f"Invalid fuel type: {fuel_type}"
        
        # Check for valid countries
        if "country" in kwargs:
            country = kwargs["country"]
            if country and country not in self.electricity_emission_factors:
                errors["country"] = f"Emission factor not available for: {country}"
        
        # Check for valid vehicle types
        if "vehicle_type" in kwargs:
            vehicle_type = kwargs["vehicle_type"]
            if vehicle_type and vehicle_type not in self.vehicle_emission_factors:
                errors["vehicle_type"] = f"Invalid vehicle type: {vehicle_type}"
        
        return errors
