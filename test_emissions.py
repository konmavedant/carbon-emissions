"""
Unit tests for the Carbon Emissions Calculator.
Tests emission calculations, validation, and model functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from emissions_calculator import EmissionsCalculator
from ml_models import MLModels
from utils import (
    format_number, validate_positive_number, calculate_percentage_change,
    validate_email, get_emission_benchmark
)

class TestEmissionsCalculator(unittest.TestCase):
    """Test cases for EmissionsCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = EmissionsCalculator()
    
    def test_scope1_emissions_fuel_only(self):
        """Test Scope 1 emissions calculation with fuel only."""
        emissions = self.calc.calculate_scope1_emissions(
            fuel_type="Gasoline",
            fuel_quantity=100.0,
            vehicle_type="Car - Gasoline",
            vehicle_km=0.0
        )
        
        expected = 100.0 * 2.31  # 100 liters * 2.31 kg CO2e/liter
        self.assertEqual(emissions, expected)
    
    def test_scope1_emissions_vehicle_only(self):
        """Test Scope 1 emissions calculation with vehicle only."""
        emissions = self.calc.calculate_scope1_emissions(
            fuel_type="Gasoline",
            fuel_quantity=0.0,
            vehicle_type="Car - Gasoline",
            vehicle_km=1000.0
        )
        
        expected = 1000.0 * 0.12  # 1000 km * 0.12 kg CO2e/km
        self.assertEqual(emissions, expected)
    
    def test_scope1_emissions_combined(self):
        """Test Scope 1 emissions calculation with both fuel and vehicle."""
        emissions = self.calc.calculate_scope1_emissions(
            fuel_type="Diesel",
            fuel_quantity=50.0,
            vehicle_type="Car - Diesel",
            vehicle_km=500.0
        )
        
        fuel_emissions = 50.0 * 2.68  # Diesel factor
        vehicle_emissions = 500.0 * 0.11  # Diesel car factor
        expected = fuel_emissions + vehicle_emissions
        
        self.assertEqual(emissions, expected)
    
    def test_scope2_emissions_valid_country(self):
        """Test Scope 2 emissions calculation with valid country."""
        emissions = self.calc.calculate_scope2_emissions(
            electricity_kwh=300.0,
            country="United States"
        )
        
        expected = 300.0 * 0.386  # US grid factor
        self.assertEqual(emissions, round(expected, 2))
    
    def test_scope2_emissions_invalid_country(self):
        """Test Scope 2 emissions calculation with invalid country."""
        with self.assertRaises(ValueError):
            self.calc.calculate_scope2_emissions(
                electricity_kwh=300.0,
                country="Invalid Country"
            )
    
    def test_scope2_emissions_zero_consumption(self):
        """Test Scope 2 emissions calculation with zero consumption."""
        emissions = self.calc.calculate_scope2_emissions(
            electricity_kwh=0.0,
            country="United States"
        )
        
        self.assertEqual(emissions, 0.0)
    
    def test_business_emissions_calculation(self):
        """Test business emissions calculation."""
        emissions = self.calc.calculate_business_emissions(
            business_travel_km=1000.0,
            waste_kg=50.0
        )
        
        travel_emissions = 1000.0 * 0.15
        waste_emissions = 50.0 * 0.5
        expected = travel_emissions + waste_emissions
        
        self.assertEqual(emissions, expected)
    
    def test_total_emissions_calculation(self):
        """Test total emissions calculation and breakdown."""
        scope1 = 100.0
        scope2 = 200.0
        business = 50.0
        
        result = self.calc.calculate_total_emissions(scope1, scope2, business)
        
        self.assertEqual(result['scope1'], scope1)
        self.assertEqual(result['scope2'], scope2)
        self.assertEqual(result['business'], business)
        self.assertEqual(result['total'], 350.0)
        self.assertEqual(result['scope1_percentage'], 28.6)
        self.assertEqual(result['scope2_percentage'], 57.1)
        self.assertEqual(result['business_percentage'], 14.3)
    
    def test_emission_factor_retrieval(self):
        """Test emission factor retrieval."""
        factor = self.calc.get_emission_factor("fuel", "Gasoline")
        self.assertEqual(factor, 2.31)
        
        factor = self.calc.get_emission_factor("invalid", "item")
        self.assertIsNone(factor)
    
    def test_available_options(self):
        """Test available options retrieval."""
        fuel_options = self.calc.get_available_options("fuel")
        self.assertIn("Gasoline", fuel_options)
        self.assertIn("Diesel", fuel_options)
        
        invalid_options = self.calc.get_available_options("invalid")
        self.assertEqual(invalid_options, [])
    
    def test_input_validation(self):
        """Test input validation."""
        errors = self.calc.validate_inputs(
            fuel_quantity=-10.0,
            electricity_kwh=300.0,
            fuel_type="Invalid Fuel",
            country="Invalid Country"
        )
        
        self.assertIn("fuel_quantity", errors)
        self.assertIn("fuel_type", errors)
        self.assertIn("country", errors)
        self.assertNotIn("electricity_kwh", errors)

class TestMLModels(unittest.TestCase):
    """Test cases for MLModels class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.ml = MLModels()
        self.ml.model_dir = self.temp_dir
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_user_clustering(self):
        """Test user clustering functionality."""
        user_profile = {
            'electricity_kwh': 300.0,
            'fuel_quantity': 50.0,
            'vehicle_km': 800.0,
            'business_travel_km': 200.0,
            'waste_kg': 20.0
        }
        
        cluster = self.ml.get_user_cluster(user_profile)
        self.assertIn(cluster, ["Low Emitter", "Medium Emitter", "High Emitter"])
    
    def test_emission_prediction(self):
        """Test emission prediction functionality."""
        # Create sample historical data
        dates = [datetime.now() - timedelta(days=30*i) for i in range(6, 0, -1)]
        emissions = [400, 420, 380, 450, 410, 430]
        
        historical_data = pd.DataFrame({
            'date': dates,
            'emissions': emissions
        })
        
        prediction = self.ml.predict_future_emissions(historical_data)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        user_profile = {
            'electricity_kwh': 400.0,
            'fuel_quantity': 80.0,
            'vehicle_km': 1200.0,
            'business_travel_km': 300.0,
            'waste_kg': 30.0,
            'total_emissions': 600.0
        }
        
        cluster = "High Emitter"
        recommendations = self.ml.generate_recommendations(user_profile, cluster)
        
        self.assertIsInstance(recommendations, str)
        self.assertGreater(len(recommendations), 20)
    
    def test_model_performance(self):
        """Test model performance metrics."""
        performance = self.ml.get_model_performance()
        
        self.assertIn("prediction_mse", performance)
        self.assertIn("prediction_r2", performance)
        self.assertIn("n_clusters", performance)
        self.assertIn("cluster_distribution", performance)
        
        self.assertGreaterEqual(performance["prediction_r2"], 0)
        self.assertEqual(performance["n_clusters"], 3)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_format_number(self):
        """Test number formatting."""
        self.assertEqual(format_number(0), "0")
        self.assertEqual(format_number(1.234), "1.23")
        self.assertEqual(format_number(1234.56), "1,234.56")
        self.assertEqual(format_number(1234567.89), "1.23M")
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        self.assertIsNone(validate_positive_number(10.5, "test"))
        self.assertIsNone(validate_positive_number(0, "test"))
        self.assertIsNotNone(validate_positive_number(-5, "test"))
        self.assertIsNotNone(validate_positive_number("not_a_number", "test"))
        self.assertIsNotNone(validate_positive_number(float('nan'), "test"))
    
    def test_validate_email(self):
        """Test email validation."""
        self.assertTrue(validate_email("test@example.com"))
        self.assertTrue(validate_email("user.name+tag@domain.co.uk"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertFalse(validate_email("@example.com"))
        self.assertFalse(validate_email("test@"))
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        self.assertEqual(calculate_percentage_change(100, 110), 10.0)
        self.assertEqual(calculate_percentage_change(100, 90), -10.0)
        self.assertEqual(calculate_percentage_change(0, 50), 100.0)
        self.assertEqual(calculate_percentage_change(0, 0), 0.0)
    
    def test_get_emission_benchmark(self):
        """Test emission benchmarking."""
        # Test low emissions
        result = get_emission_benchmark(100)  # 1,200 kg/year
        self.assertEqual(result["classification"], "Very Low")
        self.assertEqual(result["icon"], "🟢")
        
        # Test high emissions
        result = get_emission_benchmark(1000)  # 12,000 kg/year
        self.assertEqual(result["classification"], "High")
        self.assertEqual(result["icon"], "🔴")
        
        # Check calculation
        self.assertEqual(result["annual_emissions"], 12000.0)
        self.assertIn("vs_global_average", result)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.calc = EmissionsCalculator()
        self.ml = MLModels()
    
    def test_full_calculation_workflow(self):
        """Test complete calculation workflow."""
        # Calculate emissions
        scope1 = self.calc.calculate_scope1_emissions(
            fuel_type="Gasoline",
            fuel_quantity=80.0,
            vehicle_type="Car - Gasoline",
            vehicle_km=1000.0
        )
        
        scope2 = self.calc.calculate_scope2_emissions(
            electricity_kwh=350.0,
            country="United States"
        )
        
        business = self.calc.calculate_business_emissions(
            business_travel_km=500.0,
            waste_kg=25.0
        )
        
        total = scope1 + scope2 + business
        
        # Test ML workflow
        user_profile = {
            'electricity_kwh': 350.0,
            'fuel_quantity': 80.0,
            'vehicle_km': 1000.0,
            'business_travel_km': 500.0,
            'waste_kg': 25.0,
            'total_emissions': total
        }
        
        cluster = self.ml.get_user_cluster(user_profile)
        recommendations = self.ml.generate_recommendations(user_profile, cluster)
        
        # Verify results
        self.assertGreater(scope1, 0)
        self.assertGreater(scope2, 0)
        self.assertGreater(business, 0)
        self.assertGreater(total, 0)
        self.assertIn(cluster, ["Low Emitter", "Medium Emitter", "High Emitter"])
        self.assertIsInstance(recommendations, str)
        self.assertGreater(len(recommendations), 10)
    
    def test_error_handling(self):
        """Test error handling in the system."""
        # Test invalid inputs
        with self.assertRaises(ValueError):
            self.calc.calculate_scope2_emissions(300.0, "Invalid Country")
        
        # Test validation
        errors = self.calc.validate_inputs(
            fuel_quantity=-10.0,
            electricity_kwh="invalid"
        )
        self.assertGreater(len(errors), 0)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
