"""
Machine Learning models for carbon emissions prediction and analysis.
Includes regression models, clustering, and LLM-powered recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """
    Machine Learning models for emissions prediction, clustering, and recommendations.
    """
    
    def __init__(self):
        """Initialize ML models and components."""
        self.prediction_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.model_dir = "models"
        self.llm_pipeline = None
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize and load models
        self._initialize_models()
        self._load_or_train_models()
    
    def _initialize_models(self):
        """Initialize ML models with default parameters."""
        self.prediction_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.clustering_model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )
        
        # Initialize LLM pipeline for recommendations
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM pipeline for generating recommendations."""
        # For now, using rule-based recommendations instead of LLM
        # to avoid dependency issues with transformers
        self.llm_pipeline = None
    
    def _generate_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic training data for the models.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features DataFrame, targets array)
        """
        np.random.seed(42)
        
        # Generate realistic feature distributions
        electricity_kwh = np.random.exponential(300, n_samples)  # Average 300 kWh
        fuel_quantity = np.random.exponential(50, n_samples)     # Average 50 liters
        vehicle_km = np.random.exponential(800, n_samples)       # Average 800 km
        business_travel_km = np.random.exponential(200, n_samples)  # Average 200 km
        waste_kg = np.random.exponential(20, n_samples)          # Average 20 kg
        
        # Create features DataFrame
        features = pd.DataFrame({
            'electricity_kwh': electricity_kwh,
            'fuel_quantity': fuel_quantity,
            'vehicle_km': vehicle_km,
            'business_travel_km': business_travel_km,
            'waste_kg': waste_kg
        })
        
        # Calculate emissions based on realistic factors
        emissions = (
            electricity_kwh * 0.4 +      # Electricity emissions
            fuel_quantity * 2.3 +        # Fuel emissions
            vehicle_km * 0.12 +          # Vehicle emissions
            business_travel_km * 0.15 +  # Travel emissions
            waste_kg * 0.5 +             # Waste emissions
            np.random.normal(0, 20, n_samples)  # Add noise
        )
        
        # Ensure non-negative emissions
        emissions = np.maximum(emissions, 0)
        
        return features, emissions
    
    def _load_or_train_models(self):
        """Load existing models or train new ones if they don't exist."""
        prediction_model_path = os.path.join(self.model_dir, "prediction_model.joblib")
        clustering_model_path = os.path.join(self.model_dir, "clustering_model.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        if (os.path.exists(prediction_model_path) and 
            os.path.exists(clustering_model_path) and
            os.path.exists(scaler_path)):
            # Load existing models
            self.prediction_model = joblib.load(prediction_model_path)
            self.clustering_model = joblib.load(clustering_model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # Train new models
            self._train_models()
    
    def _train_models(self):
        """Train the ML models with generated data."""
        # Generate training data
        features, emissions = self._generate_training_data()
        
        # Prepare features for clustering (normalize)
        features_scaled = self.scaler.fit_transform(features)
        
        # Train clustering model
        self.clustering_model.fit(features_scaled)
        
        # Train prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            features, emissions, test_size=0.2, random_state=42
        )
        
        self.prediction_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.prediction_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        # Save models
        self._save_models()
    
    def _save_models(self):
        """Save trained models to disk."""
        joblib.dump(self.prediction_model, os.path.join(self.model_dir, "prediction_model.joblib"))
        joblib.dump(self.clustering_model, os.path.join(self.model_dir, "clustering_model.joblib"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
    
    def predict_future_emissions(self, historical_data: pd.DataFrame) -> float:
        """
        Predict future emissions based on historical data.
        
        Args:
            historical_data: DataFrame with historical emissions data
            
        Returns:
            Predicted emissions for next period
        """
        if len(historical_data) < 2:
            return historical_data['emissions'].iloc[-1] if len(historical_data) > 0 else 0
        
        # Use the last emission value as base
        recent_emission = historical_data['emissions'].iloc[-1]
        
        # Calculate trend
        trend = (historical_data['emissions'].iloc[-1] - historical_data['emissions'].iloc[0]) / len(historical_data)
        
        # Predict next month (add trend with some randomness)
        prediction = recent_emission + trend + np.random.normal(0, recent_emission * 0.1)
        
        return max(0, round(prediction, 2))
    
    def get_user_cluster(self, user_profile: Dict) -> str:
        """
        Classify user into emission cluster.
        
        Args:
            user_profile: Dictionary with user's emission profile
            
        Returns:
            Cluster label (Low/Medium/High Emitter)
        """
        # Prepare features
        features = pd.DataFrame([{
            'electricity_kwh': user_profile.get('electricity_kwh', 0),
            'fuel_quantity': user_profile.get('fuel_quantity', 0),
            'vehicle_km': user_profile.get('vehicle_km', 0),
            'business_travel_km': user_profile.get('business_travel_km', 0),
            'waste_kg': user_profile.get('waste_kg', 0)
        }])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        cluster_id = self.clustering_model.predict(features_scaled)[0]
        
        # Map cluster ID to meaningful labels
        cluster_labels = {
            0: "Low Emitter",
            1: "Medium Emitter",
            2: "High Emitter"
        }
        
        return cluster_labels.get(cluster_id, "Unknown")
    
    def generate_recommendations(self, user_profile: Dict, cluster: str) -> str:
        """
        Generate personalized recommendations using LLM or rule-based system.
        
        Args:
            user_profile: User's emission profile
            cluster: User's emission cluster
            
        Returns:
            Personalized recommendation text
        """
        total_emissions = user_profile.get('total_emissions', 0)
        
        # Use rule-based system if LLM is not available
        if not self.llm_pipeline:
            return self._generate_rule_based_recommendations(user_profile, cluster)
        
        # Generate prompt for LLM
        prompt = self._create_recommendation_prompt(user_profile, cluster)
        
        try:
            # Generate recommendations using LLM
            response = self.llm_pipeline(prompt, max_length=100, num_return_sequences=1)
            recommendation = response[0]['generated_text'].replace(prompt, '').strip()
            
            if len(recommendation) < 20:  # Fallback if response is too short
                return self._generate_rule_based_recommendations(user_profile, cluster)
            
            return recommendation
        
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_rule_based_recommendations(user_profile, cluster)
    
    def _create_recommendation_prompt(self, user_profile: Dict, cluster: str) -> str:
        """Create a prompt for LLM-based recommendations."""
        total_emissions = user_profile.get('total_emissions', 0)
        electricity = user_profile.get('electricity_kwh', 0)
        vehicle_km = user_profile.get('vehicle_km', 0)
        
        prompt = f"""You are an environmental consultant. A {cluster.lower()} with {total_emissions:.0f} kg CO2e monthly emissions needs advice. 
        They use {electricity:.0f} kWh electricity and drive {vehicle_km:.0f} km monthly. 
        Provide 2-3 specific actionable recommendations to reduce emissions:"""
        
        return prompt
    
    def _generate_rule_based_recommendations(self, user_profile: Dict, cluster: str) -> str:
        """Generate recommendations using rule-based system."""
        recommendations = []
        
        electricity = user_profile.get('electricity_kwh', 0)
        vehicle_km = user_profile.get('vehicle_km', 0)
        fuel_quantity = user_profile.get('fuel_quantity', 0)
        business_travel = user_profile.get('business_travel_km', 0)
        total_emissions = user_profile.get('total_emissions', 0)
        
        # Electricity recommendations
        if electricity > 300:
            recommendations.append("💡 Switch to LED bulbs and energy-efficient appliances to reduce electricity consumption by up to 30%")
        elif electricity > 150:
            recommendations.append("🔌 Unplug devices when not in use and consider smart power strips to eliminate phantom loads")
        
        # Transportation recommendations
        if vehicle_km > 800:
            recommendations.append("🚗 Consider carpooling, public transport, or electric vehicles to reduce transportation emissions by 40-60%")
        elif vehicle_km > 400:
            recommendations.append("🚴 Combine trips and consider cycling or walking for short distances")
        
        # Fuel recommendations
        if fuel_quantity > 50:
            recommendations.append("⛽ Improve fuel efficiency by maintaining proper tire pressure and regular vehicle maintenance")
        
        # Business travel recommendations
        if business_travel > 200:
            recommendations.append("💼 Use video conferencing to reduce business travel emissions by up to 80%")
        
        # General recommendations based on cluster
        if cluster == "High Emitter":
            recommendations.append("🌱 Focus on the biggest impact areas: transportation and energy use typically offer the greatest reduction potential")
        elif cluster == "Medium Emitter":
            recommendations.append("📊 Track your progress monthly to identify trends and maintain motivation for emission reductions")
        else:  # Low Emitter
            recommendations.append("🎯 Great job! Consider renewable energy options and helping others reduce their carbon footprint")
        
        # Ensure we have recommendations
        if not recommendations:
            recommendations.append("🌍 Start with small changes: reducing energy consumption and choosing sustainable transportation options")
        
        return "\n\n".join(recommendations[:3])  # Return top 3 recommendations
    
    def get_model_performance(self) -> Dict:
        """
        Get performance metrics for the trained models.
        
        Returns:
            Dictionary with model performance metrics
        """
        # Generate test data
        features, emissions = self._generate_training_data(200)
        
        # Test prediction model
        predictions = self.prediction_model.predict(features)
        prediction_mse = mean_squared_error(emissions, predictions)
        prediction_r2 = r2_score(emissions, predictions)
        
        # Test clustering model
        features_scaled = self.scaler.transform(features)
        cluster_labels = self.clustering_model.predict(features_scaled)
        
        return {
            "prediction_mse": round(prediction_mse, 2),
            "prediction_r2": round(prediction_r2, 3),
            "n_clusters": len(np.unique(cluster_labels)),
            "cluster_distribution": {
                f"Cluster {i}": int(np.sum(cluster_labels == i)) 
                for i in range(self.clustering_model.n_clusters)
            }
        }
    
    def update_models(self, new_data: pd.DataFrame):
        """
        Update models with new data (incremental learning).
        
        Args:
            new_data: New data to update models with
        """
        if len(new_data) < 10:  # Need minimum data for updates
            return
        
        # Retrain models with new data
        features = new_data.drop('emissions', axis=1)
        emissions = new_data['emissions']
        
        # Update prediction model
        self.prediction_model.fit(features, emissions)
        
        # Update clustering model
        features_scaled = self.scaler.fit_transform(features)
        self.clustering_model.fit(features_scaled)
        
        # Save updated models
        self._save_models()
