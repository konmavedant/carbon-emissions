# Carbon Emission Calculator - Replit Project Guide

## Overview

This is a comprehensive web application for calculating carbon emissions using GHG (Greenhouse Gas) Protocol standards. The application combines traditional emission calculations with AI/ML features including prediction, user clustering, and personalized recommendations. Built with Python and Streamlit, it provides an interactive dashboard for users to calculate their carbon footprint and receive actionable insights.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Layout**: Wide layout with expandable sidebar for inputs
- **Components**: Interactive forms, real-time calculations, data visualizations
- **Styling**: Custom CSS with environmental theme (green color scheme)

### Backend Architecture
- **Core Logic**: Modular Python architecture with separation of concerns
- **Calculation Engine**: GHG Protocol-compliant emission calculations
- **ML Pipeline**: Scikit-learn models for prediction and clustering
- **AI Integration**: Hugging Face Transformers for LLM-powered recommendations

### Data Processing
- **Input Validation**: Comprehensive validation for user inputs
- **Data Transformation**: Standardization and preprocessing for ML models
- **Emission Factors**: Hard-coded emission factors based on GHG Protocol standards

## Key Components

### 1. Emissions Calculator (`emissions_calculator.py`)
**Purpose**: Core calculation engine implementing GHG Protocol standards
**Key Features**:
- Scope 1 emissions (direct fuel consumption, vehicle emissions)
- Scope 2 emissions (electricity consumption with country-specific factors)
- Business emissions (travel, waste generation)
- Modular design with separate methods for each emission scope

**Rationale**: Separating calculation logic ensures maintainability and allows for easy updates to emission factors as standards evolve.

### 2. ML Models (`ml_models.py`)
**Purpose**: AI/ML functionality for predictions and recommendations
**Components**:
- **Prediction Model**: Random Forest Regressor for forecasting future emissions
- **Clustering Model**: K-Means clustering for user categorization (Low/Medium/High emitters)
- **LLM Integration**: Transformer-based model for personalized recommendations
- **Model Persistence**: Joblib for saving/loading trained models

**Rationale**: Random Forest chosen for its robustness and interpretability. K-Means provides simple but effective user segmentation.

### 3. Streamlit Application (`app.py`)
**Purpose**: User interface and application orchestration
**Features**:
- Sidebar form inputs for emission data
- Real-time calculation and display
- Interactive charts and visualizations
- External service integration buttons

### 4. Utilities (`utils.py`)
**Purpose**: Shared helper functions and formatting utilities
**Functions**:
- Number formatting with appropriate precision
- Input validation and error handling
- Chart creation and visualization helpers

## Data Flow

1. **User Input**: Users enter emission data through Streamlit sidebar forms
2. **Validation**: Input validation ensures data quality and prevents errors
3. **Calculation**: EmissionsCalculator processes inputs using GHG Protocol formulas
4. **ML Processing**: 
   - Historical data fed to prediction model for forecasting
   - User profile analyzed for clustering classification
   - LLM generates personalized recommendations
5. **Visualization**: Results displayed through interactive charts and metrics
6. **External Integration**: Links to carbon offset and dashboard services

## External Dependencies

### Core Libraries
- **streamlit**: Web framework for the user interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and preprocessing
- **joblib**: Model serialization and persistence
- **transformers**: Hugging Face library for LLM integration
- **matplotlib**: Visualization and charting (optional)
- **torch**: PyTorch for transformer models

### External Services
- **Carbon Offset Registry**: Integration for project registration (carbonica-ledger.netlify.app)
- **Carbon Dashboard**: Project tracking integration (karbon-compass.netlify.app)

**Rationale**: Dependencies chosen for stability, performance, and ease of use. Hugging Face Transformers provides state-of-the-art LLM capabilities without requiring custom model training.

## Deployment Strategy

### Local Development
- **Setup**: Simple pip installation from requirements.txt
- **Runtime**: Streamlit development server on port 5000
- **Hot Reload**: Automatic reloading during development

### Replit Deployment
- **Environment**: Python virtual environment management
- **Port Configuration**: Streamlit server configured for Replit hosting
- **Model Storage**: Local filesystem for model persistence
- **Resource Management**: Optimized for Replit's computational limits

**Rationale**: Streamlit's built-in deployment capabilities make it ideal for Replit hosting. The modular architecture allows for easy scaling and maintenance.

## Changelog

```
Changelog:
- June 30, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

---

**Note for Code Agent**: This application currently uses a modular Python architecture with Streamlit. The ML models use synthetic data for training - you may need to enhance the data generation or integrate real datasets for production use. The application is designed to be easily extensible for additional emission scopes or enhanced ML capabilities.