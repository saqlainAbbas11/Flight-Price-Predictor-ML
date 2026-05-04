# ✈️ Flight Price Predictor

An intelligent machine learning web application that predicts flight prices based on various parameters like airline, route, cabin class, duration, and time of travel. Built with Streamlit and multiple ML models.

## 🌟 Features

- **Real-time Price Prediction**: Get instant flight price estimates based on your inputs
- **Multiple ML Models**: Compare predictions from 3 different algorithms:
  - LightGBM (Best performing)
  - XGBoost
  - Random Forest
- **Interactive Dashboard**: Visualize flight data trends and patterns
- **Model Performance Metrics**: Detailed evaluation metrics for each model
- **Dark/Light Mode**: Automatic theme detection with manual toggle
- **Beautiful UI**: Modern, responsive design with animations
- **Smart Features**:
  - Automatic airport name resolution
  - Time period detection (Morning/Afternoon/Evening/Night)
  - Similar flights comparison
  - Real data reference from dataset

## 🚀 Live Demo

[Click here to see the live application](https://flight-price-predictor-ml-saqlain-abbas.streamlit.app/)

## 📊 Model Performance

| Model | R² Score | MAE ($) | RMSE ($) | Accuracy |
|-------|----------|---------|----------|----------|
| LightGBM | 0.85+ | ~$50-70 | ~$80-100 | ~80% |
| XGBoost | 0.84+ | ~$55-75 | ~$85-105 | ~78% |
| Random Forest | 0.83+ | ~$60-80 | ~$90-110 | ~76% |

*Note: Actual performance may vary based on the dataset*

## 🛠️ Technology Stack

- **Frontend & Framework**: Streamlit
- **Machine Learning**:
  - Scikit-learn (Random Forest, Preprocessing)
  - XGBoost
  - LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: 
  - Plotly (Interactive charts)
  - Matplotlib & Seaborn (Static charts)
- **Model Persistence**: Joblib

## 📁 Project Structure
