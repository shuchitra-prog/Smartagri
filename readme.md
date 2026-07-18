# Accident-Prone Road Segment Prediction and Traffic Forecasting Using Machine Learning[cite: 3]

## Overview
This project introduces a two-stage machine-learning framework that first forecasts short-term traffic flow and then identifies accident-prone road sections[cite: 3]. It utilizes ensemble and deep learning models to process multi-source spatio-temporal data and features an interactive dashboard for real-time visualization[cite: 3].

## Key Features
*   **Multi-Source Data Integration:** Fuses data from crash records (MoRTH), traffic flow rates (Google Maps/Uber Movement), weather variables (rainfall, visibility, wind), road network topology (OpenStreetMap), and event calendars[cite: 3].
*   **Two-Stage Predictive Modeling:** Combines traffic flow prediction with accident risk prediction[cite: 3].
*   **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) values to interpret model behavior and increase transparency regarding which features (e.g., rainfall, festivals) contribute most to crash risks[cite: 3].
*   **Interactive Dashboard:** A Streamlit-based interface integrating Folium maps allows city administrators to visualize predicted congestion, accident-risk heatmaps, and the top most risky segments[cite: 3].

## Methodology

### Stage 1: Traffic Flow Prediction
*   Predicts traffic congestion levels along road segments[cite: 3].
*   Utilizes Gradient-Boosted Trees (LightGBM) to capture non-linear feature interactions and LSTM Networks to learn temporal dependencies[cite: 3].
*   The output acts as a dynamic predictor (congestion index) for the subsequent accident prediction model[cite: 3].

### Stage 2: Accident Risk Prediction
*   Estimates the likelihood of a crash occurring in a specific segment based on the predicted congestion and spatial-topological features[cite: 3].
*   Employs supervised classifiers including LightGBM, Random Forest, and Logistic Regression[cite: 3].
*   Handles class imbalance using the Synthetic Minority Oversampling Technique (SMOTE)[cite: 3].

## Project Performance
*   The models were trained and tested on datasets from Delhi, Bengaluru, and Mumbai (2019–2023)[cite: 3].
*   The proposed LightGBM model achieved an AUC of 0.87 and an F1-score of 0.81, outperforming baseline models[cite: 3].
*   SHAP analysis revealed that rainfall increased accident likelihood by 28%, festival effects by 19%, and late-evening congestion by 14%[cite: 3].

## Technology Stack
*   **Programming Language:** Python[cite: 3]
*   **Machine Learning:** Scikit-learn, LightGBM, LSTM[cite: 3]
*   **Data Processing & Geospatial:** GeoPandas, NetworkX, LabelEncoder[cite: 3]
*   **Visualization:** Streamlit, Folium, SHAP[cite: 3]

## Future Enhancements
*   Incorporate datasets from rural and semi-urban areas to improve model generalization[cite: 3].
*   Implement real-time data streaming to improve model responsiveness instead of relying on hourly level data[cite: 3].
*   Integrate Graph Neural Networks (GNNs) for richer spatio-temporal modeling[cite: 3].
*   Utilize IoT sensor feeds and CCTV video data for live traffic inputs[cite: 3].
*   Expand the dashboard to feature real-time alerting and what-if analysis simulations for proactive planning[cite: 3].

## Authors
*   **D D MOHEETH KUMAR** - SRM Institute of Science & Technology[cite: 3]
*   **SHUCHITRA E** - SRM Institute of Science & Technology[cite: 3]
*   **N NAVYA SHREE** - SRM Institute of Science & Technology[cite: 3]
*   **K S VIVIN** - SRM Institute of Science & Technology[cite: 3]
*   **Jayachandhran R** - SRM Institute of Science & Technology[cite: 3]
*   **Neelam Sanjeev Kumar** - SRM Institute of Science & Technology[cite: 3]