# Flood Prediction App 🌊

An interactive **Streamlit web application** for historical flood risk assessment and rainfall analysis across Indian states using **machine learning** and **data visualization**.

## Description

This application analyzes long-term rainfall data and evaluates flood risk patterns using a machine learning model trained on engineered climatic features. It is designed for **risk assessment and analytical insights**, not future climate forecasting.

The app is organized into two main sections:
- 🤖 **Machine Learning–based Flood Risk Assessment**
- 📊 **Rainfall Analysis & Visualization**

## Features

- **Interactive Charts**: Visualize rainfall trends over the years with Plotly.
- **Monthly Statistics**: View average, total, max, and min rainfall for any month.
- **Flood Risk Prediction**: Probability-based flood risk classificatio
- **Period Analysis**: Analyze rainfall for different periods like Jan-Feb, Mar-May, Jun-Sep, and Oct-Dec.

### 🤖 Flood Risk Prediction (Machine Learning)
- Uses a **Random Forest Classifier** trained on historical rainfall data
- Evaluates flood risk based on **seasonal and annual rainfall patterns**
- Engineered features include:
  - Annual rainfall
  - Monsoon rainfall (Jun–Sep)
  - Pre-monsoon rainfall (Mar–May)
  - Post-monsoon rainfall (Oct–Dec)
  - Monsoon anomaly (z-score)
  - Monsoon percentile (state-wise)
- Outputs:
  - Flood probability (%)
  - Risk level: **Low / Medium / High**
- Includes **model explainability** via feature importance visualization

⚠️ *Note: The ML model assesses historical flood risk and does not predict future years.*



## 🧠 Machine Learning Details

- **Model Type**: Classification
- **Algorithm**: Random Forest Classifier  
- **Training Target**: Flood risk label derived from extreme monsoon rainfall percentiles    
- **Not a time-series or forecasting model**

## 🛠 Tech Stack

- **Streamlit** – Web application framework  
- **Scikit-learn** – Machine learning  
- **Pandas** – Data processing and analysis  
- **Plotly** – Interactive visualizations  
- **Joblib** – Model persistence 

## 📌 Disclaimer

This application is intended for **educational and analytical purposes only**.  
It does **not** provide real-time flood forecasting or official disaster warnings.

---

## 🙌 Acknowledgements

- **Streamlit** – Interactive web app development  
- **Plotly** – Data visualization  
- **Pandas & Scikit-learn** – Data science and machine learning tools  
