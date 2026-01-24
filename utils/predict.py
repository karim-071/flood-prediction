import joblib
import pandas as pd

MODEL_PATH = "models/flood_risk_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def prepare_input(state_data):
    latest = state_data.sort_values("YEAR").iloc[-1]

    # Build feature vector (MATCH TRAINING FEATURES)
    return pd.DataFrame([{
        "annual_rainfall": latest["annual_rainfall"],
        "monsoon_rainfall": latest["monsoon_rainfall"],
        "pre_monsoon_rainfall": latest["pre_monsoon_rainfall"],
        "post_monsoon_rainfall": latest["post_monsoon_rainfall"],
        "monsoon_anomaly": latest["monsoon_anomaly"],
        "monsoon_percentile": latest["monsoon_percentile"]
    }])