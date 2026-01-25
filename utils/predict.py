import joblib
import pandas as pd
import os
from pathlib import Path

# Use absolute path relative to this file
CURRENT_DIR = Path(__file__).parent.parent
MODEL_PATH = CURRENT_DIR / "models" / "flood_risk_model.pkl"

def load_model():
    """Load the trained flood risk model."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Current working directory: {os.getcwd()}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading model from {MODEL_PATH}: {str(e)}"
        )

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