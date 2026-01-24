import pandas as pd

def get_feature_importance(model, feature_names):
    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return df
