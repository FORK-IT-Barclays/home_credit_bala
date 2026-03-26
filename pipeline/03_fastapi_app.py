import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Paths
DATA_DIR = Path("/home/balahero03/credit")
PIPE_DIR = DATA_DIR / "pipeline"
MODEL_DIR = PIPE_DIR / "models"
FEAT_DIR = PIPE_DIR / "output"

app = FastAPI(title="VECTOR Pre-Delinquency Intelligence Core API", version="1.0")

# Global dependencies
models = {}
meta = {}
test_features_df = None

@app.on_event("startup")
def load_models_and_data():
    global models, meta, test_features_df
    print("Loading models and metadata...")
    try:
        models['historian'] = joblib.load(MODEL_DIR / "historian_xgb.pkl")
        models['behavioral'] = joblib.load(MODEL_DIR / "behavioral_lgbm.pkl")
        models['hist_explainer'] = joblib.load(MODEL_DIR / "historian_shap_explainer.pkl")
        models['beh_explainer'] = joblib.load(MODEL_DIR / "behavioral_shap_explainer.pkl")
        
        with open(MODEL_DIR / "model_meta.json", "r") as f:
            meta = json.load(f)
            
        print("Loading test features for inference simulation...")
        # We load a subset to save memory
        test_features_df = pd.read_parquet(FEAT_DIR / "test_features.parquet")
        # Ensure ID column is string or int properly structured
        test_features_df['SK_ID_CURR'] = test_features_df['SK_ID_CURR'].astype(str)
        test_features_df.set_index('SK_ID_CURR', inplace=True)
        print("Initialization complete.")
    except Exception as e:
        print(f"Error during initialization: {e}")

class RiskRequest(BaseModel):
    customer_id: str

@app.post("/api/v1/predict")
def predict_risk(req: RiskRequest):
    if req.customer_id not in test_features_df.index:
        raise HTTPException(status_code=404, detail="Customer ID not found in feature store.")
        
    customer_data = test_features_df.loc[[req.customer_id]]
    
    # Extract features
    X_hist = customer_data[meta['historian_features']]
    X_beh = customer_data[meta['behavioral_features']]
    
    # Inference
    hist_prob = float(models['historian'].predict_proba(X_hist)[0, 1])
    beh_prob = float(models['behavioral'].predict_proba(X_beh)[0, 1])
    
    w_hist = meta['ensemble_weights']['historian']
    w_beh = meta['ensemble_weights']['behavioral']
    
    risk_score = w_hist * hist_prob + w_beh * beh_prob
    
    # Risk Tier Assignment (calibrated dynamically or statically)
    # Using simple thresholds based on historical distribution analysis
    if risk_score >= 0.6:  
        tier = 'CRITICAL'
        zone = 'Zone 9 - High Velocity & Acceleration'
    elif risk_score >= 0.4: 
        tier = 'HIGH'
        zone = 'Zone 7 - Accelerating Spiral'
    elif risk_score >= 0.2: 
        tier = 'MEDIUM'
        zone = 'Zone 4 - Early Warning Drift'
    else:              
        tier = 'LOW'
        zone = 'Zone 1 - Stable'

    # Get SHAP explanations (top 3 behaviors)
    shap_beh = models['beh_explainer'].shap_values(X_beh)
    if isinstance(shap_beh, list): shap_beh = shap_beh[1]
    
    shap_hist = models['hist_explainer'].shap_values(X_hist)
    if isinstance(shap_hist, list): shap_hist = shap_hist[1]
    
    top_behaviors = []
    if len(shap_beh) > 0:
        beh_vals = shap_beh[0]
        # Get indices of top 3 positive SHAP values
        top_indices = np.argsort(beh_vals)[-3:][::-1]
        for idx in top_indices:
            feat_name = meta['behavioral_features'][idx]
            top_behaviors.append({
                "feature": feat_name,
                "importance": float(beh_vals[idx]),
                "value": float(X_beh.iloc[0, idx]) if pd.notnull(X_beh.iloc[0, idx]) else "N/A"
            })
            
    # Mocking GenAI message based on tier
    genai_message = ""
    if tier in ['CRITICAL', 'HIGH']:
        genai_message = (
            "Hi [Customer], we noticed some recent changes in your account activity. "
            "We want to ensure you're fully supported during this time. "
            "Click here to explore flexible payment options or a payment holiday without affecting your credit score."
        )
    elif tier == 'MEDIUM':
        genai_message = (
            "Hi [Customer], just a quick check-in. It seems your recent bill payments have shifted slightly in timing. "
            "If you need to adjust your payment dates to better align with your salary, we can help with that."
        )
    
    return {
        "customer_id": req.customer_id,
        "risk_score": round(risk_score, 4),
        "risk_tier": tier,
        "risk_zone": zone,
        "historian_score": round(hist_prob, 4),
        "behavioral_score": round(beh_prob, 4),
        "top_risk_drivers": top_behaviors,
        "recommended_action": "Payment Restructure / Holiday" if tier in ['CRITICAL', 'HIGH'] else "Educational Outreach",
        "genai_outreach_draft": genai_message
    }

@app.get("/api/v1/customers")
def list_customers():
    # Return a few sample customer IDs for testing
    if test_features_df is None:
        return {"customers": []}
    sample = test_features_df.head(20).index.tolist()
    return {"customers": sample}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
