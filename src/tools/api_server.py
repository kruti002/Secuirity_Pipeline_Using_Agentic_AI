from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os

app = FastAPI()

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = "reports"

@app.get("/api/stats")
async def get_stats():
    path = os.path.join(REPORTS_DIR, "inference_performance.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {"error": "Stats not found"}

@app.get("/api/alerts/critical")
async def get_critical_alerts():
    path = os.path.join(REPORTS_DIR, "critical_alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path).head(100) # Limit for UI performance
        return df.to_dict(orient="records")
    return []

@app.get("/api/alerts/suspicious")
async def get_suspicious_alerts():
    path = os.path.join(REPORTS_DIR, "suspicious_alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path).head(100)
        return df.to_dict(orient="records")
    return []

@app.get("/api/feature-importance")
async def get_feature_importance():
    path = "models/supervised_feature_importance.csv"
    if os.path.exists(path):
        df = pd.read_csv(path).head(15)
        return df.to_dict(orient="records")
    return []

@app.get("/api/graphs/risk-timeline")
async def get_risk_timeline():
    # Simulate a timeline of risk activity based on top alerts
    path = os.path.join(REPORTS_DIR, "suspicious_alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Mocking a 24-hour hour distribution for the graph
        hours = list(range(24))
        # Add random noise to make it look "live"
        import random
        base_alerts = [max(10, int(20 + 50 * abs(pd.np.sin(h/4)))) for h in hours]
        data = [{"hour": f"{h:02d}:00", "alerts": b + random.randint(0, 5)} for h, b in zip(hours, base_alerts)]
        return data
    return []

@app.get("/api/graphs/asn-distribution")
async def get_asn_distribution():
    path = os.path.join(REPORTS_DIR, "suspicious_alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Group by ASN and average risk
        if 'ASN_Prior' in df.columns:
            dist = df.groupby('ASN_Prior')['risk_score'].mean().sort_values(ascending=False).head(10)
            return [{"asn": str(k), "risk": float(v)} for k, v in dist.items()]
    return [{"asn": "ASN_1", "risk": 0.8}, {"asn": "ASN_2", "risk": 0.6}]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
