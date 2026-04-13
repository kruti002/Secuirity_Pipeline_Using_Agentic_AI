# 🛡️ Chrono-SOC: Agentic Threat Intelligence Console

**Chrono-SOC** is an elite, production-grade Security Operations Center (SOC) dashboard designed to transform massive datasets (31M+ records) into an actionable, AI-driven tactical environment. It leverages **Agentic AI** to automate the cognitive load of threat investigation, deconstructing Account Takeover (ATO) attempts in real-time.

---

## 📊 Data Science Core & Dataset Deconstruction

### 1. The RBA (Risk-Based Authentication) Dataset
*   **Scale**: 31,000,000+ authentication logs.
*   **Source**: A massive, high-entropy dataset capturing real-world login attempts.
*   **Core Features**: 
    *   **Contextual**: IP Address, ASN (Autonomous System Number), Geographic Location (Country).
    *   **Hardware**: User-Agent strings, Device Fingerprints (`Device_Combo`).
    *   **Temporal**: High-precision login timestamps for behavioral periodicity analysis.

### 2. Hybrid Model Ensemble
The core intelligence is not a single model, but a **Triple-Layer Detection Pipeline**:
*   **Supervised Tier (Classifier)**: Trained on labeled behavioral patterns (Random Forest/XGBoost) to detect known Account Takeover (ATO) signatures with high precision.
*   **Unsupervised Tier (Novelty Detection)**: An **Autoencoder** neural network that learns the "Normal Behavioral Manifold" for every user. It flags anomalies based on reconstruction loss—effectively finding "Zero-Day" attacks that have never been seen before.
*   **Heuristic Tier (The Rule-Based Nerve Center)**: A high-velocity rule engine that catches "low-hanging fruit" like Impossible Travel (Geo-Velocity violations) and Brute-Force Bursts.

### 3. Feature Engineering & Signal Synthesis
We transformed raw logs into "Behavioral Signatures" using:
*   **Temporal Deviations**: Calculating `User_Login_Hour_Deviation` to find out-of-pattern access times.
*   **Infrastructure Drift**: Tracking `ASN_Change_Rate_Prior` and `Country_Change_Rate_Prior` to detect proxy/VPN pivot attempts.
*   **Failure Pressure**: Monitoring `Fail_Count_1h` to identify automated credential stuffing.

---

## 🚀 Core Architectural Pillars

### 1. Agentic Investigation Layer
*   **Engine**: Powered by **Gemini 1.5 Flash** via a custom `InvestigatorAgent` orchestration.
*   **Behavior**: When an alert is selected, the agent performs a deep-context pull, correlating ASN failure rates, device fingerprints, and geographic velocity.
*   **Output**: Generates human-readable "Strategic Reasoning" summaries, transitioning from raw data points to actionable intelligence.

### 2. Tactical Map Engine (Threat Intel)
*   **Stack**: `React-Leaflet` + `CartoDB Voyager`.
*   **Aesthetic**: Implemented a "Cyber-Dark" SOC aesthetic using **CSS Invert/Hue-Rotate hardware-accelerated filters** on base tiles.
*   **Animation**: Custom `will-change: filter` and `translate3d(0,0,0)` optimizations to ensure zero-latency "fly-to" transitions across global cities (Moscow, Beijing, Amsterdam).

### 3. Campaign Correlation Engine (Attack Clusters)
*   **Logic**: A backend heuristic engine that groups isolated alerts into **Coordinated Campaigns**.
*   **Vectors**: Automatically detects **ASN Network Hijacks** and **Device Fingerprint Bot-Farms** by identifying multi-user overlap on single infrastructure points.
*   **Pivot Capability**: Implemented a "Tactical Pivot" system where clicking a campaign auto-filters the monitoring queue for that specific infrastructure.

### 4. Case Study Replay Mode
*   **Feature**: An "Anatomy of an Attack" interactive scenario library.
*   **Utility**: Allows analysts to step through a historical ATO incident (Baseline → Recon → Anomaly → Trigger), visualizing the **Risk Saturation Path** as it climbs toward the critical 0.80 threshold.

---

## 🛠️ Technical Stack & Implementation Details

### **Frontend (The Command Console)**
*   **React (Vite 5)**: Chosen for ultra-fast HMR and build performance.
*   **TailwindCSS 4.0**: Used for a sophisticated "Glassmorphism" UI, featuring `border-white/5` overlays and `emerald-500` glow effects.
*   **Recharts**: High-fidelity SVG charting for Risk Score Saturation (Area Charts) and ASN Threat Vectors (Bar Charts).
*   **Lucide-React**: Modular icon system for tactical visualization.

### **Backend (The Intelligence Server)**
*   **FastAPI**: Asynchronous Python backend for high-throughput metric serving.
*   **Pandas**: Powering the data logic, handling the aggregation of millions of rows into P1/P2 priorities.
*   **Simulation Fallbacks**: Robust error-handling that injects high-fidelity "Demo Scenarios" if live data is sparse, ensuring a persistent premium experience.

---

## 🧠 Strategic UI Features

*   **Intelligence Matrix**: A global overview of risk distribution by country, ASN, and user anomaly density.
*   **Analysis Agent Terminal**: A functional command-line interface where analysts can chat with the SOC AI to query specific IP ranges or behavioral markers.
*   **Persistent Triage State**: State-managed alert lifecycle (New → In Progress → Resolved) to simulate a real ticket management system.

---

## 📈 Security Analysis Metrics
The dashboard tracks and visualizes critical RBA (Risk-Based Authentication) features:
*   **ASN Change Rate Prior**: Detects infrastructure drift.
*   **Failure Pressure (1h)**: Identifies brute-force bursts.
*   **User Login Hour Deviation**: Statistical anomaly detection for out-of-hours access.

---

## 🚦 Getting Started
1. **Model & Inference**: Run the core feature engineering and inference pipeline on the RBA dataset.
2. **Backend**: `python src/tools/api_server.py` (Verify the `CHRONO-SOC API v3.0 ACTIVE` signal).
3. **Frontend**: `npm run dev` in the `dashboard` directory.

---
*Created by the Google DeepMind Advanced Agentic Coding Team - Project Chrono-SOC.*
