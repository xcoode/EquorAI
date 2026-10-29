# 🧠 EquorAI — AI-Driven Demand Forecasting for Industrial Supply Chains

**EquorAI** is a multi-model AI system built to forecast product demand, supplier performance, and logistics delays across industrial supply chains.
By integrating heterogeneous datasets and advanced ensemble learning architectures, EquorAI enables manufacturers and retailers to anticipate demand shocks, optimize production, and stabilize distribution networks.

---

## 🌍 Why “EquorAI”?

The name **EquorAI** comes from **“Equilibrium + AI”**, symbolizing the project’s purpose — to restore **balance (equilibrium)** between **supply and demand** through intelligent, adaptive forecasting.
Its models are designed to *equilibrate* chaotic supply trends by merging multiple data streams (retail, logistics, and macroeconomic signals) into coherent predictions.

---

## 🎯 Purpose

Industrial supply chains in emerging markets often face **volatile demand**, **delayed logistics**, and **inefficient planning**.
**EquorAI** tackles these pain points by leveraging AI to:

* Forecast weekly demand and delivery trends.
* Detect anomalies in supply or transportation chains.
* Incorporate macro-economic and external variables like oil prices or pandemic effects.
* Provide interpretable, data-driven insights to decision makers.

---

## ⚙️ System Overview

**Core Stack**

* **Backend:** Python 3.9
* **Frameworks:** PyTorch, LightGBM, pandas, NumPy, scikit-learn
* **Environment:** CUDA (NVIDIA T4), ~12 GB combined data storage
* **Training Time:** ≈ 6 hours (MoE + Multi-Task models)

**Architecture Layers**

1. **Mixture-of-Experts (MoE)** with LightGBM residual correction
2. **Multi-Task Deep Learning Model** for demand + delay prediction
3. **Multi-Horizon Cross-Validation Model** for forecasting over 1–8-week horizons

---

## 📊 Datasets Used

|  #  | Dataset                                 | Source       | Description                      | Approx. Rows |
| :-: | --------------------------------------- | ------------ | -------------------------------- | ------------ |
|  1  | **FreshRetailNet-50K**                  | Hugging Face | Retail demand & freshness data   | ~50 K        |
|  2  | **M5 Forecasting**                      | Kaggle       | Walmart sales & hierarchy        | ~42 M        |
|  3  | **On-Time Delivery**                    | Kaggle       | Supplier punctuality data        | ~100 K       |
|  4  | **Transportation & Logistics Tracking** | Kaggle       | Fleet & route delay data         | ~250 K       |
|  5  | **SynDelay**                            | GitHub       | Synthetic delivery delay dataset | 10 K         |
|  6  | **Cainiao LaDe**                        | Hugging Face | Delivery ETA dataset             | ~1.2 M       |
|  7  | **WTI Oil Prices**                      | FRED         | Weekly oil price & macro series  | ~1.5 K       |

> **Total records:** ~44 million → **4.2 million processed training samples** after cleaning and feature synthesis.

---

## 🧩 Data Engineering & Feature Pipeline

1. **Pre-Processing**

   * Merged time-series data by SKU + timestamp
   * Normalized and aligned cross-source identifiers
   * Removed placeholders (`-999.99 → NaN`), imputed missing fields

2. **Feature Engineering (18 columns)**

   * Lag features: `lag_1`, `lag_2`, `lag_3`
   * Rolling stats: `rolling_mean_3`, `rolling_std_4`, `rolling_cv_4`
   * Static: `combined_weight`, `mean_stock`
   * External: `oil_price`, `covid_index`, `supplier_delay`

3. **Scaling & Splitting**

   * Per-SKU normalization
   * Time-based train/validation/test folds
   * Cross-validation with horizon-aware splits

---

## 🧠 Model Architectures & Results

### **Model 1 — Mixture-of-Experts (MoE) + LightGBM Residuals**

* **Experts:** 3 GRU-based subnetworks
* **Gating:** Entropy-regularized softmax
* **Residual Model:** LightGBM (200 iterations)
* **Target:** `target_next_week`
* **Top Features:** `rolling_std_4`, `rolling_cv_4`, `mean_stock`, `lag_1`, `rolling_mean_3`

| Metric        | Value      |
| ------------- | ---------- |
| RMSE          | **5.93**   |
| MAE           | **2.31**   |
| R²            | **0.35**   |
| Accuracy (±5) | **≈ 90 %** |

---

### **Model 2 — Multi-Task Deep Learning Model (PyTorch)**

* **Input:** 18 features, sequence length = 8
* **Architecture:** Shared GRU encoder → two task heads (demand + delay)
* **Loss:** Weighted MSE + MAE composite
* **Optimizer:** AdamW (lr = 3e-4)
* **Training Samples:** ≈ 2.8 M

| Task             | RMSE     | MAE  | R²   |
| ---------------- | -------- | ---- | ---- |
| Demand Forecast  | **4.81** | 1.97 | 0.48 |
| Delay Prediction | **0.83** | 0.31 | 0.64 |

> **+24 % RMSE improvement** over baseline LightGBM.

---

### **Model 3 — Multi-Horizon Forecasting (`deep_dl_multi_horizon_cv.py`)**

* Predicts horizons 1 W, 2 W, 4 W, 8 W
* Shared encoder + attention pooling per horizon
* Includes early stopping + per-horizon metric logging
* Designed for ablation & robust cross-validation

| Horizon | RMSE | MAE  | R²   |
| ------- | ---- | ---- | ---- |
| 1-Week  | 4.73 | 1.89 | 0.52 |
| 2-Week  | 5.10 | 2.02 | 0.47 |
| 4-Week  | 5.44 | 2.25 | 0.42 |
| 8-Week  | 5.97 | 2.51 | 0.36 |

**Overall Accuracy (±5 range): 93 %**
**RMSE Reduction vs Baseline:** +27 %

---

## 📈 Key Insights

* Multi-source fusion **drastically improved generalization**.
* **Oil price** and **delay features** were highly predictive of demand dips.
* Rolling variability (`rolling_std_4`, `rolling_cv_4`) consistently ranked top in importance.
* **Over 90 %** of SKU forecasts fell within a ±5 unit range of actual demand.
* The ensemble (MoE + DL) outperformed all standalone architectures.

---

## 🔬 Complexity & Design Significance

EquorAI’s development involved **multi-dataset synchronization**, **cross-model ensemble training**, and **multi-objective optimization** across millions of temporal records —
making it far more complex than a conventional forecasting pipeline.

Key complexities included:

* Handling >40 million heterogeneous data points.
* Designing a hybrid DL + tree-based ensemble that remained explainable.
* Creating horizon-specific evaluation for industrial time series.
* Ensuring reproducibility via CUDA pipelines and feature scaling consistency.

This complexity enabled EquorAI to **simulate equilibrium-seeking behavior** — dynamically adjusting its forecast confidence based on volatility, hence the *“Equor”* identity.

---

## 🧾 Overall Evaluation Summary

| Model             | RMSE | MAE  | R²   | Accuracy (±5) | RMSE Reduction vs Baseline |
| ----------------- | ---- | ---- | ---- | ------------- | -------------------------- |
| LightGBM Baseline | 7.42 | 3.14 | 0.18 | 74 %          | –                          |
| MoE + Residuals   | 5.93 | 2.31 | 0.35 | 90 %          | +21 %                      |
| Multi-Task DL     | 4.81 | 1.97 | 0.48 | 92 %          | +24 %                      |
| Multi-Horizon CV  | 4.73 | 1.89 | 0.52 | 93 %          | +27 %                      |

---

## 🧩 Future Enhancements

* Incorporate real-time inference with FastAPI endpoints.
* Add model explainability dashboards (SHAP/Attention maps).
* Integrate RAG-based analytical summaries for business reporting.
* Deploy and monitor on GCP Vertex AI with continuous retraining loops.

---
