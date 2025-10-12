# ⚙️ EquorAI: AI-Driven Demand Forecasting for Industrial Supply Chains

**EquorAI** is an end-to-end **AI demand forecasting system** built to help industrial and manufacturing supply chains anticipate demand fluctuations, supplier delays, and logistics bottlenecks.  
It combines **temporal deep learning (Mixture-of-Experts, LSTM, TFT)** with **gradient boosting (LightGBM residual correction)** to produce accurate, explainable, and scalable forecasts for thousands of SKUs across regions.

---

## 🎯 Purpose

Saudi manufacturers and distributors frequently experience **inventory imbalances** — from overstocking to sudden shortages — due to volatile supplier performance and logistic constraints.  
EquorAI aims to:

- Forecast product demand **1–8 weeks ahead**
- Model supplier and transportation delays
- Integrate **external market signals** (oil prices, weather, COVID impact)
- Offer **explainable forecasts** to reduce risk and improve planning efficiency

---

## 🧠 Project Summary

| Category              | Details                                                     |
| --------------------- | ----------------------------------------------------------- |
| **Project Name**      | EquorAI                                                     |
| **Domain**            | Industrial Supply Chain Forecasting                         |
| **Approach**          | Deep Learning + Gradient Boosting Hybrid                    |
| **Models Used**       | Mixture-of-Experts (GRU-based), LightGBM Residual Corrector |
| **Device**            | NVIDIA CUDA                                                 |
| **Development Stack** | Python 3.9, PyTorch, LightGBM, Pandas, Scikit-learn         |
| **Status**            | Completed (no deployment phase)                             |

---

## 🧾 Datasets Used

| #   | Dataset                                 | Source                                                                     | Notes                               |
| --- | --------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------- |
| 1   | **FreshRetailNet-50K**                  | Hugging Face                                                               | Core retail demand dataset          |
| 2   | **M5 Forecasting**                      | Kaggle                                                                     | Hierarchical sales and demand       |
| 3   | **On-Time Delivery**                    | Kaggle                                                                     | Supplier and shipment punctuality   |
| 4   | **Transportation & Logistics Tracking** | Kaggle                                                                     | Real-world logistics data           |
| 5   | **SynDelay (Synthetic Delay Dataset)**  | [GitHub](https://github.com/lx249/syndelay/blob/main/data/syndelay_v1.csv) | Delivery delay prediction           |
| 6   | **Cainiao LaDe**                        | Hugging Face                                                               | E-commerce logistics and delay data |
| 7   | **WTI Oil Prices**                      | FRED                                                                       | External economic signal            |

### 📦 Combined Dataset Summary

| Metric                             | Value                   |
| ---------------------------------- | ----------------------- |
| **Total Rows (after merging)**     | ≈ 47,886 records        |
| **User Data**                      | 20,323                  |
| **Opportunity / Transaction Data** | 27,563                  |
| **Feature Columns**                | 18 (dynamic) + 1 static |
| **Sequence Length**                | 8 weeks per SKU         |
| **Forecast Target**                | `target_next_week`      |
| **Static Column**                  | `combined_weight`       |

---

## 🧮 Data Engineering

- **Data Cleaning:**

  - Missing values handled (`NaN`, `-999.99` → null)
  - Outlier smoothing for delay and demand spikes
  - Time alignment for multi-source consistency

- **Feature Engineering:**
  - Lag and rolling features (1–8 weeks)
  - Mean and standard deviation per SKU
  - Rolling CV, trend, and mean stock
  - Exogenous features: WTI oil index, COVID mobility, weather anomalies
  - Normalization: per-SKU scaling and standardization

---

## 🧰 Model Architecture

### 🧩 Model 1 — Mixture-of-Experts (MoE) + LightGBM Residual Corrector

**Goal:** Multi-horizon demand forecasting with expert specialization.

**Architecture:**

- 3 Expert Networks — GRU-based, entropy-regularized
- 1 Gating Network — learns expert weights dynamically
- Residual correction layer via **LightGBM (200 iterations)**

**Details:**
| Component | Description |
|------------|--------------|
| **Input Sequence** | 8 timesteps × 18 features |
| **Hidden Layers** | GRU (128 → 64 → 32) |
| **Experts** | 3 GRU experts + weighted averaging |
| **Static Embedding** | `combined_weight` |
| **Output Layer** | Single regression head (`target_next_week`) |
| **Optimizer** | AdamW |
| **Loss Function** | Smooth L1 Loss |
| **Residual Correction** | LightGBM (RMSE minimization) |

**Stored Artifacts:**

- `/deep_learning_models/` contains:
  - MoE weights
  - LightGBM residuals
  - Feature scaler
  - SKU-level stats

---

## 📊 Model Metrics

| Metric                           | Value                                                                    |
| -------------------------------- | ------------------------------------------------------------------------ |
| **RMSE**                         | ≈ 5.93                                                                   |
| **MAE**                          | ≈ 2.31                                                                   |
| **R² Score**                     | ≈ 0.35                                                                   |
| **Accuracy (±5 absolute error)** | ~90% of predictions                                                      |
| **Top Features (by importance)** | `rolling_std_4`, `rolling_cv_4`, `mean_stock`, `lag_1`, `rolling_3_mean` |
| **Data Split**                   | Train/Val/Test (per SKU normalization)                                   |

---

## 🧠 Model 2 — Multi-Task PyTorch Architecture (Finalized)

**Objective:** Multi-horizon prediction for parallel forecasting tasks (demand, delay, logistics).

**Architecture Highlights:**

- Shared Encoder (LSTM / Transformer backbone)
- Multi-head output branches for different targets:
  - `demand_next_week`
  - `supplier_delay`
  - `logistics_score`
- Task-balanced loss weighting
- Dropout regularization (p = 0.2–0.3)
- Batch normalization between layers
- Device: CUDA

**Evaluation Metrics (Cross-Validation):**
| Metric | Value |
|---------|-------|
| RMSE (mean across tasks) | 4.87 |
| MAE | 2.09 |
| R² | 0.42 |
| Weighted Accuracy | 88.4% |
| Forecast Stability Index | 0.91 |

---

## 🧩 Model 3 — `deep_dl_multi_horizon_cv.py`

**Purpose:** Final multi-horizon cross-validation pipeline used for research runs and ablation studies.

**Key Traits:**

- Time-based cross-validation for robustness
- Hybrid data loader for SKU grouping
- Dynamic masking for variable sequence lengths
- Integrated metric tracking and residual analysis
- Generates residual error histograms and accuracy buckets

---

## 📈 Results Summary

| Category                 | Key Takeaway                                            |
| ------------------------ | ------------------------------------------------------- |
| **Forecasting Quality**  | ~90% within ±5 absolute error                           |
| **Error Concentration**  | Residuals normally distributed (σ ≈ 2.1)                |
| **Feature Contribution** | Lag-based > Rolling Mean > External signals             |
| **Model Stability**      | Consistent across random seeds                          |
| **Explainability**       | SHAP + feature importances aligned with supply behavior |

---

## 🧮 Development & Training Setup

- **Hardware:** NVIDIA GPU (CUDA)
- **Batch Size:** 128
- **Learning Rate:** 1e-3 (AdamW)
- **Epochs:** 50
- **Early Stopping:** patience = 5 epochs
- **Frameworks:** PyTorch Lightning, LightGBM, pandas, sklearn

---

## 🧠 Insights & Observations

- GRU-based MoE handled SKU variability effectively (reduced per-SKU variance by ~15%)
- LightGBM residual corrector stabilized long-horizon drift
- Exogenous factors (oil prices, logistics) improved correlation for delay prediction
- Feature scaling per SKU crucial for normalization consistency
- Ablation studies showed removing rolling CV or lag_1 increased RMSE by 20–25%

---

## 🔮 Future Enhancements

- Real-time inference API (FastAPI microservice)
- On-the-fly retraining via MLflow + GCS pipelines
- Integration with Streamlit dashboard for scenario visualization
- RAG-based knowledge retrieval for supply chain reasoning
- Deployment on Hugging Face Spaces / Vertex AI for end-to-end demo

---

## 🧑‍💻 Author

**Abdullah Imran**  
AI Engineer | Data Scientist  
Focus: Predictive Analytics · Deep Learning · Supply Chain Intelligence
