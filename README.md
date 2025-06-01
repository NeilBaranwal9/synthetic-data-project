# 🧪 Synthetic Data Generation Pipeline for Tabular Data (MOSTLY AI Challenge)

This repository contains a complete pipeline for generating high-quality, privacy-preserving synthetic tabular data using advanced generative models. Developed as part of a competition hosted by MOSTLY AI, the pipeline is designed to produce synthetic data that mimics real data distributions while passing stringent utility and privacy checks.

---

## 🚀 Overview

* The pipeline trains and loads an ensemble of:
  * 4 CTGAN models with different seeds
  * 1 TVAE model

* These models are orchestrated for flat-table synthetic data generation with robust sampling, metadata handling, and privacy filtering.

---

## 📊 Results

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.897388|
| DCR Share | 50.7%   |
| NNDR Ratio| 1.08    |

*These metrics reflect excellent utility and privacy trade-offs, demonstrating the effectiveness of the approach in a real-world anonymization scenario.*

---

## 🧩 Key Features

* ✅ Multi-model ensemble: Combines CTGAN and TVAE models with weight-based sampling.  
* 🔐 Privacy filtering: Applies Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR) filters.  
* 📈 Accuracy evaluation: Synthetic data is blended with real data and tested using a stacked ensemble classifier (XGBoost + LightGBM).  
* 🧼 Robust metadata handling: Avoids boolean traps, infers column types conservatively.  
* ⚙️ SMOTENC support: Balances imbalanced classes using categorical-aware oversampling.

---

## 📂 Output

* `syn_up.csv`: The raw synthetic data before filtering.  
* Final hybrid dataset used for evaluation: Combined real and privacy-filtered synthetic data.

---

## 🛠️ Technologies Used

* Python 3.8+  
* SDV (CTGANSynthesizer, TVAESynthesizer)  
* scikit-learn, XGBoost, LightGBM  
* imblearn for SMOTENC  
* SDMetrics for quality and privacy scoring

---

## 🧠 How It Works (Simplified)

* Load & preprocess the input data, including target creation.  
* Train four CTGANs and one TVAE using SDV.  
* Evaluate models using SDMetrics' quality and privacy reports.  
* Sample synthetic rows proportionally based on composite model weights.  
* Filter synthetic rows using DCR and NNDR thresholds.  
* Combine real and synthetic data.  
* Train and evaluate a classifier to validate utility.

---

## 📎 Notes

* Requires CUDA for faster CTGAN/TVAE training (optional).  
* Models are cached to `models/` to avoid retraining.  
* Privacy thresholds: DCR ≥ 0.05, NNDR ≥ 0.5 (adjustable).
