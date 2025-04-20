# 🧠 AI-Powered Thyroid Disease Diagnostic System

This is a personal project focused on building an end-to-end intelligent system to assist in the early diagnosis of thyroid-related pathologies using:

1. **Ultrasound images** – deep learning-based binary classification of benign vs malignant nodules.
2. **Hormonal blood test results** – machine learning models trained on structured patient lab data.

The goal is to develop deployable, accurate, and interpretable models that support clinicians in thyroid disease screening.

---

## 🧬 Project Modules

### 1. 📊 Hormonal Blood Test Classifier (Structured Data)

- Dataset: 600k+ samples (TSH, T3, TT4, FTI, pregnancy, age, etc.)
- Models tested:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Feature Engineering:
  - Label Encoding, Scaling, Missing Value Imputation
  - Feature selection (top 10–20 most relevant markers)
- Visualizations:
  - Feature importance (SHAP, RF), hormone distribution, t-SNE, PCA, radar plots

**Best model:** XGBoost  
**Accuracy:** 95.7% (due to heavy class imbalance)  
**Balanced recall/F1:** further tuning suggested

---

### 2. 🧠 Thyroid Ultrasound Image Classifier (Deep Learning)

- Dataset: 6005 thyroid ultrasound images from 601 patients
- Each image linked to a patient-level histopathological diagnosis
- Models used:
  - CNN (custom)
  - MobileNetV2 (transfer learning)
- Augmentation applied: rotation, zoom, flip, shifts
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC AUC

| Model        | Accuracy | F1-score | Malignant Recall | Benign Recall |
|--------------|----------|----------|------------------|----------------|
| **MobileNetV2** | 59.4%   | 0.55     | **0.82**         | 0.20           |
| **CNN**         | 51.9%   | 0.52     | 0.54             | **0.48**       |

> ✅ Final decision logic used **ensemble prediction**, prioritizing malignant detection via MobileNet.

---

## 🌐 Web Interface

An interactive web app was developed using **Flask**, allowing users to:

- Upload ultrasound images (.jpg, .png)
- Run inference using trained models
- Get a binary diagnostic output:  
  → `Benign` or `Malignant`

> Models are preloaded (`.h5`) and predictions run in real-time.

---

## 🚀 Project Setup

```bash
git clone https://github.com/yourusername/thyroid-ai-diagnosis.git
cd thyroid-ai-diagnosis
pip install -r requirements.txt
python app.py
