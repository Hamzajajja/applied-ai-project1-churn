# Corporate Churn Diagnostic System (CCDS)
**A Research Project under the Introduction to Applied AI Module**

**Author/Researcher:** Hamza Khalid  
**Academic Standing:** MPhil (2nd Semester)  
**Hosted Product Platform:** [PASTE YOUR LIVE STREAMLIT APP LINK HERE]

---

### 📋 EXECUTIVE BRIEFING: PROJECT DELIVERABLES STATUS

- [x] **Inference Core Module:** `week1-eda.ipynb` (Ingestion, standard parsing, and 8 behavioral charts)
- [x] **Statistical Validation Node:** `week2-ml-models.ipynb` (Leak-free modeling and threshold financial optimization)
- [x] **Ensemble & Grouping Grid:** `week3-optimization.ipynb` (Cross-validation loops, K-Means profiles, and PCA loadings)
- [x] **Public Web Interface Deployment:** `app.py` (Streamlit implementation featuring real-time What-If sensitivity checks)

---

### 🔍 SECTION 1: INGESTION LOGISTICS & DATA AUDITING (WEEK 1)

#### Framework Parameters
- **Historical Database:** IBM Subscriber Dataset (7,043 transaction rows, 21 initial dimensions)
- **Primary Objective Field:** `Churn` (System drop-off indicator tracking a native **26.54% imbalance rate**)

#### Core System Discoveries
1. **Contract Type Imbalance:** Customers running on Month-to-Month arrangements act as the company's highest churn vulnerability compared to multi-year contracts.
2. **Infrastructure Quality Paradox:** High-tier Fiber Optic lines show significantly higher customer drop-off metrics than legacy DSL setups, indicating pricing frictions or post-setup connection bugs.
3. **Clearing Pipeline Vulnerabilities:** Electronic Check transactions drop off much faster than auto-pay methods like Credit Cards or Bank Transfers.
4. **Data Typo Trapped:** Resolved an object text formatting error in the `TotalCharges` column. Forcing standard numeric parsing exposed **11 missing fields** belonging to brand-new accounts with 0 months of history.

---

### ⚙️ SECTION 2: PROACTIVE MODELING & LOSS OPTIMIZATION (WEEK 2)

#### Diagnostic Performance Benchmarks (Testing Split Metrics)

| Model Architecture Strategy | Global Accuracy | In-Class Precision | Sensitivity (Recall) | F1 Metric | System ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Default (Stay Baseline)** | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Standard Logistic Regression** | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| **Class-Weighted Logistic Regression** | 0.740 | 0.507 | **0.786** | 0.616 | 0.841 |
| **Unoptimized Random Forest** | 0.785 | 0.618 | 0.497 | 0.551 | 0.825 |

#### Core Engineering Takeaways
- **The Accuracy Delusion:** A naive baseline model hits an attractive **73.5% accuracy** but achieves a **0% Recall score**, failing to flag a single customer who leaves. This demonstrates why tracking overall accuracy on imbalanced data is highly dangerous.
- **Sensitivity Correction:** Activating `class_weight='balanced'` within our training pipeline successfully flags at-risk accounts, pushing model **Recall up to 78.6%**.
- **Financial Threshold Tuning:** Applied project-specified economic parameters (Loss per Missed Churn = PKR 6,000 vs. Cost of False Alarm Discount = PKR 1,000) to find our optimal decision boundary. Shifting the threshold down to **0.15** minimizes overall cash leakage.

---

## 🔮 SECTION 3: RE-SAMPLING LOOPS & BEHAVIORAL COHORTS (WEEK 3)

#### Customer Segmentation Profiles (\(k=4\) K-Means Grouping)
- **Cohort 0 (The Crisis Group):** Very new accounts (~18.4 months) on high-tier monthly pricing plans (~\$80.41). This group suffers from a catastrophic **43.0% churn velocity**, making them our highest priority for retention outreach.
- **Cohort 2 (New, Cheap Tryouts):** Short-tenure users (~9 months) on cheap base packages (~\$37.71). Displays a high **32.0% churn rate**.
- **Cohorts 1 & 3 (The Legacy Base):** Long-term loyal accounts spanning 53 to 60 months. Despite high spending profiles (~\$92.09), multi-service integration acts as a powerful retention anchor.

#### Principal Component Dimensionality Mapping (PCA)
- Variance tracking confirms substantial input column redundancy. **15 out of 30 principal components successfully retain 90% of total system variance**, showing that customer churn is a highly non-linear problem requiring regularized classifiers.

#### Production Model Selection
- **Validation Validation Framework:** 5-Fold Stratified Cross-Validation loop execution.
- **Test Performance Matrix:** Final Test ROC-AUC locks at **0.841**, matching our CV mean within a narrow margin and confirming strong generalization limits.
- **Exported Asset:** Saved our trained pipeline cleanly as **`churn_model.joblib`**.

---

### 🛡️ SECTION 4: WEB DASHBOARD IMPLEMENTATION (WEEK 4)
- **Live Hosted Link:** [PASTE YOUR LIVE STREAMLIT APP LINK HERE]
- Re-engineered the serialized pipeline binaries into a clean, responsive web interface via Streamlit. The system includes an automated **What-If Optimization Matrix** for individual profiles and a batch upload portal for bulk spreadsheet processing.
