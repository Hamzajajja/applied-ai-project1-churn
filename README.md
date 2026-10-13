# Applied AI Project 1: Customer Churn Prediction

**Student Name:** Hamza Khalid  
**Course:** Introduction to Applied AI (MPHIL 2nd  Semester)  

---

## Week 1: Exploratory Data Analysis (EDA)

### Dataset Profile
- **Source:** IBM Telco Customer Churn (Kaggle)
- **Data Size:** 7,043 customers, 21 features
- **Target Variable:** Churn (Yes/No) -> Baseline Churn Rate: **26.54%**

### Core Insights Discovered
- **Contract Risk:** "Month-to-month" contract holders display exceptionally high churn rates compared to stable long-term agreements.
- **The Fiber Optic Paradox:** Fiber optic pipeline subscribers leave at a higher rate than DSL subscribers, indicating possible pricing friction or initial connection instability.
- **Billing Warning:** Customers who pay via Electronic Checks drop off significantly faster than automated payment profiles.
- **Data Bug Fixed:** Corrected a text-to-numeric format mismatch on `TotalCharges`, which exposed 11 missing values belonging to brand-new accounts with 0 months of tenure.



## Week 2: Building and Evaluating Machine Learning Models

### Final Comparison Matrix (Test Set Results)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Stay** | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Logistic Regression (Default)** | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| **Logistic Regression (Balanced)** | 0.740 | 0.507 | **0.786** | 0.616 | 0.841 |
| **Random Forest (Baseline)** | 0.785 | 0.618 | 0.497 | 0.551 | 0.825 |

### Core Insights & Discoveries
- **The Accuracy Trap:** A baseline dummy model achieves a high accuracy of **73.5%** but has a **0% Recall rate**, completely missing every single customer who leaves. This proves that standard accuracy is useless for imbalanced datasets.
- **Catching Churners:** By switching to `class_weight='balanced'`, our Logistic Regression model successfully targets risk profiles, causing the **Recall rate to shoot up to 78.6%** (catching nearly 4 out of 5 departing customers).
- **Financial Optimization:** Implemented a business cost sweep using project parameters (Missed Churn = PKR 6,000 vs. False Alarm = PKR 1,000). The empirical optimal decision threshold sits at **0.15**, which severely limits company revenue leakage.


## Week 3: Model Optimization and Unsupervised Learning

### Strategic Segmentation Matrix (k=4 Profile)
- **Segment 1 (High-Risk Spending):** Early-stage users (18.4 months avg tenure) with steep bills (~\$80.41/mo). Churn rate sits at a catastrophic **43.0%**. This group is our primary target for retention campaigns.
- **Segment 3 (Low-Tenure Discovery):** Brand-new accounts (~9 months avg tenure) with basic plans. Churn rate sits at **32.0%**.
- **Segment 0 & 2 (High-Value Loyals):** Long-term subscribers (53 to 60 months avg tenure). Despite high spending profiles in Segment 2 (~\$92.09/mo), multiple active services protect them from churn (only 14.5% churn).

### Dimensionality Reduction Diagnostics
- **PCA Profiling:** Structural variance scanning reveals that **15 out of 30 principal components successfully retain 90% of total feature variance**, indicating a high level of feature redundancy in the raw space.

### Final Production Model Verification
- **Cross-Validation Framework:** 5-Fold Stratified Cross-Validation loop.
- **Locked Test Set Performance (Used Once):**
  - **Final Test ROC-AUC:** `0.841` (Fully stable within 2 standard deviations of Cross-Validation distribution mean)
  - **Model File Exported:** Successfully created and exported `churn_model.joblib` for immediate integration into our product app.

