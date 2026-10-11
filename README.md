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
