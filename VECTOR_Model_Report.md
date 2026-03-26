# 📊 VECTOR Model Performance & Feature Report

This report outlines the performance metrics of the **Dual-Expert AI Architecture** and provides a full dictionary of the exact features used to predict the "Financial Death Spiral".

---

## 🎯 1. Model Performance (Cross-Validated)

The architecture splits predictions between structural/long-term financial health and dynamic behavioral velocity.

| Expert Model | Algorithm | Target Scope | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Financial Historian** | XGBoost | Structural risk, income, credit history (Asset/Capital) | **0.7627** |
| **Behavioral Specialist** | LightGBM | Payment velocity, drift, DPD acceleration (Liquidity/Mgmt) | **0.6279** |
| **Combined Ensemble** | Weighted Stacking | 360° Predictive Anticipatory Risk Score | **0.7711** |

> **Notes on Accuracy:** An AUC of 0.7711 on real-world credit default data (which is highly imbalanced at 8%) is considered highly predictive for early-warning banking systems. The ensemble weight optimally relied 10% on Historical and 90% on Behavioral patterns in testing, proving that dynamic behaviors strongly dictate upcoming risk.

---

## 🧠 2. Top Driving Signals (SHAP Values)

Using **Explainable AI (SHAP)**, the system identifies exactly *why* a user is placed in a specific Risk Zone.

### 🏛️ Financial Historian (Top 10 Risk Drivers)
*These structural features define the borrower's baseline capacity.*

| Importance Rank | Feature Name | Description |
| :---: | :--- | :--- |
| 1 | `EXT_SOURCE_MEAN` | SHAP Impact: 0.4919 |
| 2 | `BUREAU_MAX_CREDIT_UTIL` | SHAP Impact: 0.1667 |
| 3 | `CREDIT_TO_GOODS_RATIO` | SHAP Impact: 0.1537 |
| 4 | `ANNUITY_TO_INCOME_RATIO` | SHAP Impact: 0.1368 |
| 5 | `CODE_GENDER` | SHAP Impact: 0.1294 |
| 6 | `EXT_SOURCE_MIN` | SHAP Impact: 0.1243 |
| 7 | `NAME_EDUCATION_TYPE` | SHAP Impact: 0.1169 |
| 8 | `EXT_SOURCE_PRODUCT` | SHAP Impact: 0.1059 |
| 9 | `AGE_YEARS` | SHAP Impact: 0.1044 |
| 10 | `OWNS_CAR` | SHAP Impact: 0.0879 |


### ⚡ Behavioral Specialist (Top 10 Risk Drivers)
*These dynamic features act as the "Velocity" and "Acceleration" triggers for the Pre-Delinquency Engine.*

| Importance Rank | Feature Name | Description |
| :---: | :--- | :--- |
| 1 | `RECENT_LATE_RATIO` | SHAP Impact: 0.0334 |
| 2 | `INST_AMT_PAYMENT_SUM` | SHAP Impact: 0.0208 |
| 3 | `RECENT_AVG_BAL` | SHAP Impact: 0.0123 |
| 4 | `POS_FUTURE_INST_MEAN` | SHAP Impact: 0.0115 |
| 5 | `POS_COUNT` | SHAP Impact: 0.0110 |
| 6 | `INST_AMT_INSTALMENT_SUM` | SHAP Impact: 0.0096 |
| 7 | `RECENT_UNDERPAID_R` | SHAP Impact: 0.0087 |
| 8 | `CC_MAX_UTILIZATION` | SHAP Impact: 0.0072 |
| 9 | `CREDIT_UTIL_RISK_VELOCITY` | SHAP Impact: 0.0068 |
| 10 | `CC_ATM_DRAWINGS_RATIO` | SHAP Impact: 0.0043 |


---

## 🗂️ 3. Full Feature Dictionary (104 Features Total)

Below is the complete list of engineered features broken down by the two respective models.

### 🏛️ Financial Historian Features (46 features)
```text
INCOME_TOTAL, CREDIT_TO_INCOME_RATIO, ANNUITY_TO_INCOME_RATIO, CREDIT_TO_GOODS_RATIO, INCOME_PER_PERSON, EMPLOYED_TO_AGE_RATIO, AGE_YEARS, EXT_SOURCE_MEAN, EXT_SOURCE_STD, EXT_SOURCE_MIN, EXT_SOURCE_PRODUCT, MISSING_EXT_SOURCES, TOTAL_DOCS_PROVIDED, SOCIAL_CIRCLE_DEFAULT_MEAN, OBS_SOCIAL_CIRCLE_MEAN, ASSET_OWNERSHIP_SCORE, REGION_RATING, REGION_CITY_DIFF, CNT_CHILDREN, CHILDREN_INCOME_BURDEN, CODE_GENDER, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, BUREAU_CREDIT_COUNT, BUREAU_AVG_OVERDUE_DAYS, BUREAU_MAX_OVERDUE_DAYS, BUREAU_MAX_AMT_OVERDUE, BUREAU_SUM_DEBT, BUREAU_AVG_CREDIT_UTIL, BUREAU_MAX_CREDIT_UTIL, BUREAU_AVG_DAYS_CREDIT, BUREAU_CREDIT_ENDDATE_MEAN, DAYS_EMPLOYED_RATIO, EMPLOYMENT_GAP, YOUNG_BORROWER, SENIOR_BORROWER, OWNS_CAR, OWNS_REALTY, HOUR_APPR_PROCESS, NIGHT_APPLICATION, NAME_HOUSING_TYPE, NAME_CONTRACT_TYPE, BUREAU_ACTIVE_COUNT, BUREAU_CLOSED_COUNT, BUREAU_PROLONG_COUNT
```

### ⚡ Behavioral/Velocity Features (58 features)
```text
BUREAU_AVG_OVERDUE_DAYS, BUREAU_MAX_OVERDUE_DAYS, BUREAU_TOTAL_OVERDUE, BUREAU_MAX_AMT_OVERDUE, BUREAU_OVERDUE_RATIO, BUREAU_RECENT_OVERDUE_MONTHS, BUREAU_BAD_MONTHS, CREDIT_UTIL_RISK_VELOCITY, INST_COUNT, INST_MEAN_DELAY, INST_MAX_DELAY, INST_STD_DELAY, INST_LATE_COUNT, INST_LATE_RATIO, INST_EARLY_RATIO, INST_UNDERPAID_COUNT, INST_UNDERPAID_RATIO, INST_SHORTFALL_SUM, INST_SHORTFALL_MEAN, INST_AMT_PAYMENT_SUM, INST_AMT_INSTALMENT_SUM, PAYMENT_TIMING_ERRATICISM, RECENT_MEAN_DELAY, RECENT_LATE_RATIO, RECENT_UNDERPAID_R, BILL_DRIFT_VELOCITY, CC_COUNT, CC_AVG_BALANCE, CC_MAX_BALANCE, CC_MIN_BALANCE, CC_AVG_UTILIZATION, CC_MAX_UTILIZATION, CC_BALANCE_STD, CC_TOTAL_DRAWINGS, CC_ATM_DRAWINGS_RATIO, CC_DPD_COUNT, CC_MAX_DPD, CC_AVG_MIN_PAYMENT, RECENT_AVG_UTIL, RECENT_DPD_COUNT, RECENT_AVG_BAL, CREDIT_EXHAUST_VELOCITY, LIQUIDITY_MOMENTUM, POS_COUNT, POS_DPD_MEAN, POS_DPD_MAX, POS_DPD_COUNT, POS_DPD_DEF_COUNT, POS_COMPLETED_RATIO, POS_FUTURE_INST_MAX, POS_FUTURE_INST_MEAN, POS_RECENT_DPD_MEAN, POS_RECENT_DPD_COUNT, DPD_VELOCITY, VECTOR_RISK_VELOCITY, VECTOR_RISK_ACCELERATION, VECTOR_RISK_ZONE, VECTOR_STRESS_SCORE
```

---
*Report generated automatically from pipeline execution artifacts.*
