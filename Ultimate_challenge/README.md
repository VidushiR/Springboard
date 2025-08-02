
---

##  Part 1 – Exploratory Data Analysis

**Objective**: Analyze user login patterns from the `logins.json` file and identify trends based on 15-minute time intervals.

**Key Actions**:
- Resampled login timestamps into 15-minute bins
- Visualized daily/weekly patterns using line plots and heatmaps
- Identified peak login hours and weekday/weekend trends
- Highlighted demand cycles and user activity behavior

 **Conclusion**: Strong daily cycles were identified, with increased weekend nighttime activity and structured weekday usage.

---

##  Part 2 – Experiment and Metrics Design

**Objective**: Propose a strategy to test whether toll reimbursement increases driver activity across two cities (Gotham and Metropolis).

**Key Actions**:
- Chose the **percentage of cross-city trips** as the success metric
- Proposed an A/B test with a treatment group receiving toll reimbursements
- Suggested using statistical tests (e.g., Chi-square or t-test) to assess impact
- Provided interpretation guidelines and operational recommendations

 **Conclusion**: The experiment design would provide actionable insight into driver flexibility and cost-effectiveness of toll reimbursement.

---

##  Part 3 – Predictive Modeling

**Objective**: Predict user retention (active in the past 30 days) using the `ultimate_data_challenge.json` dataset.

**Key Actions**:
- Cleaned and engineered features (e.g., `retained` flag, date parsing)
- Conducted EDA on city, phone, and ride behavior
- Trained a **Random Forest classifier** with 0.82 ROC AUC
- Evaluated performance using accuracy, F1 score, and precision/recall
- Interpreted top features influencing retention (e.g., average trip distance, weekday usage)

 **Conclusion**: Weekday usage and long-distance trips are strong predictors of retention. These insights can guide marketing and user engagement strategies.

---

##  Tools & Libraries

- Python (pandas, numpy, seaborn, matplotlib)
- Scikit-learn (Random Forest, metrics, train_test_split)
- Jupyter Notebook

---
##  Note

All data is simulated and used for educational purposes. Dataset files were provided as part of the take-home challenge and should be deleted after submission.

