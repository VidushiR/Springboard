# Capstone Two — Click-Through Rate (CTR) Prediction Project

This project was completed as part of the Springboard Data Science Career Track.  
The objective is to predict the likelihood that a user clicks on an online advertisement, using machine learning models trained on simulated real-world ad interaction data.

---

##  Problem Overview

- **Problem Type**: Binary Classification (CTR Prediction)
- **Target Variable**: Click (1 if user clicked, 0 otherwise)
- **Business Goal**: Optimize online ad placements to reduce wasted impressions and increase ad engagement.

---

## Repository Structure

CapstoneTwo_CTRprediction/
│
├── data/
│ └── processed/
│ ├── cleaned_ctr_prediction_data.csv
│ ├── cleaned_no_outliers.csv
│ ├── capped_outliers.csv
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
│
├── notebooks/
│ ├── data_wrangling.ipynb
│ ├── EDAforCTRPrediction.ipynb
│ ├── CTR-Pre-processingAndTrainingDataDevelopment.ipynb
│ └── CTRPredictionModeling.ipynb
│
├── reports/
│ └── model_evaluation_summary.csv
│
├── models/
├── scripts/
├── config/
├── README.md
├── requirements.txt
└── .gitignore



---

## 1. Dataset

- Contains user, ad, and contextual features.
- Includes ad impression logs, user demographics, ad campaign features, and time-based contextual variables.
- Raw data cleaned and preprocessed into multiple versions: capped outliers, cleaned without outliers, and fully processed train/test sets.

---

## 2. Data Preprocessing

- Missing values handled.
- Outlier detection and capping performed.
- Categorical variables encoded.
- Numerical features scaled.
- Data split into training and testing sets.

---

## 3. Exploratory Data Analysis (EDA)

- Visualized CTR distribution across features such as:
  - Campaign IDs
  - Ad positions
  - Device types
  - Time of day
- Identified key feature correlations influencing click behavior.

---

## 4. Modeling & Machine Learning

Multiple models evaluated:

- Logistic Regression (Baseline)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Gradient Boosting (XGBoost, LightGBM)

### Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss

Final model evaluation results are stored in:  
[`model_evaluation_summary.csv`](reports/model_evaluation_summary.csv)

---

## 5. Results & Key Takeaways

- Gradient Boosting models (XGBoost, LightGBM) provided the best performance.
- Outlier removal and strong feature engineering improved predictive power.
- The final models generalized well across test data after hyperparameter tuning.

---

## 6. Challenges & Business Considerations

- Data imbalance between click and no-click records.
- Cold start problem for new users and ads.
- Importance of minimizing false positives to optimize ad spend.
- Model interpretability and deployment considerations for real-world business use.

---

## 7. Future Work

- Apply additional deep learning models.
- Deploy models with real-time scoring APIs.
- Add advanced feature interactions and time-decay features.
- Implement online learning for continuously updated models.

---

## 8. Tools & Technologies

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- XGBoost, LightGBM
- Jupyter Notebook

---

## 9. Acknowledgements

- Springboard Data Science Career Track
- Kaggle community datasets
- Mentors and advisors for their valuable feedback

---

*This project demonstrates my ability to apply end-to-end machine learning pipelines for real-world business problems using both statistical learning and modern ML techniques.*

