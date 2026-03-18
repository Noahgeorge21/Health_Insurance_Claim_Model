# Health Insurance Claim Prediction System

This project implements a full-stack data science solution to predict health insurance premiums based on individual health markers and demographic data. It includes a comprehensive exploratory analysis, a machine learning pipeline with multiple tuned algorithms, and an interactive Streamlit web application for real-time predictions.

## 📋 Project Overview
The goal of this project is to accurately estimate medical insurance costs. By identifying key cost drivers such as BMI, smoking status, and age, the model provides actionable insights into how specific lifestyle factors impact financial risk in healthcare.

## 📊 Data Lifecycle & Exploratory Data Analysis (EDA)
The dataset contains 1,338 records with features including Age, Gender, BMI, Blood Pressure, Diabetic Status, Children, Smoking Status, and Region.

### 1. Data Cleaning & Preprocessing
* **Handling Missing Values:** Addressed null entries in age and other features to ensure model stability.
* **Deduplication:** Verified dataset integrity by removing redundant records.
* **Outlier Detection:** Identified extreme values in BMI and Blood Pressure that could skew linear relationships.

### 2. Analysis Insights
* **Primary Cost Drivers:** Smoking status showed the highest correlation with claim amounts.
* **The "Obesity-Smoker" Interaction:** EDA revealed a critical non-linear jump in costs for individuals who are both smokers and have a BMI ≥ 30.
* **Demographic Trends:** Analyzed cost distributions across different age groups and regions, noting that medical charges generally increase with age and the number of dependents.

## 🤖 Data Science Methodology
The model development followed a rigorous iterative process to move beyond baseline performance.

### Feature Engineering
* **Categorical Encoding:** Applied Label Encoding for binary features (Gender, Smoker, Diabetic) to prepare them for mathematical processing.
* **Interaction Terms:** Created the `is_obese_smoker` feature based on EDA findings to capture the synergistic effect of high BMI and smoking.
* **Feature Scaling:** Utilized `StandardScaler` to normalize numerical features, ensuring distance-based models like SVR were not biased by feature magnitude.

### Model Selection & Tuning
We evaluated and hyperparameter-tuned five distinct algorithms using `GridSearchCV` with 5-fold cross-validation:
1.  **Linear Regression:** Established a baseline for linear relationships.
2.  **Polynomial Regression (Deg 2 & 3):** Captured non-linear curvatures in health data.
3.  **Random Forest Regressor:** Leveraged ensemble learning to handle high-variance data.
4.  **Support Vector Regressor (SVR):** Utilized RBF kernels for complex spatial mapping.
5.  **XGBoost Regressor:** Optimized using gradient boosting with L1/L2 regularization.

## 🏆 Model Results & Evaluation
After extensive testing, the **XGBoost Regressor** was identified as the best-performing model.

### Why XGBoost?
* **Non-Linear Mapping:** It successfully captured the "stair-step" cost increases (like the Smoker/Obese interaction) that linear models struggled with.
* **Regularization:** Built-in L1 and L2 penalties prevented overfitting, ensuring the model generalizes well to new data.
* **Efficiency:** Optimized gradient boosting provided the highest accuracy with the lowest computational overhead.

### Evaluation Metrics
We used three primary metrics to validate the model's fitness:
* **R-Squared ($R^2$):** Measured the proportion of variance explained by the model. The final model achieved a high $R^2$ of .83, significantly outperforming the $0.74$ baseline.
* **Mean Absolute Error (MAE):** Provided the average dollar amount the prediction typically deviated from the actual claim.
* **Root Mean Squared Error (RMSE):** Penalized larger errors to ensure the model didn't make massive miscalculations on high-value claims.

## 💻 Streamlit Web Application
The project concludes with a deployment-ready web interface built with Streamlit.

* **User Input:** An intuitive sidebar and form layout allow users to input personal health metrics.
* **Dynamic Processing:** The app loads the serialized `scaler.pkl` and `label_encoders` to transform user input in real-time.
* **Instant Prediction:** By calling the `best_model.pkl`, the app provides an immediate dollar-amount estimate for insurance claims based on the trained logic.

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost streamlit joblib
    ```
2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Technologies Used
* **Languages:** Python (Pandas, NumPy)
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment:** Streamlit, Joblib