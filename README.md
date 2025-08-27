# Retail Price Predictor

## 1. Introduction
**Retail Price Predictor** is a machine learning project designed to predict the optimal discounted price of products based on historical sales data, product attributes, and user reviews. The system leverages structured numerical and categorical data to forecast prices intelligently, helping retailers and e-commerce platforms optimize pricing strategies, increase revenue, and remain competitive in a dynamic market.

---

## 2. Problem Statement
Pricing plays a critical role in influencing consumer purchase decisions and business revenue. Retailers often struggle to set the optimal price that maximizes both sales and profit. Traditional manual pricing strategies are time-consuming and prone to human bias. The **Retail Price Predictor** project aims to automate and optimize product pricing using machine learning, ensuring accurate, data-driven price predictions.

---

## 3. Project Goal
The main goal of this project is to build a machine learning model that can:
1. Predict discounted prices for retail products based on their characteristics.
2. Handle both categorical and numerical features effectively.
3. Allow businesses to make informed pricing decisions automatically.
4. Provide a pipeline that includes preprocessing, model training, and prediction, ensuring end-to-end functionality.

---

## 4. Tech Stack
- **Python** – Main programming language.
- **Pandas & NumPy** – Data manipulation and numerical computations.
- **Scikit-learn** – Machine learning algorithms and preprocessing.
- **Joblib** – Save and load trained models and preprocessing objects.
- **Jupyter Notebook** – Interactive development and experimentation environment.

---

## 5. Project Workflow and Files
The project is organized into five main scripts/notebooks:

### 5.1 01_data_preprocessing.ipynb
- **Purpose:** Clean and preprocess raw retail data for modeling.
- **Steps:**
  1. Load raw product sales data (CSV format).
  2. Extract numerical columns such as `actual_price_num`, `discounted_price_num`, `rating_num`, etc.
  3. Extract categorical columns such as `category`, `product_name`, `user_id`, etc.
  4. Handle missing values using imputation.
  5. Feature engineering to compute additional metrics like `discount_pct_calc`, `discount_amount`, and `popularity_score`.
- **Outcome:** Cleaned and processed dataset ready for model training.

### 5.2 02_feature_engineering.ipynb
- **Purpose:** Transform features for machine learning.
- **Steps:**
  1. Split categorical and numerical features.
  2. Encode categorical features using one-hot encoding.
  3. Standardize/normalize numerical features if required.
  4. Ensure consistent feature naming for training and inference.
- **Outcome:** Preprocessed dataset ready to feed into regression models.

### 5.3 03_train_test_split.ipynb
- **Purpose:** Split the dataset for training and evaluation.
- **Steps:**
  1. Use `train_test_split` to create an 80-20 train-test split.
  2. Ensure stratification or shuffling for balanced distribution.
  3. Separate features (`X`) and target (`y` = `discounted_price_num`).
- **Outcome:** `X_train`, `X_test`, `y_train`, `y_test` datasets for model development.

### 5.4 04_model.ipynb
- **Purpose:** Train and evaluate the machine learning model.
- **Steps:**
  1. Impute missing values for categorical and numerical features.
  2. One-hot encode categorical variables.
  3. Combine numerical and encoded categorical features.
  4. Train **Random Forest Regressor** on the processed dataset.
  5. Evaluate performance using **MAE, RMSE, R²**.
  6. Save trained model and preprocessing objects using Joblib.
- **Outcome:** Trained Random Forest model and preprocessing pipeline saved for inference.

### 5.5 05_predict.ipynb
- **Purpose:** Load the saved model and predict prices for new products.
- **Steps:**
  1. Load new product data for prediction.
  2. Apply saved preprocessing objects (imputers, encoder) on new data.
  3. Generate predictions using the saved Random Forest model.
  4. Save predicted prices to `outputs/predicted_prices.csv`.
- **Outcome:** Predicted discounted prices for new products, ready for business use.

---

## 6. Model Details
- **Algorithm:** Random Forest Regressor.
- **Why Random Forest:** Handles mixed numerical and categorical data, robust to outliers, performs well with non-linear relationships.
- **Target Variable:** `discounted_price_num` – the discounted selling price of products.
- **Evaluation Metrics:**
  - **MAE (Mean Absolute Error):** Measures average absolute prediction error.
  - **RMSE (Root Mean Squared Error):** Penalizes large prediction errors.
  - **R² Score:** Indicates the proportion of variance explained by the model.

---

## 7. Output
- Predicted discounted prices for new products.
- CSV file `predicted_prices.csv` in the `outputs/` folder with:
  - Product identifiers
  - Original attributes
  - Predicted discounted price

---

## 8. Key Takeaways
1. End-to-end machine learning pipeline: preprocessing → feature engineering → training → prediction.
2. Automatic handling of missing values and categorical encoding ensures consistency.
3. Model predicts optimal discounted prices, aiding retail pricing decisions.
4. Scalable to new product data without retraining.
5. Demonstrates practical use of machine learning in retail and e-commerce analytics.

---

## 9. How to Run
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Run notebooks in sequence:
   - `01_data_preprocessing.ipynb`
   - `02_feature_engineering.ipynb`
   - `03_train_test_split.ipynb`
   - `04_model.ipynb`
   - `05_predict.ipynb`
4. Check the `outputs/` folder for predicted prices CSV file.

---

## 10. Future Improvements
- Integrate more features like seasonality, competitor prices, and sales trends.
- Experiment with gradient boosting or XGBoost models for higher accuracy.
- Deploy as a web application for real-time price prediction.
- Add visualization dashboards for pricing insights.

---

**Project Name:** Retail Price Predictor  
**Goal:** Predict optimal discounted prices to support data-driven retail pricing strategies.
