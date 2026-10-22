# Project Report: AI Electricity Bill Forecaster

## 1. Abstract
The AI Electricity Bill Forecaster is an educational AI/ML application that estimates a household's next monthly electricity bill using historical energy usage, previous bills and seasonal information. It also includes an appliance-use scenario simulator that estimates how hypothetical changes in appliance use could affect energy consumption and cost. The project demonstrates regression, time-aware validation, data visualization and practical model communication.

## 2. Problem Statement
Households may find it difficult to anticipate their electricity costs or understand which usage changes could have the greatest impact. This project provides a data-driven estimate and a transparent calculator for exploring possible reductions.

## 3. Objectives
- Explore historical monthly electricity usage and bills.
- Build a regression model for next-month bill forecasting.
- Evaluate predictions with MAE, RMSE and R² using a chronological test split.
- Model seasonal patterns and recent history.
- Estimate potential savings under appliance-use scenarios.
- Present results in an accessible interactive dashboard.

## 4. Technologies
Python, pandas, NumPy, scikit-learn, Streamlit and Plotly.

## 5. Dataset
The included CSV contains 48 synthetic monthly records, with month, kWh usage, bill in INR, average temperature and household size. It is generated for demonstration and reproducibility. It must not be represented as real household or utility data. For a real-world study, use permissioned and anonymized monthly records and document the tariff structure.

## 6. Methodology
1. Load and sort records by month.
2. Create month-of-year sine/cosine features to represent seasonality.
3. Create lagged usage and bill features and a rolling usage average. Exclude current-month bill and usage from predictors to prevent target leakage.
4. Remove rows that lack required lag history.
5. Split chronologically, using earlier rows for training and later rows for testing.
6. Train a Random Forest regressor.
7. Evaluate on the held-out period with Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R².
8. Fit the model on the available historical training portion and forecast the next month using latest known values as a baseline.
9. Calculate appliance scenario savings using wattage, hours, days and a user-specified marginal rate.

## 7. Appliance Scenario Formula
Energy reduction (kWh) = Power (W) / 1000 × reduced hours per day × days per month.

Estimated savings (₹) = Energy reduction (kWh) × assumed marginal rate (₹/kWh).

This is a simplified scenario calculator. Actual bills may include fixed charges, slab tariffs, taxes, subsidies and time-of-use rates. Appliance power can also vary. Therefore the result is an estimate rather than a guarantee.

## 8. Evaluation
The dashboard reports MAE, RMSE and R² on held-out months. MAE expresses average absolute error in rupees; RMSE penalizes larger errors more strongly; R² describes fit relative to a mean baseline and can be negative on poor forecasts. The small synthetic dataset limits the strength of any conclusions. Report the metrics displayed by the running application rather than inventing or claiming a target accuracy.

## 9. Expected Outcomes
- An interactive monthly bill forecast.
- Visual comparison of monthly kWh and bills.
- A view of actual versus predicted held-out bills.
- An appliance scenario table showing estimated saved kWh and rupees.
- A clear explanation of model performance and assumptions.

## 10. Limitations
- Demonstration data are synthetic.
- Monthly history may be too short to capture long-term changes.
- Future usage is not known and the forecast uses latest observed values as a baseline.
- Tariff structures and appliance power variability are simplified.
- Forecasts should not be used as official billing estimates.

## 11. Future Scope
- Add weather forecasts and local tariff slabs.
- Compare Random Forest with linear regression and gradient boosting.
- Add prediction intervals or uncertainty estimates.
- Support daily smart-meter data and more rigorous rolling-origin validation.
- Add downloadable prediction and scenario reports.

## 12. Conclusion
The project demonstrates a complete introductory ML workflow, from data preparation and feature engineering to model evaluation and a user-facing dashboard. Its distinctive feature is the combination of forecasting and explainable, assumption-based appliance savings scenarios.
