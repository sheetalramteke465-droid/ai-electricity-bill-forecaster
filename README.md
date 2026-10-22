# AI Electricity Bill Forecaster

A practical AI/ML mini project that forecasts the next monthly electricity bill and compares hypothetical appliance-use changes.

## Features
- Monthly bill and usage charts
- Random Forest regression forecast
- Chronological train/test evaluation with MAE, RMSE and R²
- Appliance-use scenario simulator with estimated energy and rupee savings
- Optional CSV upload for custom monthly data
- Transparent limitations and tariff assumptions

## Important dataset note
`data/sample_electricity_bills.csv` is **synthetic demonstration data**. It is not actual customer or utility data. Replace it with real monthly data for meaningful household-specific forecasts.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

## Custom CSV format
Required columns:
- `month` — parseable date, preferably YYYY-MM-DD
- `kwh` — monthly energy use in kWh
- `bill_inr` — monthly bill in Indian rupees

Optional columns:
- `average_temperature_c`
- `household_size`

Use at least 15 monthly records for a basic demonstration; more history is preferable.

## Model approach
A Random Forest regressor uses month-of-year cyclical features, time index, lagged bills/usage and a rolling usage average. Current-month bill and usage are excluded from predictors to avoid target leakage. Data are split chronologically (earlier rows for training, later rows for testing). Metrics are reported on held-out months.

## Appliance scenario assumptions
Estimated saved kWh = appliance power (kW) × reduced hours/day × days/month.
Estimated savings = saved kWh × assumed marginal energy rate (₹/kWh).
This is a simplified estimate and does not model all tariff slabs, taxes, fixed charges, appliance duty cycles or behavioral changes. Savings are not guaranteed.

## Suggested GitHub repository name
`ai-electricity-bill-forecaster`

## Suggested repository contents
- `app.py`
- `data/sample_electricity_bills.csv`
- `requirements.txt`
- `README.md`
- `PROJECT_REPORT.md`
- `.gitignore`
