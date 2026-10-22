import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AI Electricity Bill Forecaster", page_icon="⚡", layout="wide")
st.title("⚡ AI Electricity Bill Forecaster")
st.caption("Forecast monthly electricity costs and compare hypothetical appliance-use changes.")
st.info("The included dataset is synthetic demonstration data, not real utility records. Forecasts and savings are estimates, not guarantees.")

DATA_PATH = Path(__file__).parent / "data" / "sample_electricity_bills.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    return df.sort_values("month").reset_index(drop=True)

def features(df):
    x = df.copy()
    x["month_num"] = x["month"].dt.month
    x["month_sin"] = np.sin(2*np.pi*x["month_num"]/12)
    x["month_cos"] = np.cos(2*np.pi*x["month_num"]/12)
    x["time_index"] = np.arange(len(x))
    x["kwh_lag1"] = x["kwh"].shift(1)
    x["bill_lag1"] = x["bill_inr"].shift(1)
    x["kwh_lag2"] = x["kwh"].shift(2)
    x["bill_lag2"] = x["bill_inr"].shift(2)
    x["kwh_rolling3"] = x["kwh"].shift(1).rolling(3).mean()
    return x

df = load_data()
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Use your own CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        custom = pd.read_csv(uploaded, parse_dates=["month"])
        required = {"month", "kwh", "bill_inr"}
        if not required.issubset(custom.columns):
            st.sidebar.error("CSV must include: month, kwh, bill_inr.")
        else:
            if "average_temperature_c" not in custom: custom["average_temperature_c"] = 28
            if "household_size" not in custom: custom["household_size"] = 4
            df = custom.sort_values("month").reset_index(drop=True)
            st.sidebar.success("Custom CSV loaded for this session.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

feat = features(df)
# Exclude current-month kWh and bill from predictors to avoid target leakage.
# The forecast uses only historical lags plus calendar/household context.
feature_cols = ["average_temperature_c", "household_size", "month_sin", "month_cos",
                "time_index", "kwh_lag1", "bill_lag1", "kwh_lag2", "bill_lag2",
                "kwh_rolling3"]
model_df = feat.dropna(subset=feature_cols).copy()

if len(model_df) < 12:
    st.error("Please provide at least 15 monthly records for a useful demonstration.")
    st.stop()

# Chronological split; no random shuffling
split = max(8, int(len(model_df)*0.8))
train, test = model_df.iloc[:split], model_df.iloc[split:]
model = RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=2, random_state=42)
model.fit(train[feature_cols], train["bill_inr"])
pred_test = model.predict(test[feature_cols])
mae = mean_absolute_error(test["bill_inr"], pred_test)
rmse = np.sqrt(mean_squared_error(test["bill_inr"], pred_test))
r2 = r2_score(test["bill_inr"], pred_test) if len(test) > 1 else float("nan")

latest = df.iloc[-1]
next_date = latest["month"] + pd.offsets.MonthBegin(1)
next_row = pd.DataFrame([{
    "month": next_date, "kwh": latest["kwh"], "bill_inr": latest["bill_inr"],
    "average_temperature_c": latest.get("average_temperature_c", 28),
    "household_size": latest.get("household_size", 4)
}])
history_plus = pd.concat([df, next_row], ignore_index=True)
next_feat = features(history_plus).iloc[[-1]].copy()
next_feat["time_index"] = len(df)-1
forecast = float(model.predict(next_feat[feature_cols])[0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast next bill", f"₹{forecast:,.0f}")
c2.metric("Latest recorded bill", f"₹{latest['bill_inr']:,.0f}")
c3.metric("Test MAE", f"₹{mae:,.0f}")
c4.metric("Test RMSE", f"₹{rmse:,.0f}")

st.subheader("Historical usage and bill")
left, right = st.columns(2)
with left:
    st.plotly_chart(px.line(df, x="month", y="kwh", markers=True, title="Monthly electricity use (kWh)"), use_container_width=True)
with right:
    st.plotly_chart(px.line(df, x="month", y="bill_inr", markers=True, title="Monthly electricity bill (₹)"), use_container_width=True)

st.subheader("Forecast evaluation")
eval_df = test[["month", "bill_inr"]].copy()
eval_df["predicted_bill"] = pred_test
eval_long = eval_df.melt(id_vars="month", var_name="series", value_name="rupees")
st.plotly_chart(px.line(eval_long, x="month", y="rupees", color="series", markers=True,
                        title="Actual vs predicted bills on held-out months"), use_container_width=True)
st.caption(f"Test-set R²: {r2:.3f}. Metrics can be unstable with small datasets. A chronological split better reflects forecasting than random splitting.")

st.subheader("Appliance usage scenario simulator")
st.write("Adjust estimated usage reductions. The simulator converts saved kWh to rupees using your assumed marginal energy rate.")
rate = st.number_input("Energy rate (₹ per kWh)", min_value=0.0, max_value=100.0, value=7.2, step=0.5)
st.caption("Illustrative rate only. Actual tariffs may include slabs, fixed charges, taxes, subsidies and time-of-use pricing.")
appliances = [
    ("Air conditioner", 1500, 2.0, 30),
    ("Water heater / geyser", 2000, 0.5, 30),
    ("Refrigerator", 150, 0.0, 30),
    ("Lighting", 60, 1.0, 30),
    ("Washing machine", 500, 0.5, 12),
]
scenario_rows = []
total_saving_kwh = 0
for name, watts, default_hours, days in appliances:
    if name == "Refrigerator":
        reduction_hours = st.slider(f"{name}: reduce equivalent run-time (hours/day)", 0.0, 1.0, 0.0, 0.1, key=name)
    else:
        reduction_hours = st.slider(f"{name}: reduce usage (hours/day)", 0.0, max(0.5, default_hours*2), min(default_hours, max(0.5, default_hours*2)), 0.1, key=name)
    saved = watts/1000 * reduction_hours * days
    total_saving_kwh += saved
    scenario_rows.append({"Appliance": name, "Power (W)": watts, "Reduced hours/day": reduction_hours,
                          "Days/month": days, "Estimated saved kWh": round(saved, 2),
                          "Estimated savings (₹)": round(saved*rate, 2)})
scenario_df = pd.DataFrame(scenario_rows)
estimated_savings = total_saving_kwh * rate
st.dataframe(scenario_df, use_container_width=True, hide_index=True)
s1, s2, s3 = st.columns(3)
s1.metric("Potential energy reduction", f"{total_saving_kwh:.1f} kWh/month")
s2.metric("Estimated monthly savings", f"₹{estimated_savings:,.0f}")
s3.metric("Illustrative adjusted bill", f"₹{max(0, forecast-estimated_savings):,.0f}",
          delta=f"-₹{estimated_savings:,.0f}")
st.warning("Scenario savings are arithmetic estimates, not ML predictions. They assume the listed power rating and marginal rate, and may overstate savings if appliance power varies or tariff slabs/fixed charges apply.")

with st.expander("View dataset"):
    st.dataframe(df, use_container_width=True)
with st.expander("Methodology and limitations"):
    st.markdown("""
    - Model: Random Forest regression using month seasonality, time index, current-record values and lagged usage/bill features.
    - Validation: chronological 80/20 split; no random shuffling.
    - Metrics: MAE, RMSE and R² on held-out months.
    - The one-month forecast uses the latest known values as a baseline because future household usage is not known.
    - The included data are synthetic and intended for demonstrating the workflow. Replace them with real monthly records before drawing household-specific conclusions.
    - This is an educational mini project, not a utility billing tool or financial guarantee.
    """)
