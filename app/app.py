import os
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "daily_regional_demand.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import forecast_seasonal_naive
from src.capacity import create_capacity_plan


def create_history_forecast_chart(history: pd.DataFrame, plan: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for data, column, label, dash in [
        (history.tail(60), "incoming_requests", "Observed arrivals", "solid"),
        (plan, "predicted_requests", "Forecast arrivals", "dash"),
        (plan, "daily_capacity", "Expected processing capacity", "dot"),
    ]:
        figure.add_scatter(x=data["date"], y=data[column], mode="lines", name=label, line={"dash": dash})
    figure.update_layout(
        title="Daily arrivals and expected processing capacity",
        xaxis_title="Date",
        yaxis_title="Requests per day",
        hovermode="x unified",
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="Workforce Capacity Planning", page_icon="📊", layout="wide")
    st.title("Workforce Demand Forecasting & Capacity Planning")
    st.caption(
        "Synthetic demo using weekly seasonal-naïve forecasts. "
        "Capacity estimates cover incoming requests and assume one interchangeable staff pool. "
        "Existing backlog and service levels are outside this calculation."
    )

    history_path = Path(os.environ.get("WORKFORCE_HISTORY_PATH", DEFAULT_HISTORY_PATH))
    try:
        history = pd.read_csv(history_path, parse_dates=["date"])
        required = {"date", "region", "incoming_requests"}
        if history.empty or not required.issubset(history.columns):
            raise ValueError("Expected date, region, and incoming_requests in the prepared history.")
    except (OSError, ValueError) as error:
        st.error(f"Could not load prepared history. Run the preparation command in the README. {error}")
        return

    st.sidebar.header("Scenario planning")
    region = st.sidebar.selectbox("Region", options=sorted(history["region"].unique()))
    horizon = st.sidebar.slider("Planning horizon (days)", min_value=7, max_value=30, value=14)
    scheduled_staff = st.sidebar.number_input("Scheduled staff", min_value=0, max_value=100, value=9)
    absence_percent = st.sidebar.slider("Expected staff absence (%)", min_value=0, max_value=100, value=7)
    productivity = st.sidebar.number_input(
        "Requests per available person-day", min_value=0.0, max_value=20.0, value=4.0, step=0.25
    )

    try:
        forecast = forecast_seasonal_naive(history, region, horizon)
        plan = create_capacity_plan(forecast, scheduled_staff, absence_percent / 100, productivity)
    except ValueError as error:
        st.error(str(error))
        return

    origin = forecast["forecast_origin"].iloc[0]
    first_day, last_day = plan["date"].iloc[[0, -1]]
    st.caption(
        f"Forecast origin **{origin:%Y-%m-%d}** · Planning horizon **{horizon} days** "
        f"({first_day:%Y-%m-%d} to {last_day:%Y-%m-%d})"
    )
    st.caption(
        f"Scenario assumes **{scheduled_staff} scheduled staff**, **{absence_percent}% absence**, "
        f"and **{productivity:g} requests per available person-day** on each day."
    )
    daily_capacity = plan["daily_capacity"].iloc[0]
    peak_headcount = plan["required_scheduled_staff"].max()
    forecast_metric, available_metric, capacity_metric, headcount_metric = st.columns(4)
    forecast_metric.metric("Forecast arrivals", f"{plan['predicted_requests'].sum():,.0f}")
    available_metric.metric("Expected available staff", f"{plan['expected_available_staff'].iloc[0]:,.2f}")
    capacity_metric.metric("Expected capacity per day", f"{daily_capacity:,.2f}")
    headcount_metric.metric(
        "Peak required scheduled staff",
        "Infeasible" if plan["required_scheduled_staff"].isna().any() else f"{peak_headcount:,}",
    )
    overload_days = int(plan["capacity_gap"].lt(0).sum())
    if overload_days:
        st.warning(f"Forecast arrivals exceed expected capacity on {overload_days} of {horizon} days.")
    if plan["required_scheduled_staff"].isna().any():
        st.info("No finite headcount can cover positive arrivals at the selected absence and productivity.")

    region_history = history.loc[history["region"].eq(region)].sort_values("date")
    st.plotly_chart(create_history_forecast_chart(region_history, plan), width="stretch")
    st.subheader("Daily capacity plan")
    columns = {
        "date": "Date",
        "predicted_requests": "Forecast arrivals",
        "daily_capacity": "Expected processing capacity",
        "capacity_gap": "Capacity minus arrivals",
        "required_scheduled_staff": "Required scheduled staff",
        "additional_scheduled_staff": "Additional scheduled staff",
    }
    table = plan[list(columns)].rename(columns=columns)
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(table.round(2), width="stretch", hide_index=True)
    st.download_button(
        "Export workforce plan",
        data=plan.to_csv(index=False).encode("utf-8"),
        file_name=f"{region.lower().replace(' ', '_')}_capacity_plan.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
