from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast import (
    forecast_region,
    load_historical_data,
)


st.set_page_config(
    page_title="Workforce Capacity Planning",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_historical_data() -> pd.DataFrame:
    """Load and cache historical workforce data."""
    return load_historical_data()


@st.cache_data
def generate_forecast(
    region: str,
    forecast_days: int,
    available_staff: int,
    absence_rate: float,
) -> pd.DataFrame:
    """Generate and cache the regional demand forecast."""
    return forecast_region(
        region=region,
        forecast_days=forecast_days,
        available_staff=available_staff,
        staff_absence_rate=absence_rate,
    )


def create_capacity_plan(
    forecast: pd.DataFrame,
    available_staff: int,
    productivity_per_staff: float,
) -> pd.DataFrame:
    """Compare forecast demand with daily processing capacity."""
    plan = forecast.copy()

    plan["daily_capacity"] = (
        available_staff * productivity_per_staff
    )

    plan["capacity_gap"] = (
        plan["daily_capacity"]
        - plan["predicted_requests"]
    )

    plan["utilization_percentage"] = (
        plan["predicted_requests"]
        / plan["daily_capacity"]
        * 100
    )

    plan["required_staff"] = (
        plan["predicted_requests"]
        / productivity_per_staff
    ).apply(lambda value: int(-(-value // 1)))

    plan["additional_staff_required"] = (
        plan["required_staff"] - available_staff
    ).clip(lower=0)

    plan["capacity_status"] = pd.cut(
        plan["utilization_percentage"],
        bins=[
            -float("inf"),
            80,
            100,
            float("inf"),
        ],
        labels=[
            "Healthy",
            "Near Capacity",
            "Capacity Risk",
        ],
    )

    return plan


def create_forecast_chart(
    capacity_plan: pd.DataFrame,
) -> go.Figure:
    """Create demand-versus-capacity visualization."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=capacity_plan["date"],
            y=capacity_plan["predicted_requests"],
            mode="lines+markers",
            name="Predicted Demand",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=capacity_plan["date"],
            y=capacity_plan["daily_capacity"],
            mode="lines",
            name="Available Capacity",
            line={"dash": "dash"},
        )
    )

    figure.update_layout(
        title="Forecast Demand vs Available Capacity",
        xaxis_title="Date",
        yaxis_title="Requests",
        hovermode="x unified",
        height=480,
    )

    return figure


def create_staffing_chart(
    capacity_plan: pd.DataFrame,
    available_staff: int,
) -> go.Figure:
    """Create required-versus-available staffing chart."""
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=capacity_plan["date"],
            y=capacity_plan["required_staff"],
            name="Required Staff",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=capacity_plan["date"],
            y=[available_staff] * len(capacity_plan),
            mode="lines",
            name="Available Staff",
            line={"dash": "dash"},
        )
    )

    figure.update_layout(
        title="Required Staff vs Available Staff",
        xaxis_title="Date",
        yaxis_title="Staff",
        hovermode="x unified",
        height=430,
    )

    return figure


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title(
        "Workforce Demand Forecasting & Capacity Planning"
    )

    st.caption(
        """
        Predict future operational demand, evaluate workforce
        capacity, and identify staffing risks using a tuned
        XGBoost forecasting model.
        """
    )

    historical_data = get_historical_data()

    regions = sorted(
        historical_data["region"].unique()
    )

    st.sidebar.header("Scenario Planning")

    region = st.sidebar.selectbox(
        "Region",
        options=regions,
    )

    region_history = historical_data[
        historical_data["region"] == region
    ].sort_values("date")

    default_staff = int(
        round(
            region_history
            .tail(30)["available_staff"]
            .median()
        )
    )

    default_absence_rate = float(
        region_history
        .tail(30)["staff_absence_rate"]
        .mean()
    )

    forecast_days = st.sidebar.slider(
        "Forecast horizon",
        min_value=7,
        max_value=30,
        value=14,
        step=1,
    )

    available_staff = st.sidebar.number_input(
        "Available staff",
        min_value=1,
        max_value=100,
        value=default_staff,
        step=1,
    )

    productivity_per_staff = st.sidebar.number_input(
        "Requests completed per staff member per day",
        min_value=0.5,
        max_value=20.0,
        value=4.0,
        step=0.25,
    )

    absence_percentage = st.sidebar.slider(
        "Expected staff absence (%)",
        min_value=0,
        max_value=30,
        value=int(round(default_absence_rate * 100)),
        step=1,
    )

    absence_rate = absence_percentage / 100

    forecast = generate_forecast(
        region=region,
        forecast_days=forecast_days,
        available_staff=int(available_staff),
        absence_rate=absence_rate,
    )

    capacity_plan = create_capacity_plan(
        forecast=forecast,
        available_staff=int(available_staff),
        productivity_per_staff=float(
            productivity_per_staff
        ),
    )

    total_demand = capacity_plan[
        "predicted_requests"
    ].sum()

    total_capacity = capacity_plan[
        "daily_capacity"
    ].sum()

    average_utilization = capacity_plan[
        "utilization_percentage"
    ].mean()

    risk_days = int(
        (
            capacity_plan["capacity_status"]
            == "Capacity Risk"
        ).sum()
    )

    recent_historical_demand = (
        region_history
        .tail(forecast_days)["incoming_requests"]
        .sum()
    )

    if recent_historical_demand > 0:
        demand_change_percentage = (
            (
                total_demand
                - recent_historical_demand
            )
            / recent_historical_demand
            * 100
        )
    else:
        demand_change_percentage = 0.0

    total_shortage = max(
        0.0,
        total_demand - total_capacity,
    )

    total_surplus = max(
        0.0,
        total_capacity - total_demand,
    )

    peak_required_staff = int(
        capacity_plan["required_staff"].max()
    )

    additional_peak_staff = max(
        0,
        peak_required_staff - int(available_staff),
    )

    required_productivity = (
        total_demand
        / (
            int(available_staff)
            * forecast_days
        )
    )

    highest_demand_row = capacity_plan.loc[
        capacity_plan[
            "predicted_requests"
        ].idxmax()
    ]

    highest_demand_date = highest_demand_row[
        "date"
    ].strftime("%d %b %Y")

    highest_daily_demand = highest_demand_row[
        "predicted_requests"
    ]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        label="Forecasted Requests",
        value=f"{total_demand:,.0f}",
        delta=(
            f"{demand_change_percentage:+.1f}% vs previous {forecast_days} days"
        ),
    )

    metric_2.metric(
        label="Forecast Capacity",
        value=f"{total_capacity:,.0f}",
        delta=(
            f"{total_capacity - total_demand:+,.0f} "
            "request balance"
        ),
        delta_color=(
            "normal"
            if total_capacity >= total_demand
            else "inverse"
        ),
    )

    metric_3.metric(
        label="Average Utilization",
        value=f"{average_utilization:.1f}%",
        delta=(
            "Above target"
            if average_utilization > 100
            else (
                "Near capacity"
                if average_utilization > 80
                else "Healthy"
            )
        ),
        delta_color=(
            "inverse"
            if average_utilization > 100
            else "normal"
        ),
    )

    metric_4.metric(
        label="Capacity Risk",
        value=f"{risk_days} days",
        delta=(
            f"Peak requirement: "
            f"{peak_required_staff} staff"
        ),
        delta_color=(
            "inverse"
            if risk_days > 0
            else "normal"
        ),
    )

    if total_demand > total_capacity:
        st.error(
            f"""
            **HIGH CAPACITY RISK**

            Forecast demand exceeds available capacity by
            approximately **{total_shortage:,.0f} requests**
            across the selected {forecast_days}-day period.
            """
        )

    elif risk_days > 0:
        st.warning(
            f"""
            **LOCALIZED CAPACITY RISK**

            Total capacity is sufficient for the complete forecast
            period, but demand exceeds daily capacity on
            **{risk_days} days**.
            """
        )

    else:
        st.success(
            f"""
            **HEALTHY CAPACITY POSITION**

            Available capacity exceeds forecast demand by
            approximately **{total_surplus:,.0f} requests**,
            with no projected capacity-risk days.
            """
        )

    st.subheader("Executive Recommendation")

    recommendation_1, recommendation_2 = st.columns(2)

    with recommendation_1:
        if additional_peak_staff > 0:
            st.info(
                f"""
                **Staffing scenario**

                Increase available staffing from
                **{int(available_staff)}** to approximately
                **{peak_required_staff} staff members** during
                peak-demand days.

                This represents up to
                **{additional_peak_staff} additional staff
                members**.
                """
            )
        else:
            st.info(
                f"""
                **Staffing scenario**

                Current staffing of
                **{int(available_staff)} staff members**
                is sufficient for the forecast period.
                """
            )

    with recommendation_2:
        if required_productivity > productivity_per_staff:
            productivity_increase = (
                (
                    required_productivity
                    / productivity_per_staff
                )
                - 1
            ) * 100

            st.info(
                f"""
                **Productivity scenario**

                Without increasing headcount, average productivity
                would need to rise from
                **{productivity_per_staff:.2f}** to approximately
                **{required_productivity:.2f} requests per staff
                member per day**.

                Required improvement:
                **{productivity_increase:.1f}%**.
                """
            )
        else:
            st.info(
                f"""
                **Productivity scenario**

                The current productivity assumption of
                **{productivity_per_staff:.2f} requests per staff
                member per day** is sufficient for total forecast
                demand.
                """
            )

    st.caption(
        f"Peak demand is forecast for {highest_demand_date}, "
        f"with approximately {highest_daily_demand:.0f} "
        "incoming requests."
    )

    st.plotly_chart(
        create_history_forecast_chart(
            region_history=region_history,
            capacity_plan=capacity_plan,
            history_days=60,
        ),
        width="stretch",
    )

    st.plotly_chart(
        create_staffing_chart(
            capacity_plan,
            int(available_staff),
        ),
        width="stretch",
    )

    st.subheader("Daily capacity plan")

    display_columns = [
        "date",
        "predicted_requests",
        "daily_capacity",
        "capacity_gap",
        "utilization_percentage",
        "required_staff",
        "additional_staff_required",
        "capacity_status",
    ]

    display_table = capacity_plan[
        display_columns
    ].copy()

    display_table["date"] = (
        display_table["date"]
        .dt.strftime("%Y-%m-%d")
    )

    numeric_columns = [
        "predicted_requests",
        "daily_capacity",
        "capacity_gap",
        "utilization_percentage",
    ]

    display_table[numeric_columns] = (
        display_table[numeric_columns].round(1)
    )

    display_table = display_table.rename(
        columns={
            "date": "Date",
            "predicted_requests": "Forecast Requests",
            "daily_capacity": "Daily Capacity",
            "capacity_gap": "Capacity Gap",
            "utilization_percentage": "Utilization %",
            "required_staff": "Required Staff",
            "additional_staff_required": "Additional Staff",
            "capacity_status": "Status",
        }
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )

    download_data = capacity_plan.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Export Workforce Plan",
        data=download_data,
        file_name=(
            f"{region.lower().replace(' ', '_')}"
            f"_capacity_plan.csv"
        ),
        mime="text/csv",
    )

    with st.expander("Model information"):
        st.write(
            """
            **Selected model:** Tuned XGBoost regressor

            **Holdout performance**

            - MAE: 5.78 requests
            - RMSE: 8.82 requests
            - WAPE: 15.84%
            - Forecast bias: 0.54 requests
            - Training rows: 2,928
            - Test rows: 270
            - Validation: expanding-window time-series CV

            Hyperparameters were selected using expanding-window,
            date-based cross-validation. The latest 90 days were
            retained as an untouched test period.
            """
        )
def create_history_forecast_chart(
    region_history: pd.DataFrame,
    capacity_plan: pd.DataFrame,
    history_days: int = 60,
) -> go.Figure:
    """Show recent historical demand together with the future forecast."""
    history = (
        region_history
        .sort_values("date")
        .tail(history_days)
        .copy()
    )

    forecast_start = capacity_plan["date"].min()

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["incoming_requests"],
            mode="lines",
            name="Historical Demand",
            line={"width": 2},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=capacity_plan["date"],
            y=capacity_plan["predicted_requests"],
            mode="lines+markers",
            name="Forecast Demand",
            line={
                "dash": "dash",
                "width": 3,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=capacity_plan["date"],
            y=capacity_plan["daily_capacity"],
            mode="lines",
            name="Available Capacity",
            line={
                "dash": "dot",
                "width": 2,
            },
        )
    )

    figure.add_vline(
        x=forecast_start.timestamp() * 1000,
        line_dash="dash",
        annotation_text="Forecast starts",
        annotation_position="top",
    )

    figure.update_layout(
        title="Historical Demand and Future Forecast",
        xaxis_title="Date",
        yaxis_title="Incoming Requests",
        hovermode="x unified",
        height=500,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure



if __name__ == "__main__":
    main()