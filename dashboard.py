"""Streamlit dashboard for the Harare house price project.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/harare_listings.csv")
RNG = 42

SUBURBS = {
    "Borrowdale":       {"lat": -17.740, "lon": 31.100, "tier": "premium"},
    "Glen Lorne":       {"lat": -17.745, "lon": 31.180, "tier": "premium"},
    "Mt Pleasant":      {"lat": -17.780, "lon": 31.020, "tier": "premium"},
    "Ballantyne Park":  {"lat": -17.760, "lon": 31.080, "tier": "premium"},
    "Highlands":        {"lat": -17.790, "lon": 31.070, "tier": "upper-mid"},
    "Vainona":          {"lat": -17.780, "lon": 31.080, "tier": "upper-mid"},
    "Greendale":        {"lat": -17.790, "lon": 31.130, "tier": "upper-mid"},
    "Avondale":         {"lat": -17.790, "lon": 31.030, "tier": "upper-mid"},
    "Marlborough":      {"lat": -17.780, "lon": 30.950, "tier": "middle"},
    "Avonlea":          {"lat": -17.770, "lon": 30.990, "tier": "middle"},
    "Belvedere":        {"lat": -17.850, "lon": 31.020, "tier": "middle"},
    "Hatfield":         {"lat": -17.870, "lon": 31.090, "tier": "middle"},
    "Mabelreign":       {"lat": -17.810, "lon": 31.000, "tier": "middle"},
    "Mufakose":         {"lat": -17.910, "lon": 30.960, "tier": "township"},
    "Glen View":        {"lat": -17.930, "lon": 30.970, "tier": "township"},
    "Kambuzuma":        {"lat": -17.860, "lon": 30.960, "tier": "township"},
    "Budiriro":         {"lat": -17.940, "lon": 30.940, "tier": "township"},
}
CBD_LAT, CBD_LON = -17.831, 31.045

st.set_page_config(page_title="Harare House Prices", page_icon=":house:", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #27AE60 0%, #0F5132 100%);
        padding: 36px 32px; border-radius: 16px; color: white;
        margin: -10px 0 24px 0;
        box-shadow: 0 12px 32px rgba(39, 174, 96, 0.25);
    }
    .hero h1 { margin: 0; font-size: 38px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { margin: 8px 0 0 0; font-size: 17px; opacity: 0.92; }
    .stat {
        background: white; padding: 22px 24px; border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-top: 4px solid #27AE60; height: 100%;
    }
    .stat .label { font-size: 12px; color: #7F8C8D; text-transform: uppercase; letter-spacing: 1px; }
    .stat .value { font-size: 30px; font-weight: 800; color: #145A32; margin: 4px 0; }
    .stat .sub   { font-size: 13px; color: #95A5A6; }
    .insight {
        background: linear-gradient(180deg, #FFFFFF 0%, #E8F8EF 100%);
        border-left: 4px solid #27AE60; padding: 18px 22px; border-radius: 10px; margin: 8px 0;
    }
    .insight .head { font-size: 12px; color: #27AE60; font-weight: 700; letter-spacing: 1px; }
    .insight .body { font-size: 16px; color: #145A32; margin-top: 4px; line-height: 1.5; }
    </style>

    <div class="hero">
      <h1>:house: Harare House Prices</h1>
      <p>Map the city, see what really drives prices, price a house in any suburb.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def big_stat(label, value, sub=""):
    return f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def insight(head, body):
    return f'<div class="insight"><div class="head">{head}</div><div class="body">{body}</div></div>'


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading Harare listings...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("No data at data/harare_listings.csv - run the notebook once to generate it.")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training stacked ensemble...")
def train_model(df: pd.DataFrame):
    y = df["price_usd"]
    X = df.drop(columns=["price_usd"])
    categorical = ["suburb", "tier"]
    numeric = [c for c in X.columns if c not in categorical]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RNG)),
        ("gb", GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=RNG)),
    ]
    pipe = Pipeline([
        ("prep", pre),
        ("model", StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return pipe, metrics


df = load_data()
model, metrics = train_model(df)

median_price = float(df["price_usd"].median())
mean_premium = float(df.loc[df["tier"] == "premium", "price_usd"].mean())
mean_township = float(df.loc[df["tier"] == "township", "price_usd"].mean())
gap_x = mean_premium / max(mean_township, 1)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(big_stat("Listings", f"{len(df):,}"), unsafe_allow_html=True)
c2.markdown(big_stat("Median price", f"${median_price:,.0f}"), unsafe_allow_html=True)
c3.markdown(big_stat("Model R²", f"{metrics['r2']:.3f}",
                      f"RMSE ${metrics['rmse']:,.0f}"), unsafe_allow_html=True)
c4.markdown(big_stat("Premium / township gap", f"{gap_x:.1f}×",
                      f"${mean_premium:,.0f} vs ${mean_township:,.0f}"),
            unsafe_allow_html=True)


tab_map, tab_drivers, tab_predict = st.tabs([
    ":world_map: Map of Harare",
    ":chart_with_upwards_trend: What drives price?",
    ":dart: Price my house",
])

# --------------------------------------------------------------------------- #
with tab_map:
    st.subheader("Every listing, plotted on Harare")
    st.caption("Yellow = expensive, dark = cheap. Hover for details.")

    tier_filter = st.multiselect(
        "Show tiers",
        options=["premium", "upper-mid", "middle", "township"],
        default=["premium", "upper-mid", "middle", "township"],
    )
    f = df[df["tier"].isin(tier_filter)]
    sample = f.sample(min(3000, len(f)), random_state=RNG)

    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude", color="price_usd", size="size_sqm",
        color_continuous_scale="Viridis", size_max=14, zoom=11,
        center={"lat": -17.83, "lon": 31.04},
        mapbox_style="carto-positron",
        hover_name="suburb",
        hover_data={
            "price_usd": ":$,.0f", "size_sqm": True, "bedrooms": True,
            "tier": True, "latitude": False, "longitude": False,
        },
        labels={"price_usd": "Price (USD)"},
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    suburb_median = df.groupby("suburb")["price_usd"].median().sort_values(ascending=False)
    top_suburb = suburb_median.index[0]
    bottom_suburb = suburb_median.index[-1]
    multiplier = suburb_median.iloc[0] / suburb_median.iloc[-1]
    st.markdown(
        insight(
            "WHERE THE MONEY LIVES",
            f"<b>{top_suburb}</b> (median <b>${suburb_median.iloc[0]:,.0f}</b>) is "
            f"<b>{multiplier:.1f}×</b> more expensive than <b>{bottom_suburb}</b> "
            f"(median <b>${suburb_median.iloc[-1]:,.0f}</b>). The north-east of Harare "
            "concentrates the high end; the southern townships sit at the affordable end.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_drivers:
    st.subheader("Price by suburb")
    order = df.groupby("suburb")["price_usd"].median().sort_values().index
    fig = px.box(
        df, x="price_usd", y="suburb", color="tier",
        category_orders={"suburb": order.tolist()},
        color_discrete_map={
            "premium": "#27AE60", "upper-mid": "#3498DB",
            "middle": "#F39C12", "township": "#95A5A6",
        },
        labels={"price_usd": "Price (USD)", "suburb": ""},
    )
    fig.update_layout(height=580, xaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Amenities that lift the price")
    feat_effects = []
    for col, label in [("has_borehole", "Borehole"), ("has_solar", "Solar backup"),
                        ("has_pool", "Pool"), ("walled", "Walled / secured")]:
        with_it = df[df[col] == 1]["price_usd"].median()
        without = df[df[col] == 0]["price_usd"].median()
        if without > 0:
            uplift = (with_it - without) / without * 100
            feat_effects.append({"feature": label, "with": with_it, "without": without, "uplift_%": uplift})
    eff = pd.DataFrame(feat_effects).sort_values("uplift_%", ascending=True)

    fig = px.bar(
        eff, x="uplift_%", y="feature", orientation="h",
        color="uplift_%", color_continuous_scale="Greens",
        text=eff["uplift_%"].map(lambda x: f"+{x:.0f}%"),
        labels={"uplift_%": "Median uplift vs houses without"},
    )
    fig.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    borehole_uplift = next((e["uplift_%"] for e in feat_effects if e["feature"] == "Borehole"), 0)
    solar_uplift = next((e["uplift_%"] for e in feat_effects if e["feature"] == "Solar backup"), 0)
    st.markdown(
        insight(
            "ZIMBABWEAN PREMIUMS",
            f"A <b>borehole</b> lifts the median price by <b>+{borehole_uplift:.0f}%</b>, "
            f"a <b>solar backup</b> by <b>+{solar_uplift:.0f}%</b>. With load-shedding and "
            "water shortages routine, these aren't luxuries — they're insurance, and buyers pay for them.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Price a house in a Harare suburb")

    col1, col2, col3 = st.columns(3)
    with col1:
        suburb = st.selectbox("Suburb", list(SUBURBS.keys()))
        bedrooms = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1, 6, 2)
    with col2:
        size_sqm = st.slider("House size (m²)", 40, 900, 220)
        plot_size_sqm = st.slider("Plot size (m²)", 200, 8000, 1200, step=50)
        house_age = st.slider("House age (years)", 0, 70, 15)
    with col3:
        has_borehole = st.toggle("Borehole", value=True)
        has_solar = st.toggle("Solar backup", value=False)
        has_pool = st.toggle("Pool", value=False)
        walled = st.toggle("Walled / secured yard", value=True)

    tier = SUBURBS[suburb]["tier"]
    lat = SUBURBS[suburb]["lat"]
    lon = SUBURBS[suburb]["lon"]
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, CBD_LAT, CBD_LON])
    a = (np.sin((lat2 - lat1)/2)**2
          + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2)
    distance_to_cbd_km = float(2 * R * np.arcsin(np.sqrt(a)))

    record = pd.DataFrame([{
        "suburb": suburb, "tier": tier,
        "latitude": lat, "longitude": lon,
        "distance_to_cbd_km": round(distance_to_cbd_km, 2),
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "size_sqm": size_sqm, "plot_size_sqm": plot_size_sqm,
        "house_age": house_age,
        "has_pool": int(has_pool), "has_borehole": int(has_borehole),
        "has_solar": int(has_solar), "walled": int(walled),
    }])
    pred = float(model.predict(record)[0])
    suburb_median_price = float(df[df["suburb"] == suburb]["price_usd"].median())
    diff_pct = (pred - suburb_median_price) / max(suburb_median_price, 1) * 100

    st.markdown(f"### Predicted listing price")
    st.markdown(
        f"<div style='font-size:64px; font-weight:800; color:#27AE60; margin: -10px 0;'>${pred:,.0f}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"For comparison, the median {suburb} listing is **${suburb_median_price:,.0f}**. "
        f"This house comes in **{diff_pct:+.1f}%** vs that median."
    )

    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=pred,
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, max(pred, suburb_median_price) * 1.4]},
            "bar": {"color": "#27AE60"},
            "threshold": {
                "line": {"color": "#1B4F72", "width": 3},
                "thickness": 0.85,
                "value": suburb_median_price,
            },
            "steps": [{"range": [0, suburb_median_price], "color": "#E8F8EF"}],
        },
        title={"text": f"vs {suburb} median (dark line)"},
    ))
    fig.update_layout(height=130, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
