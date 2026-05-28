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

# Typical gross rental yield in Harare residential is 6-8% depending on suburb tier.
# Premium suburbs trend lower (capital growth focus), townships trend higher (cashflow focus).
YIELD_BY_TIER = {
    "premium": 0.060,
    "upper-mid": 0.068,
    "middle": 0.075,
    "township": 0.085,
}

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

# A calm, professional palette — not the marketing-hero template look.
INK = "#1f2933"
MUTED = "#6b7280"
ACCENT = "#2f7d4f"   # muted Harare green, used sparingly
PLOT_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Harare house prices",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def note(text: str) -> None:
    """A quiet analyst's note — sentence-case, no shouting, woven into the page."""
    st.markdown(
        f'<div style="border-left:3px solid #d7dbe0; background:#f7f8fa; '
        f'padding:11px 16px; margin:4px 0 22px 0; color:#3a434d; '
        f'font-size:15px; line-height:1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=INK, size=13),
    )
    return fig


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading Harare listings...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("No data at data/harare_listings.csv - run the notebook once to generate it.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Derived analytics columns
    df["price_per_sqm"] = df["price_usd"] / df["size_sqm"].clip(lower=1)
    df["yield_assumption"] = df["tier"].map(YIELD_BY_TIER).fillna(0.07)
    df["est_monthly_rent_usd"] = df["price_usd"] * df["yield_assumption"] / 12.0
    # Mark "below market" by suburb median
    sub_med = df.groupby("suburb")["price_usd"].transform("median")
    df["pct_vs_suburb_median"] = (df["price_usd"] - sub_med) / sub_med * 100
    return df


@st.cache_resource(show_spinner="Training stacked ensemble...")
def train_model(df: pd.DataFrame):
    # Drop derived columns so the model only sees raw listing features (matches original)
    drop_cols = ["price_usd", "price_per_sqm", "yield_assumption",
                  "est_monthly_rent_usd", "pct_vs_suburb_median"]
    y = df["price_usd"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
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
median_ppsqm = float(df["price_per_sqm"].median())

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div style="margin:-6px 0 6px 0;">
      <div style="font-size:13px; letter-spacing:.8px; color:{MUTED};
                  text-transform:uppercase;">Harare &middot; residential property</div>
      <h1 style="margin:2px 0 4px 0; font-size:30px; font-weight:700; color:{INK};">
        What a house costs across the city</h1>
      <div style="font-size:16px; color:{MUTED};">
        Mapping listings, comparing suburbs, and pricing a house — built on
        location, size and the resilience features Harare buyers pay for.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Listings", f"{len(df):,}")
c2.metric("Median price", f"${median_price:,.0f}")
c3.metric("Median $/m²", f"${median_ppsqm:,.0f}")
c4.metric("Model R²", f"{metrics['r2']:.3f}", f"RMSE ${metrics['rmse']:,.0f}",
          delta_color="off")
c5.metric("Premium / township gap", f"{gap_x:.1f}×",
          f"${mean_premium:,.0f} vs ${mean_township:,.0f}", delta_color="off")

note(
    f"The whole story of Harare prices is location. A typical premium-suburb house "
    f"runs about <b>${mean_premium:,.0f}</b>, against <b>${mean_township:,.0f}</b> in "
    f"the townships — roughly {gap_x:.1f}× the price for what's often a similar amount "
    f"of brick. The map and suburb comparisons below pull that gap apart, and the "
    f"predictor lets you price a specific house against its own neighbourhood."
)

tab_market, tab_drivers, tab_price = st.tabs([
    "The market",
    "What drives price",
    "Price a house",
])

# --------------------------------------------------------------------------- #
with tab_market:
    st.markdown("#### Every listing, plotted on Harare")
    st.caption("Lighter points are pricier, darker are cheaper. Hover for details.")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        tier_filter = st.multiselect(
            "Show tiers",
            options=["premium", "upper-mid", "middle", "township"],
            default=["premium", "upper-mid", "middle", "township"],
        )
    with col_f2:
        price_range = st.slider(
            "Price range (USD)",
            min_value=int(df["price_usd"].min()),
            max_value=int(df["price_usd"].max()),
            value=(int(df["price_usd"].min()), int(df["price_usd"].max())),
            step=5000,
        )

    f = df[
        df["tier"].isin(tier_filter)
        & df["price_usd"].between(price_range[0], price_range[1])
    ]

    if len(f) == 0:
        st.warning("No listings match those filters. Widen the range.")
    else:
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
        note(
            f"The high end clusters in the north-east — <b>{top_suburb}</b> sits at a "
            f"median <b>${suburb_median.iloc[0]:,.0f}</b>, about <b>{multiplier:.1f}×</b> "
            f"what you'd pay in <b>{bottom_suburb}</b> (<b>${suburb_median.iloc[-1]:,.0f}</b>) "
            f"in the southern townships. If you're buying for capital growth, the north-east "
            f"is where it has historically held; if you're after rental yield, the "
            f"south-west gives a better rent-to-price ratio — cheaper to get into and "
            f"faster to pay back, though with higher tenant turnover."
        )

    # --- Suburb compare -----------------------------------------------------
    st.markdown("#### Compare suburbs head-to-head")

    all_suburbs = sorted(df["suburb"].unique().tolist())
    default_picks = [s for s in ["Borrowdale", "Avondale", "Hatfield", "Glen View"]
                     if s in all_suburbs][:4]
    if not default_picks:
        default_picks = all_suburbs[:3]
    picks = st.multiselect(
        "Pick 2-6 suburbs to compare",
        options=all_suburbs,
        default=default_picks,
    )

    if len(picks) < 2:
        st.info("Pick at least two suburbs to compare.")
    else:
        comp = df[df["suburb"].isin(picks)].copy()
        agg = (
            comp.groupby("suburb")
            .agg(
                listings=("price_usd", "size"),
                median_price=("price_usd", "median"),
                median_ppsqm=("price_per_sqm", "median"),
                median_size=("size_sqm", "median"),
                median_rent=("est_monthly_rent_usd", "median"),
                tier=("tier", lambda s: s.mode().iloc[0]),
            )
            .reset_index()
            .sort_values("median_price", ascending=False)
        )

        tier_color = {
            "premium": "#27AE60", "upper-mid": "#3498DB",
            "middle": "#F39C12", "township": "#95A5A6",
        }

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                agg, x="suburb", y="median_price", color="tier",
                color_discrete_map=tier_color,
                text=agg["median_price"].map(lambda x: f"${x/1000:,.0f}k"),
                labels={"median_price": "Median price (USD)", "suburb": ""},
                title="Median listing price by suburb",
            )
            fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        with col_b:
            fig = px.bar(
                agg, x="suburb", y="median_ppsqm", color="tier",
                color_discrete_map=tier_color,
                text=agg["median_ppsqm"].map(lambda x: f"${x:,.0f}"),
                labels={"median_ppsqm": "Median $/m²", "suburb": ""},
                title="Median price per m² — strips out house size",
            )
            fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

        st.markdown("**Side-by-side stats**")
        display = agg.copy()
        display["median_price"] = display["median_price"].map(lambda x: f"${x:,.0f}")
        display["median_ppsqm"] = display["median_ppsqm"].map(lambda x: f"${x:,.0f}")
        display["median_size"] = display["median_size"].map(lambda x: f"{x:,.0f} m²")
        display["median_rent"] = display["median_rent"].map(lambda x: f"${x:,.0f}/mo")
        display = display.rename(columns={
            "suburb": "Suburb", "tier": "Tier", "listings": "Listings",
            "median_price": "Median price", "median_ppsqm": "Median $/m²",
            "median_size": "Typical size", "median_rent": "Est. rent",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        top_ppsqm = agg.sort_values("median_ppsqm", ascending=False).iloc[0]
        cheap_ppsqm = agg.sort_values("median_ppsqm").iloc[0]
        ratio = top_ppsqm["median_ppsqm"] / max(cheap_ppsqm["median_ppsqm"], 1)
        note(
            f"Price per square metre is the honest number to compare on — it strips out "
            f"house size. In your selection, <b>{top_ppsqm['suburb']}</b> trades at "
            f"<b>${top_ppsqm['median_ppsqm']:,.0f}/m²</b> against "
            f"<b>{cheap_ppsqm['suburb']}</b> at <b>${cheap_ppsqm['median_ppsqm']:,.0f}/m²</b>, "
            f"a <b>{ratio:.1f}×</b> spread that is almost entirely about location. It's also "
            f"the figure worth negotiating on."
        )

    # --- Best value & yield -------------------------------------------------
    st.divider()
    st.markdown("#### Investment view — yield and best value")

    st.markdown("##### Rental yield calculator")
    st.caption(
        "Harare residential gross yields typically run 6-8%, with premium suburbs at the lower end "
        "(capital-growth tenants) and townships at the higher end (cash-yield investors)."
    )

    col_y1, col_y2, col_y3 = st.columns(3)
    with col_y1:
        calc_price = st.number_input("Purchase price (USD)", 5_000, 2_000_000, 120_000, step=5_000)
    with col_y2:
        calc_rent = st.number_input("Expected rent ($/month)", 50, 20_000, 750, step=50)
    with col_y3:
        calc_costs = st.slider("Annual costs as % of rent (rates, repairs, voids)", 5, 40, 20)

    annual_rent = calc_rent * 12
    net_rent = annual_rent * (1 - calc_costs / 100)
    gross_yield = annual_rent / max(calc_price, 1) * 100
    net_yield = net_rent / max(calc_price, 1) * 100
    payback = calc_price / max(net_rent, 1)

    yc1, yc2, yc3 = st.columns(3)
    yc1.metric("Gross yield", f"{gross_yield:.2f}%", f"${annual_rent:,.0f} rent/yr",
               delta_color="off")
    yc2.metric("Net yield", f"{net_yield:.2f}%", f"after {calc_costs}% costs",
               delta_color="off")
    yc3.metric("Cash payback", f"{payback:.1f} yrs", "ignoring capital growth",
               delta_color="off")

    if net_yield >= 7:
        verdict = (
            f"At <b>{net_yield:.1f}%</b> net yield this comfortably beats the Harare median — "
            "a strong cash-flow buy. The sensible play is to hold it for the rent and "
            "refinance into the next purchase."
        )
    elif net_yield >= 5:
        verdict = (
            f"A net yield of <b>{net_yield:.1f}%</b> is about typical for Harare. Whether it's "
            "a good buy comes down to capital growth — is the suburb gentrifying, or flat?"
        )
    else:
        verdict = (
            f"At <b>{net_yield:.1f}%</b> net you're really buying for capital growth, not rent. "
            "Only worth it if you have a genuine reason to think prices climb from here."
        )
    note(verdict)

    st.markdown("##### Estimated rent landscape across Harare")
    rent_by_tier = df.groupby("tier").agg(
        median_price=("price_usd", "median"),
        median_rent=("est_monthly_rent_usd", "median"),
        yield_used=("yield_assumption", "first"),
    ).reset_index()
    rent_by_tier["yield_used"] = (rent_by_tier["yield_used"] * 100).round(1).astype(str) + "%"
    rent_by_tier["median_price"] = rent_by_tier["median_price"].map(lambda x: f"${x:,.0f}")
    rent_by_tier["median_rent"] = rent_by_tier["median_rent"].map(lambda x: f"${x:,.0f}/mo")
    rent_by_tier = rent_by_tier.rename(columns={
        "tier": "Tier", "median_price": "Median price",
        "median_rent": "Est. monthly rent", "yield_used": "Yield assumed",
    })
    st.dataframe(rent_by_tier, use_container_width=True, hide_index=True)

    st.markdown("##### Best-value listings — under-priced for their suburb")
    st.caption("Properties priced 10%+ below the median for their own suburb, ranked by discount.")

    discount_threshold = st.slider("Minimum discount vs suburb median (%)", 5, 40, 15)
    bv = df[df["pct_vs_suburb_median"] <= -discount_threshold].copy()
    bv = bv.sort_values("pct_vs_suburb_median").head(25)
    if len(bv) == 0:
        st.info("No listings below that discount threshold. Lower the slider.")
    else:
        display_bv = bv[[
            "suburb", "tier", "price_usd", "size_sqm", "bedrooms",
            "price_per_sqm", "pct_vs_suburb_median", "est_monthly_rent_usd",
        ]].copy()
        display_bv["price_usd"] = display_bv["price_usd"].map(lambda x: f"${x:,.0f}")
        display_bv["price_per_sqm"] = display_bv["price_per_sqm"].map(lambda x: f"${x:,.0f}")
        display_bv["est_monthly_rent_usd"] = display_bv["est_monthly_rent_usd"].map(lambda x: f"${x:,.0f}")
        display_bv["pct_vs_suburb_median"] = display_bv["pct_vs_suburb_median"].map(lambda x: f"{x:+.1f}%")
        display_bv = display_bv.rename(columns={
            "suburb": "Suburb", "tier": "Tier", "price_usd": "Price",
            "size_sqm": "Size (m²)", "bedrooms": "Beds",
            "price_per_sqm": "$/m²", "pct_vs_suburb_median": "Vs suburb median",
            "est_monthly_rent_usd": "Est. rent/mo",
        })
        st.dataframe(display_bv, use_container_width=True, hide_index=True)
        note(
            f"These {len(bv)} listings are priced well below the median for their own "
            f"suburb. Some will be cheap for a reason — hidden defects — but others are "
            f"genuine mispricings: stale listings, motivated sellers, agents who haven't "
            f"refreshed the price. It's a sensible viewing shortlist. For the top few, "
            f"check the title deed is clean, there are no structural issues, and the rent "
            f"you'd actually get matches the estimate above. If all three hold, you have "
            f"an offer to make."
        )

# --------------------------------------------------------------------------- #
with tab_drivers:
    st.markdown("#### Price by suburb")
    order = df.groupby("suburb")["price_usd"].median().sort_values().index
    fig = px.box(
        df, x="price_usd", y="suburb", color="tier",
        category_orders={"suburb": order.tolist()},
        color_discrete_map={
            "premium": ACCENT, "upper-mid": "#5b8aa6",
            "middle": "#c98a3a", "township": "#8a929b",
        },
        labels={"price_usd": "Price (USD)", "suburb": ""},
    )
    fig.update_layout(xaxis_tickformat="$,.0f")
    st.plotly_chart(style_fig(fig, 580), use_container_width=True)

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
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    borehole_uplift = next((e["uplift_%"] for e in feat_effects if e["feature"] == "Borehole"), 0)
    solar_uplift = next((e["uplift_%"] for e in feat_effects if e["feature"] == "Solar backup"), 0)
    note(
        f"The amenities that move price most are the ones that solve Zimbabwe's everyday "
        f"problems. A borehole adds about <b>+{borehole_uplift:.0f}%</b> to the median, a "
        f"solar backup roughly <b>+{solar_uplift:.0f}%</b>. With load-shedding and water "
        f"cuts routine, these read less as luxuries and more as insurance — and buyers pay "
        f"for them. If you're renovating to sell, drilling a borehole and fitting a solar "
        f"inverter does more for the price than cosmetic finishes do."
    )

    st.markdown("#### Size vs price")
    fig = px.scatter(
        df.sample(min(2000, len(df)), random_state=RNG),
        x="size_sqm", y="price_usd", color="tier",
        color_discrete_map={
            "premium": ACCENT, "upper-mid": "#5b8aa6",
            "middle": "#c98a3a", "township": "#8a929b",
        },
        opacity=0.55, trendline="ols",
        labels={"size_sqm": "House size (m²)", "price_usd": "Price (USD)"},
    )
    fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    note(
        "Within a tier, bigger houses cost more, as you'd expect. But the gap between "
        "tiers at the same size is the bigger story — a 200m² house in a premium suburb "
        "outsells the same 200m² house in a township several times over. Location is the "
        "dominant variable here; size is secondary."
    )

# --------------------------------------------------------------------------- #
with tab_price:
    st.markdown("#### Price a house in a Harare suburb")

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
    pred_ppsqm = pred / max(size_sqm, 1)
    suburb_ppsqm = float(df[df["suburb"] == suburb]["price_per_sqm"].median())

    st.markdown("##### Predicted listing price")
    st.markdown(
        f"<div style='font-size:56px; font-weight:700; color:{INK}; margin: -6px 0;'>${pred:,.0f}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"For comparison, the median {suburb} listing is **${suburb_median_price:,.0f}**. "
        f"This house comes in **{diff_pct:+.1f}%** vs that median. "
        f"$/m²: **${pred_ppsqm:,.0f}** (suburb median ${suburb_ppsqm:,.0f})."
    )

    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=pred,
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, max(pred, suburb_median_price) * 1.4]},
            "bar": {"color": ACCENT},
            "threshold": {
                "line": {"color": INK, "width": 3},
                "thickness": 0.85,
                "value": suburb_median_price,
            },
            "steps": [{"range": [0, suburb_median_price], "color": "#eef4f0"}],
        },
        title={"text": f"vs {suburb} median (dark line)"},
    ))
    st.plotly_chart(style_fig(fig, 130), use_container_width=True)

    # Investment quick-take using the predicted price
    yield_assumed = YIELD_BY_TIER.get(tier, 0.07)
    est_rent = pred * yield_assumed / 12
    st.markdown("##### Investment quick-take")
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Est. monthly rent", f"${est_rent:,.0f}",
               f"at {yield_assumed*100:.1f}% gross yield", delta_color="off")
    qc2.metric("Annual gross rent", f"${est_rent*12:,.0f}")
    qc3.metric("Cash payback", f"{(pred / max(est_rent*12*0.8, 1)):.1f} yrs",
               "net of 20% costs", delta_color="off")

    if diff_pct < -10:
        note(
            f"This configuration prices <b>{abs(diff_pct):.1f}%</b> below the {suburb} "
            f"median. If you can actually buy it at this price, you're getting in under "
            f"the local benchmark — worth a closer look."
        )
    elif diff_pct > 10:
        note(
            f"This sits <b>{diff_pct:.1f}%</b> above the {suburb} median. Make sure the "
            f"features genuinely justify the premium — buyers in this suburb have "
            f"alternatives, and an over-priced listing tends to sit."
        )
    else:
        note(
            f"This lands within about <b>±10%</b> of the {suburb} median — a market-rate "
            f"price. There's usually some room to negotiate, but don't expect a steal."
        )
