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
    .action {
        background: linear-gradient(180deg, #FFFFFF 0%, #FEF5E7 100%);
        border-left: 4px solid #E67E22; padding: 16px 20px; border-radius: 10px; margin: 8px 0;
    }
    .action .head { font-size: 12px; color: #E67E22; font-weight: 700; letter-spacing: 1px; }
    .action .body { font-size: 15px; color: #6E4D11; margin-top: 4px; line-height: 1.5; }
    </style>

    <div class="hero">
      <h1>:house: Harare House Prices</h1>
      <p>Map the city, see what really drives prices, price a house and check the investment case.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def big_stat(label, value, sub=""):
    return f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def insight(head, body):
    return f'<div class="insight"><div class="head">{head}</div><div class="body">{body}</div></div>'


def action(head, body):
    return f'<div class="action"><div class="head">{head}</div><div class="body">{body}</div></div>'


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

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(big_stat("Listings", f"{len(df):,}"), unsafe_allow_html=True)
c2.markdown(big_stat("Median price", f"${median_price:,.0f}"), unsafe_allow_html=True)
c3.markdown(big_stat("Median $/m²", f"${median_ppsqm:,.0f}"), unsafe_allow_html=True)
c4.markdown(big_stat("Model R²", f"{metrics['r2']:.3f}",
                      f"RMSE ${metrics['rmse']:,.0f}"), unsafe_allow_html=True)
c5.markdown(big_stat("Premium / township gap", f"{gap_x:.1f}×",
                      f"${mean_premium:,.0f} vs ${mean_township:,.0f}"),
            unsafe_allow_html=True)


tab_map, tab_suburb, tab_drivers, tab_value, tab_predict = st.tabs([
    ":world_map: Map of Harare",
    ":bar_chart: Suburb compare",
    ":chart_with_upwards_trend: What drives price?",
    ":moneybag: Best value & yield",
    ":dart: Price my house",
])

# --------------------------------------------------------------------------- #
with tab_map:
    st.subheader("Every listing, plotted on Harare")
    st.caption("Yellow = expensive, dark = cheap. Hover for details.")

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
        st.markdown(
            action(
                "WHAT TO DO WITH THIS",
                "If you are an investor chasing capital growth, anchor in the north-east. "
                "If you are chasing rental yield, the south-west delivers a higher rent-to-price ratio — "
                "lower ticket price, faster payback, but higher tenant turnover.",
            ),
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
with tab_suburb:
    st.subheader("Compare suburbs head-to-head")

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
            )
            fig.update_layout(height=380, yaxis_tickformat="$,.0f",
                              title="Median listing price by suburb")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.bar(
                agg, x="suburb", y="median_ppsqm", color="tier",
                color_discrete_map=tier_color,
                text=agg["median_ppsqm"].map(lambda x: f"${x:,.0f}"),
                labels={"median_ppsqm": "Median $/m²", "suburb": ""},
            )
            fig.update_layout(height=380, yaxis_tickformat="$,.0f",
                              title="Median price per m² — strips out house size")
            st.plotly_chart(fig, use_container_width=True)

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
        st.markdown(
            insight(
                "PRICE PER SQUARE METRE TELLS THE TRUTH",
                f"In your selection, <b>{top_ppsqm['suburb']}</b> trades at "
                f"<b>${top_ppsqm['median_ppsqm']:,.0f}/m²</b> vs "
                f"<b>{cheap_ppsqm['suburb']}</b> at <b>${cheap_ppsqm['median_ppsqm']:,.0f}/m²</b> — "
                f"a <b>{ratio:.1f}×</b> spread. Headline price hides house size; $/m² makes the "
                "premium for location explicit and is the metric to negotiate on.",
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
    st.markdown(
        action(
            "WHAT TO DO WITH THIS",
            "If you are renovating to sell, prioritise drilling a borehole and adding a solar inverter "
            "before cosmetic finishes — the data says the market pays for resilience, not for paint.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Size vs price — the relationship")
    fig = px.scatter(
        df.sample(min(2000, len(df)), random_state=RNG),
        x="size_sqm", y="price_usd", color="tier",
        color_discrete_map={
            "premium": "#27AE60", "upper-mid": "#3498DB",
            "middle": "#F39C12", "township": "#95A5A6",
        },
        opacity=0.55, trendline="ols",
        labels={"size_sqm": "House size (m²)", "price_usd": "Price (USD)"},
    )
    fig.update_layout(height=420, yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        insight(
            "SIZE EXPLAINS PART OF IT — NOT ALL OF IT",
            "Within the same tier, bigger houses sell for more (predictably). But the gap "
            "<i>between</i> tiers at the same size is large — a 200m² house in a premium suburb "
            "outsells the same 200m² house in a township several times over. <b>Location is "
            "the dominant variable</b>, size is the secondary one.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_value:
    st.subheader("Investment view — yield and best value")

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
    yc1.markdown(big_stat("Gross yield", f"{gross_yield:.2f}%",
                          f"${annual_rent:,.0f} rent/yr"), unsafe_allow_html=True)
    yc2.markdown(big_stat("Net yield", f"{net_yield:.2f}%",
                          f"after {calc_costs}% costs"), unsafe_allow_html=True)
    yc3.markdown(big_stat("Cash payback", f"{payback:.1f} yrs",
                          "ignoring capital growth"), unsafe_allow_html=True)

    if net_yield >= 7:
        verdict = ("STRONG CASH-FLOW DEAL",
                   f"At <b>{net_yield:.1f}%</b> net yield this comfortably beats the Harare median. "
                   "Hold for cash, refinance for the next purchase.")
    elif net_yield >= 5:
        verdict = ("FAIR — TYPICAL HARARE YIELD",
                   f"<b>{net_yield:.1f}%</b> net is in line with the wider market. The investment case "
                   "depends on expected capital growth — is the suburb gentrifying?")
    else:
        verdict = ("WEAK CASH YIELD",
                   f"At <b>{net_yield:.1f}%</b> net you are buying for capital growth, not rent. "
                   "Only worth it if you have a strong reason to believe prices rise from here.")
    st.markdown(insight(verdict[0], verdict[1]), unsafe_allow_html=True)

    st.markdown("---")
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

    st.markdown("---")
    st.markdown("##### :gem: Best-value listings — under-priced for their suburb")
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
        st.markdown(
            insight(
                "WHY THIS LIST MATTERS",
                f"These {len(bv)} listings are priced materially below the median for their own suburb. "
                "Some will have hidden defects (the reason for the discount), but some are genuine "
                "mispricings — old listings, motivated sellers, or stale agents. "
                "<b>This is your viewing shortlist.</b>",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            action(
                "NEXT STEP FOR AN INVESTOR",
                "View the top 5. For each, confirm: (1) clean title deed, (2) no structural issues, "
                "(3) realistic rent matches the estimate above. If yes on all three, you have an offer to make.",
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
    pred_ppsqm = pred / max(size_sqm, 1)
    suburb_ppsqm = float(df[df["suburb"] == suburb]["price_per_sqm"].median())

    st.markdown(f"### Predicted listing price")
    st.markdown(
        f"<div style='font-size:64px; font-weight:800; color:#27AE60; margin: -10px 0;'>${pred:,.0f}</div>",
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

    # Investment quick-take using the predicted price
    yield_assumed = YIELD_BY_TIER.get(tier, 0.07)
    est_rent = pred * yield_assumed / 12
    st.markdown("##### Investment quick-take")
    qc1, qc2, qc3 = st.columns(3)
    qc1.markdown(big_stat("Est. monthly rent", f"${est_rent:,.0f}",
                          f"at {yield_assumed*100:.1f}% gross yield"), unsafe_allow_html=True)
    qc2.markdown(big_stat("Annual gross rent", f"${est_rent*12:,.0f}"), unsafe_allow_html=True)
    qc3.markdown(big_stat("Cash payback", f"{(pred / max(est_rent*12*0.8, 1)):.1f} yrs",
                          "net of 20% costs"), unsafe_allow_html=True)

    if diff_pct < -10:
        st.markdown(
            insight(
                "PRICED BELOW SUBURB MEDIAN",
                f"This configuration prices <b>{abs(diff_pct):.1f}%</b> below the {suburb} median. "
                "If you can buy it at this price, you are getting in below the local benchmark.",
            ),
            unsafe_allow_html=True,
        )
    elif diff_pct > 10:
        st.markdown(
            insight(
                "PRICED ABOVE SUBURB MEDIAN",
                f"This configuration sits <b>{diff_pct:.1f}%</b> above the {suburb} median. "
                "Make sure the features justify the premium — buyers in this suburb have alternatives.",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            insight(
                "ROUGHLY ON THE MARKET",
                f"This sits within <b>±10%</b> of the {suburb} median — a market-rate price. "
                "Negotiating room exists but don't expect a steal.",
            ),
            unsafe_allow_html=True,
        )
