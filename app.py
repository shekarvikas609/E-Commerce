"""Second Order: Olist customer segments, retention and 30-day demand outlook.

    streamlit run app.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import cohorts, pipeline, ui  # noqa: E402
from src.forecast import MODEL_LABELS  # noqa: E402
from src.rfm import PLAYBOOK, SEGMENTS  # noqa: E402

st.set_page_config(page_title="Second Order: Olist customers and demand", page_icon="◐", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
CFG = {"displayModeBar": False}


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config=CFG, theme=None)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load() -> dict:
    return pipeline.load_outputs()


if not pipeline.processed_ready():
    with st.spinner("First run: building the dataset, models and forecasts (about a minute)..."):
        pipeline.run(progress=lambda *_: None)
    st.cache_data.clear()

D = load()
M, cust_all, orders_all = D["metrics"], D["customers"], D["orders"]
labels = M["series_labels"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand">Second Order<small>Which Olist customers come back, and what will sell next.</small></div>',
        unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    picked = st.multiselect("States", sorted(cust_all["state"].dropna().unique()), placeholder="All states",
                            help="Filters the customer and order views. Forecasts always cover all states.")
    st.markdown(
        f'<div class="side-note">Data: {M["start"]} to {M["end"]}<br>{M["n_customers"]:,} customers, {M["n_orders"]:,} purchases.<br>'
        f'Revenue is item price plus freight. Same-day orders from one customer count as one purchase.</div>',
        unsafe_allow_html=True)
    if st.button("Rebuild from data/raw"):
        with st.spinner("Rebuilding..."):
            pipeline.run(progress=lambda *_: None)
        st.cache_data.clear()
        st.rerun()

cust = cust_all[cust_all["state"].isin(picked)] if picked else cust_all
orders = orders_all[orders_all["state"].isin(picked)] if picked else orders_all
if cust.empty:
    st.warning("No customers match this filter.")
    st.stop()

# ---------------------------------------------------------------- derived numbers
seg = cust.groupby("segment").agg(
    customers=("monetary", "size"), revenue=("monetary", "sum"), aov=("aov", "mean"),
    recency=("recency", "mean"), p=("p_repurchase", "mean"), risk=("risk_index", "mean"),
    exp_rev=("exp_rev_90d", "sum"),
).reindex(SEGMENTS).fillna(0)
seg["rev_share"] = seg["revenue"] / seg["revenue"].sum()
seg["cust_share"] = seg["customers"] / seg["customers"].sum()
seg["tier"] = np.select([seg["risk"] >= 60, seg["risk"] >= 40], ["High", "Medium"], "Low")
seg.loc[seg["customers"] == 0, "tier"] = "n/a"

one_time = (cust["frequency"] == 1).mean()
oneoff_rev = seg.loc[["Big spenders", "Lapsed high-value"], "rev_share"].sum()

hist, fut = D["history"], D["future"]
fc_orders = fut[fut["series"] == "orders"]["yhat"].sum()
fc_rev = fut[fut["series"] == "revenue"]["yhat"].sum()

# ---------------------------------------------------------------- hero + ribbon
if M["source"] != "olist":
    st.markdown(
        '<div class="banner">Showing generated demo data that mimics the Olist schema. Put the real CSVs in '
        '<b>data/raw</b> and press <b>Rebuild from data/raw</b> in the sidebar.</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="hero"><h1>{one_time:.0%} of customers have ordered only once</h1>'
    f'<p>Big spenders and lapsed high-value buyers hold {oneoff_rev:.0%} of revenue, and every one of them is a one-time buyer. '
    f'The next 30 days are forecast at {fc_orders:,.0f} orders and {ui.brl(fc_rev)} in revenue.</p></div>',
    unsafe_allow_html=True)

rib = go.Figure()
for s in SEGMENTS:
    cs, rs = seg.loc[s, "cust_share"] * 100, seg.loc[s, "rev_share"] * 100
    rib.add_bar(
        y=["Customers", "Revenue"], x=[cs, rs], name=s, orientation="h",
        marker=dict(color=ui.SEG_COLORS[s], line=dict(color=ui.SURFACE, width=2)),
        text=[f"{cs:.0f}%" if cs >= 6 else "", f"{rs:.0f}%" if rs >= 6 else ""], textposition="inside",
        insidetextanchor="middle", textfont=dict(color="#ffffff", size=13),
        hovertemplate=f"<b>{s}</b><br>%{{x:.1f}}% of %{{y}}<extra></extra>")
rib.update_layout(barmode="stack", bargap=.22)
ui.style(rib, h=210)
rib.update_xaxes(visible=False, range=[0, 100])
rib.update_yaxes(autorange="reversed", automargin=True, tickfont=dict(color=ui.INK2, size=13))
rib.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, x=0, traceorder="normal", font=dict(color=ui.INK2, size=12)))
show(rib)

# ---------------------------------------------------------------- KPI strip
end = orders["purchase_ts"].max().normalize() + pd.Timedelta(days=1)
last30 = orders[orders["purchase_ts"] >= end - pd.Timedelta(days=30)]
prev30 = orders[(orders["purchase_ts"] >= end - pd.Timedelta(days=60)) & (orders["purchase_ts"] < end - pd.Timedelta(days=30))]
rev_all = orders["revenue"].sum()
repeat_rev = orders.loc[orders["order_no"] > 1, "revenue"].sum() / rev_all
st.markdown(ui.kpi_strip([
    ("Revenue", ui.brl(rev_all), ui.delta_text(last30["revenue"].sum(), prev30["revenue"].sum(), "last 30 days vs prior 30")),
    ("Purchases", f"{len(orders):,}", ui.delta_text(len(last30), len(prev30), "last 30 days vs prior 30")),
    ("Customers", f"{len(cust):,}", f"<b>{(cust['frequency'] >= 2).mean():.1%}</b> bought more than once"),
    ("Average order", ui.brl(orders["revenue"].mean(), short=False), f"<b>{repeat_rev:.1%}</b> of revenue is repeat purchases"),
    ("Next 30 days", f"{fc_orders:,.0f}", f"forecast purchases, <b>{ui.brl(fc_rev)}</b> revenue"),
]), unsafe_allow_html=True)

tab_over, tab_seg, tab_ret, tab_fc, tab_act = st.tabs(
    ["Overview", "Segments", "Retention", "Forecast", "Actions"])

# ================================================================ Overview
with tab_over:
    c1, c2 = st.columns([1.45, 1], gap="large")
    with c1:
        st.markdown(ui.section("Monthly revenue", "First purchases versus repeat purchases. Partial months at the edges are left out."), unsafe_allow_html=True)
        o = orders.copy()
        o["month"] = o["purchase_ts"].dt.to_period("M")
        lo, hi = orders_all["purchase_ts"].min(), orders_all["purchase_ts"].max()
        full = [m for m in o["month"].unique() if m.start_time >= lo.normalize() and m.end_time <= hi + pd.Timedelta(days=1)]
        o = o[o["month"].isin(full)]
        m = o.assign(kind=np.where(o["order_no"] == 1, "First purchase", "Repeat purchase")) \
            .groupby(["month", "kind"])["revenue"].sum().unstack(fill_value=0)
        x = m.index.astype(str)
        fig = go.Figure()
        for kind, col in [("First purchase", ui.SEG_COLORS["Champions"]), ("Repeat purchase", ui.SEG_COLORS["Big spenders"])]:
            fig.add_bar(x=x, y=m.get(kind, 0), name=kind, marker=dict(color=col, line=dict(color=ui.SURFACE, width=1.5), cornerradius=3),
                        hovertemplate="%{x}<br>" + kind + ": R$ %{y:,.0f}<extra></extra>")
        fig.update_layout(barmode="stack", bargap=.28)
        ui.style(fig, h=340)
        fig.update_layout(legend=dict(traceorder="normal"))
        fig.update_yaxes(tickprefix="R$ ", tickformat="~s")
        show(fig)
    with c2:
        st.markdown(ui.section("How fast customers come back", "Share who placed a second order within N days of the first."), unsafe_allow_html=True)
        rc = cohorts.repeat_curve(orders)
        fig = go.Figure(go.Scatter(
            x=rc["days"], y=rc["pct_repeat"], mode="lines", line=dict(color=ui.BLUE, width=2.5), name="Placed a 2nd order",
            hovertemplate="Within %{x} days: %{y:.2f}%<extra></extra>"))
        if len(rc):
            fig.add_annotation(x=rc["days"].iloc[-1], y=rc["pct_repeat"].iloc[-1], text=f"{rc['pct_repeat'].iloc[-1]:.1f}%",
                               showarrow=False, xanchor="left", xshift=6, font=dict(color=ui.INK, size=13))
        ui.style(fig, h=340, legend=False)
        fig.update_xaxes(title=dict(text="Days since first order", font=dict(color=ui.MUTED, size=11)))
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        show(fig)

# ================================================================ Segments
with tab_seg:
    st.markdown(ui.section("The eight segments", "Recency and spend are scored 1 to 5 by quintile. Frequency uses fixed steps because almost everyone has ordered once."), unsafe_allow_html=True)
    st.markdown(ui.segment_table(seg), unsafe_allow_html=True)
    st.markdown(
        '<p class="note" style="margin-top:.8rem">Churn risk is relative: customers are ranked on their predicted chance of '
        'not ordering again within 90 days, and a segment is High when its average rank is in the top 40%.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(ui.section("Revenue by segment"), unsafe_allow_html=True)
        s2 = seg.sort_values("revenue")
        fig = go.Figure(go.Bar(
            y=s2.index, x=s2["revenue"], orientation="h", marker=dict(color=[ui.SEG_COLORS[s] for s in s2.index], cornerradius=4),
            text=[ui.brl(v) for v in s2["revenue"]], textposition="outside", textfont=dict(color=ui.INK2, size=12), cliponaxis=False,
            hovertemplate="%{y}<br>R$ %{x:,.0f}<extra></extra>"))
        ui.style(fig, h=330, legend=False)
        fig.update_xaxes(visible=False, range=[0, s2["revenue"].max() * 1.22])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        show(fig)
    with c2:
        st.markdown(ui.section("Repurchase chance in the next 90 days"), unsafe_allow_html=True)
        s3 = seg.sort_values("p")
        avg = cust["p_repurchase"].mean() * 100
        fig = go.Figure(go.Bar(
            y=s3.index, x=s3["p"] * 100, orientation="h", marker=dict(color=[ui.SEG_COLORS[s] for s in s3.index], cornerradius=4),
            text=[f"{v:.1%}" for v in s3["p"]], textposition="outside", textfont=dict(color=ui.INK2, size=12), cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.2f}% expected to order again<extra></extra>"))
        fig.add_vline(x=avg, line=dict(color=ui.MUTED, width=1, dash="dot"))
        fig.add_annotation(x=avg, y=1.02, yref="paper", text=f"All customers {avg:.1f}%", showarrow=False, xanchor="left", xshift=4, font=dict(color=ui.MUTED, size=11))
        ui.style(fig, h=330, legend=False, ymargin=30)
        fig.update_xaxes(visible=False, range=[0, max(s3["p"].max() * 100 * 1.25, avg * 1.4)])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        show(fig)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(ui.section("Where repurchase happens", "Average repurchase chance (%) by recency and spend score."), unsafe_allow_html=True)
        piv = cust.pivot_table(index="R", columns="M", values="p_repurchase", aggfunc="mean").reindex(index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5]) * 100
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=["Lowest", "2", "3", "4", "Highest"], y=["Most recent", "4", "3", "2", "Least recent"],
            colorscale=ui.SEQ, xgap=3, ygap=3, showscale=False, texttemplate="%{z:.1f}", textfont=dict(color=ui.INK, size=12),
            hovertemplate="Recency %{y}, spend %{x}<br>%{z:.2f}%<extra></extra>"))
        ui.style(fig, h=320, legend=False)
        fig.update_xaxes(type="category", title=dict(text="Spend score", font=dict(color=ui.MUTED, size=11)), side="bottom")
        fig.update_yaxes(type="category", autorange="reversed", title=dict(text="Recency score", font=dict(color=ui.MUTED, size=11)), tickfont=dict(color=ui.INK2, size=11))
        show(fig)
    with c4:
        st.markdown(ui.section("What drives the prediction", "Share of the model's gain by feature."), unsafe_allow_html=True)
        imp = list(M["importance"].items())[:8][::-1]
        fig = go.Figure(go.Bar(
            y=[k for k, _ in imp], x=[v * 100 for _, v in imp], orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
            text=[f"{v:.0%}" for _, v in imp], textposition="outside", textfont=dict(color=ui.INK2, size=12), cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1f}% of model gain<extra></extra>"))
        ui.style(fig, h=320, legend=False)
        fig.update_xaxes(visible=False, range=[0, max(v for _, v in imp) * 100 * 1.25])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        show(fig)

    st.markdown(ui.section("Look inside a segment"), unsafe_allow_html=True)
    pick = st.selectbox("Segment", SEGMENTS, index=SEGMENTS.index("Big spenders"), label_visibility="collapsed")
    sub = cust[cust["segment"] == pick]
    pb = PLAYBOOK[pick]
    e1, e2 = st.columns([1, 1.2], gap="large")
    with e1:
        st.markdown(
            f'<div class="callout"><b style="color:{ui.INK}">{html.escape(pick)}</b>: {html.escape(pb["who"])}<br>'
            f'{len(sub):,} customers, average order {ui.brl(sub["aov"].mean(), short=False)}, '
            f'last ordered {sub["recency"].mean():.0f} days ago on average.</div>', unsafe_allow_html=True)
    with e2:
        tc = sub["top_category"].value_counts().head(6)[::-1]
        fig = go.Figure(go.Bar(
            y=[c.replace("_", " ") for c in tc.index], x=tc.values, orientation="h", marker=dict(color=ui.SEG_COLORS[pick], cornerradius=4),
            text=[f"{v:,}" for v in tc.values], textposition="outside", textfont=dict(color=ui.INK2, size=12), cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} customers<extra></extra>"))
        ui.style(fig, h=250, legend=False)
        fig.update_xaxes(visible=False, range=[0, tc.max() * 1.2 if len(tc) else 1])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        st.markdown('<p class="note" style="margin:0 0 .2rem">Top categories by customers</p>', unsafe_allow_html=True)
        show(fig)

# ================================================================ Retention
with tab_ret:
    ret, sizes = cohorts.cohort_retention(orders)
    st.markdown(ui.section("Cohort retention", "Each row is the group of customers who first bought in that month. Cells show the share who ordered again k months later."), unsafe_allow_html=True)
    zmax = float(np.nanmax(ret.values)) if ret.size and not np.isnan(ret.values).all() else 1
    fig = go.Figure(go.Heatmap(
        z=ret.values, x=[f"+{int(c)}" for c in ret.columns], y=[f"{i}  ({int(sizes[i]):,})" for i in ret.index],
        colorscale=ui.SEQ, zmin=0, zmax=max(zmax, .5), xgap=2, ygap=2, showscale=False,
        texttemplate="%{z:.1f}", textfont=dict(color=ui.INK, size=10), hoverongaps=False,
        hovertemplate="Cohort %{y}<br>Month %{x}: %{z:.2f}% ordered again<extra></extra>"))
    ui.style(fig, h=max(380, 24 * len(ret) + 80), legend=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=ui.INK2, size=11))
    fig.update_xaxes(type="category", side="top", title=dict(text="Months since first purchase", font=dict(color=ui.MUTED, size=11)))
    show(fig)
    top_cell = np.nanmax(ret.values) if ret.size else 0
    st.markdown(
        f'<div class="callout">Even the best cell is only {top_cell:.1f}%. On Olist, coming back is the exception in every cohort, '
        f'so retention work is mostly about lifting a very small base rate.</div>', unsafe_allow_html=True)

    gaps = cohorts.second_order_gaps(orders)
    if len(gaps):
        st.markdown(ui.section("Time to the second order", f"{len(gaps):,} customers placed a second order. Half of them did it within {gaps.median():.0f} days."), unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=gaps, xbins=dict(start=0, end=420, size=14), marker=dict(color=ui.BLUE, line=dict(color=ui.SURFACE, width=1.5)),
                                     hovertemplate="%{x} days: %{y:,} customers<extra></extra>"))
        fig.add_vline(x=gaps.median(), line=dict(color=ui.INK2, width=1, dash="dot"))
        fig.add_annotation(x=gaps.median(), y=1.04, yref="paper", text=f"Median {gaps.median():.0f} days", showarrow=False, xanchor="left", xshift=4, font=dict(color=ui.INK2, size=11))
        ui.style(fig, h=300, legend=False, ymargin=20)
        fig.update_xaxes(title=dict(text="Days between first and second order", font=dict(color=ui.MUTED, size=11)))
        fig.update_yaxes(title=dict(text="Customers", font=dict(color=ui.MUTED, size=11)))
        show(fig)

# ================================================================ Forecast
with tab_fc:
    st.markdown(ui.section("Next 30 days", "Pick a series. Each one is backtested on its own last 90 days, and the best model on that backtest makes the forecast."), unsafe_allow_html=True)
    key = st.selectbox("Series", list(labels), format_func=lambda k: labels[k], label_visibility="collapsed")
    h = hist[hist["series"] == key]
    f = fut[fut["series"] == key]
    bt = D["backtest"][D["backtest"]["series"] == key]
    lb = D["leaderboard"][D["leaderboard"]["series"] == key].set_index("model")
    chosen = lb.index[lb["chosen"]][0]
    is_money = key == "revenue"
    fmt = (lambda v: ui.brl(v)) if is_money else (lambda v: f"{v:,.0f}")

    tot, last30v, prev30v = f["yhat"].sum(), h["y"].tail(30).sum(), h["y"].iloc[-60:-30].sum()
    naive_w, best_w = lb.loc["naive", "wape"], lb.loc[chosen, "wape"]
    gain = (naive_w - best_w) / naive_w if naive_w else 0
    st.markdown(ui.kpi_strip([
        ("Forecast total", fmt(tot), f"typically within <b>{lb.loc[chosen, 'total_err']:.0%}</b> on 30-day totals in backtest"),
        ("Versus last 30 days", f"{(tot / last30v - 1):+.1%}" if last30v else "n/a", f"last 30 days: {fmt(last30v)}"),
        ("Model used", MODEL_LABELS[chosen], f"daily error (WAPE) <b>{best_w:.1%}</b>"),
        ("Against seasonal naive", f"{gain:+.0%}" if chosen != "naive" else "none", f"naive WAPE {naive_w:.1%}" + (", it won here" if chosen == "naive" else "")),
    ]), unsafe_allow_html=True)

    hh = h.tail(150)
    fig = go.Figure()
    fig.add_scatter(x=f["ds"], y=f["hi"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=f["ds"], y=f["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,0.20)", name="80% range", hoverinfo="skip")
    fig.add_scatter(x=hh["ds"], y=hh["y"], mode="lines", name="Actual", line=dict(color="#e8e6df", width=1.6),
                    hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:,.0f}<extra></extra>")
    bt_in = bt[bt["ds"] >= hh["ds"].min()]
    fig.add_scatter(x=bt_in["ds"], y=bt_in[chosen], mode="lines", name="Backtest (past predictions)", line=dict(color=ui.MUTED, width=1.4, dash="dot"),
                    hovertemplate="%{x|%d %b %Y}<br>Backtest: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=f["ds"], y=f["yhat"], mode="lines", name="Forecast", line=dict(color=ui.BLUE, width=2.4),
                    hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:,.0f}<extra></extra>")
    fig.add_vline(x=h["ds"].max() + pd.Timedelta(hours=12), line=dict(color=ui.MUTED, width=1, dash="dot"))
    ui.style(fig, h=400)
    fig.update_layout(hovermode="x unified")
    if is_money:
        fig.update_yaxes(tickprefix="R$ ", tickformat="~s")
    show(fig)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(ui.section("Model comparison", "Daily error on the last 90 days. Lower is better."), unsafe_allow_html=True)
        lbs = lb.sort_values("wape", ascending=False)
        fig = go.Figure(go.Bar(
            y=[MODEL_LABELS[m] for m in lbs.index], x=lbs["wape"] * 100, orientation="h",
            marker=dict(color=[ui.MODEL_COLORS[m] for m in lbs.index], cornerradius=4),
            text=[f"{v:.1%}" + ("  chosen" if c else "") for v, c in zip(lbs["wape"], lbs["chosen"])], textposition="outside",
            textfont=dict(color=ui.INK2, size=12), cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% WAPE<extra></extra>"))
        ui.style(fig, h=260, legend=False)
        fig.update_xaxes(visible=False, range=[0, lbs["wape"].max() * 100 * 1.4])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        show(fig)
    with c2:
        st.markdown(ui.section("Expected repeat revenue by segment", "Existing customers only, next 90 days: repurchase chance times average order."), unsafe_allow_html=True)
        s4 = seg.sort_values("exp_rev")
        fig = go.Figure(go.Bar(
            y=s4.index, x=s4["exp_rev"], orientation="h", marker=dict(color=[ui.SEG_COLORS[s] for s in s4.index], cornerradius=4),
            text=[ui.brl(v) for v in s4["exp_rev"]], textposition="outside", textfont=dict(color=ui.INK2, size=12), cliponaxis=False,
            hovertemplate="%{y}<br>R$ %{x:,.0f}<extra></extra>"))
        ui.style(fig, h=300, legend=False)
        fig.update_xaxes(visible=False, range=[0, s4["exp_rev"].max() * 1.25])
        fig.update_yaxes(tickfont=dict(color=ui.INK2, size=12))
        show(fig)
    st.markdown(
        '<p class="note">Why segments are not forecast as time series: a segment is defined by recent activity, so slicing history by today\'s segments '
        'would leak the future into the past. The segment view above uses the repurchase model instead.</p>', unsafe_allow_html=True)

# ================================================================ Actions
with tab_act:
    st.markdown(ui.section("What to do with each segment", "Ordered by revenue held."), unsafe_allow_html=True)
    rows = []
    for s in seg.sort_values("revenue", ascending=False).index:
        pb = PLAYBOOK[s]
        rows.append(
            f'<div class="pb"><div class="bar4" style="background:{ui.SEG_COLORS[s]}"></div>'
            f'<div><h4>{html.escape(s)}</h4><div class="m">{int(seg.loc[s, "customers"]):,} customers<br>{ui.brl(seg.loc[s, "revenue"])} held, {seg.loc[s, "p"]:.1%} repurchase chance</div></div>'
            f'<div class="a">{html.escape(pb["action"])}<div class="m" style="margin-top:.3rem">{html.escape(pb["who"])}</div></div>'
            f'<div class="g">{html.escape(pb["goal"])}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)

    st.markdown(ui.section("Export a customer list", "Customer IDs are Olist's anonymised ids."), unsafe_allow_html=True)
    a1, a2 = st.columns([1, 1.4], gap="large")
    with a1:
        sel_seg = st.selectbox("Segment to export", SEGMENTS, index=SEGMENTS.index("Lapsed high-value"))
    with a2:
        tiers = st.multiselect("Churn risk", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    ex = cust[(cust["segment"] == sel_seg) & (cust["risk_tier"].isin(tiers))].sort_values("monetary", ascending=False)
    cols = ["customer_unique_id", "state", "top_category", "frequency", "monetary", "recency", "p_repurchase", "risk_tier"]
    ex = ex[cols].rename(columns={"monetary": "total_spend", "recency": "days_since_last_order", "p_repurchase": "repurchase_chance_90d"})
    st.markdown(f'<p class="note">{len(ex):,} customers, {ui.brl(ex["total_spend"].sum())} historical spend. Showing the top 200 by spend.</p>', unsafe_allow_html=True)
    st.dataframe(ex.head(200), hide_index=True, height=300, column_config={
        "total_spend": st.column_config.NumberColumn("Total spend", format="R$ %.0f"),
        "repurchase_chance_90d": st.column_config.NumberColumn("Repurchase chance", format="percent"),
    })
    st.download_button("Download full list (CSV)", ex.to_csv(index=False).encode(), file_name=f"{sel_seg.lower().replace(' ', '_')}_customers.csv", mime="text/csv")

# ================================================================ Method
st.markdown("&nbsp;", unsafe_allow_html=True)
cm = M["churn"]
with st.expander("How this was built and how far to trust it"):
    st.markdown(f"""
**Data.** Olist public e-commerce dataset, {M['start']} to {M['end']}. Canceled and unavailable orders are dropped. Orders by the same customer on the same day are merged into one purchase.

**Segments.** Recency and spend are scored 1 to 5 by quintile. Frequency uses fixed steps (1, 2, 3, 4+ orders), because about 97% of customers order once.

**Repurchase model.** XGBoost estimates whether a customer orders again in the next {cm['horizon_days']} days, using spend, recency, delivery and review history. It is trained on monthly snapshots and validated on the latest snapshot with a complete outcome window ({cm['val_snapshot']}).
On that holdout the AUC is **{cm['auc']:.2f}**, and the top 10% of customers reorder **{cm['top_decile_lift']:.1f}x** as often as average. That is a modest signal: only {cm['base_rate']:.1%} of customers reorder in any 90 days, so treat the scores as a way to rank customers, not to predict individuals. The model slightly over-predicts on average ({cm['mean_pred']:.1%} predicted against {cm['base_rate']:.1%} actual).

**Forecasts.** Seasonal naive, Prophet, XGBoost and a Prophet plus XGBoost blend are compared on three consecutive 30-day backtest windows, each trained only on earlier data. WAPE is total absolute error divided by total actual demand. Shaded ranges come from backtest error and are approximate.
Daily Olist demand is noisy and the history is under two years, so there is no reliable yearly seasonality. The one holiday spike in the data (Black Friday 2017) is a single observation.
""")
