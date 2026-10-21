import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Stock Quality & Valuation Analyzer", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).with_name("universe.csv")
# Replace this with the ticker used in your 60 Second Pitch before deploying.
PITCH_TICKER_TO_EXCLUDE = "REPLACE_WITH_PITCH_TICKER"

REQUIRED = ["ticker", "name", "sector", "market_cap", "total_assets", "net_income",
            "operating_cash_flow", "gross_profit", "book_equity", "total_liabilities",
            "current_assets", "current_liabilities", "retained_earnings", "ebit", "revenue",
            "cash", "total_debt"]

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("universe.csv is missing required columns: " + ", ".join(missing))
    for c in REQUIRED:
        if c not in ["ticker", "name", "sector"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].fillna(df["ticker"])
    df["sector"] = df["sector"].fillna("Unknown")
    return df

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Course slide formula: scaled total accruals = (net income - operating cash flow) / total assets.
    assets = d["total_assets"].where(d["total_assets"] > 0)
    d["STA"] = (d["net_income"] - d["operating_cash_flow"]) / assets
    # Gross profitability and book-to-market are BUY-side measures listed in the course slides.
    d["GPA"] = d["gross_profit"] / assets
    d["Book_to_Market"] = d["book_equity"] / d["market_cap"].where(d["market_cap"] > 0)
    # Operating-asset proxy: total assets less cash; operating-liability proxy: total liabilities less debt.
    # Confirm this operational definition against your course notes before submission.
    operating_assets = d["total_assets"] - d["cash"]
    operating_liabilities = d["total_liabilities"] - d["total_debt"]
    d["Scaled_NOA"] = (operating_assets - operating_liabilities) / assets
    # Standard Altman Z-score proxy for non-financial public manufacturers. Use only as context, not as the SELL rule.
    wc = d["current_assets"] - d["current_liabilities"]
    market_equity = d["market_cap"]
    d["Altman_Z"] = (
        1.2 * (wc / assets)
        + 1.4 * (d["retained_earnings"] / assets)
        + 3.3 * (d["ebit"] / assets)
        + 0.6 * (market_equity / d["total_liabilities"].where(d["total_liabilities"] > 0))
        + 1.0 * (d["revenue"] / assets)
    )
    return d

st.title("📊 Stock Quality & Valuation Analyzer")
st.caption("Educational stock screen · Public-company financial statements · Interactive BUY and SELL signals")

if not DATA_PATH.exists():
    st.error("universe.csv was not found. Generate it using build_universe.py in Google Colab, then upload it beside app.py in your GitHub repository.")
    st.stop()
try:
    raw = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if PITCH_TICKER_TO_EXCLUDE != "REPLACE_WITH_PITCH_TICKER":
    raw = raw[raw["ticker"] != PITCH_TICKER_TO_EXCLUDE.upper()]

df = calculate_metrics(raw)

with st.sidebar:
    st.header("Screening controls")
    st.caption("Changing a control reruns the screen and updates the tables and chart.")
    sectors = sorted(df["sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect("Sectors", sectors, default=sectors)
    valid_caps = df["market_cap"].dropna() / 1e9
    cap_min = float(max(0, np.nanmin(valid_caps))) if len(valid_caps) else 0.0
    cap_max = float(max(cap_min + 0.1, np.nanmax(valid_caps))) if len(valid_caps) else 1000.0
    min_cap = st.slider("Minimum market cap (US$ billions)", min_value=0.0, max_value=max(1.0, cap_max), value=min(0.3, max(1.0, cap_max)), step=0.1)
    st.divider()
    st.subheader("SELL screen · earnings quality")
    sta_pct = st.slider("STA red-flag percentile", 50, 99, 80, help="Flags companies in the highest selected percentile of scaled total accruals.")
    noa_pct = st.slider("Scaled NOA red-flag percentile", 50, 99, 80, help="Flags companies in the highest selected percentile of scaled net operating assets.")
    sell_flags_needed = st.slider("Minimum SELL red flags", 1, 2, 1)
    st.divider()
    st.subheader("BUY screen · quality and value")
    gpa_pct = st.slider("Minimum GPA percentile", 50, 95, 75)
    bm_pct = st.slider("Minimum book-to-market percentile", 50, 95, 75)
    buy_both = st.checkbox("Require both BUY signals", value=True)

filtered = df[df["sector"].isin(selected_sectors)].copy()
filtered = filtered[filtered["market_cap"] >= min_cap * 1e9]

# Calculate percentiles on the currently eligible peer group, not on a fixed outside sample.
if filtered.empty:
    st.warning("No companies meet these sector and market-cap filters. Broaden the filters to continue.")
    st.stop()

sta_cut = filtered["STA"].quantile(sta_pct / 100) if filtered["STA"].notna().any() else np.nan
noa_cut = filtered["Scaled_NOA"].quantile(noa_pct / 100) if filtered["Scaled_NOA"].notna().any() else np.nan
gpa_cut = filtered["GPA"].quantile(gpa_pct / 100) if filtered["GPA"].notna().any() else np.nan
bm_cut = filtered["Book_to_Market"].quantile(bm_pct / 100) if filtered["Book_to_Market"].notna().any() else np.nan
filtered["STA_Flag"] = filtered["STA"].notna() & (filtered["STA"] >= sta_cut)
filtered["NOA_Flag"] = filtered["Scaled_NOA"].notna() & (filtered["Scaled_NOA"] >= noa_cut)
filtered["SELL_Flag_Count"] = filtered[["STA_Flag", "NOA_Flag"]].sum(axis=1)
filtered["SELL_Candidate"] = filtered["SELL_Flag_Count"] >= sell_flags_needed
filtered["GPA_Flag"] = filtered["GPA"].notna() & (filtered["GPA"] >= gpa_cut)
filtered["BM_Flag"] = filtered["Book_to_Market"].notna() & (filtered["Book_to_Market"] >= bm_cut)
filtered["BUY_Candidate"] = (filtered["GPA_Flag"] & filtered["BM_Flag"]) if buy_both else (filtered["GPA_Flag"] | filtered["BM_Flag"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies screened", f"{len(filtered):,}")
c2.metric("SELL candidates", f"{int(filtered['SELL_Candidate'].sum()):,}")
c3.metric("BUY candidates", f"{int(filtered['BUY_Candidate'].sum()):,}")
c4.metric("Missing STA values", f"{int(filtered['STA'].isna().sum()):,}")
st.caption(f"Dynamic peer-group cutoffs: STA ≥ {sta_cut:.3f} · Scaled NOA ≥ {noa_cut:.3f} · GPA ≥ {gpa_cut:.3f} · B/M ≥ {bm_cut:.3f}")

sell_tab, buy_tab, chart_tab, company_tab, method_tab = st.tabs(["SELL candidates", "BUY candidates", "Visual analysis", "Company drill-down", "Methodology"])
with sell_tab:
    st.subheader("Potential earnings-quality concerns")
    st.write("SELL flags are based on high scaled total accruals and/or high scaled net operating assets. These are warning signals for further investigation, not proof of manipulation or an automatic sell instruction.")
    sell = filtered[filtered["SELL_Candidate"]].sort_values(["SELL_Flag_Count", "STA"], ascending=False)
    cols = ["ticker", "name", "sector", "market_cap", "STA", "Scaled_NOA", "Altman_Z", "SELL_Flag_Count"]
    st.dataframe(sell[cols], use_container_width=True, hide_index=True)
    st.download_button("Download SELL candidates CSV", sell[cols].to_csv(index=False), "sell_candidates.csv", "text/csv")
with buy_tab:
    st.subheader("Potential quality-at-a-reasonable-value candidates")
    st.write("BUY candidates rank well on gross profitability and book-to-market relative to the currently eligible peer group. High book-to-market can also reflect distress, so inspect debt, cash flow, and business conditions.")
    buy = filtered[filtered["BUY_Candidate"]].sort_values(["GPA", "Book_to_Market"], ascending=False)
    cols = ["ticker", "name", "sector", "market_cap", "GPA", "Book_to_Market", "STA", "Altman_Z"]
    st.dataframe(buy[cols], use_container_width=True, hide_index=True)
    st.download_button("Download BUY candidates CSV", buy[cols].to_csv(index=False), "buy_candidates.csv", "text/csv")
with chart_tab:
    chart_df = filtered.dropna(subset=["GPA", "Book_to_Market"]).copy()
    chart_df["Screen result"] = np.select([chart_df["SELL_Candidate"], chart_df["BUY_Candidate"]], ["SELL candidate", "BUY candidate"], default="Neither")
    fig = px.scatter(chart_df, x="Book_to_Market", y="GPA", color="Screen result", hover_name="ticker", hover_data=["name", "sector", "STA", "Scaled_NOA"], title="Gross profitability vs. book-to-market")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The chart is a peer comparison, not a forecast of future returns. Extreme values may distort the axes.")
with company_tab:
    options = sorted(filtered["ticker"].dropna().unique().tolist())
    ticker = st.selectbox("Choose a company", options)
    r = filtered.loc[filtered["ticker"] == ticker].iloc[0]
    st.subheader(f"{r['name']} ({r['ticker']})")
    st.write(f"Sector: {r['sector']} · Market capitalization: ${r['market_cap']/1e9:,.2f} billion")
    metrics = st.columns(4)
    for col, label, key in zip(metrics, ["STA", "Scaled NOA", "GPA", "Book-to-market"], ["STA", "Scaled_NOA", "GPA", "Book_to_Market"]):
        value = r[key]
        col.metric(label, "N/A" if pd.isna(value) else f"{value:.3f}")
    reasons = []
    if r["STA_Flag"]: reasons.append("STA is in the highest selected percentile of the peer group.")
    if r["NOA_Flag"]: reasons.append("Scaled NOA is in the highest selected percentile of the peer group.")
    if r["GPA_Flag"]: reasons.append("Gross profitability is in the highest selected percentile of the peer group.")
    if r["BM_Flag"]: reasons.append("Book-to-market is in the highest selected percentile of the peer group.")
    if not reasons: reasons = ["This company does not currently meet any selected percentile flag."]
    st.markdown("**Why this company is flagged (or not):**")
    for reason in reasons: st.write("- " + reason)
    st.write("**Interpretation prompt:** Compare these ratios with competitors, read the latest annual report, inspect cash-flow trends and debt obligations, and consider whether the apparent signal is explained by the company's business model.")
with method_tab:
    st.subheader("Formulas and limitations")
    st.markdown("""
- **Scaled total accruals (STA):** `(Net income − operating cash flow) / total assets`. 
- **Gross profitability (GPA):** `gross profit / total assets`. Higher values indicate more gross profit per dollar of assets.
- **Book-to-market (B/M):** `book equity / market capitalization`. Higher values mean accounting equity is larger relative to market value; it is not automatically evidence of undervaluation.
- **Scaled net operating assets (Scaled NOA):** `(total assets − cash − (total liabilities − total debt)) / total assets`. 
- **Altman Z:** the app includes a commonly used public-manufacturer specification as context only. Different versions apply to different company types; it is not used to construct the BUY rule.

""")
