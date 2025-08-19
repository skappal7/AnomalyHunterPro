import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------ Page Setup ------------------------
st.set_page_config(page_title="Anomaly Hunter Pro", layout="wide", page_icon="📊")
st.title("🚀 Anomaly Hunter Pro")

st.markdown(
    """
Welcome to **Anomaly Hunter Pro** 🕵️‍♂️ — your interactive anomaly detection tool.

- Upload a CSV (large files are processed via **chunking**) **or use sample data**.
- Select a **categorical column** (e.g., Agent, Department) and **numeric metrics**.
- Choose **IsolationForest** or **LocalOutlierFactor**.
- Click **Run** to see charts, narratives, and downloads.
"""
)

# ------------------------ Session State ------------------------
ss = st.session_state
ss.setdefault("data_loaded", False)
ss.setdefault("df", pd.DataFrame())
ss.setdefault("analysis_results", None)
ss.setdefault("current_method", None)
ss.setdefault("all_columns", [])
ss.setdefault("uploaded_bytes", None)
ss.setdefault("file_hash", None)

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("🧭 How to Use")
    st.markdown(
        """
1) Pick data source  
2) Select **categorical** + **numeric** columns  
3) Pick algorithm  
4) **Run** analysis  
5) Download CSV/PDF
"""
    )
    if st.button("🔄 Reset All", help="Clear state and restart", key="reset_all"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ------------------------ Helpers ------------------------
@st.cache_data(show_spinner=False)
def load_csv_in_chunks_from_bytes(
    file_bytes: bytes, chunksize: int = 200_000, usecols=None, downcast: bool = True
) -> pd.DataFrame:
    """Load CSV from bytes using chunking; optionally filter columns & downcast numerics."""
    try:
        if not file_bytes:
            return pd.DataFrame()
        f = BytesIO(file_bytes)
        chunks = []
        total_rows = 0
        prog = st.progress(0, text="Loading data in chunks…")

        for i, chunk in enumerate(
            pd.read_csv(f, low_memory=False, chunksize=chunksize, usecols=usecols)
        ):
            if downcast:
                for col in chunk.select_dtypes(include=["float64", "float32"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="float", errors="coerce")
                for col in chunk.select_dtypes(include=["int64", "int32"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="integer", errors="coerce")
            chunks.append(chunk)
            total_rows += len(chunk)
            prog.progress(min(0.02 + i * 0.03, 1.0), text=f"Loaded ~{total_rows:,} rows…")
        prog.empty()
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    agents = [f"Agent_{i}" for i in range(1, 31)]
    departments = ["Electronics", "Clothing", "Groceries", "Home", "Toys", "Beauty"]
    weeks = pd.date_range(start="2024-01-01", periods=60, freq="W")

    rows = []
    for week in weeks:
        for dept in departments:
            base = np.random.randint(150, 260)
            for agent in agents:
                sales = np.random.poisson(lam=base)
                r = np.random.rand()
                if r < 0.03:
                    sales *= np.random.randint(3, 6)      # spikes
                elif r < 0.06:
                    sales = max(0, int(sales * np.random.uniform(0.1, 0.4)))  # dropouts
                rows.append([agent, dept, week, sales])
    return pd.DataFrame(rows, columns=["Agent", "Department", "Week", "Sales"])

def ensure_loaded_subset(required_cols):
    """Load/trim df to contain required columns; for uploaded path use chunked loader."""
    # For sample data, df is already in memory
    if ss.get("source") == "sample":
        if ss.df.empty:
            ss.df = load_sample_data()
            ss.data_loaded = True
        # Trim to required cols if present
        missing = [c for c in required_cols if c not in ss.df.columns]
        if missing:
            raise ValueError(f"Columns not found in sample data: {', '.join(missing)}")
        return ss.df[required_cols].copy()

    # For upload, use bytes with chunking (only required cols)
    if not ss.uploaded_bytes:
        raise ValueError("No uploaded file available.")
    df = load_csv_in_chunks_from_bytes(ss.uploaded_bytes, usecols=list(set(required_cols)))
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Check file and column selections.")
    # Final sanity check
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Selected columns not found in uploaded file: {', '.join(missing)}")
    ss.data_loaded = True
    return df[required_cols].copy()

def run_analysis(df_in: pd.DataFrame, categorical_col: str, numeric_cols: list, method: str):
    # numeric coercion & cleaning
    df = df_in.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols, how="all")
    if len(df) < 10:
        raise ValueError("After cleaning, insufficient data remains (need at least 10 rows).")

    for col in numeric_cols:
        med = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(med) else med)

    X = df[numeric_cols].values
    if method == "IsolationForest":
        contamination = min(0.1, max(0.01, 50 / len(X)))
        model = IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42, n_jobs=1
        )
        model.fit(X)
        labels = model.predict(X)
        scores = -model.score_samples(X)
        reason = "**Why flagged?** IsolationForest isolates records that require fewer random splits."
    else:
        n_neighbors = min(20, max(5, int(np.sqrt(len(X)))))
        contamination = min(0.1, max(0.01, 50 / len(X)))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=1)
        labels = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_
        reason = "**Why flagged?** LOF compares local density and flags sparse neighborhoods."

    df["anomaly"] = pd.Series(labels, index=df.index).map({1: "Normal", -1: "Anomaly"})
    df["anomaly_score"] = scores

    ss.analysis_results = dict(
        df_analysis=df,
        method=method,
        reason_blurb=reason,
        categorical_col=categorical_col,
        numeric_cols=numeric_cols,
    )
    ss.current_method = method

# ------------------------ Data Source ------------------------
source = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data"], horizontal=True, key="source_radio")
ss.source = "upload" if source == "Upload CSV" else "sample"

if ss.source == "sample":
    if not ss.data_loaded or ss.df.empty:
        with st.spinner("Loading sample data..."):
            ss.df = load_sample_data()
            ss.data_loaded = True
    st.success(f"✅ Sample data ready: {ss.df.shape[0]:,} rows, {ss.df.shape[1]:,} columns")
    available_columns = ss.df.columns.tolist()
else:
    uploaded = st.file_uploader("📂 Upload CSV", type=["csv"], help="Large files load via chunking.")
    if uploaded is not None:
        # Stable bytes + hash
        uploaded_bytes = uploaded.getvalue()
        file_hash = hash(uploaded_bytes)
        if ss.file_hash != file_hash:
            ss.file_hash = file_hash
            ss.uploaded_bytes = uploaded_bytes
            ss.data_loaded = False
            ss.analysis_results = None
            # Peek columns
            try:
                peek = pd.read_csv(BytesIO(uploaded_bytes), nrows=200)
                ss.all_columns = peek.columns.tolist()
                st.caption(f"Detected columns: {', '.join(ss.all_columns)}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        else:
            if ss.all_columns:
                st.caption(f"Using previously detected columns: {', '.join(ss.all_columns)}")

    available_columns = ss.all_columns if ss.all_columns else []

# ------------------------ Column Selection ------------------------
if ss.source == "sample" or (ss.source == "upload" and available_columns):
    categorical_col = st.selectbox(
        "📌 Select Categorical Column", options=available_columns, key="cat_col"
    )

    # Guess numeric choices
    if ss.source == "sample":
        numeric_guess = ss.df.select_dtypes(include=np.number).columns.tolist()
    else:
        try:
            # Use a small sample from uploaded bytes to infer numeric columns
            sdf = pd.read_csv(BytesIO(ss.uploaded_bytes), nrows=500) if ss.uploaded_bytes else pd.DataFrame()
            numeric_guess = sdf.select_dtypes(include=np.number).columns.tolist()
        except Exception:
            numeric_guess = []

    numeric_cols = st.multiselect(
        "📊 Select Numeric Columns",
        options=(numeric_guess if numeric_guess else available_columns),
        help="Metrics used by the model(s)",
        key="num_cols",
    )

    # ------------------------ Algorithm & Run UI (always visible after column pick) ------------------------
    algo_disabled = not (categorical_col and numeric_cols)
    method = st.radio(
        "⚙️ Choose Anomaly Detection Method",
        ["IsolationForest", "LocalOutlierFactor"],
        horizontal=True,
        key="method_radio",
        disabled=algo_disabled,
        help="IsolationForest isolates global outliers; LOF finds local density outliers.",
    )

    run_disabled = algo_disabled or (method is None)
    run_clicked = st.button(
        "🔍 Run Anomaly Analysis",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
        key="run_analysis_btn",
    )

    # ------------------------ Run Analysis ------------------------
    if run_clicked:
        try:
            required_cols = [categorical_col] + list(numeric_cols)
            df_subset = ensure_loaded_subset(required_cols)
            run_analysis(df_subset, categorical_col, list(numeric_cols), method)
            st.success("✅ Analysis completed!")
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")

    # ------------------------ Results ------------------------
    if ss.analysis_results:
        res = ss.analysis_results
        df_analysis = res["df_analysis"]
        categorical_col = res["categorical_col"]
        numeric_cols = res["numeric_cols"]
        method = res["method"]
        reason_blurb = res["reason_blurb"]

        st.subheader("📈 Anomaly Distribution")
        counts = df_analysis["anomaly"].value_counts()
        total = int(counts.sum())
        anomalies = int(counts.get("Anomaly", 0))
        pct = (anomalies / total * 100) if total > 0 else 0.0

        if total > 0:
            cdf = pd.DataFrame({"label": counts.index, "count": counts.values})
            cdf["percent"] = (cdf["count"] / total * 100).round(2)
            fig_pie = px.pie(
                cdf,
                names="label",
                values="count",
                hole=0.55,
                hover_data=["percent"],
                title=f"Anomalies: {anomalies:,} / {total:,}  (≈ {pct:.2f}%)",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Share of records flagged as anomalous.")

            # Narrative
            st.subheader("🧠 Narrative Insight")
            try:
                grp = df_analysis.groupby([categorical_col, "anomaly"]).size().unstack(fill_value=0)
                for col in ("Anomaly", "Normal"):
                    if col not in grp.columns:
                        grp[col] = 0
                denom = (grp["Anomaly"] + grp["Normal"]).replace(0, np.nan)
                grp["anomaly_rate"] = (grp["Anomaly"] / denom).fillna(0.0)

                top_rate = grp.sort_values("anomaly_rate", ascending=False).head(5)
                lines = [
                    f"Out of **{total:,}** records, **{anomalies:,}** (≈ **{pct:.2f}%**) were flagged using **{method}**.",
                    reason_blurb,
                ]
                if not top_rate.empty:
                    leader = str(top_rate.index[0])
                    leader_rate = float(top_rate.iloc[0]["anomaly_rate"] * 100)
                    lines.append(f"Highest anomaly rate: **{leader}** at **{leader_rate:.1f}%** of its records.")

                strongest = df_analysis.nlargest(3, "anomaly_score")
                if not strongest.empty:
                    examples = ", ".join(
                        f"{row[categorical_col]} (score={row['anomaly_score']:.2f})"
                        for _, row in strongest.iterrows()
                    )
                    lines.append(f"Most extreme outliers by score: {examples}.")
                st.markdown("\n".join(f"- {x}" for x in lines))
            except Exception as e:
                st.warning(f"Could not generate narrative: {e}")
                grp = None  # ensure defined

            # Top categories
            st.subheader(f"🏆 Top 5 {categorical_col} by Anomaly Rate")
            try:
                if grp is not None and not grp.empty:
                    top_table = (
                        grp.sort_values("anomaly_rate", ascending=False)
                        .head(5)
                        .reset_index()[[categorical_col, "Anomaly", "Normal", "anomaly_rate"]]
                    )
                    if not top_table.empty:
                        top_table["Anomaly Rate (%)"] = (top_table["anomaly_rate"] * 100).round(2)
                        st.dataframe(
                            top_table[[categorical_col, "Anomaly", "Normal", "Anomaly Rate (%)"]],
                            use_container_width=True,
                        )

                        bar = px.bar(top_table, x=categorical_col, y="Anomaly Rate (%)", text="Anomaly Rate (%)")
                        bar.update_traces(texttemplate="%{text}", textposition="outside")
                        st.plotly_chart(bar, use_container_width=True)
                    else:
                        st.info("Not enough data to compute top categories.")
            except Exception as e:
                st.warning(f"Could not compute top categories: {e}")

            # Results table
            st.subheader("📋 Results Table")
            view_cols = [categorical_col] + list(numeric_cols) + ["anomaly", "anomaly_score"]
            safe_cols = [c for c in view_cols if c in df_analysis.columns]
            st.dataframe(df_analysis[safe_cols], use_container_width=True, height=420)

            # Viz
            if len(numeric_cols) >= 2 and all(c in df_analysis.columns for c in numeric_cols[:2]):
                try:
                    fig = px.scatter(
                        df_analysis,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        color="anomaly",
                        hover_data=[categorical_col, "anomaly_score"],
                        title="Anomalies vs Normal Data",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate scatter plot: {e}")
            elif len(numeric_cols) >= 1 and numeric_cols[0] in df_analysis.columns:
                try:
                    fig = px.histogram(
                        df_analysis, x=numeric_cols[0], color="anomaly", title="Value Distribution with Anomalies"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate histogram: {e}")

            # Downloads
            c1, c2 = st.columns(2)
            with c1:
                try:
                    output_csv = BytesIO()
                    df_analysis[safe_cols].to_csv(output_csv, index=False)
                    st.download_button(
                        "⬇️ Download Results (CSV)",
                        data=output_csv.getvalue(),
                        file_name="anomaly_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"CSV download failed: {e}")

            with c2:
                try:
                    def build_pdf():
                        buf = BytesIO()
                        doc = SimpleDocTemplate(buf, pagesize=A4)
                        styles = getSampleStyleSheet()
                        elems = []
                        elems.append(Paragraph("<b>Anomaly Hunter Pro – Insights Report</b>", styles["Title"]))
                        elems.append(Spacer(1, 12))
                        elems.append(Paragraph(f"Total Rows: {total:,}", styles["Normal"]))
                        elems.append(Paragraph(f"Anomalies: {anomalies:,} (≈ {pct:.2f}%)", styles["Normal"]))
                        elems.append(Paragraph(f"Method: {method}", styles["Normal"]))
                        elems.append(Spacer(1, 10))

                        if grp is not None and not grp.empty:
                            top_tbl = grp.sort_values("anomaly_rate", ascending=False).head(10)
                            data_table = [[categorical_col, "Anomaly", "Normal", "Anomaly Rate"]]
                            for idx, r in top_tbl.iterrows():
                                data_table.append(
                                    [str(idx), str(int(r["Anomaly"])), str(int(r["Normal"])), f"{r['anomaly_rate']*100:.1f}%"]
                                )
                            t = Table(data_table)
                            t.setStyle(
                                TableStyle(
                                    [
                                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                    ]
                                )
                            )
                            elems.append(t)
                        doc.build(elems)
                        buf.seek(0)
                        return buf

                    pdf_buf = build_pdf()
                    st.download_button(
                        "⬇️ Download Insights (PDF)",
                        data=pdf_buf,
                        file_name="anomaly_insights_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}")

else:
    st.info("Please upload a CSV or use sample data to continue.")

# ------------------------ Footer ------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Developed with 💗 CE Innovation Team 2025</div>",
    unsafe_allow_html=True,
)
