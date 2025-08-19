import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Anomaly Hunter Pro", layout="wide", page_icon="📊")
st.title("🚀 Anomaly Hunter Pro")

st.markdown(
    """
Welcome to **Anomaly Hunter Pro** 🕵️‍♂️ — your interactive anomaly detection tool.

- Upload a CSV (large files are processed via **chunking**) **or use sample data**.
- Select a **categorical column** (e.g., Agent, Department) and **numeric metrics**.
- Choose a method (**IsolationForest** or **LocalOutlierFactor**).
- Get **visual anomaly distribution**, **narrative data stories**, **ranked category tables**, and download **results & insights** (CSV/PDF).
"""
)

# Sidebar Wizard Guide
with st.sidebar:
    st.header("🧭 Step-by-Step Guide")
    st.markdown(
        """
1. **Choose Data**: Upload CSV or use sample.
2. **Pick Columns**: Categorical + numeric metrics.
3. **Method**: IsolationForest (global) or LOF (local).
4. **Run**: See charts, tables, and narrative.
5. **Download**: Results & PDF insights.

ℹ️ Chunking and numeric downcasting are used for stability with large files.
"""
    )

# -----------------------------
# Data source selection
# -----------------------------
option = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data"], horizontal=True)

@st.cache_data(show_spinner=False)
def load_csv_in_chunks(file, chunksize=200_000, usecols=None, downcast=True):
    """Load CSV in chunks; optionally select columns and downcast numerics."""
    try:
        chunks = []
        progress_bar = st.progress(0, text="Loading data in chunks…")
        total_rows = 0
        for i, chunk in enumerate(pd.read_csv(file, low_memory=False, chunksize=chunksize, usecols=usecols)):
            if downcast:
                for col in chunk.select_dtypes(include=["float64"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="float")
                for col in chunk.select_dtypes(include=["int64"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
            chunks.append(chunk)
            total_rows += len(chunk)
            progress_bar.progress(min(0.02 + i * 0.03, 1.0), text=f"Loaded ~{total_rows:,} rows…")
        progress_bar.empty()
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_sample_data():
    np.random.seed(42)
    agents = [f"Agent_{i}" for i in range(1, 31)]
    departments = ["Electronics", "Clothing", "Groceries", "Home", "Toys", "Beauty"]
    weeks = pd.date_range(start="2024-01-01", periods=60, freq="W")

    data = []
    for week in weeks:
        for dept in departments:
            base = np.random.randint(150, 260)
            for agent in agents:
                sales = np.random.poisson(lam=base)
                r = np.random.rand()
                if r < 0.03:
                    sales *= np.random.randint(3, 6)
                elif r < 0.06:
                    sales = max(0, int(sales * np.random.uniform(0.1, 0.4)))
                data.append([agent, dept, week, sales])

    df = pd.DataFrame(data, columns=["Agent", "Department", "Week", "Sales"])
    return df

# Upload flow (sample a few rows for columns)
uploaded_file = None
sample_for_columns = None

if option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"], help="Upload a CSV. Large files are processed via chunking.")
    if uploaded_file is not None:
        with st.spinner("Sampling a few rows to list columns…"):
            sample_for_columns = pd.read_csv(uploaded_file, nrows=5)
        st.caption(f"Detected columns: {', '.join(sample_for_columns.columns.astype(str))}")
else:
    st.info("Using dynamically generated sample dataset with multiple agents, departments, and weeks.")

# Column selection
if option == "Use Sample Data":
    df = load_sample_data()
    all_columns = df.columns.tolist()
else:
    if uploaded_file is None:
        df = pd.DataFrame(); all_columns = []
    else:
        all_columns = sample_for_columns.columns.tolist() if sample_for_columns is not None else []

if all_columns:
    categorical_col = st.selectbox("📌 Select Categorical Column", options=all_columns, help="e.g., Agent or Department")
    # Guess numeric columns
    if option == "Use Sample Data":
        numeric_guess = df.select_dtypes(include=np.number).columns.tolist()
    elif sample_for_columns is not None:
        numeric_guess = sample_for_columns.select_dtypes(include=np.number).columns.tolist()
    else:
        numeric_guess = []
    numeric_cols = st.multiselect("📊 Select Numeric Columns", options=numeric_guess if numeric_guess else all_columns, help="Metrics used by the model(s)")

    # Efficient load path for uploaded files
    if option == "Upload CSV" and uploaded_file is not None:
        usecols_selected = list({categorical_col, *numeric_cols}) if numeric_cols else [categorical_col]
        if st.button("⚡ Load Data Efficiently (Selected Columns Only)"):
            df = load_csv_in_chunks(uploaded_file, usecols=usecols_selected)
    if option == "Upload CSV" and uploaded_file is not None and 'df' not in locals():
        df = load_csv_in_chunks(uploaded_file)

# Proceed if df exists
if 'df' in locals() and not df.empty and 'categorical_col' in locals():
    st.success(f"✅ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    method = st.radio("⚙️ Choose Anomaly Detection Method", ["IsolationForest", "LocalOutlierFactor"], horizontal=True, help="IsolationForest isolates global outliers; LOF finds sparse local densities")

    if numeric_cols:
        # Prepare data
        df_analysis = df[[categorical_col] + numeric_cols].copy()
        df_analysis[numeric_cols] = df_analysis[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df_analysis[numeric_cols] = df_analysis[numeric_cols].fillna(df_analysis[numeric_cols].median())

        X = df_analysis[numeric_cols].values

        if method == "IsolationForest":
            model = IsolationForest(n_estimators=256, contamination='auto', random_state=42, n_jobs=-1)
            model.fit(X)
            labels = model.predict(X)
            scores = -model.score_samples(X)  # higher => more anomalous
            reason_blurb = (
                "**Why flagged?** IsolationForest isolates records that require fewer random splits; these have uncommon metric combinations."
            )
        else:
            n_neighbors = min(35, max(5, int(np.sqrt(len(X)))))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto', n_jobs=-1)
            labels = lof.fit_predict(X)
            scores = -lof.negative_outlier_factor_  # higher => more anomalous
            reason_blurb = (
                "**Why flagged?** LOF compares local density—points in sparse neighborhoods relative to their neighbors are flagged."
            )

        df_analysis["anomaly"] = pd.Series(labels, index=df_analysis.index).map({1: "Normal", -1: "Anomaly"})
        df_analysis["anomaly_score"] = pd.Series(scores, index=df_analysis.index)

        # -----------------------------
        # Distribution (percentage, donut)
        # -----------------------------
        st.subheader("📈 Anomaly Distribution")
        counts = df_analysis["anomaly"].value_counts()
        total = int(counts.sum()) if not counts.empty else 0
        anomalies = int(counts.get("Anomaly", 0))
        pct = float((anomalies / total * 100)) if total else 0.0
        pct = max(0.0, min(100.0, pct))  # clamp for safety

        cdf = pd.DataFrame({"label": counts.index, "count": counts.values})
        cdf["percent"] = (cdf["count"] / total * 100).round(2) if total else 0
        fig_pie = px.pie(cdf, names="label", values="count", hole=0.55, hover_data=["percent"], title=f"Anomalies: {anomalies:,} / {total:,}  (≈ {pct:.2f}%)")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Share of records flagged as anomalous. Percentages are relative to total analyzed rows.")

        # -----------------------------
        # Narrative data story
        # -----------------------------
        st.subheader("🧠 Narrative Insight")
        grp = df_analysis.groupby([categorical_col, "anomaly"]).size().unstack(fill_value=0)
        if "Anomaly" not in grp.columns: grp["Anomaly"] = 0
        if "Normal" not in grp.columns: grp["Normal"] = 0
        grp["anomaly_rate"] = (grp["Anomaly"] / (grp["Anomaly"] + grp["Normal"]).clip(lower=1)).clip(0,1)
        top_rate = grp.sort_values("anomaly_rate", ascending=False).head(5)

        story_lines = [
            f"Out of **{total:,}** records, **{anomalies:,}** (≈ **{pct:.2f}%**) were flagged using **{method}**.",
            reason_blurb,
        ]
        if not top_rate.empty:
            leader = top_rate.index[0]
            story_lines.append(
                f"Highest anomaly rate: **{leader}** at **{(top_rate.iloc[0]['anomaly_rate']*100):.1f}%** of its records."
            )
        strongest = df_analysis.sort_values("anomaly_score", ascending=False).head(3)
        if not strongest.empty:
            examples = ", ".join([f"{row[categorical_col]} (score={row['anomaly_score']:.2f})" for _, row in strongest.iterrows()])
            story_lines.append(f"Most extreme outliers by score: {examples}.")
        st.markdown("\n".join([f"- {s}" for s in story_lines]))

        # -----------------------------
        # Top 5 categories table + bar
        # -----------------------------
        st.subheader(f"🏆 Top 5 {categorical_col} by Anomaly Rate")
        top_table = (
            grp.sort_values("anomaly_rate", ascending=False)
            .head(5)
            .reset_index()[[categorical_col, "Anomaly", "Normal", "anomaly_rate"]]
        )
        top_table["anomaly_rate"] = (top_table["anomaly_rate"] * 100).round(2)
        st.dataframe(top_table.rename(columns={"anomaly_rate": "Anomaly Rate (%)"}), use_container_width=True)

        bar = px.bar(top_table, x=categorical_col, y="Anomaly Rate (%)", text="Anomaly Rate (%)")
        bar.update_traces(texttemplate="%{text}", textposition="outside")
        st.plotly_chart(bar, use_container_width=True)

        # -----------------------------
        # Results Table (fast) + Styled preview with pill badges
        # -----------------------------
        st.subheader("📋 Results Table")
        view_cols = [categorical_col] + numeric_cols + ["anomaly", "anomaly_score"]

        # Fast dataframe for full data (no HTML styling)
        st.dataframe(df_analysis[view_cols], use_container_width=True, height=420)

        # Styled HTML preview (first 150 rows) with pill badges
        st.caption("Styled preview (first 150 rows) with pill badges")
        preview = df_analysis[view_cols].head(150).copy()
        def pill_html(label: str) -> str:
            if isinstance(label, str) and label.lower().startswith("anomaly"):
                return "<span style=\"background:linear-gradient(90deg,#ff4d4f,#ff7875);color:#fff;border-radius:999px;padding:2px 10px;\">Anomaly</span>"
            return "<span style=\"background:linear-gradient(90deg,#1890ff,#69c0ff);color:#fff;border-radius:999px;padding:2px 10px;\">Normal</span>"
        preview["anomaly_pill"] = preview["anomaly"].apply(pill_html)
        # Build HTML table
        cols_html = ''.join([f"<th style='padding:6px 10px;text-align:left'>{c}</th>" for c in [categorical_col] + numeric_cols + ["anomaly_pill", "anomaly_score"]])
        rows_html = []
        for _, r in preview.iterrows():
            cells = []
            for c in [categorical_col] + numeric_cols:
                cells.append(f"<td style='padding:6px 10px'>{r[c]}</td>")
            cells.append(f"<td style='padding:6px 10px;text-align:center'>{r['anomaly_pill']}</td>")
            cells.append(f"<td style='padding:6px 10px'>{r['anomaly_score']:.2f}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        html_table = f"""
        <div style='overflow:auto; max-height:420px; border:1px solid #eee; border-radius:8px;'>
        <table style='border-collapse:separate;border-spacing:0;width:100%'>
          <thead style='position:sticky;top:0;background:#fafafa'><tr>{cols_html}</tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """
        st.markdown(html_table, unsafe_allow_html=True)

        # -----------------------------
        # 2D visualization (>=2 metrics) or histogram
        # -----------------------------
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                df_analysis,
                x=numeric_cols[0], y=numeric_cols[1], color="anomaly",
                hover_data=[categorical_col, "anomaly_score"],
                title="Anomalies vs Normal Data (colored by flag)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df_analysis, x=numeric_cols[0], color="anomaly", title="Distribution of Values with Anomalies Highlighted")
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Downloads
        # -----------------------------
        output_csv = BytesIO(); df_analysis.to_csv(output_csv, index=False)
        st.download_button("⬇️ Download Results as CSV", data=output_csv.getvalue(), file_name="anomaly_results.csv", mime="text/csv")

        def generate_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("<b>Anomaly Hunter Pro – Insights Report</b>", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Total Rows: {total:,}", styles["Normal"]))
            elements.append(Paragraph(f"Anomalies: {anomalies:,} (≈ {pct:.2f}%)", styles["Normal"]))
            elements.append(Paragraph(f"Method: {method}", styles["Normal"]))
            elements.append(Spacer(1, 10))
            top_tbl = grp.sort_values("anomaly_rate", ascending=False).head(10)[["Anomaly", "Normal", "anomaly_rate"]]
            data_table = [[categorical_col, "Anomaly", "Normal", "Anomaly Rate"]] + [
                [idx, int(r["Anomaly"]), int(r["Normal"]), f"{r['anomaly_rate']*100:.1f}%"] for idx, r in top_tbl.iterrows()
            ]
            t = Table(data_table)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ]))
            elements.append(t)
            doc.build(elements)
            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf()
        st.download_button("⬇️ Download Insights Report (PDF)", data=pdf_buffer, file_name="anomaly_insights_report.pdf", mime="application/pdf")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed with 💗 CE Innovation Team 2025</div>", unsafe_allow_html=True)
