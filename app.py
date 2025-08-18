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

st.set_page_config(page_title="Anomaly Hunter Pro", layout="wide", page_icon="📊")
st.title("🚀 Anomaly Hunter Pro")

st.markdown("""
Welcome to **Anomaly Hunter Pro** 🕵️‍♂️ — your interactive anomaly detection tool.

- Upload a CSV (supports >1GB with chunked reading) **or use sample data**.
- Select a **categorical column** (e.g., Agent, Department).
- Select one or more **numeric metrics**.
- Choose a detection method (**IsolationForest** or **LocalOutlierFactor**).
- View **dynamic visualizations**, **insights**, and **download results** as CSV or PDF.

💡 *Tooltips and guidance are provided throughout to help you interpret the results.*
""")

# Sidebar Wizard Guide
with st.sidebar:
    st.header("🧭 Step-by-Step Guide")
    st.markdown("""
    1. **Upload Data**: Upload a CSV file (supports >1GB) or use sample data.
    2. **Choose Columns**: Select categorical & numeric columns.
    3. **Select Method**: Pick IsolationForest or LOF.
    4. **Analyze**: View anomalies in plots & tables.
    5. **Download**: Export results & insights as CSV/PDF.
    
    ℹ️ Hover over tooltips in the main panel for more guidance.
    """)

# File uploader or sample data option
option = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data"], horizontal=True)

@st.cache_data(show_spinner=False)
def load_csv_in_chunks(file, chunksize=200000):
    """Load large CSVs in chunks with progress bar."""
    try:
        chunks = []
        progress_bar = st.progress(0, text="Loading data in chunks...")
        total = 0
        for i, chunk in enumerate(pd.read_csv(file, low_memory=False, chunksize=chunksize)):
            chunks.append(chunk)
            total += len(chunk)
            progress_bar.progress(min((i+1)*0.05, 1.0), text=f"Loaded {total:,} rows...")
        progress_bar.empty()
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_sample_data():
    np.random.seed(42)
    agents = [f"Agent_{i}" for i in range(1, 21)]
    departments = ["Electronics", "Clothing", "Groceries", "Home", "Toys"]
    weeks = pd.date_range(start="2024-01-01", periods=50, freq="W")

    data = []
    for week in weeks:
        for dept in departments:
            for agent in agents:
                sales = np.random.poisson(lam=200)
                if np.random.rand() < 0.05:
                    sales *= np.random.randint(3, 6)
                data.append([agent, dept, week, sales])

    df = pd.DataFrame(data, columns=["Agent", "Department", "Week", "Sales"])
    return df

if option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"], help="Upload a CSV file for anomaly detection. Supports large files.")
    if uploaded_file:
        df = load_csv_in_chunks(uploaded_file)
    else:
        df = pd.DataFrame()
else:
    st.info("Using dynamically generated sample dataset with multiple agents, departments, and weeks.")
    df = load_sample_data()

if not df.empty:
    st.success(f"✅ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    # Column selection
    categorical_col = st.selectbox("📌 Select Categorical Column", options=df.columns, help="Choose the column representing categories, e.g., Agents or Departments.")
    numeric_cols = st.multiselect("📊 Select Numeric Columns", options=df.select_dtypes(include=np.number).columns, help="Choose numeric metrics for anomaly detection.")

    method = st.radio("⚙️ Choose Anomaly Detection Method", ["IsolationForest", "LocalOutlierFactor"], horizontal=True, help="IsolationForest is efficient for large data; LOF is good for local density-based anomalies.")

    if numeric_cols:
        df_analysis = df[[categorical_col] + numeric_cols].copy()
        df_analysis[numeric_cols] = df_analysis[numeric_cols].fillna(df_analysis[numeric_cols].median())

        X = df_analysis[numeric_cols].values

        if method == "IsolationForest":
            model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
            df_analysis["anomaly"] = model.fit_predict(X)
        else:
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
            df_analysis["anomaly"] = model.fit_predict(X)

        df_analysis["anomaly"] = df_analysis["anomaly"].map({1: "Normal", -1: "Anomaly"})

        # Insights
        st.subheader("📊 Insights Summary")
        anomaly_counts = df_analysis["anomaly"].value_counts()
        top_anomalies = df_analysis[df_analysis["anomaly"] == "Anomaly"][categorical_col].value_counts().head(5)
        st.write("**Anomaly Distribution:**", anomaly_counts.to_dict())
        st.write("**Top 5 Anomalous Categories:**")
        st.write(top_anomalies)

        # Results Table
        st.subheader("📋 Results Table")
        st.dataframe(df_analysis[[categorical_col] + numeric_cols + ["anomaly"]])

        # Visualization
        if len(numeric_cols) >= 2:
            fig = px.scatter(df_analysis, x=numeric_cols[0], y=numeric_cols[1], color="anomaly", hover_data=[categorical_col], title="Anomalies vs Normal Data")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df_analysis, x=numeric_cols[0], color="anomaly", title="Distribution of Values with Anomalies Highlighted")
            st.plotly_chart(fig, use_container_width=True)

        # Download results CSV
        output_csv = BytesIO()
        df_analysis.to_csv(output_csv, index=False)
        st.download_button("⬇️ Download Results as CSV", data=output_csv.getvalue(), file_name="anomaly_results.csv", mime="text/csv")

        # Download Insights Report (PDF)
        def generate_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("<b>Anomaly Hunter Pro – Insights Report</b>", styles['Title']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Total Rows: {len(df_analysis)}", styles['Normal']))
            elements.append(Paragraph(f"Anomaly Counts: {anomaly_counts.to_dict()}", styles['Normal']))
            elements.append(Spacer(1, 12))

            data_table = [[categorical_col, "Count"]] + [[idx, val] for idx, val in top_anomalies.items()]
            t = Table(data_table)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
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
