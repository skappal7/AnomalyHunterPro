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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_method' not in st.session_state:
    st.session_state.current_method = None

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
    
    # Reset button in sidebar
    if st.button("🔄 Reset All", help="Clear all data and start fresh"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Data source selection
option = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data"], horizontal=True)

@st.cache_data(show_spinner=False)
def load_csv_in_chunks(file, chunksize=200_000, usecols=None, downcast=True):
    """Load CSV in chunks; optionally select columns and downcast numerics."""
    try:
        chunks = []
        progress_bar = st.progress(0, text="Loading data in chunks…")
        total_rows = 0
        
        file.seek(0)
        
        for i, chunk in enumerate(pd.read_csv(file, low_memory=False, chunksize=chunksize, usecols=usecols)):
            if downcast:
                for col in chunk.select_dtypes(include=["float64"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="float", errors='coerce')
                for col in chunk.select_dtypes(include=["int64"]).columns:
                    chunk[col] = pd.to_numeric(chunk[col], downcast="integer", errors='coerce')
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

# Handle data loading
if option == "Use Sample Data":
    if not st.session_state.data_loaded or st.session_state.df.empty:
        with st.spinner("Loading sample data..."):
            st.session_state.df = load_sample_data()
            st.session_state.data_loaded = True
        st.success(f"✅ Sample data loaded: {st.session_state.df.shape[0]:,} rows, {st.session_state.df.shape[1]:,} columns")

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"], help="Upload a CSV. Large files are processed via chunking.")
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_hash = str(hash(uploaded_file.read()))
        uploaded_file.seek(0)  # Reset after hashing
        
        if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
            # New file uploaded, reset everything
            st.session_state.file_hash = file_hash
            st.session_state.data_loaded = False
            st.session_state.analysis_results = None
            
            try:
                with st.spinner("Analyzing file structure..."):
                    uploaded_file.seek(0)
                    sample_for_columns = pd.read_csv(uploaded_file, nrows=5)
                st.session_state.all_columns = sample_for_columns.columns.tolist()
                st.caption(f"Detected columns: {', '.join(st.session_state.all_columns)}")
                
                # Store file for later use
                uploaded_file.seek(0)
                st.session_state.uploaded_file_content = uploaded_file.read()
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        
        else:
            # Same file, show existing columns
            st.caption(f"Using previously loaded file with columns: {', '.join(st.session_state.all_columns)}")

# Column selection (only if we have data structure)
if option == "Use Sample Data" or (option == "Upload CSV" and 'all_columns' in st.session_state):
    
    if option == "Use Sample Data":
        all_columns = st.session_state.df.columns.tolist()
    else:
        all_columns = st.session_state.all_columns
    
    # Column selection
    categorical_col = st.selectbox("📌 Select Categorical Column", options=all_columns, help="e.g., Agent or Department")
    
    if option == "Use Sample Data":
        numeric_guess = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    else:
        # For uploaded files, we need to guess from sample
        try:
            sample_df = pd.DataFrame()
            if 'uploaded_file_content' in st.session_state:
                sample_df = pd.read_csv(BytesIO(st.session_state.uploaded_file_content), nrows=100)
            numeric_guess = sample_df.select_dtypes(include=np.number).columns.tolist()
        except:
            numeric_guess = []
    
    numeric_cols = st.multiselect(
        "📊 Select Numeric Columns", 
        options=numeric_guess if numeric_guess else all_columns, 
        help="Metrics used by the model(s)"
    )

    # Auto-load data when columns are selected
    if categorical_col and numeric_cols:
        
        # Load uploaded file data if needed
        if option == "Upload CSV" and not st.session_state.data_loaded:
            required_cols = list(set([categorical_col] + numeric_cols))
            
            with st.spinner("Loading your data..."):
                try:
                    file_content = BytesIO(st.session_state.uploaded_file_content)
                    st.session_state.df = load_csv_in_chunks(file_content, usecols=required_cols)
                    st.session_state.data_loaded = True
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    st.stop()
        
        # Validate data
        if st.session_state.data_loaded and not st.session_state.df.empty:
            # Check if columns exist
            missing_cols = []
            if categorical_col not in st.session_state.df.columns:
                missing_cols.append(categorical_col)
            for col in numeric_cols:
                if col not in st.session_state.df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                st.error(f"Selected columns not found in data: {', '.join(missing_cols)}")
                st.stop()
            
            if len(st.session_state.df) < 10:
                st.error("Need at least 10 rows for anomaly detection.")
                st.stop()
            
            st.success(f"✅ Data ready for analysis: {st.session_state.df.shape[0]:,} rows, {len([categorical_col] + numeric_cols)} selected columns")
            
            # Method selection
            method = st.radio(
                "⚙️ Choose Anomaly Detection Method", 
                ["IsolationForest", "LocalOutlierFactor"], 
                horizontal=True, 
                help="IsolationForest isolates global outliers; LOF finds sparse local densities"
            )
            
            # Check if we need to rerun analysis (method changed or no results)
            need_analysis = (
                st.session_state.analysis_results is None or 
                st.session_state.current_method != method or
                st.session_state.analysis_results.get('categorical_col') != categorical_col or
                st.session_state.analysis_results.get('numeric_cols') != numeric_cols
            )
            
            # Analysis button
            if need_analysis:
                if st.button("🔍 Run Anomaly Analysis", type="primary", use_container_width=True):
                    with st.spinner(f"Running {method} analysis..."):
                        try:
                            # Prepare data
                            required_cols = [categorical_col] + numeric_cols
                            df_analysis = st.session_state.df[required_cols].copy()
                            
                            # Convert and clean numeric columns
                            for col in numeric_cols:
                                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
                            
                            df_analysis = df_analysis.dropna(subset=numeric_cols, how='all')
                            
                            if len(df_analysis) < 10:
                                st.error("After cleaning, insufficient data remains (need at least 10 rows).")
                                st.stop()
                            
                            # Fill NaN values
                            for col in numeric_cols:
                                median_val = df_analysis[col].median()
                                if pd.isna(median_val):
                                    df_analysis[col] = 0
                                else:
                                    df_analysis[col] = df_analysis[col].fillna(median_val)

                            X = df_analysis[numeric_cols].values

                            # Run anomaly detection
                            if method == "IsolationForest":
                                contamination = min(0.1, max(0.01, 50/len(X)))
                                model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=1)
                                model.fit(X)
                                labels = model.predict(X)
                                scores = -model.score_samples(X)
                                reason_blurb = "**Why flagged?** IsolationForest isolates records that require fewer random splits; these have uncommon metric combinations."
                            else:
                                n_neighbors = min(20, max(5, int(np.sqrt(len(X)))))
                                contamination = min(0.1, max(0.01, 50/len(X)))
                                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=1)
                                labels = lof.fit_predict(X)
                                scores = -lof.negative_outlier_factor_
                                reason_blurb = "**Why flagged?** LOF compares local density—points in sparse neighborhoods relative to their neighbors are flagged."

                            df_analysis["anomaly"] = pd.Series(labels).map({1: "Normal", -1: "Anomaly"})
                            df_analysis["anomaly_score"] = scores

                            # Store results in session state
                            st.session_state.analysis_results = {
                                'df_analysis': df_analysis,
                                'method': method,
                                'reason_blurb': reason_blurb,
                                'categorical_col': categorical_col,
                                'numeric_cols': numeric_cols
                            }
                            st.session_state.current_method = method
                            
                            st.success("✅ Analysis completed!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error in anomaly detection: {e}")
                            st.info("Please check your data and try again.")
            
            else:
                # Show change method option
                if st.session_state.current_method != method:
                    st.info(f"Method changed from {st.session_state.current_method} to {method}")
                    if st.button("🔄 Rerun Analysis with New Method", type="secondary", use_container_width=True):
                        st.session_state.analysis_results = None  # Force rerun
                        st.rerun()
                else:
                    st.success("✅ Analysis results ready!")
            
            # Display results if available
            if st.session_state.analysis_results:
                df_analysis = st.session_state.analysis_results['df_analysis']
                method = st.session_state.analysis_results['method']
                reason_blurb = st.session_state.analysis_results['reason_blurb']
                
                # Distribution analysis
                st.subheader("📈 Anomaly Distribution")
                counts = df_analysis["anomaly"].value_counts()
                total = counts.sum()
                anomalies = counts.get("Anomaly", 0)
                pct = (anomalies / total * 100) if total > 0 else 0.0

                if total > 0:
                    cdf = pd.DataFrame({"label": counts.index, "count": counts.values})
                    cdf["percent"] = (cdf["count"] / total * 100).round(2)
                    fig_pie = px.pie(
                        cdf, names="label", values="count", hole=0.55, 
                        hover_data=["percent"], 
                        title=f"Anomalies: {anomalies:,} / {total:,}  (≈ {pct:.2f}%)"
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption("Share of records flagged as anomalous. Percentages are relative to total analyzed rows.")

                    # Narrative insight
                    st.subheader("🧠 Narrative Insight")
                    try:
                        grp = df_analysis.groupby([categorical_col, "anomaly"]).size().unstack(fill_value=0)
                        if "Anomaly" not in grp.columns: 
                            grp["Anomaly"] = 0
                        if "Normal" not in grp.columns: 
                            grp["Normal"] = 0
                        
                        grp["anomaly_rate"] = grp["Anomaly"] / (grp["Anomaly"] + grp["Normal"]).clip(lower=1)
                        top_rate = grp.sort_values("anomaly_rate", ascending=False).head(5)

                        story_lines = [
                            f"Out of **{total:,}** records, **{anomalies:,}** (≈ **{pct:.2f}%**) were flagged using **{method}**.",
                            reason_blurb,
                        ]
                        
                        if not top_rate.empty and len(top_rate) > 0:
                            leader = top_rate.index[0]
                            leader_rate = top_rate.iloc[0]['anomaly_rate'] * 100
                            story_lines.append(f"Highest anomaly rate: **{leader}** at **{leader_rate:.1f}%** of its records.")
                        
                        strongest = df_analysis.nlargest(3, "anomaly_score")
                        if not strongest.empty:
                            examples = ", ".join([
                                f"{row[categorical_col]} (score={row['anomaly_score']:.2f})" 
                                for _, row in strongest.iterrows()
                            ])
                            story_lines.append(f"Most extreme outliers by score: {examples}.")
                        
                        st.markdown("\n".join([f"- {s}" for s in story_lines]))
                    except Exception as e:
                        st.warning(f"Could not generate narrative insight: {e}")

                    # Top categories table
                    st.subheader(f"🏆 Top 5 {categorical_col} by Anomaly Rate")
                    try:
                        if 'grp' in locals() and not grp.empty:
                            top_table = (
                                grp.sort_values("anomaly_rate", ascending=False)
                                .head(5)
                                .reset_index()[[categorical_col, "Anomaly", "Normal", "anomaly_rate"]]
                            )
                            if not top_table.empty:
                                top_table["Anomaly Rate (%)"] = (top_table["anomaly_rate"] * 100).round(2)
                                display_table = top_table[[categorical_col, "Anomaly", "Normal", "Anomaly Rate (%)"]]
                                st.dataframe(display_table, use_container_width=True)

                                bar = px.bar(
                                    top_table, x=categorical_col, y="Anomaly Rate (%)", 
                                    text="Anomaly Rate (%)"
                                )
                                bar.update_traces(texttemplate="%{text}", textposition="outside")
                                st.plotly_chart(bar, use_container_width=True)
                            else:
                                st.info("Not enough data to compute top categories.")
                    except Exception as e:
                        st.warning(f"Could not generate category analysis: {e}")

                    # Results table
                    st.subheader("📋 Results Table")
                    view_cols = [categorical_col] + numeric_cols + ["anomaly", "anomaly_score"]
                    st.dataframe(df_analysis[view_cols], use_container_width=True, height=420)

                    # Visualization
                    if len(numeric_cols) >= 2:
                        try:
                            fig = px.scatter(
                                df_analysis,
                                x=numeric_cols[0], y=numeric_cols[1], color="anomaly",
                                hover_data=[categorical_col, "anomaly_score"],
                                title="Anomalies vs Normal Data (colored by flag)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate scatter plot: {e}")
                    else:
                        try:
                            fig = px.histogram(
                                df_analysis, x=numeric_cols[0], color="anomaly", 
                                title="Distribution of Values with Anomalies Highlighted"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate histogram: {e}")

                    # Downloads
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            output_csv = BytesIO()
                            df_analysis.to_csv(output_csv, index=False)
                            st.download_button(
                                "⬇️ Download Results as CSV", 
                                data=output_csv.getvalue(), 
                                file_name="anomaly_results.csv", 
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Could not generate CSV download: {e}")

                    with col2:
                        try:
                            def generate_pdf():
                                try:
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
                                    
                                    if 'grp' in locals() and not grp.empty:
                                        top_tbl = grp.sort_values("anomaly_rate", ascending=False).head(10)
                                        data_table = [[categorical_col, "Anomaly", "Normal", "Anomaly Rate"]]
                                        for idx, r in top_tbl.iterrows():
                                            data_table.append([
                                                str(idx), str(int(r["Anomaly"])), 
                                                str(int(r["Normal"])), f"{r['anomaly_rate']*100:.1f}%"
                                            ])
                                        
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
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                                    return None

                            pdf_buffer = generate_pdf()
                            if pdf_buffer:
                                st.download_button(
                                    "⬇️ Download Insights Report (PDF)", 
                                    data=pdf_buffer, 
                                    file_name="anomaly_insights_report.pdf", 
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.warning(f"Could not generate PDF download: {e}")
        else:
            st.info("Please upload a valid CSV file or select sample data to continue.")
    
    else:
        if not categorical_col:
            st.info("👆 Please select a categorical column to continue.")
        elif not numeric_cols:
            st.info("👆 Please select at least one numeric column to continue.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed with 💗 CE Innovation Team 2025</div>", unsafe_allow_html=True)
