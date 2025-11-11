import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
import tempfile
import os
from io import BytesIO
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Anomaly Hunter Pro", layout="wide", page_icon="📊")

# Professional Minitab-style theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * { font-family: 'Roboto', sans-serif; }
    
    .main .block-container {
        max-width: 1400px;
        padding: 1rem 2rem 3rem 2rem;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background-color: #495057;
        color: white;
        border: none;
        width: 100%;
        padding: 0.5rem;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #343a40;
    }
    
    h1, h2, h3 { color: #212529; font-weight: 500; }
    h1 { font-size: 1.75rem; margin-bottom: 1rem; }
    h2 { font-size: 1.25rem; margin: 1.5rem 0 0.75rem 0; }
    h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem 0; }
    
    .stButton button {
        background-color: #495057;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border-radius: 0.25rem;
    }
    
    .stButton button:hover {
        background-color: #343a40;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 500;
        color: #212529;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .info-box {
        background-color: #e7f3ff;
        border-left: 3px solid #0056b3;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 3px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border: 1px solid #dee2e6;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        background-color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        border-bottom: none;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True,
        'parquet_path': None,
        'duckdb_con': None,
        'data_loaded': False,
        'row_count': 0,
        'column_info': {},
        'results_df': None,
        'temp_dir': tempfile.mkdtemp(),
        'engineered_features': []
    })

# Core functions
def get_file_type(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    return {'csv': 'csv', '.txt': 'csv', '.xlsx': 'xlsx', '.xls': 'xlsx', '.parquet': 'parquet'}.get(ext)

def convert_to_parquet(file, file_type: str) -> Tuple[Optional[str], int, int]:
    try:
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        temp_file = os.path.join(st.session_state.temp_dir, 'temp_data')
        
        with open(temp_file, 'wb') as f:
            f.write(file.getvalue())
        
        con = duckdb.connect(':memory:')
        
        if file_type in ['csv', 'txt']:
            con.execute(f"""
                COPY (SELECT * FROM read_csv_auto('{temp_file}', sample_size=-1, ignore_errors=true, normalize_names=true))
                TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
        else:
            df = pd.read_excel(temp_file, engine='openpyxl' if file_type == 'xlsx' else 'xlrd')
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
            con.register('temp_table', df)
            con.execute(f"COPY temp_table TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        
        row_count = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{parquet_path}')").fetchone()[0]
        col_count = len(con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1").description)
        
        con.close()
        os.remove(temp_file)
        
        return parquet_path, row_count, col_count
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, 0, 0

def get_duckdb_con():
    if st.session_state.duckdb_con is None:
        st.session_state.duckdb_con = duckdb.connect(':memory:')
    return st.session_state.duckdb_con

def get_column_stats(con, parquet_path: str) -> Dict:
    schema = con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0").description
    column_info = {}
    
    for col_name, col_type, *_ in schema:
        stats = con.execute(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT "{col_name}") as distinct_count,
                COUNT(*) - COUNT("{col_name}") as null_count
            FROM parquet_scan('{parquet_path}')
        """).fetchone()
        
        column_info[col_name] = {
            'type': str(col_type),
            'total': stats[0],
            'distinct': stats[1],
            'nulls': stats[2],
            'null_pct': (stats[2] / stats[0] * 100) if stats[0] > 0 else 0
        }
    
    return column_info

def detect_anomalies_statistical(con, parquet_path: str, cols: List[str], method: str, threshold: float) -> Optional[pd.DataFrame]:
    try:
        if method == 'zscore':
            zscore_parts = [f'ABS(("{col}" - AVG("{col}") OVER ()) / NULLIF(STDDEV("{col}") OVER (), 0)) as zscore_{col}' for col in cols]
            query = f"""
                SELECT *, 
                    {', '.join(zscore_parts)},
                    GREATEST({', '.join([f'zscore_{col}' for col in cols])}) as max_score
                FROM parquet_scan('{parquet_path}')
            """
            df = con.execute(query).df()
            df['anomaly'] = (df['max_score'] > threshold).astype(int)
            df['anomaly_score'] = df['max_score']
            
        elif method == 'iqr':
            percentiles = {}
            for col in cols:
                q1, q3 = con.execute(f"""
                    SELECT quantile_cont("{col}", 0.25), quantile_cont("{col}", 0.75)
                    FROM parquet_scan('{parquet_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                iqr = q3 - q1
                percentiles[col] = {'lower': q1 - 1.5 * iqr, 'upper': q3 + 1.5 * iqr}
            
            conditions = [f'("{col}" < {percentiles[col]["lower"]} OR "{col}" > {percentiles[col]["upper"]})' for col in cols]
            query = f"""
                SELECT *, CASE WHEN {' OR '.join(conditions)} THEN 1 ELSE 0 END as anomaly
                FROM parquet_scan('{parquet_path}')
            """
            df = con.execute(query).df()
            
            scores = []
            for col in cols:
                lower, upper = percentiles[col]['lower'], percentiles[col]['upper']
                score = np.maximum(np.maximum(lower - df[col], 0), np.maximum(df[col] - upper, 0))
                scores.append(score)
            df['anomaly_score'] = np.max(scores, axis=0)
            
        else:  # MAD
            mad_values = {}
            for col in cols:
                med, mad_val = con.execute(f"""
                    SELECT median("{col}"), median(ABS("{col}" - median("{col}")))
                    FROM parquet_scan('{parquet_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                mad_values[col] = {'median': med, 'mad': mad_val if mad_val > 0 else 1}
            
            mad_parts = [f'ABS(("{col}" - {mad_values[col]["median"]}) / ({mad_values[col]["mad"]} * 1.4826)) as mad_{col}' for col in cols]
            query = f"""
                SELECT *, {', '.join(mad_parts)}, GREATEST({', '.join([f'mad_{col}' for col in cols])}) as max_score
                FROM parquet_scan('{parquet_path}')
            """
            df = con.execute(query).df()
            df['anomaly'] = (df['max_score'] > threshold).astype(int)
            df['anomaly_score'] = df['max_score']
        
        return df
    except Exception as e:
        st.error(f"Statistical detection error: {str(e)}")
        return None

def detect_anomalies_ml(df: pd.DataFrame, cols: List[str], method: str, contamination: float) -> Optional[pd.DataFrame]:
    try:
        df_clean = df.copy()
        for col in cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        X = StandardScaler().fit_transform(df_clean[cols].values)
        
        models = {
            'isolation_forest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1, max_samples=256),
            'lof': LocalOutlierFactor(n_neighbors=min(20, len(df_clean)-1), contamination=contamination, n_jobs=-1),
            'one_class_svm': OneClassSVM(nu=contamination, kernel='rbf', gamma='auto'),
            'elliptic_envelope': EllipticEnvelope(contamination=contamination, random_state=42)
        }
        
        model = models[method]
        predictions = model.fit_predict(X)
        
        if hasattr(model, 'score_samples'):
            scores = model.score_samples(X)
        elif hasattr(model, 'negative_outlier_factor_'):
            scores = model.negative_outlier_factor_
        else:
            scores = predictions
        
        df_clean['anomaly'] = (predictions == -1).astype(int)
        df_clean['anomaly_score'] = np.abs(scores)
        
        return df_clean
    except Exception as e:
        st.error(f"ML detection error: {str(e)}")
        return None

def create_visualizations(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], date_col: Optional[str]) -> Dict:
    figs = {}
    
    # Distribution pie
    anom_counts = df['anomaly'].value_counts()
    figs['pie'] = go.Figure(data=[go.Pie(
        labels=['Normal', 'Anomaly'],
        values=[anom_counts.get(0, 0), anom_counts.get(1, 0)],
        hole=0.3,
        marker=dict(colors=['#6c757d', '#dc3545']),
        textfont=dict(size=14, color='white')
    )])
    figs['pie'].update_layout(title="Distribution", height=350, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
    
    # Score histogram
    if 'anomaly_score' in df.columns:
        figs['histogram'] = px.histogram(
            df, x='anomaly_score', color='anomaly', nbins=40,
            color_discrete_map={0: '#6c757d', 1: '#dc3545'}
        )
        figs['histogram'].update_layout(title="Score Distribution", height=350, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    
    # Timeline
    if date_col and date_col in df.columns:
        try:
            df_time = df.copy()
            df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
            df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)
            
            time_agg = df_time.groupby(pd.Grouper(key=date_col, freq='D'))['anomaly'].sum().reset_index()
            figs['timeline'] = go.Figure(data=[go.Scatter(
                x=time_agg[date_col], y=time_agg['anomaly'],
                mode='lines', line=dict(color='#dc3545', width=2), fill='tozeroy'
            )])
            figs['timeline'].update_layout(title="Timeline", height=350, margin=dict(l=40, r=20, t=40, b=40))
        except:
            pass
    
    # Feature scatter
    if len(numeric_cols) >= 2:
        sample_df = df.sample(min(1000, len(df)))
        figs['scatter'] = px.scatter(
            sample_df, x=numeric_cols[0], y=numeric_cols[1], color='anomaly',
            color_discrete_map={0: '#6c757d', 1: '#dc3545'}, opacity=0.6
        )
        figs['scatter'].update_layout(title=f"{numeric_cols[0]} vs {numeric_cols[1]}", height=400, margin=dict(l=40, r=20, t=40, b=40))
    
    # Category bars
    if cat_cols:
        for cat_col in cat_cols[:2]:
            if cat_col in df.columns:
                cat_agg = df.groupby(cat_col)['anomaly'].agg(['sum', 'count']).reset_index()
                cat_agg = cat_agg.nlargest(10, 'sum')
                figs[f'cat_{cat_col}'] = go.Figure(data=[go.Bar(
                    x=cat_agg[cat_col], y=cat_agg['sum'], marker_color='#dc3545'
                )])
                figs[f'cat_{cat_col}'].update_layout(title=f"Anomalies by {cat_col}", height=350, margin=dict(l=40, r=20, t=40, b=40))
    
    return figs

def engineer_feature(con, parquet_path: str, feature_def: Dict) -> bool:
    try:
        col_name = feature_def['name']
        expression = feature_def['expression']
        
        # Create new parquet with engineered feature
        query = f"""
            COPY (
                SELECT *, ({expression}) as "{col_name}"
                FROM parquet_scan('{parquet_path}')
            ) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        con.execute(query)
        
        st.session_state.engineered_features.append(col_name)
        return True
    except Exception as e:
        st.error(f"Feature engineering error: {str(e)}")
        return False

# UI
st.title("📊 Anomaly Hunter Pro")

with st.sidebar:
    st.header("Data Upload")
    
    uploaded_file = st.file_uploader("Select file", type=['csv', 'xlsx', 'xls', 'txt', 'parquet'])
    
    if uploaded_file and not st.session_state.data_loaded:
        file_type = get_file_type(uploaded_file.name)
        if file_type:
            with st.spinner("Loading..."):
                parquet_path, rows, cols = convert_to_parquet(uploaded_file, file_type)
                if parquet_path:
                    st.session_state.parquet_path = parquet_path
                    st.session_state.row_count = rows
                    st.session_state.data_loaded = True
                    con = get_duckdb_con()
                    st.session_state.column_info = get_column_stats(con, parquet_path)
                    st.success(f"Loaded: {rows:,} rows")
                    st.rerun()
    
    if st.session_state.data_loaded:
        st.divider()
        st.header("Feature Selection")
        
        numeric_cols = [col for col, info in st.session_state.column_info.items() 
                       if any(t in info['type'].upper() for t in ['INT', 'DOUBLE', 'DECIMAL', 'FLOAT'])]
        
        categorical_cols = [col for col, info in st.session_state.column_info.items()
                          if 'VARCHAR' in info['type'].upper() and info['distinct'] < 100]
        
        date_cols = [col for col, info in st.session_state.column_info.items()
                    if any(t in info['type'].upper() for t in ['DATE', 'TIMESTAMP'])]
        
        selected_numeric = st.multiselect("Numeric features", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        selected_categorical = st.multiselect("Categorical (optional)", categorical_cols)
        selected_date = st.selectbox("Date column (optional)", [None] + date_cols)
        
        st.divider()
        st.header("Detection Method")
        
        method_type = st.radio("Type", ["Statistical", "Machine Learning"])
        
        if method_type == "Statistical":
            stat_method = st.selectbox("Algorithm", ["Z-Score", "IQR", "MAD"])
            threshold = st.slider("Threshold", 1.5, 5.0, 3.0, 0.5)
            method_config = ('stat', stat_method.split()[0].lower(), threshold)
        else:
            ml_method = st.selectbox("Algorithm", ["Isolation Forest", "LOF", "One-Class SVM", "Elliptic Envelope"])
            contamination = st.slider("Expected anomaly rate", 0.01, 0.30, 0.10, 0.01)
            method_map = {
                "Isolation Forest": "isolation_forest",
                "LOF": "lof",
                "One-Class SVM": "one_class_svm",
                "Elliptic Envelope": "elliptic_envelope"
            }
            method_config = ('ml', method_map[ml_method], contamination)
        
        st.divider()
        
        if st.button("Run Detection", use_container_width=True):
            if not selected_numeric:
                st.error("Select at least one numeric feature")
            else:
                with st.spinner("Detecting anomalies..."):
                    con = get_duckdb_con()
                    method_cat, method_name, param = method_config
                    
                    if method_cat == 'stat':
                        df_results = detect_anomalies_statistical(con, st.session_state.parquet_path, selected_numeric, method_name, param)
                    else:
                        df_full = con.execute(f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}') LIMIT 100000").df()
                        df_results = detect_anomalies_ml(df_full, selected_numeric, method_name, param)
                    
                    if df_results is not None:
                        st.session_state.results_df = df_results
                        st.session_state.selected_numeric = selected_numeric
                        st.session_state.selected_categorical = selected_categorical
                        st.session_state.selected_date = selected_date
                        st.success("Analysis complete")
                        st.rerun()
        
        if st.session_state.data_loaded:
            st.divider()
            if st.button("Clear Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['temp_dir', 'initialized']:
                        del st.session_state[key]
                st.session_state.data_loaded = False
                st.rerun()

# Main area
if not st.session_state.data_loaded:
    st.info("Upload a file to begin analysis")
    
elif st.session_state.results_df is None:
    st.subheader("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.row_count:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.column_info)}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        numeric_count = len([c for c, i in st.session_state.column_info.items() if any(t in i['type'].upper() for t in ['INT', 'DOUBLE', 'DECIMAL'])])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{numeric_count}</div><div class="metric-label">Numeric</div></div>', unsafe_allow_html=True)
    with col4:
        cat_count = len([c for c, i in st.session_state.column_info.items() if 'VARCHAR' in i['type'].upper()])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{cat_count}</div><div class="metric-label">Categorical</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Feature engineering section
    with st.expander("⚙️ Feature Engineering", expanded=False):
        st.write("Create new features using DuckDB SQL expressions")
        
        feature_name = st.text_input("Feature name", placeholder="new_feature")
        
        feature_expr = st.text_area(
            "SQL expression",
            placeholder='Example: col1 + col2\nExample: CASE WHEN col1 > 100 THEN 1 ELSE 0 END\nExample: col1 / NULLIF(col2, 0)',
            height=100
        )
        
        st.caption("Available columns: " + ", ".join(st.session_state.column_info.keys()))
        
        if st.button("Add Feature", use_container_width=True):
            if feature_name and feature_expr:
                con = get_duckdb_con()
                if engineer_feature(con, st.session_state.parquet_path, {'name': feature_name, 'expression': feature_expr}):
                    st.session_state.column_info = get_column_stats(con, st.session_state.parquet_path)
                    st.success(f"Created: {feature_name}")
                    st.rerun()
            else:
                st.error("Please provide both feature name and expression")
    
    st.divider()
    
    con = get_duckdb_con()
    sample_df = con.execute(f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}') LIMIT 100").df()
    st.dataframe(sample_df, use_container_width=True, height=400)

else:
    df_results = st.session_state.results_df
    numeric_cols = st.session_state.selected_numeric
    cat_cols = st.session_state.selected_categorical
    date_col = st.session_state.selected_date
    
    total = len(df_results)
    anomalies = int(df_results['anomaly'].sum())
    anomaly_rate = (anomalies / total * 100) if total > 0 else 0
    
    st.subheader("Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{anomalies:,}</div><div class="metric-label">Anomalies</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{anomaly_rate:.2f}%</div><div class="metric-label">Anomaly Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total - anomalies:,}</div><div class="metric-label">Normal</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    if anomalies > 0:
        st.markdown(f'<div class="warning-box"><b>Found {anomalies:,} anomalies</b> ({anomaly_rate:.2f}%) across {len(numeric_cols)} features: {", ".join(numeric_cols)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box"><b>Clean dataset:</b> No anomalies detected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Data"])
    
    with tab1:
        figs = create_visualizations(df_results, numeric_cols, cat_cols, date_col)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'pie' in figs:
                st.plotly_chart(figs['pie'], use_container_width=True)
        with col2:
            if 'histogram' in figs:
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        if 'timeline' in figs:
            st.plotly_chart(figs['timeline'], use_container_width=True)
        
        if 'scatter' in figs:
            st.plotly_chart(figs['scatter'], use_container_width=True)
    
    with tab2:
        if anomalies > 0:
            st.subheader("Feature Impact")
            
            feature_stats = []
            for col in numeric_cols:
                if col in df_results.columns:
                    anom_mean = df_results[df_results['anomaly'] == 1][col].mean()
                    norm_mean = df_results[df_results['anomaly'] == 0][col].mean()
                    diff = abs(anom_mean - norm_mean)
                    feature_stats.append({
                        'Feature': col,
                        'Anomaly Avg': f"{anom_mean:.2f}",
                        'Normal Avg': f"{norm_mean:.2f}",
                        'Difference': f"{diff:.2f}"
                    })
            
            if feature_stats:
                st.dataframe(pd.DataFrame(feature_stats), use_container_width=True, hide_index=True)
        
        if cat_cols:
            st.subheader("Categorical Analysis")
            for cat_col in cat_cols:
                if cat_col in df_results.columns:
                    figs = create_visualizations(df_results, numeric_cols, cat_cols, date_col)
                    if f'cat_{cat_col}' in figs:
                        st.plotly_chart(figs[f'cat_{cat_col}'], use_container_width=True)
    
    with tab3:
        st.dataframe(df_results, use_container_width=True, height=500)
    
    st.divider()
    
    # Export
    st.subheader("Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_buffer = BytesIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        parquet_buffer = BytesIO()
        df_results.to_parquet(parquet_buffer, compression='snappy')
        st.download_button(
            "Download Parquet",
            data=parquet_buffer.getvalue(),
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )
    
    with col3:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, sheet_name='Results', index=False)
            
            metadata = pd.DataFrame([
                ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Records', total],
                ['Anomalies', anomalies],
                ['Anomaly Rate', f'{anomaly_rate:.2f}%'],
                ['Features', ', '.join(numeric_cols)]
            ])
            metadata.to_excel(writer, sheet_name='Metadata', index=False, header=False)
        
        excel_buffer.seek(0)
        st.download_button(
            "Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
