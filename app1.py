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

# CRITICAL: Set page config FIRST before ANY other Streamlit commands
st.set_page_config(page_title="Anomaly Hunter Pro", layout="wide", page_icon="🎯")

# CRITICAL: Initialize session state IMMEDIATELY after page config
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.parquet_path = None
    st.session_state.duckdb_con = None
    st.session_state.data_loaded = False
    st.session_state.row_count = 0
    st.session_state.column_info = {}
    st.session_state.results_df = None
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.selected_numeric = []
    st.session_state.selected_categorical = []
    st.session_state.selected_date = None
    st.session_state.engineered_features = []
    st.session_state.analysis_history = []

# Professional eye-candy UI theme with better readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        max-width: 1600px;
        padding: 1.5rem 2rem 3rem 2rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #2d3748 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        width: 100%;
        padding: 0.65rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    h1 {
        color: #1a202c;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #1a202c;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #2d3748;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
    }
    
    p, li, span, label, div {
        color: #2d3748 !important;
    }
    
    .stMarkdown {
        color: #2d3748;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #ebf4ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .info-box * {
        color: #1e40af !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
    }
    
    .warning-box * {
        color: #92400e !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);
    }
    
    .success-box * {
        color: #065f46 !important;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #ef4444;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1);
    }
    
    .error-box * {
        color: #991b1b !important;
    }
    
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f7fafc;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        color: #4a5568;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #edf2f7;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_file_type(filename: str) -> Optional[str]:
    """Determine file type from extension"""
    ext = Path(filename).suffix.lower()
    mapping = {
        '.csv': 'csv',
        '.txt': 'csv',
        '.xlsx': 'xlsx',
        '.xls': 'xlsx',
        '.parquet': 'parquet'
    }
    return mapping.get(ext)

def convert_to_parquet(file, file_type: str) -> Tuple[Optional[str], int, int]:
    """Convert uploaded file to Parquet format using DuckDB"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📁 Reading file...")
        progress_bar.progress(10)
        
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        temp_input = os.path.join(st.session_state.temp_dir, f'temp_input.{file_type}')
        
        # Save uploaded file
        with open(temp_input, 'wb') as f:
            f.write(file.getvalue())
        
        progress_bar.progress(30)
        status_text.text("🔄 Converting to optimized format...")
        
        # Create DuckDB connection
        con = duckdb.connect(':memory:')
        
        if file_type == 'csv':
            # Read CSV with auto-detection
            query = f"""
                COPY (
                    SELECT * FROM read_csv_auto(
                        '{temp_input}',
                        sample_size=-1,
                        ignore_errors=true,
                        normalize_names=true,
                        header=true
                    )
                ) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
            con.execute(query)
            
        elif file_type == 'xlsx':
            # Read Excel with pandas then convert
            df = pd.read_excel(temp_input, engine='openpyxl')
            # Normalize column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
            con.register('temp_table', df)
            con.execute(f"COPY temp_table TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
            
        elif file_type == 'parquet':
            # Already parquet, just copy
            import shutil
            shutil.copy(temp_input, parquet_path)
        
        progress_bar.progress(70)
        status_text.text("📊 Analyzing data...")
        
        # Get row and column counts
        row_count = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{parquet_path}')").fetchone()[0]
        col_count = len(con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1").description)
        
        con.close()
        
        # Cleanup temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)
        
        progress_bar.progress(100)
        status_text.text("✅ File loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return parquet_path, row_count, col_count
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading file: {str(e)}")
        if os.path.exists(temp_input):
            os.remove(temp_input)
        return None, 0, 0

def get_duckdb_con():
    """Get or create DuckDB connection - FIXED to check key existence"""
    if 'duckdb_con' not in st.session_state or st.session_state.duckdb_con is None:
        st.session_state.duckdb_con = duckdb.connect(':memory:')
    return st.session_state.duckdb_con

def get_column_stats(con, parquet_path: str) -> Dict:
    """Get column statistics and types"""
    try:
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
    except Exception as e:
        st.error(f"Error analyzing columns: {str(e)}")
        return {}

def detect_anomalies_statistical(con, parquet_path: str, cols: List[str], method: str, threshold: float) -> Optional[pd.DataFrame]:
    """Statistical anomaly detection using DuckDB SQL - COMPLETELY FIXED"""
    df = None
    try:
        if method == 'zscore':
            zscore_parts = [f'ABS(("{col}" - AVG("{col}") OVER ()) / NULLIF(STDDEV("{col}") OVER (), 0)) as zscore_{col}' 
                           for col in cols]
            query = f"""
                SELECT *, 
                    {', '.join(zscore_parts)},
                    GREATEST({', '.join([f'COALESCE(zscore_{col}, 0)' for col in cols])}) as max_score
                FROM parquet_scan('{parquet_path}')
            """
            df = con.execute(query).df()
            df['anomaly'] = (df['max_score'] > threshold).astype(int)
            df['anomaly_score'] = df['max_score']
            
        elif method == 'iqr':
            # Get percentiles for each column
            percentiles = {}
            for col in cols:
                result = con.execute(f"""
                    SELECT 
                        quantile_cont("{col}", 0.25) as q1,
                        quantile_cont("{col}", 0.75) as q3
                    FROM parquet_scan('{parquet_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                q1, q3 = result[0], result[1]
                iqr = q3 - q1
                percentiles[col] = {'lower': q1 - 1.5 * iqr, 'upper': q3 + 1.5 * iqr}
            
            # Load full data
            query = f"SELECT * FROM parquet_scan('{parquet_path}')"
            df = con.execute(query).df()
            
            # Add anomaly flag based on IQR bounds
            df['anomaly'] = 0
            for col in cols:
                lower, upper = percentiles[col]['lower'], percentiles[col]['upper']
                df.loc[(df[col] < lower) | (df[col] > upper), 'anomaly'] = 1
            
            # Calculate anomaly scores
            scores = []
            for col in cols:
                lower, upper = percentiles[col]['lower'], percentiles[col]['upper']
                score = np.maximum(np.maximum(lower - df[col], 0), np.maximum(df[col] - upper, 0))
                scores.append(score)
            df['anomaly_score'] = np.max(scores, axis=0) if scores else 0
            
        elif method == 'mad':
            # FIXED: Calculate median and MAD separately to avoid nested aggregates
            mad_values = {}
            for col in cols:
                # First get median
                median_result = con.execute(f"""
                    SELECT median("{col}") as med
                    FROM parquet_scan('{parquet_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                median_val = median_result[0]
                
                # Then calculate MAD using the median value
                mad_result = con.execute(f"""
                    SELECT median(ABS("{col}" - {median_val})) as mad_val
                    FROM parquet_scan('{parquet_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                mad_val = mad_result[0]
                
                mad_values[col] = {
                    'median': median_val, 
                    'mad': mad_val if mad_val and mad_val > 0 else 1
                }
            
            # Build query with pre-calculated values
            mad_parts = [f'ABS(("{col}" - {mad_values[col]["median"]}) / ({mad_values[col]["mad"]} * 1.4826)) as mad_{col}' 
                        for col in cols]
            query = f"""
                SELECT *, 
                    {', '.join(mad_parts)},
                    GREATEST({', '.join([f'COALESCE(mad_{col}, 0)' for col in cols])}) as max_score
                FROM parquet_scan('{parquet_path}')
            """
            df = con.execute(query).df()
            df['anomaly'] = (df['max_score'] > threshold).astype(int)
            df['anomaly_score'] = df['max_score']
        
        return df
        
    except Exception as e:
        st.error(f"Statistical detection error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def detect_anomalies_ml(df: pd.DataFrame, cols: List[str], method: str, contamination: float) -> Optional[pd.DataFrame]:
    """Machine learning anomaly detection"""
    try:
        df_clean = df.copy()
        
        # Handle missing values
        for col in cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Prepare features
        X = df_clean[cols].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Select model
        models = {
            'isolation_forest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1, max_samples=256),
            'lof': LocalOutlierFactor(n_neighbors=min(20, len(df_clean)-1), contamination=contamination, n_jobs=-1),
            'one_class_svm': OneClassSVM(nu=contamination, kernel='rbf', gamma='auto'),
            'elliptic_envelope': EllipticEnvelope(contamination=contamination, random_state=42)
        }
        
        model = models[method]
        predictions = model.fit_predict(X_scaled)
        
        # Get scores
        if hasattr(model, 'score_samples'):
            scores = model.score_samples(X_scaled)
        elif hasattr(model, 'negative_outlier_factor_'):
            scores = model.negative_outlier_factor_
        else:
            scores = predictions.astype(float)
        
        df_clean['anomaly'] = (predictions == -1).astype(int)
        df_clean['anomaly_score'] = np.abs(scores)
        df_clean['confidence_percentile'] = pd.Series(df_clean['anomaly_score']).rank(pct=True) * 100
        
        return df_clean
        
    except Exception as e:
        st.error(f"ML detection error: {str(e)}")
        return None

def create_visualizations(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], date_col: Optional[str]) -> Dict:
    """Create comprehensive visualizations"""
    figs = {}
    
    try:
        # Distribution pie chart
        anom_counts = df['anomaly'].value_counts()
        colors = ['#10b981', '#ef4444']
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[anom_counts.get(0, 0), anom_counts.get(1, 0)],
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textfont=dict(size=16, color='white', family='Inter'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        fig_pie.update_layout(
            title=dict(text="Anomaly Distribution", font=dict(size=20, family='Inter', color='#2d3748')),
            height=400,
            showlegend=True,
            legend=dict(font=dict(size=14)),
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        figs['pie'] = fig_pie
        
        # Score histogram
        if 'anomaly_score' in df.columns:
            fig_hist = go.Figure()
            
            # Normal distribution
            normal_scores = df[df['anomaly'] == 0]['anomaly_score']
            fig_hist.add_trace(go.Histogram(
                x=normal_scores,
                name='Normal',
                marker_color='#10b981',
                opacity=0.7,
                nbinsx=40
            ))
            
            # Anomaly distribution
            anom_scores = df[df['anomaly'] == 1]['anomaly_score']
            fig_hist.add_trace(go.Histogram(
                x=anom_scores,
                name='Anomaly',
                marker_color='#ef4444',
                opacity=0.7,
                nbinsx=40
            ))
            
            fig_hist.update_layout(
                title=dict(text="Anomaly Score Distribution", font=dict(size=20, family='Inter', color='#2d3748')),
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                height=400,
                barmode='overlay',
                hovermode='x unified',
                legend=dict(font=dict(size=14)),
                margin=dict(l=40, r=20, t=60, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            figs['histogram'] = fig_hist
        
        # Timeline
        if date_col and date_col in df.columns:
            try:
                df_time = df.copy()
                df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)
                
                if len(df_time) > 0:
                    time_agg = df_time.groupby(pd.Grouper(key=date_col, freq='D'))['anomaly'].agg(['sum', 'count']).reset_index()
                    time_agg['anomaly_rate'] = (time_agg['sum'] / time_agg['count'] * 100).fillna(0)
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=time_agg[date_col],
                        y=time_agg['sum'],
                        mode='lines+markers',
                        name='Anomaly Count',
                        line=dict(color='#ef4444', width=3),
                        marker=dict(size=8),
                        fill='tozeroy',
                        fillcolor='rgba(239, 68, 68, 0.2)',
                        hovertemplate='<b>Date</b>: %{x}<br><b>Anomalies</b>: %{y}<extra></extra>'
                    ))
                    
                    fig_ts.update_layout(
                        title=dict(text="Anomaly Timeline", font=dict(size=20, family='Inter', color='#2d3748')),
                        xaxis_title="Date",
                        yaxis_title="Anomaly Count",
                        height=400,
                        hovermode='x unified',
                        margin=dict(l=40, r=20, t=60, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    figs['timeline'] = fig_ts
            except:
                pass
        
        # Feature scatter plot
        if len(numeric_cols) >= 2:
            sample_df = df.sample(min(2000, len(df))) if len(df) > 2000 else df
            
            fig_scatter = go.Figure()
            
            # Normal points
            normal_data = sample_df[sample_df['anomaly'] == 0]
            fig_scatter.add_trace(go.Scatter(
                x=normal_data[numeric_cols[0]],
                y=normal_data[numeric_cols[1]],
                mode='markers',
                name='Normal',
                marker=dict(color='#10b981', size=8, opacity=0.6, line=dict(width=0)),
                hovertemplate=f'<b>{numeric_cols[0]}</b>: %{{x}}<br><b>{numeric_cols[1]}</b>: %{{y}}<extra></extra>'
            ))
            
            # Anomaly points
            anom_data = sample_df[sample_df['anomaly'] == 1]
            fig_scatter.add_trace(go.Scatter(
                x=anom_data[numeric_cols[0]],
                y=anom_data[numeric_cols[1]],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#ef4444', size=10, opacity=0.8, 
                          line=dict(width=1, color='white'),
                          symbol='diamond'),
                hovertemplate=f'<b>{numeric_cols[0]}</b>: %{{x}}<br><b>{numeric_cols[1]}</b>: %{{y}}<extra></extra>'
            ))
            
            fig_scatter.update_layout(
                title=dict(text=f"{numeric_cols[0]} vs {numeric_cols[1]}", font=dict(size=20, family='Inter', color='#2d3748')),
                xaxis_title=numeric_cols[0],
                yaxis_title=numeric_cols[1],
                height=500,
                hovermode='closest',
                legend=dict(font=dict(size=14)),
                margin=dict(l=40, r=20, t=60, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            figs['scatter'] = fig_scatter
        
        # Box plots for features
        if len(numeric_cols) > 0:
            fig_box = go.Figure()
            
            for col in numeric_cols[:4]:
                if col in df.columns:
                    # Normal box
                    fig_box.add_trace(go.Box(
                        y=df[df['anomaly'] == 0][col],
                        name=f'{col} (Normal)',
                        marker_color='#10b981',
                        boxmean='sd'
                    ))
                    
                    # Anomaly box
                    fig_box.add_trace(go.Box(
                        y=df[df['anomaly'] == 1][col],
                        name=f'{col} (Anomaly)',
                        marker_color='#ef4444',
                        boxmean='sd'
                    ))
            
            fig_box.update_layout(
                title=dict(text="Feature Distribution Comparison", font=dict(size=20, family='Inter', color='#2d3748')),
                yaxis_title="Value",
                height=500,
                showlegend=True,
                legend=dict(font=dict(size=12)),
                margin=dict(l=40, r=20, t=60, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            figs['boxplot'] = fig_box
        
        # Category bar charts
        if cat_cols:
            for cat_col in cat_cols[:2]:
                if cat_col in df.columns:
                    cat_agg = df.groupby(cat_col)['anomaly'].agg(['sum', 'count']).reset_index()
                    cat_agg['rate'] = (cat_agg['sum'] / cat_agg['count'] * 100).round(2)
                    cat_agg = cat_agg.nlargest(10, 'sum')
                    
                    fig_cat = go.Figure()
                    fig_cat.add_trace(go.Bar(
                        x=cat_agg[cat_col],
                        y=cat_agg['sum'],
                        marker_color='#ef4444',
                        text=cat_agg['sum'],
                        textposition='outside',
                        hovertemplate=f'<b>%{{x}}</b><br>Anomalies: %{{y}}<br>Rate: {cat_agg["rate"]}%<extra></extra>'
                    ))
                    
                    fig_cat.update_layout(
                        title=dict(text=f"Top Anomalies by {cat_col}", font=dict(size=20, family='Inter', color='#2d3748')),
                        xaxis_title=cat_col,
                        yaxis_title="Anomaly Count",
                        height=400,
                        margin=dict(l=40, r=20, t=60, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    figs[f'cat_{cat_col}'] = fig_cat
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
    
    return figs

def engineer_feature(con, parquet_path: str, feature_name: str, expression: str) -> bool:
    """Create new feature using SQL expression"""
    try:
        # Create temp table with new feature
        query = f"""
            CREATE OR REPLACE TABLE temp_data AS 
            SELECT *, ({expression}) as "{feature_name}"
            FROM parquet_scan('{parquet_path}')
        """
        con.execute(query)
        
        # Export back to parquet
        con.execute(f"COPY temp_data TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        
        return True
    except Exception as e:
        st.error(f"Feature engineering failed: {str(e)}")
        return False

# Main UI
st.title("🎯 Anomaly Hunter Pro")
st.markdown("<p style='color: #4a5568; font-size: 1.1rem; font-weight: 500;'>Enterprise-Grade Anomaly Detection Platform • Powered by Advanced ML & Statistical Methods</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'txt', 'parquet'],
        help="Upload CSV, Excel, TXT, or Parquet files (max 200MB)"
    )
    
    if uploaded_file and not st.session_state.data_loaded:
        file_type = get_file_type(uploaded_file.name)
        
        if file_type:
            parquet_path, rows, cols = convert_to_parquet(uploaded_file, file_type)
            
            if parquet_path:
                st.session_state.parquet_path = parquet_path
                st.session_state.row_count = rows
                st.session_state.data_loaded = True
                
                con = get_duckdb_con()
                st.session_state.column_info = get_column_stats(con, parquet_path)
                
                st.markdown(f'<div class="success-box"><b>✅ Loaded:</b> {rows:,} rows × {cols} columns</div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
        else:
            st.error("❌ Unsupported file format")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        
        # Get column lists - FIXED: Increased categorical limit to 500
        numeric_cols = [col for col, info in st.session_state.column_info.items() 
                       if any(t in info['type'].upper() for t in ['INT', 'DOUBLE', 'DECIMAL', 'FLOAT', 'NUMERIC'])]
        
        categorical_cols = [col for col, info in st.session_state.column_info.items()
                          if 'VARCHAR' in info['type'].upper() and info['distinct'] < 500]  # FIXED: Was 100
        
        date_cols = [col for col, info in st.session_state.column_info.items()
                    if any(t in info['type'].upper() for t in ['DATE', 'TIMESTAMP'])]
        
        # Feature selection
        st.markdown("### 📊 Features")
        selected_numeric = st.multiselect(
            "Numeric features *",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            help="Select numeric columns for anomaly detection"
        )
        
        selected_categorical = st.multiselect(
            "Categorical (optional)",
            categorical_cols,
            help="For grouping and segmentation analysis"
        )
        
        selected_date = st.selectbox(
            "Date column (optional)",
            [None] + date_cols,
            help="For timeline visualization"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Method")
        
        method_type = st.radio(
            "Method Type",
            ["Statistical", "Machine Learning"],
            help="Statistical: Fast, interpretable | ML: Advanced patterns"
        )
        
        if method_type == "Statistical":
            stat_method = st.selectbox(
                "Algorithm",
                ["Z-Score", "IQR", "MAD"],
                help="Z-Score: Standard dev | IQR: Box plot | MAD: Robust median"
            )
            
            threshold = st.slider(
                "Sensitivity",
                1.5, 5.0, 3.0, 0.5,
                help="Lower = More sensitive (more anomalies)"
            )
            
            method_config = ('stat', stat_method.split()[0].lower(), threshold)
            
        else:
            ml_method = st.selectbox(
                "Algorithm",
                ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"],
                help="Choose ML algorithm based on data characteristics"
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate",
                0.01, 0.30, 0.10, 0.01,
                format="%.2f",
                help="Expected proportion of anomalies (0.10 = 10%)"
            )
            
            method_map = {
                "Isolation Forest": "isolation_forest",
                "Local Outlier Factor": "lof",
                "One-Class SVM": "one_class_svm",
                "Elliptic Envelope": "elliptic_envelope"
            }
            
            method_config = ('ml', method_map[ml_method], contamination)
        
        st.markdown("---")
        
        # Run detection button
        if st.button("🚀 Run Detection", use_container_width=True):
            if not selected_numeric:
                st.error("⚠️ Please select at least one numeric feature")
            else:
                with st.spinner("🔍 Detecting anomalies..."):
                    con = get_duckdb_con()
                    method_cat, method_name, param = method_config
                    
                    if method_cat == 'stat':
                        df_results = detect_anomalies_statistical(
                            con, st.session_state.parquet_path, 
                            selected_numeric, method_name, param
                        )
                    else:
                        # Load data for ML
                        df_full = con.execute(
                            f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}') LIMIT 100000"
                        ).df()
                        df_results = detect_anomalies_ml(
                            df_full, selected_numeric, method_name, param
                        )
                    
                    if df_results is not None:
                        st.session_state.results_df = df_results
                        st.session_state.selected_numeric = selected_numeric
                        st.session_state.selected_categorical = selected_categorical
                        st.session_state.selected_date = selected_date
                        
                        # Save to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now(),
                            'method': f"{method_cat}_{method_name}",
                            'features': len(selected_numeric),
                            'anomalies': int(df_results['anomaly'].sum())
                        })
                        
                        st.success("✅ Analysis complete!")
                        time.sleep(0.5)
                        st.rerun()
        
        # Clear button - FIXED to properly reset state
        if st.session_state.data_loaded:
            st.markdown("---")
            if st.button("🗑️ Clear Data", use_container_width=True):
                # Close DuckDB connection first
                if 'duckdb_con' in st.session_state and st.session_state.duckdb_con is not None:
                    try:
                        st.session_state.duckdb_con.close()
                    except:
                        pass
                
                # Reset critical keys
                st.session_state.parquet_path = None
                st.session_state.duckdb_con = None
                st.session_state.data_loaded = False
                st.session_state.row_count = 0
                st.session_state.column_info = {}
                st.session_state.results_df = None
                st.session_state.selected_numeric = []
                st.session_state.selected_categorical = []
                st.session_state.selected_date = None
                
                st.rerun()

# Main content area
if not st.session_state.data_loaded:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Anomaly Hunter Pro</h3>
        <p><b>Get started in 3 simple steps:</b></p>
        <ol style="margin-left: 1.5rem;">
            <li>📁 Upload your data file (CSV, Excel, Parquet, or TXT)</li>
            <li>⚙️ Select features and detection method in the sidebar</li>
            <li>🚀 Click "Run Detection" and analyze results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Detection Methods**
        - Z-Score Analysis
        - IQR (Box Plot)
        - MAD (Robust)
        - Isolation Forest
        - Local Outlier Factor
        - One-Class SVM
        - Elliptic Envelope
        """)
    
    with col2:
        st.markdown("""
        **📊 Visualizations**
        - Distribution charts
        - Score histograms
        - Timeline analysis
        - Feature scatter plots
        - Box plot comparisons
        - Category breakdowns
        """)
    
    with col3:
        st.markdown("""
        **💾 Export Options**
        - CSV format
        - Parquet format
        - Excel with metadata
        - Feature engineering
        - Batch processing
        - Custom thresholds
        """)
    
    st.markdown("---")
    
    st.markdown("## 📚 Supported File Formats")
    
    format_col1, format_col2, format_col3, format_col4 = st.columns(4)
    
    with format_col1:
        st.markdown("**📄 CSV**")
        st.markdown("Standard comma-separated values")
    
    with format_col2:
        st.markdown("**📊 Excel**")
        st.markdown("XLSX and XLS formats")
    
    with format_col3:
        st.markdown("**📦 Parquet**")
        st.markdown("Optimized columnar format")
    
    with format_col4:
        st.markdown("**📝 Text**")
        st.markdown("Tab or comma delimited")

elif st.session_state.results_df is None:
    # Data loaded but no analysis yet
    st.markdown("## 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.row_count:,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.column_info)}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_count = len([c for c, i in st.session_state.column_info.items() 
                           if any(t in i['type'].upper() for t in ['INT', 'DOUBLE', 'DECIMAL'])])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{numeric_count}</div>
            <div class="metric-label">Numeric Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cat_count = len([c for c, i in st.session_state.column_info.items() 
                        if 'VARCHAR' in i['type'].upper()])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cat_count}</div>
            <div class="metric-label">Categorical</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature engineering section
    with st.expander("⚙️ Feature Engineering", expanded=False):
        st.markdown("Create custom features using DuckDB SQL expressions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            feature_name = st.text_input(
                "Feature Name",
                placeholder="new_feature",
                help="Name for your new feature"
            )
            
            feature_expr = st.text_area(
                "SQL Expression",
                placeholder="Examples:\n• col1 + col2\n• CASE WHEN amount > 1000 THEN 1 ELSE 0 END\n• col1 / NULLIF(col2, 0)",
                height=120,
                help="Use DuckDB SQL syntax"
            )
            
            st.caption(f"**Available columns:** {', '.join(list(st.session_state.column_info.keys())[:10])}" + 
                      (f" ... and {len(st.session_state.column_info) - 10} more" if len(st.session_state.column_info) > 10 else ""))
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Create Feature", use_container_width=True):
                if feature_name and feature_expr:
                    con = get_duckdb_con()
                    if engineer_feature(con, st.session_state.parquet_path, feature_name, feature_expr):
                        st.session_state.column_info = get_column_stats(con, st.session_state.parquet_path)
                        st.session_state.engineered_features.append(feature_name)
                        st.success(f"✅ Created: {feature_name}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("⚠️ Please provide both feature name and expression")
    
    st.markdown("---")
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    
    con = get_duckdb_con()
    sample_df = con.execute(f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}') LIMIT 100").df()
    st.dataframe(sample_df, use_container_width=True, height=400)
    
    # Column details
    with st.expander("📊 Column Details"):
        col_details = pd.DataFrame([
            {
                'Column': col,
                'Type': info['type'],
                'Distinct Values': info['distinct'],
                'Null Count': info['nulls'],
                'Null %': f"{info['null_pct']:.1f}%"
            }
            for col, info in st.session_state.column_info.items()
        ])
        st.dataframe(col_details, use_container_width=True, hide_index=True)

else:
    # Results view
    df_results = st.session_state.results_df
    numeric_cols = st.session_state.selected_numeric
    cat_cols = st.session_state.selected_categorical
    date_col = st.session_state.selected_date
    
    total = len(df_results)
    anomalies = int(df_results['anomaly'].sum())
    anomaly_rate = (anomalies / total * 100) if total > 0 else 0
    normal = total - anomalies
    
    st.markdown("## 🎯 Detection Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{anomalies:,}</div>
            <div class="metric-label">Anomalies Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{anomaly_rate:.2f}%</div>
            <div class="metric-label">Anomaly Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{normal:,}</div>
            <div class="metric-label">Normal Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary box
    if anomalies > 0:
        high_priority = len(df_results[df_results['anomaly_score'] > df_results['anomaly_score'].quantile(0.95)]) if 'anomaly_score' in df_results.columns else 0
        st.markdown(f"""
        <div class="warning-box">
            <b>🔍 Analysis Summary</b><br>
            • Found <b>{anomalies:,} anomalies</b> ({anomaly_rate:.2f}% of dataset)<br>
            • Analyzed <b>{len(numeric_cols)} features</b>: {', '.join(numeric_cols)}<br>
            • High priority records (>95th percentile): <b>{high_priority}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <b>✅ Clean Dataset</b><br>
            No significant anomalies detected. Consider adjusting sensitivity or trying different features.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analysis", "📋 Data Table", "💾 Export"])
    
    with tab1:
        st.markdown("### Visualizations")
        
        figs = create_visualizations(df_results, numeric_cols, cat_cols, date_col)
        
        # Two column layout
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pie' in figs:
                st.plotly_chart(figs['pie'], use_container_width=True)
        
        with col2:
            if 'histogram' in figs:
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        # Full width charts
        if 'timeline' in figs:
            st.plotly_chart(figs['timeline'], use_container_width=True)
        
        if 'scatter' in figs:
            st.plotly_chart(figs['scatter'], use_container_width=True)
        
        if 'boxplot' in figs:
            st.plotly_chart(figs['boxplot'], use_container_width=True)
    
    with tab2:
        st.markdown("### Feature Impact Analysis")
        
        if anomalies > 0:
            # Calculate feature statistics
            feature_stats = []
            for col in numeric_cols:
                if col in df_results.columns:
                    anom_data = df_results[df_results['anomaly'] == 1][col]
                    norm_data = df_results[df_results['anomaly'] == 0][col]
                    
                    anom_mean = anom_data.mean()
                    norm_mean = norm_data.mean()
                    diff = abs(anom_mean - norm_mean)
                    pct_diff = (diff / norm_mean * 100) if norm_mean != 0 else 0
                    
                    feature_stats.append({
                        'Feature': col,
                        'Anomaly Mean': f"{anom_mean:.2f}",
                        'Normal Mean': f"{norm_mean:.2f}",
                        'Difference': f"{diff:.2f}",
                        'Impact %': f"{pct_diff:.1f}%"
                    })
            
            if feature_stats:
                st.dataframe(
                    pd.DataFrame(feature_stats),
                    use_container_width=True,
                    hide_index=True
                )
        
        # Categorical analysis
        if cat_cols:
            st.markdown("---")
            st.markdown("### Categorical Breakdown")
            
            for cat_col in cat_cols:
                if cat_col in df_results.columns:
                    figs = create_visualizations(df_results, numeric_cols, cat_cols, date_col)
                    if f'cat_{cat_col}' in figs:
                        st.plotly_chart(figs[f'cat_{cat_col}'], use_container_width=True)
                        
                        # Top categories table
                        cat_stats = df_results.groupby(cat_col)['anomaly'].agg(['sum', 'count']).reset_index()
                        cat_stats['rate'] = (cat_stats['sum'] / cat_stats['count'] * 100).round(2)
                        cat_stats.columns = [cat_col, 'Anomalies', 'Total', 'Rate %']
                        cat_stats = cat_stats.nlargest(10, 'Anomalies')
                        
                        st.dataframe(cat_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Complete Results")
        
        # Add filters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            filter_anomaly = st.selectbox(
                "Filter by",
                ["All Records", "Anomalies Only", "Normal Only"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Anomaly Score (High to Low)", "Anomaly Score (Low to High)"] if 'anomaly_score' in df_results.columns else ["Original Order"]
            )
        
        # Apply filters
        display_df = df_results.copy()
        
        if filter_anomaly == "Anomalies Only":
            display_df = display_df[display_df['anomaly'] == 1]
        elif filter_anomaly == "Normal Only":
            display_df = display_df[display_df['anomaly'] == 0]
        
        if 'anomaly_score' in display_df.columns and sort_by != "Original Order":
            ascending = sort_by == "Anomaly Score (Low to High)"
            display_df = display_df.sort_values('anomaly_score', ascending=ascending)
        
        st.dataframe(display_df, use_container_width=True, height=500)
        
        st.info(f"📊 Showing {len(display_df):,} of {total:,} records")
    
    with tab4:
        st.markdown("### Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_buffer = BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            parquet_buffer = BytesIO()
            df_results.to_parquet(parquet_buffer, compression='snappy')
            st.download_button(
                "📥 Download Parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        with col3:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_results.to_excel(writer, sheet_name='Results', index=False)
                
                # Metadata sheet
                metadata = pd.DataFrame([
                    ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['Total Records', total],
                    ['Anomalies', anomalies],
                    ['Anomaly Rate', f'{anomaly_rate:.2f}%'],
                    ['Features Used', ', '.join(numeric_cols)],
                    ['', ''],
                    ['Feature Statistics', '']
                ])
                
                if anomalies > 0:
                    for col in numeric_cols:
                        if col in df_results.columns:
                            anom_mean = df_results[df_results['anomaly'] == 1][col].mean()
                            norm_mean = df_results[df_results['anomaly'] == 0][col].mean()
                            metadata = pd.concat([metadata, pd.DataFrame([[col, f'Anom: {anom_mean:.2f} | Normal: {norm_mean:.2f}']])], ignore_index=True)
                
                metadata.to_excel(writer, sheet_name='Metadata', index=False, header=False)
            
            excel_buffer.seek(0)
            st.download_button(
                "📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Export statistics
        st.markdown("### Export Summary")
        st.info(f"""
        **Export includes:**
        - {len(df_results.columns)} columns (original data + anomaly flags + scores)
        - {anomalies:,} anomaly records flagged
        - Analysis metadata (Excel only)
        - Ready for further analysis in Python, R, Excel, or BI tools
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem;'>
    <p style='font-size: 14px; font-weight: 300;'>
        🎯 <b>Anomaly Hunter Pro</b> v2.0 • Enterprise Edition<br>
        Powered by DuckDB • Apache Arrow • Scikit-learn • Plotly<br>
        <i>Built for Data Analysts & Data Scientists</i>
    </p>
</div>
""", unsafe_allow_html=True)
