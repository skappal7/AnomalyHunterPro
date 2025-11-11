import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
import os
from io import BytesIO
import time
from datetime import datetime
from typing import Any
import requests
import json

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from scipy import stats

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="Anomaly Hunter Pro 🎯",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# Initialize session state IMMEDIATELY after page config
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.parquet_path = None
    st.session_state.duckdb_con = None
    st.session_state.data_loaded = False
    st.session_state.row_count = 0
    st.session_state.column_info = {}
    st.session_state.analysis_complete = False
    st.session_state.results_df = None
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.selected_numeric_cols = []
    st.session_state.selected_categorical_cols = []
    st.session_state.selected_date_col = None
    st.session_state.selected_method = None
    st.session_state.anomaly_history = []  # NEW: Track detection runs
    st.session_state.column_correlations = {}  # NEW: Feature correlations

# Minimalist CSS with fixed scaling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        text-align: center;
        border-left: 4px solid #6366f1;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0.25rem 0;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        letter-spacing: 0.02em;
        font-size: 0.875rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_metric_card(label: str, value: Any) -> str:
    """Create a metric card with hover effect"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def get_file_type(filename: str) -> str | None:
    """Determine file type from extension"""
    ext = Path(filename).suffix.lower()
    match ext:
        case '.csv': return 'csv'
        case '.txt': return 'txt'
        case '.xlsx': return 'xlsx'
        case '.xls': return 'xls'
        case '.parquet': return 'parquet'
        case _: return None

def convert_to_parquet(file, file_type: str) -> tuple[str | None, int, int]:
    """Convert files to Parquet using DuckDB - handles large files efficiently"""
    try:
        progress = st.progress(0, text="🔄 Reading file...")
        
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        temp_csv = os.path.join(st.session_state.temp_dir, 'temp_data.csv')
        
        progress.progress(20, text="📦 Preparing file...")
        with open(temp_csv, 'wb') as f:
            f.write(file.getvalue())
        
        progress.progress(40, text="💾 Converting with DuckDB...")
        
        con = duckdb.connect(':memory:')
        
        match file_type:
            case 'csv' | 'txt':
                con.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{temp_csv}', 
                                                       sample_size=-1,
                                                       ignore_errors=true,
                                                       normalize_names=true))
                    TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
                """)
                
            case 'xlsx' | 'xls':
                df = pd.read_excel(file, engine='openpyxl' if file_type == 'xlsx' else 'xlrd')
                # Normalize column names
                df.columns = df.columns.str.strip().str.replace(' ', '_')
                con.register('temp_table', df)
                con.execute(f"COPY temp_table TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
                
            case 'parquet':
                with open(parquet_path, 'wb') as f:
                    f.write(file.getvalue())
                con.execute(f"SELECT COUNT(*) as cnt FROM parquet_scan('{parquet_path}')")
                row_count = con.fetchone()[0]
                con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1")
                col_count = len(con.fetchone())
                con.close()
                
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                
                progress.progress(100, text="✅ File ready!")
                time.sleep(0.5)
                progress.empty()
                return parquet_path, row_count, col_count
                
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        progress.progress(80, text="📊 Getting file info...")
        
        con.execute(f"SELECT COUNT(*) as cnt FROM parquet_scan('{parquet_path}')")
        row_count = con.fetchone()[0]
        
        con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1")
        result = con.fetchone()
        col_count = len(result) if result else 0
        
        con.close()
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        progress.progress(100, text="✅ Conversion complete!")
        time.sleep(0.5)
        progress.empty()
        
        return parquet_path, row_count, col_count
        
    except Exception as e:
        st.error(f"❌ Error converting file: {e}")
        import traceback
        st.error(traceback.format_exc())
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        return None, 0, 0

def initialize_duckdb():
    """Initialize DuckDB connection"""
    if st.session_state.duckdb_con is None:
        st.session_state.duckdb_con = duckdb.connect(':memory:')
    return st.session_state.duckdb_con

def get_column_stats(con, parquet_path: str) -> dict:
    """Get comprehensive column statistics using DuckDB"""
    try:
        query = f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0"
        schema = con.execute(query).description
        
        column_info = {}
        for col_name, col_type, *_ in schema:
            stats_query = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT "{col_name}") as distinct_count,
                COUNT(*) - COUNT("{col_name}") as null_count
            FROM parquet_scan('{parquet_path}')
            """
            stats = con.execute(stats_query).fetchone()
            
            column_info[col_name] = {
                'type': str(col_type),
                'total': stats[0],
                'distinct': stats[1],
                'nulls': stats[2],
                'null_pct': (stats[2] / stats[0] * 100) if stats[0] > 0 else 0
            }
        
        return column_info
    except Exception as e:
        st.error(f"Error getting column stats: {e}")
        return {}

def get_sample_data(con, parquet_path: str, limit: int = 1000) -> pd.DataFrame:
    """Get sample data for preview"""
    query = f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT {limit}"
    return con.execute(query).df()

def explain_contamination(rate: float, total_records: int) -> str:
    """Generate user-friendly explanation of contamination rate"""
    expected_anomalies = int(total_records * rate)
    
    if rate <= 0.01:
        sensitivity = "very strict"
        use_case = "credit card fraud, critical systems"
    elif rate <= 0.05:
        sensitivity = "strict"
        use_case = "financial transactions, quality control"
    elif rate <= 0.10:
        sensitivity = "balanced"
        use_case = "general business data, customer behavior"
    elif rate <= 0.20:
        sensitivity = "lenient"
        use_case = "exploratory analysis, noisy data"
    else:
        sensitivity = "very lenient"
        use_case = "highly variable data"
    
    return f"""
    **What this means:** You're telling the algorithm to expect ~**{expected_anomalies:,} unusual records** 
    out of {total_records:,} total ({rate*100:.1f}%).
    
    **Sensitivity:** {sensitivity.title()} - typical for {use_case}
    
    **💡 Tip:** If you're unsure, start with 0.10 (10%) and adjust based on results.
    """

# =============================================================================
# NEW: FEATURE CORRELATION ANALYSIS
# =============================================================================

def calculate_feature_correlations(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Calculate correlation matrix for numeric features"""
    try:
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated pairs (> 0.8 or < -0.8)
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr.append({
                        'feature1': numeric_cols[i],
                        'feature2': numeric_cols[j],
                        'correlation': corr_val
                    })
        
        return {
            'matrix': corr_matrix,
            'high_correlations': high_corr
        }
    except:
        return {}

# =============================================================================
# ANOMALY DETECTION ENGINE (ENHANCED WITH NEW METHODS)
# =============================================================================

DETECTION_METHODS = {
    'isolation_forest': {
        'class': IsolationForest,
        'params': {'contamination': 'auto', 'random_state': 42, 'n_jobs': -1, 'max_samples': 256},
        'type': 'ml',
        'description': 'Best for: High-dimensional data, outliers in sparse regions'
    },
    'lof': {
        'class': LocalOutlierFactor,
        'params': {'n_neighbors': 20, 'contamination': 'auto', 'n_jobs': -1},
        'type': 'ml',
        'description': 'Best for: Dense clusters, local density-based anomalies'
    },
    'one_class_svm': {
        'class': OneClassSVM,
        'params': {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'auto'},
        'type': 'ml',
        'description': 'Best for: Clean boundaries, non-linear decision surfaces'
    },
    'elliptic_envelope': {
        'class': EllipticEnvelope,
        'params': {'contamination': 0.1, 'random_state': 42},
        'type': 'ml',
        'description': 'Best for: Gaussian-distributed data, multivariate outliers'
    },
}

def detect_anomalies_statistical(con, parquet_path: str, numeric_cols: list, method: str = 'zscore', threshold: float = 3) -> pd.DataFrame | None:
    """SQL-based statistical anomaly detection - FIXED VERSION"""
    try:
        if method == 'zscore':
            select_parts = [f"""
                ABS(("{col}" - AVG("{col}") OVER ()) / NULLIF(STDDEV("{col}") OVER (), 0)) as zscore_{col}
            """ for col in numeric_cols]
            
            query = f"""
            SELECT *, 
                {', '.join(select_parts)},
                GREATEST({', '.join([f'zscore_{col}' for col in numeric_cols])}) as max_zscore
            FROM parquet_scan('{parquet_path}')
            """
            
            df = con.execute(query).df()
            df['anomaly'] = (df['max_zscore'] > threshold).astype(int)
            df['anomaly_score'] = df['max_zscore']
            
        elif method == 'iqr':
            # FIXED: Use quantile_cont instead of percentile_cont with WITHIN GROUP
            percentiles = {}
            for col in numeric_cols:
                pct_query = f"""
                SELECT 
                    quantile_cont("{col}", 0.25) as q1,
                    quantile_cont("{col}", 0.75) as q3
                FROM parquet_scan('{parquet_path}')
                WHERE "{col}" IS NOT NULL
                """
                q1, q3 = con.execute(pct_query).fetchone()
                iqr = q3 - q1
                percentiles[col] = {'lower': q1 - 1.5 * iqr, 'upper': q3 + 1.5 * iqr}
            
            conditions = [f'("{col}" < {percentiles[col]["lower"]} OR "{col}" > {percentiles[col]["upper"]})' 
                         for col in numeric_cols]
            
            query = f"""
            SELECT *,
                CASE WHEN {' OR '.join(conditions)} THEN 1 ELSE 0 END as anomaly
            FROM parquet_scan('{parquet_path}')
            """
            
            df = con.execute(query).df()
            
            # Calculate anomaly score
            scores = []
            for col in numeric_cols:
                lower, upper = percentiles[col]['lower'], percentiles[col]['upper']
                score = np.maximum(np.maximum(lower - df[col], 0), np.maximum(df[col] - upper, 0))
                scores.append(score)
            
            df['anomaly_score'] = np.max(scores, axis=0)
        
        elif method == 'mad':  # NEW: Median Absolute Deviation
            mad_values = {}
            for col in numeric_cols:
                mad_query = f"""
                SELECT 
                    median("{col}") as med,
                    median(ABS("{col}" - median("{col}"))) as mad_val
                FROM parquet_scan('{parquet_path}')
                WHERE "{col}" IS NOT NULL
                """
                med, mad_val = con.execute(mad_query).fetchone()
                mad_values[col] = {'median': med, 'mad': mad_val if mad_val > 0 else 1}
            
            select_parts = [f"""
                ABS(("{col}" - {mad_values[col]['median']}) / ({mad_values[col]['mad']} * 1.4826)) as mad_score_{col}
            """ for col in numeric_cols]
            
            query = f"""
            SELECT *,
                {', '.join(select_parts)},
                GREATEST({', '.join([f'mad_score_{col}' for col in numeric_cols])}) as max_mad
            FROM parquet_scan('{parquet_path}')
            """
            
            df = con.execute(query).df()
            df['anomaly'] = (df['max_mad'] > threshold).astype(int)
            df['anomaly_score'] = df['max_mad']
        
        return df
    except Exception as e:
        st.error(f"Error in statistical detection: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def detect_anomalies_ml(df: pd.DataFrame, numeric_cols: list, method: str, contamination: float = 0.1) -> pd.DataFrame | None:
    """ML-based anomaly detection with factory pattern - ENHANCED"""
    try:
        df_clean = df.copy()
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        X = df_clean[numeric_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        
        config = DETECTION_METHODS[method]
        model_class = config['class']
        params = config['params'].copy()
        
        if 'contamination' in params:
            params['contamination'] = contamination
        if 'nu' in params:
            params['nu'] = contamination
        
        if method == 'lof':
            params['n_neighbors'] = min(params['n_neighbors'], len(df_clean) - 1)
        
        model = model_class(**params)
        predictions = model.fit_predict(X_scaled)
        
        if hasattr(model, 'score_samples'):
            scores = model.score_samples(X_scaled)
        elif hasattr(model, 'negative_outlier_factor_'):
            scores = model.negative_outlier_factor_
        else:
            scores = predictions
        
        df_clean['anomaly'] = (predictions == -1).astype(int)
        df_clean['anomaly_score'] = np.abs(scores)
        
        # NEW: Add confidence percentile
        df_clean['confidence_percentile'] = pd.Series(df_clean['anomaly_score']).rank(pct=True) * 100
        
        return df_clean
    except Exception as e:
        st.error(f"Error in ML detection: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# =============================================================================
# NEW: ENSEMBLE DETECTION
# =============================================================================

def detect_anomalies_ensemble(df: pd.DataFrame, numeric_cols: list, contamination: float = 0.1) -> pd.DataFrame | None:
    """Ensemble method combining multiple detection algorithms"""
    try:
        results = []
        methods_to_use = ['isolation_forest', 'lof', 'elliptic_envelope']
        
        for method in methods_to_use:
            result = detect_anomalies_ml(df.copy(), numeric_cols, method, contamination)
            if result is not None:
                results.append(result['anomaly'])
        
        if not results:
            return None
        
        # Voting: flag as anomaly if 2+ methods agree
        df_result = df.copy()
        vote_matrix = np.column_stack(results)
        df_result['anomaly'] = (vote_matrix.sum(axis=1) >= 2).astype(int)
        df_result['ensemble_votes'] = vote_matrix.sum(axis=1)
        df_result['anomaly_score'] = vote_matrix.sum(axis=1) / len(methods_to_use)
        
        return df_result
    except Exception as e:
        st.error(f"Error in ensemble detection: {e}")
        return None

# =============================================================================
# INSIGHTS & VISUALIZATIONS (ENHANCED)
# =============================================================================

def generate_insights(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, method: str) -> dict:
    """Generate comprehensive insights - ENHANCED"""
    try:
        total = len(df)
        anomalies = df['anomaly'].sum()
        pct = (anomalies / total * 100) if total > 0 else 0
        
        insights = {
            'total_records': total,
            'anomalies': anomalies,
            'anomaly_rate': pct,
            'normal_records': total - anomalies,
            'method': method
        }
        
        # Feature importance
        if anomalies > 0:
            feature_stats = {}
            for col in numeric_cols:
                if col in df.columns:
                    anom_mean = df[df['anomaly'] == 1][col].mean()
                    norm_mean = df[df['anomaly'] == 0][col].mean()
                    feature_stats[col] = {
                        'anomaly_mean': anom_mean,
                        'normal_mean': norm_mean,
                        'difference': abs(anom_mean - norm_mean),
                        'pct_difference': abs((anom_mean - norm_mean) / norm_mean * 100) if norm_mean != 0 else 0
                    }
            insights['feature_importance'] = feature_stats
        
        # Categorical insights
        if categorical_cols and anomalies > 0:
            cat_insights = {}
            for cat_col in categorical_cols:
                if cat_col in df.columns:
                    cat_stats = df.groupby(cat_col).agg({
                        'anomaly': ['sum', 'count']
                    }).reset_index()
                    cat_stats.columns = [cat_col, 'anomaly_count', 'total_count']
                    cat_stats['anomaly_rate'] = (cat_stats['anomaly_count'] / cat_stats['total_count'] * 100)
                    cat_stats = cat_stats.sort_values('anomaly_count', ascending=False)
                    
                    cat_insights[cat_col] = {
                        'top_categories': cat_stats.to_dict('records'),
                        'highest_rate': cat_stats.nlargest(1, 'anomaly_rate').to_dict('records')[0] if len(cat_stats) > 0 else None
                    }
            insights['categorical_insights'] = cat_insights
        
        # NEW: Anomaly distribution stats
        if 'anomaly_score' in df.columns and anomalies > 0:
            anom_scores = df[df['anomaly'] == 1]['anomaly_score']
            insights['score_stats'] = {
                'min': float(anom_scores.min()),
                'max': float(anom_scores.max()),
                'median': float(anom_scores.median()),
                'p95': float(anom_scores.quantile(0.95))
            }
        
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return {'total_records': 0, 'anomalies': 0, 'anomaly_rate': 0}

def create_visualizations(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, date_col: str | None) -> dict:
    """Create comprehensive visualizations - ENHANCED"""
    figs = {}
    
    try:
        # Pie chart
        anom_counts = df['anomaly'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[anom_counts.get(0, 0), anom_counts.get(1, 0)],
            hole=0.4,
            marker=dict(colors=['#10b981', '#ef4444']),
            textfont_size=14
        )])
        fig_pie.update_layout(
            title="Anomaly Distribution",
            height=400,
            showlegend=True
        )
        figs['pie'] = fig_pie
        
        # Score histogram with better binning
        if 'anomaly_score' in df.columns:
            fig_hist = px.histogram(
                df, 
                x='anomaly_score',
                color='anomaly',
                nbins=50,
                title="Anomaly Score Distribution",
                color_discrete_map={0: '#10b981', 1: '#ef4444'},
                labels={'anomaly': 'Type'}
            )
            fig_hist.update_layout(height=400, showlegend=True)
            figs['histogram'] = fig_hist
        
        # Timeseries
        if date_col and date_col in df.columns:
            try:
                df_time = df.copy()
                df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                df_time = df_time.dropna(subset=[date_col])
                
                if len(df_time) > 0:
                    df_time = df_time.sort_values(date_col)
                    time_agg = df_time.groupby(pd.Grouper(key=date_col, freq='D'))['anomaly'].agg(['sum', 'count']).reset_index()
                    time_agg['anomaly_rate'] = (time_agg['sum'] / time_agg['count'] * 100)
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=time_agg[date_col],
                        y=time_agg['sum'],
                        mode='lines+markers',
                        name='Anomaly Count',
                        line=dict(color='#ef4444', width=2),
                        fill='tozeroy'
                    ))
                    fig_ts.update_layout(
                        title="Anomaly Timeline",
                        xaxis_title="Date",
                        yaxis_title="Anomaly Count",
                        height=400,
                        hovermode='x unified'
                    )
                    figs['timeseries'] = fig_ts
            except:
                pass
        
        # Scatter plots for top 2 features
        if len(numeric_cols) >= 2 and 'anomaly' in df.columns:
            fig_scatter = px.scatter(
                df.sample(min(1000, len(df))),
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='anomaly',
                color_discrete_map={0: '#10b981', 1: '#ef4444'},
                title=f"Feature Relationship: {numeric_cols[0]} vs {numeric_cols[1]}",
                opacity=0.6
            )
            fig_scatter.update_layout(height=500)
            figs['scatter'] = fig_scatter
        
        # Categorical bar charts
        if categorical_cols:
            for cat_col in categorical_cols[:2]:
                if cat_col in df.columns:
                    cat_agg = df.groupby(cat_col)['anomaly'].agg(['sum', 'count']).reset_index()
                    cat_agg['anomaly_rate'] = (cat_agg['sum'] / cat_agg['count'] * 100)
                    cat_agg = cat_agg.nlargest(10, 'sum')
                    
                    fig_cat = go.Figure()
                    fig_cat.add_trace(go.Bar(
                        x=cat_agg[cat_col],
                        y=cat_agg['sum'],
                        name='Anomaly Count',
                        marker_color='#ef4444'
                    ))
                    fig_cat.update_layout(
                        title=f"Top Anomalies by {cat_col}",
                        xaxis_title=cat_col,
                        yaxis_title="Anomaly Count",
                        height=400
                    )
                    figs[f'category_bar_{cat_col}'] = fig_cat
        
        # Box plots for numeric features
        if len(numeric_cols) > 0:
            fig_box = go.Figure()
            for col in numeric_cols[:4]:
                if col in df.columns:
                    fig_box.add_trace(go.Box(
                        y=df[df['anomaly'] == 0][col],
                        name=f'{col} (Normal)',
                        marker_color='#10b981'
                    ))
                    fig_box.add_trace(go.Box(
                        y=df[df['anomaly'] == 1][col],
                        name=f'{col} (Anomaly)',
                        marker_color='#ef4444'
                    ))
            fig_box.update_layout(
                title="Feature Distribution Comparison",
                height=500,
                showlegend=True
            )
            figs['boxplot'] = fig_box
        
        # NEW: Correlation heatmap
        if len(numeric_cols) >= 2:
            corr_cols = [col for col in numeric_cols if col in df.columns]
            if len(corr_cols) >= 2:
                corr_matrix = df[corr_cols].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig_corr.update_layout(
                    title="Feature Correlation Matrix",
                    height=500
                )
                figs['correlation'] = fig_corr
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
    
    return figs

# =============================================================================
# PDF REPORT GENERATION (ENHANCED)
# =============================================================================

def generate_pdf_report(insights: dict, categorical_cols: list, numeric_cols: list) -> BytesIO:
    """Generate comprehensive PDF report"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=30
        )
        story.append(Paragraph("🎯 Anomaly Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Records', f"{insights['total_records']:,}"],
            ['Anomalies Detected', f"{insights['anomalies']:,}"],
            ['Anomaly Rate', f"{insights['anomaly_rate']:.2f}%"],
            ['Detection Method', insights.get('method', 'N/A')],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Feature Analysis
        if 'feature_importance' in insights and insights['feature_importance']:
            story.append(Paragraph("Feature Analysis", styles['Heading2']))
            feat_data = [['Feature', 'Anomaly Avg', 'Normal Avg', 'Difference']]
            
            for feat, stats in sorted(insights['feature_importance'].items(), 
                                     key=lambda x: x[1]['difference'], reverse=True):
                feat_data.append([
                    feat,
                    f"{stats['anomaly_mean']:.2f}",
                    f"{stats['normal_mean']:.2f}",
                    f"{stats['difference']:.2f}"
                ])
            
            feat_table = Table(feat_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            feat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            story.append(feat_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("Recommended Actions", styles['Heading2']))
        recommendations = f"""
        <para>
        1. <b>Immediate Review:</b> Prioritize {insights['anomalies']} flagged records<br/>
        2. <b>Investigation Focus:</b> Analyze top differentiating features<br/>
        3. <b>Validation:</b> Manually review sample of flagged records<br/>
        4. <b>Monitoring:</b> Set up alerts for anomaly rate > {insights['anomaly_rate']*1.5:.1f}%<br/>
        5. <b>Documentation:</b> Record findings for future reference
        </para>
        """
        story.append(Paragraph(recommendations, styles['BodyText']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# =============================================================================
# NEW: CHART NARRATIVES
# =============================================================================

def get_chart_narrative(chart_type: str) -> str:
    """Provide contextual narratives for each chart type"""
    narratives = {
        'pie': """
        **What you're seeing:** The overall split between normal and anomalous records in your dataset.
        
        **How to interpret:** 
        - Green = Normal records (expected behavior)
        - Red = Anomalies (unusual patterns)
        - Check if the ratio matches your domain expectations
        """,
        'histogram': """
        **What you're seeing:** Distribution of anomaly confidence scores across all records.
        
        **How to interpret:**
        - Higher scores = More confident anomaly detection
        - Look for clear separation between normal (left) and anomalous (right) distributions
        - Overlapping distributions may indicate borderline cases requiring manual review
        """,
        'timeseries': """
        **What you're seeing:** Anomaly occurrences over time to identify patterns or spikes.
        
        **How to interpret:**
        - Spikes indicate periods with unusual activity
        - Consistent patterns may reveal systemic issues
        - Random distribution suggests isolated incidents
        """,
        'scatter': """
        **What you're seeing:** Relationship between two key features, colored by anomaly status.
        
        **How to interpret:**
        - Red points in sparse regions = outliers
        - Clusters of red points = systematic anomalies
        - Use this to identify which feature combinations drive anomalies
        """,
        'category': """
        **What you're seeing:** Which categories contain the most anomalies.
        
        **How to interpret:**
        - Tallest bars = Categories with most anomalies
        - Compare to category sizes (some may naturally have more records)
        - Focus investigation on unexpected high-anomaly categories
        """,
        'boxplot': """
        **What you're seeing:** Feature value distributions for normal vs. anomalous records.
        
        **How to interpret:**
        - Compare box positions (medians) between normal and anomaly
        - Longer boxes = More variability
        - Points outside whiskers = Extreme values
        """
    }
    return narratives.get(chart_type, "")

# =============================================================================
# NEW: EXPORT WITH METADATA
# =============================================================================

def export_results_with_metadata(df: pd.DataFrame, insights: dict, method: str) -> BytesIO:
    """Export results with analysis metadata sheet"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Results sheet
        df.to_excel(writer, sheet_name='Anomaly Results', index=False)
        
        # Metadata sheet
        metadata = pd.DataFrame([
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detection Method', method],
            ['Total Records', insights['total_records']],
            ['Anomalies Found', insights['anomalies']],
            ['Anomaly Rate', f"{insights['anomaly_rate']:.2f}%"],
            ['', ''],
            ['Top Features by Impact', '']
        ])
        
        if 'feature_importance' in insights:
            for feat, stats in sorted(insights['feature_importance'].items(), 
                                     key=lambda x: x[1]['difference'], reverse=True)[:5]:
                metadata = pd.concat([metadata, pd.DataFrame([[feat, f"Δ {stats['difference']:.2f}"]])], ignore_index=True)
        
        metadata.to_excel(writer, sheet_name='Analysis Metadata', index=False, header=False)
    
    buffer.seek(0)
    return buffer

# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Anomaly Hunter Pro</h1>
    <p>Enterprise-Grade Anomaly Detection | Powered by Advanced ML & Statistical Methods</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your data",
        type=['csv', 'xlsx', 'xls', 'txt', 'parquet'],
        help="Supports CSV, Excel, TXT, and Parquet files up to 200MB"
    )
    
    if uploaded_file:
        file_type = get_file_type(uploaded_file.name)
        
        if file_type and not st.session_state.data_loaded:
            with st.spinner("🔄 Processing file..."):
                parquet_path, rows, cols = convert_to_parquet(uploaded_file, file_type)
                
                if parquet_path:
                    st.session_state.parquet_path = parquet_path
                    st.session_state.row_count = rows
                    st.session_state.data_loaded = True
                    
                    con = initialize_duckdb()
                    st.session_state.column_info = get_column_stats(con, parquet_path)
                    
                    st.success(f"✅ Loaded: {rows:,} rows × {cols} columns")
                    st.rerun()
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("## ⚙️ Detection Settings")
        
        # Column selection
        numeric_cols = [col for col, info in st.session_state.column_info.items() 
                       if 'INT' in info['type'].upper() or 'DOUBLE' in info['type'].upper() or 'DECIMAL' in info['type'].upper()]
        
        categorical_cols = [col for col, info in st.session_state.column_info.items()
                          if 'VARCHAR' in info['type'].upper() and info['distinct'] < 50]
        
        date_cols = [col for col, info in st.session_state.column_info.items()
                    if 'DATE' in info['type'].upper() or 'TIMESTAMP' in info['type'].upper()]
        
        st.session_state.selected_numeric_cols = st.multiselect(
            "🔢 Numeric Features",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            help="Select features to analyze for anomalies"
        )
        
        st.session_state.selected_categorical_cols = st.multiselect(
            "📋 Categorical Features (optional)",
            categorical_cols,
            help="For grouping and segmentation"
        )
        
        if date_cols:
            st.session_state.selected_date_col = st.selectbox(
                "📅 Date/Time Column (optional)",
                [None] + date_cols,
                help="For time-based analysis"
            )
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Method")
        
        method_category = st.radio(
            "Category",
            ["Statistical", "Machine Learning", "Ensemble"],
            help="Statistical: Fast, interpretable | ML: Advanced pattern detection | Ensemble: Best of both"
        )
        
        if method_category == "Statistical":
            stat_method = st.selectbox(
                "Algorithm",
                ["Z-Score", "IQR (Interquartile Range)", "MAD (Median Absolute Deviation)"],
                help="Z-Score: Standard deviations | IQR: Box plot method | MAD: Robust to outliers"
            )
            
            threshold = st.slider(
                "Sensitivity Threshold",
                min_value=1.5,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Lower = More sensitive (more anomalies)"
            )
            
            st.session_state.selected_method = ('statistical', stat_method.split()[0].lower(), threshold)
            
        elif method_category == "Machine Learning":
            ml_method = st.selectbox(
                "Algorithm",
                ["Isolation Forest", "Local Outlier Factor (LOF)", "One-Class SVM", "Elliptic Envelope"],
                help="Different ML approaches for anomaly detection"
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate",
                min_value=0.01,
                max_value=0.30,
                value=0.10,
                step=0.01,
                format="%.2f",
                help="Expected proportion of anomalies (0.10 = 10%)"
            )
            
            st.session_state.contamination_rate = contamination
            
            method_map = {
                "Isolation Forest": "isolation_forest",
                "Local Outlier Factor (LOF)": "lof",
                "One-Class SVM": "one_class_svm",
                "Elliptic Envelope": "elliptic_envelope"
            }
            
            st.session_state.selected_method = ('ml', method_map[ml_method], contamination)
            
            # Show method description
            method_key = method_map[ml_method]
            if method_key in DETECTION_METHODS:
                st.info(f"ℹ️ {DETECTION_METHODS[method_key]['description']}")
            
        else:  # Ensemble
            contamination = st.slider(
                "Expected Anomaly Rate",
                min_value=0.01,
                max_value=0.30,
                value=0.10,
                step=0.01,
                format="%.2f"
            )
            
            st.session_state.contamination_rate = contamination
            st.session_state.selected_method = ('ensemble', 'ensemble', contamination)
            
            st.info("🎯 Ensemble combines Isolation Forest, LOF, and Elliptic Envelope - flags anomalies when 2+ methods agree")
        
        st.markdown("---")
        
        # Run detection
        if st.button("🚀 RUN DETECTION", use_container_width=True):
            if not st.session_state.selected_numeric_cols:
                st.error("⚠️ Select at least one numeric feature")
            else:
                with st.spinner("🔍 Detecting anomalies..."):
                    con = initialize_duckdb()
                    
                    method_type, method_name, param = st.session_state.selected_method
                    
                    if method_type == 'statistical':
                        df_results = detect_anomalies_statistical(
                            con,
                            st.session_state.parquet_path,
                            st.session_state.selected_numeric_cols,
                            method_name,
                            param
                        )
                    elif method_type == 'ensemble':
                        df_full = get_sample_data(con, st.session_state.parquet_path, limit=100000)
                        df_results = detect_anomalies_ensemble(
                            df_full,
                            st.session_state.selected_numeric_cols,
                            param
                        )
                    else:  # ML
                        df_full = get_sample_data(con, st.session_state.parquet_path, limit=100000)
                        df_results = detect_anomalies_ml(
                            df_full,
                            st.session_state.selected_numeric_cols,
                            method_name,
                            param
                        )
                    
                    if df_results is not None:
                        st.session_state.results_df = df_results
                        st.session_state.analysis_complete = True
                        
                        # Save to history
                        st.session_state.anomaly_history.append({
                            'timestamp': datetime.now(),
                            'method': f"{method_type}_{method_name}",
                            'anomaly_count': int(df_results['anomaly'].sum()),
                            'anomaly_rate': float(df_results['anomaly'].sum() / len(df_results) * 100)
                        })
                        
                        st.rerun()
        
        # Clear button
        if st.session_state.data_loaded:
            if st.button("🗑️ Clear Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'temp_dir':
                        del st.session_state[key]
                st.rerun()

# Main area
if not st.session_state.data_loaded:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Anomaly Hunter Pro</h3>
        <p><b>Get started in 3 steps:</b></p>
        <ol>
            <li>📁 Upload your CSV, Excel, or Parquet file (up to 200MB)</li>
            <li>⚙️ Select features and detection method</li>
            <li>🚀 Run detection and analyze results</li>
        </ol>
        <p><b>New in this version:</b></p>
        <ul>
            <li>🆕 <b>Ensemble Detection</b> - Combines multiple ML methods for higher accuracy</li>
            <li>🆕 <b>MAD Method</b> - Robust statistical detection using Median Absolute Deviation</li>
            <li>🆕 <b>Correlation Analysis</b> - Understand feature relationships</li>
            <li>🆕 <b>Enhanced Visualizations</b> - Correlation heatmaps and improved charts</li>
            <li>🆕 <b>Confidence Scores</b> - Percentile rankings for anomaly prioritization</li>
            <li>✅ <b>Fixed SQL Error</b> - Resolved DuckDB percentile syntax issue</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data preview
    st.markdown("### 📊 What kind of data works best?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✅ Great for:**
        - Transaction logs
        - Sensor readings
        - Customer behavior data
        - Quality metrics
        - Financial records
        """)
    
    with col2:
        st.markdown("""
        **📋 Required:**
        - At least 100 rows
        - 1+ numeric columns
        - Clean data (few nulls)
        - Consistent format
        """)
    
    with col3:
        st.markdown("""
        **🎯 Use Cases:**
        - Fraud detection
        - Quality control
        - System monitoring
        - Customer segmentation
        - Process optimization
        """)

else:
    # Data loaded - show preview
    if not st.session_state.analysis_complete:
        st.markdown("### 📊 Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("Total Rows", f"{st.session_state.row_count:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Columns", len(st.session_state.column_info)), unsafe_allow_html=True)
        with col3:
            numeric_count = len([c for c, i in st.session_state.column_info.items() if 'INT' in i['type'].upper() or 'DOUBLE' in i['type'].upper()])
            st.markdown(create_metric_card("Numeric", numeric_count), unsafe_allow_html=True)
        with col4:
            categorical_count = len([c for c, i in st.session_state.column_info.items() if 'VARCHAR' in i['type'].upper()])
            st.markdown(create_metric_card("Categorical", categorical_count), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sample data
        con = initialize_duckdb()
        sample_df = get_sample_data(con, st.session_state.parquet_path, limit=10)
        st.dataframe(sample_df, use_container_width=True)
        
        # Column details
        with st.expander("📋 Column Details"):
            col_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': info['type'],
                    'Distinct': info['distinct'],
                    'Nulls': f"{info['null_pct']:.1f}%"
                }
                for col, info in st.session_state.column_info.items()
            ])
            st.dataframe(col_df, use_container_width=True)
    
    else:
        # Show results
        df_results = st.session_state.results_df
        numeric_cols = st.session_state.selected_numeric_cols
        categorical_cols = st.session_state.selected_categorical_cols
        date_col = st.session_state.selected_date_col
        
        method_type, method_name, _ = st.session_state.selected_method
        method = f"{method_type}_{method_name}"
        
        insights = generate_insights(df_results, categorical_cols, numeric_cols, method)
        
        st.markdown("### 🎯 Detection Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("Total Records", f"{insights['total_records']:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Anomalies", f"{insights['anomalies']:,}"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Anomaly Rate", f"{insights['anomaly_rate']:.2f}%"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Method", method.replace('_', ' ').title()), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key findings
        if insights['anomalies'] > 0:
            st.markdown(f"""
            <div class="warning-box">
                <b>🔍 Key Findings:</b><br>
                • Detected <b>{insights['anomalies']:,} anomalies</b> ({insights['anomaly_rate']:.2f}% of dataset)<br>
                • Analyzed <b>{len(numeric_cols)} features</b>: {', '.join(numeric_cols)}<br>
                {'• High priority: ' + str(len(df_results[df_results['anomaly_score'] > df_results['anomaly_score'].quantile(0.95)])) + ' records with scores >95th percentile' if 'anomaly_score' in df_results.columns else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <b>✅ Clean Dataset:</b> No significant anomalies detected. Consider adjusting sensitivity or trying different features.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Visual Analysis")
        
        figs = create_visualizations(df_results, numeric_cols, categorical_cols, date_col)
        
        # Two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pie' in figs:
                with st.expander("ℹ️ Chart Insights", expanded=False):
                    st.markdown(get_chart_narrative('pie'))
                st.plotly_chart(figs['pie'], use_container_width=True)
        
        with col2:
            if 'histogram' in figs:
                with st.expander("ℹ️ Chart Insights", expanded=False):
                    st.markdown(get_chart_narrative('histogram'))
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        # Full width charts
        if 'timeseries' in figs:
            with st.expander("ℹ️ Chart Insights", expanded=False):
                st.markdown(get_chart_narrative('timeseries'))
            st.plotly_chart(figs['timeseries'], use_container_width=True)
        
        if 'scatter' in figs:
            with st.expander("ℹ️ Chart Insights", expanded=False):
                st.markdown(get_chart_narrative('scatter'))
            st.plotly_chart(figs['scatter'], use_container_width=True)
        
        if 'correlation' in figs:
            st.plotly_chart(figs['correlation'], use_container_width=True)
        
        # Category charts
        for key in figs.keys():
            if key.startswith('category_bar_'):
                with st.expander("ℹ️ Chart Insights", expanded=False):
                    st.markdown(get_chart_narrative('category'))
                st.plotly_chart(figs[key], use_container_width=True)
        
        if 'boxplot' in figs:
            with st.expander("📦 Box Plot Comparison"):
                st.markdown(get_chart_narrative('boxplot'))
                st.plotly_chart(figs['boxplot'], use_container_width=True)
        
        st.markdown("---")
        
        # Feature importance
        if 'feature_importance' in insights and insights['feature_importance']:
            st.markdown("### 📊 Feature Impact Analysis")
            
            feat_df = pd.DataFrame([
                {
                    'Feature': feat,
                    'Anomaly Avg': f"{stats['anomaly_mean']:.2f}",
                    'Normal Avg': f"{stats['normal_mean']:.2f}",
                    'Difference': f"{stats['difference']:.2f}",
                    'Impact %': f"{stats['pct_difference']:.1f}%"
                }
                for feat, stats in sorted(insights['feature_importance'].items(), 
                                         key=lambda x: x[1]['difference'], reverse=True)
            ])
            
            st.dataframe(feat_df, use_container_width=True)
        
        # Categorical insights
        if 'categorical_insights' in insights and insights['categorical_insights']:
            st.markdown("---")
            st.markdown("### 📋 Categorical Breakdown")
            
            for cat_col, cat_data in insights['categorical_insights'].items():
                with st.expander(f"🏆 {cat_col} Analysis", expanded=True):
                    if cat_data['top_categories']:
                        cat_df = pd.DataFrame(cat_data['top_categories'])
                        cat_df['anomaly_rate'] = cat_df['anomaly_rate'].round(2)
                        st.dataframe(cat_df.head(10), use_container_width=True)
                        
                        if cat_data['highest_rate']:
                            highest = cat_data['highest_rate']
                            st.info(f"🥇 Highest: **{highest[cat_col]}** with **{highest['anomaly_rate']:.2f}%** anomaly rate")
        
        # Full results
        with st.expander("📋 Full Results Table (Click to expand)"):
            st.dataframe(df_results, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Action recommendations
        if insights['anomalies'] > 0:
            st.markdown("### 🎯 Recommended Actions")
            
            high_priority = len(df_results[df_results['anomaly_score'] > df_results['anomaly_score'].quantile(0.95)]) if 'anomaly_score' in df_results.columns else insights['anomalies']
            
            st.markdown(f"""
            **1️⃣ IMMEDIATE (Next 24-48 hours)**
            - Export and review **{high_priority} high-priority records** (top 5% by score)
            - Assign to investigation team with clear ownership
            - Document initial findings in tracking system
            
            **2️⃣ INVESTIGATION FOCUS**
            """)
            
            if 'feature_importance' in insights and insights['feature_importance']:
                top_feat = max(insights['feature_importance'].items(), key=lambda x: x[1]['difference'])
                st.markdown(f"- **Primary driver:** {top_feat[0]} (avg anomaly: {top_feat[1]['anomaly_mean']:.2f} vs normal: {top_feat[1]['normal_mean']:.2f})")
            
            if 'categorical_insights' in insights and insights['categorical_insights']:
                for cat_col, cat_data in list(insights['categorical_insights'].items())[:1]:
                    if cat_data['highest_rate']:
                        highest = cat_data['highest_rate']
                        st.markdown(f"- **Concentrated in:** {cat_col} = '{highest[cat_col]}' ({highest['anomaly_rate']:.1f}% anomaly rate)")
            
            st.markdown(f"""
            **3️⃣ PREVENTION & MONITORING**
            - Set threshold alerts when rate exceeds {insights['anomaly_rate']*1.5:.1f}% (1.5x current)
            - Create automated review queue for recurring patterns
            - Schedule weekly anomaly trend analysis
            
            **4️⃣ VALIDATION**
            - Sample 20 random flagged records for manual review
            - If >80% are genuine issues → Model is well-calibrated ✓
            - If <50% are genuine → Increase expected anomaly rate and re-run
            """)
        
        else:
            st.markdown("### ✅ Clean Dataset")
            st.markdown("""
            No significant anomalies detected. Consider:
            - Lowering the sensitivity threshold
            - Selecting different features
            - Using a different detection method (try Z-Score with threshold 2.5)
            """)
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_buffer = BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 CSV",
                data=csv_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            excel_buffer = export_results_with_metadata(df_results, insights, method)
            st.download_button(
                "📥 Excel + Metadata",
                data=excel_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            parquet_buffer = BytesIO()
            df_results.to_parquet(parquet_buffer, compression='snappy')
            st.download_button(
                "📥 Parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        with col4:
            pdf_buffer = generate_pdf_report(insights, categorical_cols, numeric_cols)
            if pdf_buffer:
                st.download_button(
                    "📥 PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 1.5rem;'>
    <p style='font-size: 14px; font-weight: 300;'>
        🎯 <b>Anomaly Hunter Pro v2.0</b> | Enhanced Edition<br>
        Powered by DuckDB • PyArrow • Scikit-learn • Plotly<br>
        <i>Built for Data Analysts, QA Teams, and Data Scientists</i>
    </p>
</div>
""", unsafe_allow_html=True)
