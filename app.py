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
from scipy import stats

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="Anomaly Detection Pro 🎯",
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

# Minimalist CSS with fixed scaling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Force proper scaling */
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
    
    .info-box, .success-box {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    
    .info-box {
        border-left: 4px solid #f59e0b;
    }
    
    .success-box {
        border-left: 4px solid #10b981;
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
    
    /* Sidebar fixes */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        font-size: 0.875rem;
    }
    
    /* Improved expander in sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: white;
        font-size: 0.9rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: 2px solid #e2e8f0;
        font-size: 0.875rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: 2px solid transparent;
    }
    
    /* Better form labels */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Radio button horizontal layout */
    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
    
    .lottie-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOTTIE ANIMATION LOADER
# =============================================================================

def load_lottie_url(url: str) -> dict | None:
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def display_lottie(lottie_json: dict | None, height: int = 200, key: str = None):
    """Display Lottie animation using streamlit-lottie"""
    if lottie_json:
        try:
            from streamlit_lottie import st_lottie
            st_lottie(lottie_json, height=height, key=key)
        except ImportError:
            # Fallback if streamlit-lottie not installed
            st.info("💡 Install streamlit-lottie for animations: `pip install streamlit-lottie`")

# Lottie animation URLs (free from LottieFiles)
LOTTIE_URLS = {
    'upload': 'https://lottie.host/6dadc34a-b444-48fb-a796-e6b8b5a54172/QtBAFnRjNv.json',
    'processing': 'https://lottie.host/74287700-bcf8-489b-a4da-7f749e114c3c/mxSexILitR.json',
    'success': 'https://lottie.host/68f99417-882d-48e7-990d-1db441c73c7e/7zNWZQiOK0.json',
    'empty': 'https://lottie.host/415990e8-43a1-44db-a967-cb4f989c7dda/kZDZtv4i60.json',
    'hero': 'https://lottie.host/35b4b211-d909-41e5-8ea6-cfd26d9ee338/Un0LjC6t8a.json'
}




# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_file_type(file) -> str | None:
    """Detect file type from uploaded file using match/case"""
    filename = file.name.lower()
    match Path(filename).suffix:
        case '.csv': return 'csv'
        case '.xlsx': return 'xlsx'
        case '.xls': return 'xls'
        case '.txt': return 'txt'
        case '.parquet': return 'parquet'
        case _: return None

def convert_to_parquet(file, file_type: str) -> tuple[str | None, int, int]:
    """Convert files to Parquet using DuckDB - handles large files efficiently"""
    try:
        progress = st.progress(0, text="🔄 Reading file...")
        
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        temp_csv = os.path.join(st.session_state.temp_dir, 'temp_data.csv')
        
        # Save uploaded file temporarily
        progress.progress(20, text="📦 Preparing file...")
        with open(temp_csv, 'wb') as f:
            f.write(file.getvalue())
        
        progress.progress(40, text="💾 Converting with DuckDB...")
        
        # Use DuckDB to convert directly - handles any size file
        con = duckdb.connect(':memory:')
        
        match file_type:
            case 'csv' | 'txt':
                # DuckDB auto-detects CSV format
                con.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{temp_csv}', 
                                                       sample_size=-1,
                                                       ignore_errors=true))
                    TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
                """)
                
            case 'xlsx' | 'xls':
                # For Excel, read with pandas then let DuckDB handle it
                df = pd.read_excel(file, engine='openpyxl' if file_type == 'xlsx' else 'xlrd')
                con.register('temp_table', df)
                con.execute(f"COPY temp_table TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
                
            case 'parquet':
                # Already parquet, just copy
                with open(parquet_path, 'wb') as f:
                    f.write(file.getvalue())
                con.execute(f"SELECT COUNT(*) as cnt FROM parquet_scan('{parquet_path}')")
                row_count = con.fetchone()[0]
                con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1")
                col_count = len(con.fetchone())
                con.close()
                
                # Cleanup
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                
                progress.progress(100, text="✅ File ready!")
                time.sleep(0.5)
                progress.empty()
                return parquet_path, row_count, col_count
                
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        progress.progress(80, text="📊 Getting file info...")
        
        # Get row and column counts
        con.execute(f"SELECT COUNT(*) as cnt FROM parquet_scan('{parquet_path}')")
        row_count = con.fetchone()[0]
        
        con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 1")
        result = con.fetchone()
        col_count = len(result) if result else 0
        
        con.close()
        
        # Cleanup temp file
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
        
        # Cleanup on error
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
    **What this means:** This is your **sensitivity guide**. The algorithm will use ~**{expected_anomalies:,}** 
    as a reference point ({rate*100:.1f}%), but can find MORE or FEWER anomalies based on actual data patterns.
    
    **Sensitivity:** {sensitivity.title()} - typical for {use_case}
    
    **💡 How it works:** The algorithm calculates anomaly scores for ALL records, then uses this rate to set 
    an intelligent threshold. Records scoring significantly above this threshold are flagged as anomalies.
    
    **Key difference:** Unlike the old version, this won't force exactly {rate*100:.1f}% to be anomalies - 
    it discovers what's genuinely unusual in your data.
    """

# =============================================================================
# ANOMALY DETECTION ENGINE (FACTORY PATTERN)
# =============================================================================

# Detection methods configuration
DETECTION_METHODS = {
    'isolation_forest': {
        'class': IsolationForest,
        'params': {'contamination': 'auto', 'random_state': 42, 'n_jobs': -1},
        'type': 'ml'
    },
    'lof': {
        'class': LocalOutlierFactor,
        'params': {'n_neighbors': 20, 'contamination': 'auto', 'n_jobs': -1},
        'type': 'ml'
    },
    'one_class_svm': {
        'class': OneClassSVM,
        'params': {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'auto'},
        'type': 'ml'
    },
}

def detect_anomalies_statistical(con, parquet_path: str, numeric_cols: list, method: str = 'zscore', threshold: float = 3) -> tuple[pd.DataFrame | None, dict]:
    """SQL-based statistical anomaly detection - returns (dataframe, metadata)"""
    try:
        if method == 'zscore':
            select_parts = [f"""
                ABS(("{col}" - AVG("{col}") OVER ()) / NULLIF(STDDEV("{col}") OVER (), 0)) as zscore_{col.replace(' ', '_')}
            """ for col in numeric_cols]
            
            query = f"""
            SELECT *, 
                {', '.join(select_parts)},
                GREATEST({', '.join([f'zscore_{col.replace(" ", "_")}' for col in numeric_cols])}) as max_zscore
            FROM parquet_scan('{parquet_path}')
            """
            
            df = con.execute(query).df()
            df['anomaly'] = (df['max_zscore'] > threshold).astype(int)
            df['anomaly_score'] = df['max_zscore']
            
            metadata = {
                'method': 'zscore',
                'threshold_value': threshold,
                'threshold_display': f"{threshold:.1f}σ (standard deviations)"
            }
            
        else:  # IQR method
            percentiles = {}
            for col in numeric_cols:
                pct_query = f"""
                SELECT 
                    percentile_cont(0.25) WITHIN GROUP (ORDER BY "{col}") as q1,
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY "{col}") as q3
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
            
            metadata = {
                'method': 'iqr',
                'threshold_value': 1.5,
                'threshold_display': "1.5×IQR range"
            }
        
        return df, metadata
    except Exception as e:
        st.error(f"Error in statistical detection: {e}")
        return None, {}

def detect_anomalies_ml(df: pd.DataFrame, numeric_cols: list, method: str, contamination: float = 0.1) -> tuple[pd.DataFrame | None, dict]:
    """ML-based anomaly detection using score-based thresholding instead of hard contamination constraint
    
    KEY IMPROVEMENT: contamination is now a GUIDE, not a hard constraint. The algorithm:
    1. Trains on all data to learn normal patterns
    2. Calculates anomaly scores for every record
    3. Uses contamination to set an intelligent threshold
    4. Flags records that score SIGNIFICANTLY above threshold
    5. Can find MORE or FEWER anomalies than expected based on actual data
    """
    try:
        # Handle missing values
        df_clean = df.copy()
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Prepare and scale features
        X = df_clean[numeric_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Get model configuration
        config = DETECTION_METHODS[method]
        model_class = config['class']
        params = config['params'].copy()
        
        # Adjust LOF n_neighbors for small datasets
        if method == 'lof':
            params['n_neighbors'] = min(params['n_neighbors'], len(df_clean) - 1)
        
        # Train models WITHOUT hard contamination constraint
        if method == 'isolation_forest':
            # Use contamination as hint but get raw scores
            params['contamination'] = contamination
            model = model_class(**params)
            model.fit(X_scaled)
            scores = model.score_samples(X_scaled)
            # Higher negative scores = more anomalous, convert to positive
            anomaly_scores = -scores
            
        elif method == 'lof':
            # LOF requires contamination, but we'll override with scores
            params['contamination'] = contamination
            params['novelty'] = False
            model = model_class(**params)
            model.fit(X_scaled)
            scores = model.negative_outlier_factor_
            # More negative = more anomalous
            anomaly_scores = -scores
            
        elif method == 'one_class_svm':
            # Train without nu constraint, use decision function
            params_train = params.copy()
            params_train.pop('nu', None)
            model = model_class(nu=0.5, **params_train)  # Use neutral nu for training
            model.fit(X_scaled)
            scores = model.decision_function(X_scaled)
            # More negative = more anomalous
            anomaly_scores = -scores
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize scores to 0-10 scale for consistency
        score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
        if score_max > score_min:
            normalized_scores = 10 * (anomaly_scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.zeros_like(anomaly_scores)
        
        # Use contamination as GUIDE to set threshold, not hard constraint
        # Calculate threshold based on expected percentile
        threshold_percentile = (1 - contamination) * 100
        score_threshold = np.percentile(normalized_scores, threshold_percentile)
        
        # Apply intelligent thresholding: flag if score significantly above threshold
        # Use 1.2x threshold as buffer to focus on truly extreme cases
        adaptive_threshold = max(score_threshold * 1.2, 6.0)  # Minimum threshold of 6.0
        
        df_clean['anomaly'] = (normalized_scores > adaptive_threshold).astype(int)
        df_clean['anomaly_score'] = normalized_scores
        
        # Store detection metadata in session state (df.attrs breaks parquet export)
        expected_count = int(len(df_clean) * contamination)
        actual_count = df_clean['anomaly'].sum()
        
        return df_clean, {
            'method': 'ml',
            'expected_anomalies': expected_count,
            'actual_anomalies': actual_count,
            'threshold_value': adaptive_threshold,
            'threshold_display': f"{adaptive_threshold:.2f} (0-10 scale)"
        }
    except Exception as e:
        st.error(f"Error in ML detection: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, {}

# =============================================================================
# INSIGHTS & VISUALIZATIONS
# =============================================================================

def generate_insights(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, method: str) -> dict:
    """Generate comprehensive insights"""
    try:
        total = len(df)
        anomalies = df['anomaly'].sum()
        pct = (anomalies / total * 100) if total > 0 else 0
        
        insights = {
            'total_records': total,
            'anomalies': anomalies,
            'anomaly_rate': pct,
            'method': method,
            'categorical_insights': {}
        }
        
        # Add detection metadata from session state if available
        if 'detection_metadata' in st.session_state:
            metadata = st.session_state.detection_metadata
            insights['expected_anomalies'] = metadata.get('expected_anomalies', None)
            insights['threshold_value'] = metadata.get('threshold_value', None)
            insights['threshold_display'] = metadata.get('threshold_display', None)
            insights['detection_method_type'] = metadata.get('method', None)
        
        # Category-level analysis
        if categorical_cols:
            for categorical_col in categorical_cols:
                if categorical_col in df.columns:
                    grp = df.groupby(categorical_col)['anomaly'].agg(['sum', 'count']).reset_index()
                    grp.columns = [categorical_col, 'anomaly_count', 'total_count']
                    grp['anomaly_rate'] = (grp['anomaly_count'] / grp['total_count'] * 100)
                    grp = grp.sort_values('anomaly_rate', ascending=False)
                    
                    insights['categorical_insights'][categorical_col] = {
                        'top_categories': grp.head(10).to_dict('records'),
                        'highest_rate': grp.iloc[0].to_dict() if len(grp) > 0 else None
                    }
        
        # Feature importance
        if numeric_cols:
            anomaly_df = df[df['anomaly'] == 1]
            normal_df = df[df['anomaly'] == 0]
            
            feature_stats = {}
            for col in numeric_cols:
                if col in df.columns:
                    feature_stats[col] = {
                        'anomaly_mean': anomaly_df[col].mean(),
                        'normal_mean': normal_df[col].mean(),
                        'anomaly_std': anomaly_df[col].std(),
                        'difference': abs(anomaly_df[col].mean() - normal_df[col].mean())
                    }
            
            insights['feature_importance'] = feature_stats
        
        # Top anomalies
        if 'anomaly_score' in df.columns:
            top_anomalies = df.nlargest(5, 'anomaly_score')
            insights['top_anomalies'] = top_anomalies.to_dict('records')
        
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return {}

def generate_visual_summary(insights: dict, categorical_cols: list, numeric_cols: list, expected_rate: float = None):
    """Generate professional visual summary with Streamlit native components"""
    total = insights['total_records']
    anomalies = insights['anomalies']
    rate = insights['anomaly_rate']
    method = insights['method']
    
    # === EXECUTIVE VERDICT ===
    st.markdown("### 🎯 EXECUTIVE SUMMARY")
    
    # Verdict box with color coding
    if expected_rate and anomalies > 0:
        expected_pct = expected_rate * 100
        expected_count = insights.get('expected_anomalies', int(total * expected_rate))
        diff = anomalies - expected_count
        diff_pct = (diff / expected_count * 100) if expected_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected", f"{expected_count:,}", f"{expected_pct:.1f}%")
        with col2:
            st.metric("Found", f"{anomalies:,}", f"{rate:.2f}%")
        with col3:
            st.metric("Difference", f"{diff:+,}", f"{diff_pct:+.1f}%", delta_color="inverse")
        
        # Verdict message
        if abs(diff_pct) < 10:
            st.success(f"✅ **On Target** - Detection aligns with expectations. Found {anomalies:,} anomalies within {abs(diff_pct):.1f}% of target.")
        elif diff > 0:
            st.error(f"🚨 **High Alert** - Found {abs(diff):,} MORE anomalies than expected ({diff_pct:+.1f}%). Investigate immediately.")
        else:
            st.info(f"✓ **Cleaner Than Expected** - Found {abs(diff):,} fewer anomalies. Data quality is better than anticipated.")
    else:
        # Simple verdict for non-ML methods
        if rate < 1:
            st.success(f"✅ **Excellent** - Only {anomalies:,} anomalies ({rate:.2f}%). Very clean dataset.")
        elif rate < 5:
            st.info(f"✓ **Good** - {anomalies:,} anomalies ({rate:.2f}%). Typical for quality data.")
        elif rate < 10:
            st.warning(f"⚠️ **Moderate** - {anomalies:,} anomalies ({rate:.2f}%). Review patterns carefully.")
        else:
            st.error(f"🚨 **High** - {anomalies:,} anomalies ({rate:.2f}%). Significant issues detected.")
    
    # Threshold display
    if 'threshold_display' in insights and insights['threshold_display']:
        st.caption(f"Detection threshold: **{insights['threshold_display']}**")
    
    st.markdown("---")
    
    # === TOP FINDING - FEATURE ===
    if 'feature_importance' in insights and insights['feature_importance'] and anomalies > 0:
        top_feature = max(insights['feature_importance'].items(), key=lambda x: x[1]['difference'])
        feature_name = top_feature[0]
        stats = top_feature[1]
        ratio = stats['anomaly_mean'] / stats['normal_mean'] if stats['normal_mean'] != 0 else 0
        
        st.markdown("### 🔥 PRIMARY RED FLAG")
        
        # Create visual alert box
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;
                    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);'>
            <h3 style='margin: 0; color: white; font-size: 1.25rem;'>📊 {feature_name}</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;'>
                This metric shows the strongest deviation between normal and anomalous records.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomaly Average", f"{stats['anomaly_mean']:,.2f}", delta=None)
        with col2:
            st.metric("Normal Average", f"{stats['normal_mean']:,.2f}", delta=None)
        with col3:
            st.metric("Multiplier", f"{ratio:.1f}x", delta=None, delta_color="off")
        
        # Interpretation
        if ratio > 2:
            st.error(f"⚠️ **Critical:** Anomalies are {ratio:.1f}x higher than normal. This is your #1 investigation priority.")
        elif ratio < 0.5:
            st.error(f"⚠️ **Critical:** Anomalies are {(1/ratio):.1f}x lower than normal. Investigate unusual drops.")
        else:
            st.warning(f"⚠️ **Notable:** Anomalies differ by {abs(ratio-1)*100:.0f}% from normal baseline.")
        
        st.markdown("---")
    
    # === WHERE TO LOOK - CATEGORY ===
    if 'categorical_insights' in insights and insights['categorical_insights'] and anomalies > 0:
        hotspot_found = False
        
        for cat_col, cat_data in insights['categorical_insights'].items():
            if cat_data['highest_rate']:
                highest = cat_data['highest_rate']
                cat_name = highest.get(cat_col, 'Unknown')
                cat_rate = highest.get('anomaly_rate', 0)
                cat_count = highest.get('anomaly_count', 0)
                
                # Only show if significantly higher than average
                if cat_rate > rate * 1.5:  # 50% higher than average
                    if not hotspot_found:
                        st.markdown("### 🎯 INVESTIGATION HOTSPOT")
                        hotspot_found = True
                    
                    # Create hotspot card
                    severity_color = "#dc2626" if cat_rate > rate * 3 else "#f59e0b"
                    
                    st.markdown(f"""
                    <div style='background: white; border-left: 6px solid {severity_color}; 
                                padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1.1rem;'>
                            🔍 {cat_col}
                        </h4>
                        <p style='margin: 0; font-size: 1.5rem; font-weight: 700; color: {severity_color};'>
                            {cat_name}
                        </p>
                        <p style='margin: 0.75rem 0 0 0; color: #64748b; font-size: 0.9rem;'>
                            <b>{cat_count:,} anomalies</b> ({cat_rate:.1f}% rate) vs {rate:.2f}% overall
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    ratio_to_avg = cat_rate / rate if rate > 0 else 0
                    
                    if ratio_to_avg > 5:
                        st.error(f"🚨 **CRITICAL HOTSPOT:** This category has {ratio_to_avg:.1f}x the average anomaly rate. Start here immediately.")
                    elif ratio_to_avg > 3:
                        st.warning(f"⚠️ **High Priority:** {ratio_to_avg:.1f}x average rate. Prioritize for investigation.")
                    else:
                        st.info(f"📍 **Above Average:** {ratio_to_avg:.1f}x average rate. Review carefully.")
                    
                    break  # Only show top hotspot
        
        if hotspot_found:
            st.markdown("---")
    
    # === QUICK ACTION ITEMS ===
    if anomalies > 0:
        st.markdown("### ⚡ IMMEDIATE ACTIONS")
        
        high_score_threshold = insights.get('threshold_value', 7.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: #eff6ff; border: 2px solid #3b82f6; padding: 1rem; border-radius: 8px;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #1e40af;'>🎯 Priority 1</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #1e293b;'>
                    <li>Export flagged records (CSV button below)</li>
                    <li>Focus on top 5% highest scores first</li>
                    <li>Assign to audit team within 24hrs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #fef3c7; border: 2px solid #f59e0b; padding: 1rem; border-radius: 8px;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #92400e;'>📋 Priority 2</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #1e293b;'>
                    <li>Review patterns in top finding</li>
                    <li>Cross-check hotspot categories</li>
                    <li>Document common themes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def create_metric_card(label: str, value: str | int) -> str:
    """Generate HTML for metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def get_chart_narrative(chart_type: str, insights: dict = None) -> str:
    """Generate educational narrative for each visualization"""
    narratives = {
        'pie': """
        **📊 Distribution Overview**
        
        This shows your overall split. The **RED slice** represents records that don't fit the normal pattern - 
        these are your outliers. If this slice is tiny (<1%), you have a clean dataset. If it's large (>10%), 
        either your data has quality issues OR the normal pattern is very diverse.
        """,
        
        'histogram': """
        **📈 Anomaly Score Distribution**
        
        Anomaly scores are like 'weirdness ratings'. Records on the **RIGHT side** (high scores) are the most 
        unusual - prioritize investigating these first. The tall bar on the left shows your normal data 
        clustered together.
        
        💡 **Tip:** Focus on records with scores >7.0 for immediate action.
        """,
        
        'timeseries': """
        **⏰ Timeline Analysis**
        
        This timeline reveals **WHEN** anomalies happen. Spikes indicate problem periods - maybe a system glitch, 
        fraud attack, or seasonal pattern. Look for:
        - Sudden spikes (incidents)
        - Gradual increases (trend changes)
        - Weekly/monthly patterns (seasonality)
        """,
        
        'scatter': """
        **🔍 Feature Relationship**
        
        Each dot is a record. **RED dots** far from the GREEN cluster are your outliers. If you see a RED cluster, 
        that's a pattern worth naming - maybe 'high-value-low-frequency' transactions.
        
        💡 **Tip:** Outliers in the corners are the most extreme cases.
        """,
        
        'category': """
        **📋 Category Rankings**
        
        This ranks categories by anomaly concentration. Categories at the **top** deserve immediate attention. 
        If one category has 5x the average rate, that's not random - investigate the relationship.
        """,
        
        'boxplot': """
        **📦 Distribution Comparison**
        
        Box plots show the spread of values. **RED boxes** (anomalies) shifted away from **GREEN boxes** (normal) 
        indicate that feature is a strong discriminator. Overlapping boxes mean that feature isn't helpful for detection.
        """
    }
    
    return narratives.get(chart_type, "")

def create_visualizations(df: pd.DataFrame, numeric_cols: list, categorical_cols: list = None, date_col: str = None) -> dict:
    """Create comprehensive visualizations"""
    figs = {}
    colors_scheme = {'normal': '#10b981', 'anomaly': '#ef4444'}
    
    try:
        # 1. Anomaly Distribution Pie
        anomaly_counts = df['anomaly'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)],
            hole=0.4,
            marker=dict(colors=[colors_scheme['normal'], colors_scheme['anomaly']]),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig_pie.update_layout(
            title=dict(text="Anomaly Distribution", font=dict(size=18, color='#1e293b')),
            template="plotly_white",
            height=400,
            showlegend=True
        )
        figs['pie'] = fig_pie
        
        # 2. Anomaly Score Distribution
        if 'anomaly_score' in df.columns:
            fig_hist = px.histogram(
                df, x='anomaly_score', color='anomaly',
                nbins=50,
                title="Anomaly Score Distribution",
                labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'},
                color_discrete_map={0: colors_scheme['normal'], 1: colors_scheme['anomaly']}
            )
            fig_hist.update_layout(template="plotly_white", height=400)
            figs['histogram'] = fig_hist
        
        # 3. Time Series (if date column exists)
        if date_col and date_col in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                df_sorted = df.sort_values(date_col)
                daily_stats = df_sorted.groupby(pd.Grouper(key=date_col, freq='D')).agg({
                    'anomaly': ['sum', 'count']
                }).reset_index()
                daily_stats.columns = [date_col, 'anomalies', 'total']
                daily_stats['anomaly_rate'] = (daily_stats['anomalies'] / daily_stats['total'] * 100).fillna(0)
                
                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_time.add_trace(
                    go.Scatter(x=daily_stats[date_col], y=daily_stats['anomalies'],
                              name="Anomaly Count", line=dict(color=colors_scheme['anomaly'], width=2),
                              fill='tozeroy'),
                    secondary_y=False,
                )
                
                fig_time.add_trace(
                    go.Scatter(x=daily_stats[date_col], y=daily_stats['anomaly_rate'],
                              name="Anomaly Rate (%)", line=dict(color='#f59e0b', width=2, dash='dash')),
                    secondary_y=True,
                )
                
                fig_time.update_layout(
                    title="Anomalies Over Time",
                    template="plotly_white",
                    height=400,
                    hovermode='x unified'
                )
                fig_time.update_xaxes(title_text="Date")
                fig_time.update_yaxes(title_text="Anomaly Count", secondary_y=False)
                fig_time.update_yaxes(title_text="Anomaly Rate (%)", secondary_y=True)
                
                figs['timeseries'] = fig_time
            except Exception as e:
                st.warning(f"Could not create time series chart: {e}")
        
        # 4. Scatter Plot (if 2+ numeric columns)
        if len(numeric_cols) >= 2:
            sample_df = df.sample(min(10000, len(df)))
            fig_scatter = px.scatter(
                sample_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='anomaly',
                hover_data=numeric_cols[:5] if len(numeric_cols) > 2 else numeric_cols,
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                color_discrete_map={0: colors_scheme['normal'], 1: colors_scheme['anomaly']},
                opacity=0.6
            )
            fig_scatter.update_layout(template="plotly_white", height=500)
            figs['scatter'] = fig_scatter
        
        # 5. Category Analysis
        if categorical_cols:
            for i, categorical_col in enumerate(categorical_cols[:3]):
                if categorical_col in df.columns:
                    grp = df.groupby(categorical_col)['anomaly'].agg(['sum', 'count']).reset_index()
                    grp.columns = [categorical_col, 'anomaly_count', 'total_count']
                    grp['anomaly_rate'] = (grp['anomaly_count'] / grp['total_count'] * 100)
                    grp = grp.sort_values('anomaly_rate', ascending=False).head(15)
                    
                    fig_bar = px.bar(
                        grp,
                        x=categorical_col,
                        y='anomaly_rate',
                        title=f"Top 15 {categorical_col} by Anomaly Rate",
                        labels={'anomaly_rate': 'Anomaly Rate (%)'},
                        color='anomaly_rate',
                        color_continuous_scale='Reds'
                    )
                    fig_bar.update_layout(template="plotly_white", height=400)
                    figs[f'category_bar_{i}'] = fig_bar
        
        # 6. Box plots
        if numeric_cols:
            fig_box = make_subplots(
                rows=1, cols=min(3, len(numeric_cols)),
                subplot_titles=numeric_cols[:3]
            )
            
            for i, col in enumerate(numeric_cols[:3], 1):
                for anomaly_val, color in [(0, colors_scheme['normal']), (1, colors_scheme['anomaly'])]:
                    data = df[df['anomaly'] == anomaly_val][col]
                    fig_box.add_trace(
                        go.Box(y=data, name='Normal' if anomaly_val == 0 else 'Anomaly',
                               marker_color=color, showlegend=(i == 1)),
                        row=1, col=i
                    )
            
            fig_box.update_layout(
                title="Feature Distribution by Anomaly Status",
                template="plotly_white",
                height=400,
                showlegend=True
            )
            figs['boxplot'] = fig_box
        
        return figs
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return figs

def generate_pdf_report(insights: dict, categorical_cols: list, numeric_cols: list) -> BytesIO | None:
    """Generate PDF report"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=30,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#8b5cf6'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph("🎯 Anomaly Detection Pro - Analysis Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(Paragraph(f"<b>Total Records:</b> {insights['total_records']:,}", styles['Normal']))
        elements.append(Paragraph(f"<b>Anomalies Detected:</b> {insights['anomalies']:,} ({insights['anomaly_rate']:.2f}%)", styles['Normal']))
        elements.append(Paragraph(f"<b>Detection Method:</b> {insights['method']}", styles['Normal']))
        
        # Show expected vs actual if available
        if 'expected_anomalies' in insights:
            elements.append(Paragraph(f"<b>Expected Anomalies:</b> {insights['expected_anomalies']:,}", styles['Normal']))
        
        elements.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Categorical insights
        if 'categorical_insights' in insights and insights['categorical_insights']:
            for cat_col, cat_data in insights['categorical_insights'].items():
                if cat_data['top_categories']:
                    elements.append(Paragraph(f"Top Categories - {cat_col}", heading_style))
                    
                    table_data = [[cat_col, 'Anomaly Count', 'Total Count', 'Anomaly Rate (%)']]
                    for cat in cat_data['top_categories'][:10]:
                        table_data.append([
                            str(cat.get(cat_col, 'N/A'))[:30],
                            str(cat.get('anomaly_count', 0)),
                            str(cat.get('total_count', 0)),
                            f"{cat.get('anomaly_rate', 0):.2f}"
                        ])
                    
                    t = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ]))
                    elements.append(t)
                    elements.append(Spacer(1, 20))
        
        # Feature importance
        if 'feature_importance' in insights and insights['feature_importance']:
            elements.append(Paragraph("Feature Analysis", heading_style))
            
            feature_data = [['Feature', 'Anomaly Mean', 'Normal Mean', 'Difference']]
            for feat, stats in insights['feature_importance'].items():
                feature_data.append([
                    feat,
                    f"{stats['anomaly_mean']:.2f}",
                    f"{stats['normal_mean']:.2f}",
                    f"{stats['difference']:.2f}"
                ])
            
            t2 = Table(feature_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(t2)
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# =============================================================================
# MAIN APP
# =============================================================================

# Header with hero animation
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Anomaly Detection Pro</h1>
        <p>Enterprise-Grade Anomaly Detection Platform - FIXED Version</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    lottie_hero = load_lottie_url(LOTTIE_URLS['hero'])
    if lottie_hero:
        display_lottie(lottie_hero, height=180, key="hero_animation")


# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Quick Guide
    
    **1.** Upload data (CSV/Excel/Parquet)  
    **2.** Select features for analysis  
    **3.** Choose detection algorithm  
    **4.** Analyze & get insights  
    **5.** Export results & reports
    """)
    
    st.markdown("---")
    
    with st.expander("🧠 Algorithm Guide"):
        st.markdown("""
        **Statistical** (Fast, Scalable)
        
        • **Z-Score** - Detects outliers >3σ  
        • **IQR** - Interquartile range method
        
        **Machine Learning** (Advanced)
        
        • **Isolation Forest** - Global anomalies  
        • **LOF** - Local neighborhood outliers  
        • **One-Class SVM** - Non-linear boundaries
        
        **🔧 FIXED:** Expected rate is now a guide, not a quota!
        """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["📂 Data Upload", "🔍 Analysis", "📊 Results"])

# =============================================================================
# TAB 1: DATA UPLOAD
# =============================================================================

with tab1:
    st.markdown("### 📤 Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'txt', 'parquet'],
            help="Supported: CSV, Excel, Text, Parquet"
        )
    
    with col2:
        use_sample = st.checkbox("Use Sample Data", help="Load demo dataset")
    
    # Lottie animation for upload state
    if not st.session_state.data_loaded and not use_sample and not uploaded_file:
        st.markdown("<div class='lottie-container'>", unsafe_allow_html=True)
        lottie_upload = load_lottie_url(LOTTIE_URLS['upload'])
        display_lottie(lottie_upload, height=250, key="upload_animation")
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("👆 Upload a file or use sample data to get started")
    
    if use_sample:
        with st.spinner("🎲 Generating sample data..."):
            np.random.seed(42)
            n_samples = 10000
            
            feature1 = np.random.normal(100, 15, n_samples)
            feature2 = np.random.normal(200, 30, n_samples)
            feature3 = np.random.normal(50, 10, n_samples)
            
            anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            feature1[anomaly_indices] *= np.random.uniform(2, 4, len(anomaly_indices))
            feature2[anomaly_indices] *= np.random.uniform(0.2, 0.5, len(anomaly_indices))
            
            df_sample = pd.DataFrame({
                'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
                'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'Support'], n_samples),
                'Feature_1': feature1,
                'Feature_2': feature2,
                'Feature_3': feature3,
                'Date': pd.date_range('2024-01-01', periods=n_samples, freq='H')
            })
            
            parquet_path = os.path.join(st.session_state.temp_dir, 'sample_data.parquet')
            df_sample.to_parquet(parquet_path, compression='snappy')
            
            st.session_state.parquet_path = parquet_path
            st.session_state.data_loaded = True
            st.session_state.row_count = len(df_sample)
            
            con = initialize_duckdb()
            st.session_state.column_info = get_column_stats(con, parquet_path)
            
            st.success("✅ Sample data loaded!")
    
    elif uploaded_file:
        file_type = detect_file_type(uploaded_file)
        
        if file_type:
            st.info(f"📄 Detected: **{file_type.upper()}**")
            
            if st.button("🚀 Process File", type="primary", use_container_width=True):
                parquet_path, row_count, col_count = convert_to_parquet(uploaded_file, file_type)
                
                if parquet_path:
                    st.session_state.parquet_path = parquet_path
                    st.session_state.data_loaded = True
                    st.session_state.row_count = row_count
                    
                    con = initialize_duckdb()
                    st.session_state.column_info = get_column_stats(con, parquet_path)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ File Processed Successfully!</h4>
                        <p><b>Rows:</b> {row_count:,} | <b>Columns:</b> {col_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("❌ Unsupported file format")
    
    # Data overview
    if st.session_state.data_loaded and st.session_state.parquet_path:
        st.markdown("---")
        st.markdown("### 📊 Data Overview")
        
        con = initialize_duckdb()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'INT', 'NUMBER']
        numeric_count = sum(1 for v in st.session_state.column_info.values() 
                          if any(ntype in v['type'].upper() for ntype in numeric_types))
        categorical_count = sum(1 for v in st.session_state.column_info.values() 
                              if 'VARCHAR' in v['type'].upper() or 'STRING' in v['type'].upper())
        
        with col1:
            st.markdown(create_metric_card("Total Rows", f"{st.session_state.row_count:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Columns", len(st.session_state.column_info)), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Numeric", numeric_count), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Categorical", categorical_count), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Column details
        with st.expander("🔍 View Column Details"):
            col_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': info['type'],
                    'Distinct': info['distinct'],
                    'Nulls': info['nulls'],
                    'Null %': f"{info['null_pct']:.2f}%"
                }
                for col, info in st.session_state.column_info.items()
            ])
            st.dataframe(col_df, use_container_width=True, height=400)
        
        # Data preview
        with st.expander("👀 Data Preview"):
            sample_df = get_sample_data(con, st.session_state.parquet_path, 1000)
            st.dataframe(sample_df, use_container_width=True, height=400)

# =============================================================================
# TAB 2: ANALYSIS
# =============================================================================

with tab2:
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ No Data Loaded</h4>
            <p>Upload data in the <b>Data Upload</b> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Empty state animation
        lottie_empty = load_lottie_url(LOTTIE_URLS['empty'])
        display_lottie(lottie_empty, height=200, key="empty_animation")
    else:
        st.markdown("### ⚙️ Configure Analysis")
        
        con = initialize_duckdb()
        
        # Column selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📋 Categorical")
            categorical_cols = [col for col, info in st.session_state.column_info.items() 
                              if 'VARCHAR' in info['type'].upper() or 'STRING' in info['type'].upper()]
            
            selected_categorical_cols = st.multiselect(
                "Choose for grouping",
                categorical_cols,
                help="Category-level analysis"
            )
        
        with col2:
            st.markdown("#### 🔢 Numeric")
            numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'INT', 'NUMBER']
            numeric_cols_available = [col for col, info in st.session_state.column_info.items() 
                                     if any(ntype in info['type'].upper() for ntype in numeric_types)]
            
            numeric_cols = st.multiselect(
                "Choose for detection",
                numeric_cols_available,
                default=numeric_cols_available[:3] if len(numeric_cols_available) >= 3 else numeric_cols_available,
                help="Features for anomaly detection"
            )
        
        with col3:
            st.markdown("#### 📅 Date (Optional)")
            date_cols = [col for col, info in st.session_state.column_info.items() 
                        if 'DATE' in info['type'].upper() or 'TIMESTAMP' in info['type'].upper()]
            
            date_col = st.selectbox(
                "Choose for time analysis",
                ['None'] + date_cols,
                help="Time-based analysis"
            )
            date_col = None if date_col == 'None' else date_col
        
        st.markdown("---")
        
        # Method selection
        st.markdown("#### 🎯 Detection Method")
        
        method_category = st.radio(
            "Category",
            ['Statistical (Fast)', 'Machine Learning (Advanced)'],
            horizontal=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        if method_category == 'Statistical (Fast)':
            with col1:
                method = st.selectbox("Algorithm", ['Z-Score', 'IQR Method'])
            with col2:
                if method == 'Z-Score':
                    threshold = st.slider("Threshold", 2.0, 5.0, 3.0, 0.5)
                else:
                    threshold = None
            
            detection_method = 'zscore' if method == 'Z-Score' else 'iqr'
            use_ml = False
        else:
            with col1:
                method = st.selectbox("Algorithm", ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM'])
            with col2:
                contamination = st.slider("Expected Anomaly Rate (Sensitivity Guide)", 0.01, 0.5, 0.1, 0.01, 
                                        help="This is a GUIDE, not a hard limit. Algorithm can find more or fewer anomalies.")
                
                # Show explanation
                with st.expander("ℹ️ What does this mean? (IMPROVED)", expanded=False):
                    st.markdown(explain_contamination(contamination, st.session_state.row_count))
            with col3:
                max_sample = min(50000, st.session_state.row_count) if method != 'Isolation Forest' else st.session_state.row_count
                use_sampling = st.checkbox("Use Sampling", value=st.session_state.row_count > 50000 and method != 'Isolation Forest')
                
                if use_sampling:
                    sample_size = st.number_input("Sample Size", 1000, max_sample, min(50000, max_sample))
                else:
                    sample_size = max_sample
                    st.info(f"Full dataset: {sample_size:,} rows")
            
            method_map = {'Isolation Forest': 'isolation_forest', 'Local Outlier Factor': 'lof', 'One-Class SVM': 'one_class_svm'}
            detection_method = method_map[method]
            use_ml = True
        
        st.markdown("---")
        
        # Run analysis
        if not numeric_cols:
            st.warning("⚠️ Select at least one numeric column")
        else:
            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                
                # Processing animation
                lottie_processing = load_lottie_url(LOTTIE_URLS['processing'])
                processing_placeholder = st.empty()
                with processing_placeholder.container():
                    st.markdown("<div class='lottie-container'>", unsafe_allow_html=True)
                    display_lottie(lottie_processing, height=150, key="processing_animation")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with st.spinner(f"🔍 Running {method}..."):
                    start_time = time.time()
                    
                    try:
                        if use_ml:
                            if use_sampling and st.session_state.row_count > sample_size:
                                query = f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}') USING SAMPLE {sample_size} ROWS"
                                df = con.execute(query).df()
                            else:
                                df = get_sample_data(con, st.session_state.parquet_path, st.session_state.row_count)
                            
                            results_df, detection_metadata = detect_anomalies_ml(df, numeric_cols, detection_method, contamination)
                            # Store metadata in session state
                            st.session_state.detection_metadata = detection_metadata
                        else:
                            results_df, detection_metadata = detect_anomalies_statistical(con, st.session_state.parquet_path, numeric_cols, detection_method, threshold if detection_method == 'zscore' else 3)
                            st.session_state.detection_metadata = detection_metadata
                        
                        if results_df is not None:
                            st.session_state.results_df = results_df
                            st.session_state.analysis_complete = True
                            st.session_state.selected_numeric_cols = numeric_cols
                            st.session_state.selected_categorical_cols = selected_categorical_cols
                            st.session_state.selected_date_col = date_col
                            st.session_state.selected_method = method
                            if use_ml:
                                st.session_state.contamination_rate = contamination
                            else:
                                st.session_state.contamination_rate = None
                            
                            elapsed_time = time.time() - start_time
                            
                            # Clear processing animation
                            processing_placeholder.empty()
                            
                            # Success animation
                            lottie_success = load_lottie_url(LOTTIE_URLS['success'])
                            display_lottie(lottie_success, height=150, key="success_animation")
                            
                            st.success(f"✅ Complete in {elapsed_time:.2f}s!")
                            
                            anomaly_count = results_df['anomaly'].sum()
                            anomaly_rate = (anomaly_count / len(results_df) * 100)
                            expected_count = int(len(results_df) * contamination) if use_ml else None
                            
                            result_msg = f"""
                            <div class="success-box">
                                <h4>📊 Quick Results</h4>
                                <p><b>Detected:</b> {anomaly_count:,} / {len(results_df):,} ({anomaly_rate:.2f}%)</p>
                            """
                            
                            if expected_count:
                                diff = anomaly_count - expected_count
                                result_msg += f"<p><b>Expected:</b> ~{expected_count:,} ({contamination*100:.1f}%)</p>"
                                if abs(diff) > expected_count * 0.1:  # More than 10% difference
                                    result_msg += f"<p><b>Difference:</b> {diff:+,} anomalies ({(diff/expected_count)*100:+.1f}%)</p>"
                            
                            result_msg += "<p>👉 View details in <b>Results</b> tab</p></div>"
                            
                            st.markdown(result_msg, unsafe_allow_html=True)
                    
                    except Exception as e:
                        processing_placeholder.empty()
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.error(traceback.format_exc())

# =============================================================================
# TAB 3: RESULTS
# =============================================================================

with tab3:
    if not st.session_state.analysis_complete:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ No Results</h4>
            <p>Run analysis in <b>Analysis</b> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Empty state animation
        lottie_empty = load_lottie_url(LOTTIE_URLS['empty'])
        display_lottie(lottie_empty, height=200, key="empty_results_animation")
    else:
        df_results = st.session_state.results_df
        numeric_cols = st.session_state.selected_numeric_cols
        categorical_cols = st.session_state.selected_categorical_cols
        date_col = st.session_state.selected_date_col
        method = st.session_state.selected_method
        
        st.markdown("### 📊 Analysis Results")
        
        insights = generate_insights(df_results, categorical_cols, numeric_cols, method)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Records", f"{insights['total_records']:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Anomalies", f"{insights['anomalies']:,}"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Anomaly Rate", f"{insights['anomaly_rate']:.2f}%"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Method", method), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Executive visual summary with professional styling
        expected_rate = st.session_state.get('contamination_rate', None)
        generate_visual_summary(insights, categorical_cols, numeric_cols, expected_rate)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        figs = create_visualizations(df_results, numeric_cols, categorical_cols, date_col)
        
        # Pie chart with narrative
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pie' in figs:
                with st.expander("ℹ️ Understanding this chart", expanded=False):
                    st.markdown(get_chart_narrative('pie'))
                st.plotly_chart(figs['pie'], use_container_width=True)
        
        with col2:
            if 'histogram' in figs:
                with st.expander("ℹ️ Understanding this chart", expanded=False):
                    st.markdown(get_chart_narrative('histogram'))
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        if 'timeseries' in figs:
            with st.expander("ℹ️ Understanding this chart", expanded=False):
                st.markdown(get_chart_narrative('timeseries'))
            st.plotly_chart(figs['timeseries'], use_container_width=True)
        
        if 'scatter' in figs:
            with st.expander("ℹ️ Understanding this chart", expanded=False):
                st.markdown(get_chart_narrative('scatter'))
            st.plotly_chart(figs['scatter'], use_container_width=True)
        
        for key in figs.keys():
            if key.startswith('category_bar_'):
                with st.expander("ℹ️ Understanding this chart", expanded=False):
                    st.markdown(get_chart_narrative('category'))
                st.plotly_chart(figs[key], use_container_width=True)
        
        if 'boxplot' in figs:
            with st.expander("📦 Box Plot Analysis"):
                st.markdown(get_chart_narrative('boxplot'))
                st.plotly_chart(figs['boxplot'], use_container_width=True)
        
        st.markdown("---")
        
        # Insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Summary")
            st.markdown(f"""
            • Analyzed **{insights['total_records']:,}** records
            • Detected **{insights['anomalies']:,}** anomalies (**{insights['anomaly_rate']:.2f}%**)
            • Features: **{', '.join(numeric_cols)}**
            """)
            
            if categorical_cols:
                st.markdown(f"• Categories: **{', '.join(categorical_cols)}**")
            if date_col:
                st.markdown(f"• Time: **{date_col}**")
        
        with col2:
            if 'feature_importance' in insights and insights['feature_importance']:
                st.markdown("#### 📊 Feature Impact")
                feature_diff = sorted(insights['feature_importance'].items(), key=lambda x: x[1]['difference'], reverse=True)
                
                for feat, stats in feature_diff[:5]:
                    st.markdown(f"""
                    • **{feat}**: Anomaly avg = {stats['anomaly_mean']:.2f}, Normal avg = {stats['normal_mean']:.2f} (Δ = {stats['difference']:.2f})
                    """)
        
        # Categorical insights
        if 'categorical_insights' in insights and insights['categorical_insights']:
            st.markdown("---")
            st.markdown("### 📋 Categorical Analysis")
            
            for cat_col, cat_data in insights['categorical_insights'].items():
                with st.expander(f"🏆 Top Categories - {cat_col}", expanded=True):
                    if cat_data['top_categories']:
                        cat_df = pd.DataFrame(cat_data['top_categories'])
                        display_cols = [cat_col, 'anomaly_count', 'total_count', 'anomaly_rate']
                        display_cols = [col for col in display_cols if col in cat_df.columns]
                        
                        if display_cols:
                            cat_df_display = cat_df[display_cols].head(10)
                            cat_df_display['anomaly_rate'] = cat_df_display['anomaly_rate'].round(2)
                            st.dataframe(cat_df_display, use_container_width=True)
                            
                            if cat_data['highest_rate']:
                                highest = cat_data['highest_rate']
                                st.info(f"🥇 Highest: **{highest[cat_col]}** at **{highest['anomaly_rate']:.2f}%**")
        
        # Full results
        with st.expander("📋 Full Results Table"):
            st.dataframe(df_results, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Actionable recommendations
        if insights['anomalies'] > 0:
            st.markdown("### 🎯 RECOMMENDED NEXT STEPS")
            
            high_score_count = len(df_results[df_results['anomaly_score'] > df_results['anomaly_score'].quantile(0.95)]) if 'anomaly_score' in df_results.columns else 0
            
            st.markdown(f"""
            Based on your analysis, here's your action plan:
            
            #### 1️⃣ IMMEDIATE ACTION
            • Export the **{insights['anomalies']:,} flagged records** using CSV button below
            • Prioritize records with highest scores (top {high_score_count} records with score >95th percentile)
            • Assign to review team within 24-48 hours
            
            #### 2️⃣ DEEP DIVE
            Focus investigation on:
            """)
            
            # Dynamic recommendations based on results
            if 'feature_importance' in insights and insights['feature_importance']:
                top_feature = max(insights['feature_importance'].items(), key=lambda x: x[1]['difference'])
                st.markdown(f"• **{top_feature[0]}**: Primary differentiator (avg {top_feature[1]['anomaly_mean']:.2f} vs {top_feature[1]['normal_mean']:.2f})")
            
            if 'categorical_insights' in insights and insights['categorical_insights']:
                for cat_col, cat_data in list(insights['categorical_insights'].items())[:1]:
                    if cat_data['highest_rate']:
                        highest = cat_data['highest_rate']
                        st.markdown(f"• **{cat_col} = '{highest[cat_col]}'**: Highest anomaly concentration ({highest['anomaly_rate']:.1f}%)")
            
            if date_col:
                st.markdown(f"• **Time patterns**: Review timeline chart for incident clusters")
            
            st.markdown(f"""
            #### 3️⃣ PREVENTION
            Consider automated rules:
            • Set up alerts when anomaly rate exceeds {insights['anomaly_rate']*1.5:.1f}% (1.5x current)
            • Create review queue for high-risk combinations identified above
            • Schedule weekly anomaly trend reports
            
            #### 4️⃣ VALIDATE
            • Manually review 20 random flagged records
            • If >80% are genuinely problematic → model is well-tuned ✓
            • If <50% are problematic → adjust sensitivity and re-run
            • Document common patterns for future reference
            """)
        else:
            st.markdown("### ✅ CLEAN DATASET DETECTED")
            st.markdown(f"""
            Great news! No significant anomalies found (0 out of {insights['total_records']:,} records).
            
            **This could mean:**
            • Your data is genuinely clean ✓
            • Expected rate was too strict - try lowering threshold
            • Selected features don't capture unusual patterns - try different columns
            
            💡 **Next Step:** Try Z-Score method with threshold 2.5 for more sensitive detection.
            """)
        
        st.markdown("---")
        
        # Export
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_buffer = BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            parquet_buffer = BytesIO()
            df_results.to_parquet(parquet_buffer, compression='snappy')
            st.download_button(
                "📥 Download Parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        with col3:
            pdf_buffer = generate_pdf_report(insights, categorical_cols, numeric_cols)
            if pdf_buffer:
                st.download_button(
                    "📥 Download PDF",
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
        🎯 <b>Anomaly Detection Pro - FIXED</b> | Developed by CE Innovations Team 2025<br>
        Powered by DuckDB • PyArrow • Scikit-learn • Plotly | Score-Based Detection ✓
    </p>
</div>
""", unsafe_allow_html=True)
