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
        padding: 1.5rem;
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
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.875rem;
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
    'upload': 'https://lottie.host/b7c3ec8e-3f5f-4c5c-8a0e-d5b8e3c7a0f1/rVK3wZ5Cxe.json',
    'processing': 'https://lottie.host/4f3b5c8e-2f1f-4c5c-9a0e-c5b8e3c7a0f2/xYz2wV5Dxf.json',
    'success': 'https://lottie.host/8a7d1e9f-3g2h-5d6e-0b1f-f6c9d4e8b1g3/pQr3xW6Eyg.json',
    'empty': 'https://lottie.host/6e5f2g0h-4h3i-6e7f-1c2g-g7d0e5f9c2h4/mNo4yX7Fzh.json',
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
    """Convert various file formats to Parquet"""
    try:
        progress = st.progress(0, text="🔄 Reading file...")
        
        # Read based on file type using match/case
        match file_type:
            case 'csv':
                df = pd.read_csv(file, low_memory=False, encoding='utf-8', encoding_errors='replace')
            case 'xlsx' | 'xls':
                engine = 'openpyxl' if file_type == 'xlsx' else 'xlrd'
                df = pd.read_excel(file, engine=engine)
            case 'txt':
                file.seek(0)
                sample = file.read(1024).decode('utf-8', errors='replace')
                file.seek(0)
                delimiter = ',' if ',' in sample else '\t' if '\t' in sample else None
                df = pd.read_csv(file, delimiter=delimiter, low_memory=False, encoding='utf-8', encoding_errors='replace')
            case 'parquet':
                df = pd.read_parquet(file)
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        progress.progress(30, text="🔧 Optimizing data types...")
        
        # Optimize data types
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
        
        # Auto-detect datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object' and df[col].notna().sum() > 0:
                try:
                    sample = df[col].dropna().iloc[0]
                    if isinstance(sample, str) and any(char.isdigit() for char in sample):
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        progress.progress(60, text="💾 Converting to Parquet format...")
        
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        progress.progress(100, text="✅ Conversion complete!")
        time.sleep(0.5)
        progress.empty()
        
        return parquet_path, df.shape[0], df.shape[1]
        
    except Exception as e:
        st.error(f"❌ Error converting file: {e}")
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

def detect_anomalies_statistical(con, parquet_path: str, numeric_cols: list, method: str = 'zscore', threshold: float = 3) -> pd.DataFrame | None:
    """SQL-based statistical anomaly detection"""
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
        
        return df
    except Exception as e:
        st.error(f"Error in statistical detection: {e}")
        return None

def detect_anomalies_ml(df: pd.DataFrame, numeric_cols: list, method: str, contamination: float = 0.1) -> pd.DataFrame | None:
    """ML-based anomaly detection with factory pattern"""
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
        
        # Update contamination if provided
        if 'contamination' in params:
            params['contamination'] = contamination
        if 'nu' in params:
            params['nu'] = contamination
        
        # Adjust LOF n_neighbors for small datasets
        if method == 'lof':
            params['n_neighbors'] = min(params['n_neighbors'], len(df_clean) - 1)
        
        # Fit and predict
        model = model_class(**params)
        predictions = model.fit_predict(X_scaled)
        
        # Get anomaly scores
        if hasattr(model, 'score_samples'):
            scores = model.score_samples(X_scaled)
        elif hasattr(model, 'negative_outlier_factor_'):
            scores = model.negative_outlier_factor_
        else:
            scores = predictions
        
        df_clean['anomaly'] = (predictions == -1).astype(int)
        df_clean['anomaly_score'] = np.abs(scores)
        
        return df_clean
    except Exception as e:
        st.error(f"Error in ML detection: {e}")
        return None

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

def create_metric_card(label: str, value: str | int) -> str:
    """Generate HTML for metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

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
        elements.append(Paragraph("🎯 Anomaly Hunter Pro - Analysis Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(Paragraph(f"<b>Total Records:</b> {insights['total_records']:,}", styles['Normal']))
        elements.append(Paragraph(f"<b>Anomalies Detected:</b> {insights['anomalies']:,} ({insights['anomaly_rate']:.2f}%)", styles['Normal']))
        elements.append(Paragraph(f"<b>Detection Method:</b> {insights['method']}", styles['Normal']))
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Anomaly Hunter Pro</h1>
    <p>Enterprise-Grade Anomaly Detection Platform</p>
</div>
""", unsafe_allow_html=True)

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
            st.balloons()
    
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
                    st.balloons()
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
                contamination = st.slider("Expected Anomaly Rate", 0.01, 0.5, 0.1, 0.01, help="0.1 = 10%")
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
                            
                            results_df = detect_anomalies_ml(df, numeric_cols, detection_method, contamination)
                        else:
                            results_df = detect_anomalies_statistical(con, st.session_state.parquet_path, numeric_cols, detection_method, threshold if detection_method == 'zscore' else 3)
                        
                        if results_df is not None:
                            st.session_state.results_df = results_df
                            st.session_state.analysis_complete = True
                            st.session_state.selected_numeric_cols = numeric_cols
                            st.session_state.selected_categorical_cols = selected_categorical_cols
                            st.session_state.selected_date_col = date_col
                            st.session_state.selected_method = method
                            
                            elapsed_time = time.time() - start_time
                            
                            # Clear processing animation
                            processing_placeholder.empty()
                            
                            # Success animation
                            lottie_success = load_lottie_url(LOTTIE_URLS['success'])
                            display_lottie(lottie_success, height=150, key="success_animation")
                            
                            st.success(f"✅ Complete in {elapsed_time:.2f}s!")
                            st.balloons()
                            
                            anomaly_count = results_df['anomaly'].sum()
                            anomaly_rate = (anomaly_count / len(results_df) * 100)
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📊 Quick Results</h4>
                                <p><b>Detected:</b> {anomaly_count:,} / {len(results_df):,} ({anomaly_rate:.2f}%)</p>
                                <p>👉 View details in <b>Results</b> tab</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        processing_placeholder.empty()
                        st.error(f"❌ Error: {e}")

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
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        figs = create_visualizations(df_results, numeric_cols, categorical_cols, date_col)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pie' in figs:
                st.plotly_chart(figs['pie'], use_container_width=True)
        
        with col2:
            if 'histogram' in figs:
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        if 'timeseries' in figs:
            st.plotly_chart(figs['timeseries'], use_container_width=True)
        
        if 'scatter' in figs:
            st.plotly_chart(figs['scatter'], use_container_width=True)
        
        for key in figs.keys():
            if key.startswith('category_bar_'):
                st.plotly_chart(figs[key], use_container_width=True)
        
        if 'boxplot' in figs:
            with st.expander("📦 Box Plot Analysis"):
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
        🎯 <b>Anomaly Hunter Pro</b> | Developed by CE Innovations Lab 2025<br>
        Powered by DuckDB • PyArrow • Scikit-learn • Plotly
    </p>
</div>
""", unsafe_allow_html=True)
