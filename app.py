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

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy import stats

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4, letter
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

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --accent-color: #FFE66D;
        --background-dark: #1A1A2E;
        --card-bg: #16213E;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        color: white;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #c21858;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0575e6;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_file_type(file):
    """Detect file type from uploaded file"""
    filename = file.name.lower()
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.xls'):
        return 'xls'
    elif filename.endswith('.txt'):
        return 'txt'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        return None

def convert_to_parquet(file, file_type):
    """Convert various file formats to Parquet with progress tracking"""
    try:
        progress = st.progress(0, text="🔄 Reading file...")
        
        # Read based on file type
        if file_type == 'csv':
            df = pd.read_csv(file, low_memory=False, encoding='utf-8', encoding_errors='replace')
        elif file_type == 'xlsx' or file_type == 'xls':
            df = pd.read_excel(file, engine='openpyxl' if file_type == 'xlsx' else 'xlrd')
        elif file_type == 'txt':
            # Try to detect delimiter
            file.seek(0)
            sample = file.read(1024).decode('utf-8', errors='replace')
            file.seek(0)
            delimiter = ',' if ',' in sample else '\t' if '\t' in sample else None
            df = pd.read_csv(file, delimiter=delimiter, low_memory=False, encoding='utf-8', encoding_errors='replace')
        elif file_type == 'parquet':
            df = pd.read_parquet(file)
        else:
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
        
        # Save as Parquet
        parquet_path = os.path.join(st.session_state.temp_dir, 'data.parquet')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        progress.progress(100, text="✅ Conversion complete!")
        time.sleep(0.5)
        progress.empty()
        
        return parquet_path, df.shape[0], df.shape[1]
        
    except Exception as e:
        st.error(f"❌ Error converting file: {str(e)}")
        return None, 0, 0

def initialize_duckdb():
    """Initialize DuckDB connection"""
    if st.session_state.duckdb_con is None:
        st.session_state.duckdb_con = duckdb.connect(':memory:')
    return st.session_state.duckdb_con

def get_column_stats(con, parquet_path):
    """Get comprehensive column statistics using DuckDB"""
    try:
        # Get column info
        query = f"""
        SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0
        """
        schema = con.execute(query).description
        
        column_info = {}
        for col_name, col_type, *_ in schema:
            # Get distinct count and null count
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
        
        # DEBUG: Print column types to console
        print("DEBUG - Column Types Detected:")
        for col, info in column_info.items():
            print(f"  {col}: {info['type']}")
        
        return column_info
        
    except Exception as e:
        st.error(f"Error getting column stats: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {}

def get_sample_data(con, parquet_path, limit=1000):
    """Get sample data for preview"""
    query = f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT {limit}"
    return con.execute(query).df()

def detect_anomalies_statistical(con, parquet_path, numeric_cols, method='zscore', threshold=3):
    """SQL-based statistical anomaly detection for large datasets"""
    try:
        if method == 'zscore':
            # Calculate z-scores for each numeric column
            select_parts = []
            for col in numeric_cols:
                select_parts.append(f"""
                    ABS(("{col}" - AVG("{col}") OVER ()) / NULLIF(STDDEV("{col}") OVER (), 0)) as zscore_{col}
                """)
            
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
            # IQR method
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
                percentiles[col] = {
                    'lower': q1 - 1.5 * iqr,
                    'upper': q3 + 1.5 * iqr
                }
            
            # Build query
            conditions = []
            for col in numeric_cols:
                conditions.append(f'("{col}" < {percentiles[col]["lower"]} OR "{col}" > {percentiles[col]["upper"]})')
            
            query = f"""
            SELECT *,
                CASE WHEN {' OR '.join(conditions)} THEN 1 ELSE 0 END as anomaly
            FROM parquet_scan('{parquet_path}')
            """
            
            df = con.execute(query).df()
            
            # Calculate anomaly score as max deviation from IQR bounds
            scores = []
            for col in numeric_cols:
                lower, upper = percentiles[col]['lower'], percentiles[col]['upper']
                score = np.maximum(
                    np.maximum(lower - df[col], 0),
                    np.maximum(df[col] - upper, 0)
                )
                scores.append(score)
            
            df['anomaly_score'] = np.max(scores, axis=0)
        
        return df
        
    except Exception as e:
        st.error(f"Error in statistical detection: {str(e)}")
        return None

def detect_anomalies_ml(df, numeric_cols, method='isolation_forest', contamination=0.1):
    """ML-based anomaly detection with preprocessing"""
    try:
        # Handle missing values
        df_clean = df.copy()
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Prepare feature matrix
        X = df_clean[numeric_cols].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply selected method
        if method == 'isolation_forest':
            model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            predictions = model.fit_predict(X_scaled)
            scores = model.score_samples(X_scaled)
            
        elif method == 'lof':
            n_neighbors = min(20, len(df_clean) - 1)
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
            predictions = model.fit_predict(X_scaled)
            scores = model.negative_outlier_factor_
            
        elif method == 'one_class_svm':
            model = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
            predictions = model.fit_predict(X_scaled)
            scores = model.score_samples(X_scaled)
            
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            predictions = model.fit_predict(X_scaled)
            # For DBSCAN, -1 indicates anomaly
            predictions = np.where(predictions == -1, -1, 1)
            scores = np.zeros(len(predictions))  # DBSCAN doesn't provide scores
        
        # Convert predictions to binary (1 = anomaly, 0 = normal)
        df_clean['anomaly'] = (predictions == -1).astype(int)
        df_clean['anomaly_score'] = np.abs(scores)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error in ML detection: {str(e)}")
        return None

def generate_insights(df, categorical_cols, numeric_cols, method):
    """Generate narrative insights from results"""
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
        
        # Category-level analysis for each categorical column
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
        
        # Feature importance (which features contribute most to anomalies)
        if len(numeric_cols) > 0:
            feature_stats = {}
            anomaly_df = df[df['anomaly'] == 1]
            normal_df = df[df['anomaly'] == 0]
            
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
        st.error(f"Error generating insights: {str(e)}")
        return {}

def create_visualizations(df, numeric_cols, categorical_cols=None, date_col=None):
    """Create comprehensive visualizations"""
    figs = {}
    
    try:
        # 1. Anomaly Distribution Pie Chart
        anomaly_counts = df['anomaly'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)],
            hole=0.4,
            marker=dict(colors=['#4ECDC4', '#FF6B6B']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        fig_pie.update_layout(
            title="Anomaly Distribution",
            template="plotly_dark",
            height=400
        )
        figs['pie'] = fig_pie
        
        # 2. Anomaly Score Distribution
        if 'anomaly_score' in df.columns:
            fig_hist = px.histogram(
                df, x='anomaly_score', color='anomaly',
                nbins=50,
                title="Anomaly Score Distribution",
                labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'},
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
            )
            fig_hist.update_layout(template="plotly_dark", height=400)
            figs['histogram'] = fig_hist
        
        # 3. Time Series Chart (if date column exists)
        if date_col and date_col in df.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Group by date
                df_sorted = df.sort_values(date_col)
                daily_stats = df_sorted.groupby(pd.Grouper(key=date_col, freq='D')).agg({
                    'anomaly': ['sum', 'count']
                }).reset_index()
                daily_stats.columns = [date_col, 'anomalies', 'total']
                daily_stats['anomaly_rate'] = (daily_stats['anomalies'] / daily_stats['total'] * 100).fillna(0)
                
                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_time.add_trace(
                    go.Scatter(x=daily_stats[date_col], y=daily_stats['anomalies'],
                              name="Anomaly Count", line=dict(color='#FF6B6B', width=2),
                              fill='tozeroy'),
                    secondary_y=False,
                )
                
                fig_time.add_trace(
                    go.Scatter(x=daily_stats[date_col], y=daily_stats['anomaly_rate'],
                              name="Anomaly Rate (%)", line=dict(color='#FFE66D', width=2, dash='dash')),
                    secondary_y=True,
                )
                
                fig_time.update_layout(
                    title="Anomalies Over Time",
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified'
                )
                fig_time.update_xaxes(title_text="Date")
                fig_time.update_yaxes(title_text="Anomaly Count", secondary_y=False)
                fig_time.update_yaxes(title_text="Anomaly Rate (%)", secondary_y=True)
                
                figs['timeseries'] = fig_time
            except Exception as e:
                st.warning(f"Could not create time series chart: {str(e)}")
        
        # 4. Scatter Plot (if 2+ numeric columns)
        if len(numeric_cols) >= 2:
            fig_scatter = px.scatter(
                df.sample(min(10000, len(df))),  # Sample for performance
                x=numeric_cols[0],
                y=numeric_cols[1],
                color='anomaly',
                hover_data=numeric_cols[:5] if len(numeric_cols) > 2 else numeric_cols,
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                opacity=0.6
            )
            fig_scatter.update_layout(template="plotly_dark", height=500)
            figs['scatter'] = fig_scatter
        
        # 5. Category Analysis (for each categorical column)
        if categorical_cols:
            for i, categorical_col in enumerate(categorical_cols[:3]):  # Limit to first 3
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
                    fig_bar.update_layout(template="plotly_dark", height=400)
                    figs[f'category_bar_{i}'] = fig_bar
        
        # 6. Parallel Coordinates (for multivariate view)
        if len(numeric_cols) >= 3:
            sample_df = df.sample(min(1000, len(df)))
            
            dimensions = []
            for col in numeric_cols[:6]:  # Limit to 6 columns for readability
                dimensions.append(dict(
                    label=col,
                    values=sample_df[col]
                ))
            
            fig_parallel = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=sample_df['anomaly'],
                        colorscale=[[0, '#4ECDC4'], [1, '#FF6B6B']],
                        showscale=True,
                        cmin=0,
                        cmax=1
                    ),
                    dimensions=dimensions
                )
            )
            fig_parallel.update_layout(
                title="Parallel Coordinates (Multivariate Analysis)",
                template="plotly_dark",
                height=500
            )
            figs['parallel'] = fig_parallel
        
        # 7. Box plots for each numeric column
        if len(numeric_cols) > 0:
            fig_box = make_subplots(
                rows=1, cols=min(3, len(numeric_cols)),
                subplot_titles=numeric_cols[:3]
            )
            
            for i, col in enumerate(numeric_cols[:3], 1):
                for anomaly_val, color in [(0, '#4ECDC4'), (1, '#FF6B6B')]:
                    data = df[df['anomaly'] == anomaly_val][col]
                    fig_box.add_trace(
                        go.Box(y=data, name='Normal' if anomaly_val == 0 else 'Anomaly',
                               marker_color=color, showlegend=(i == 1)),
                        row=1, col=i
                    )
            
            fig_box.update_layout(
                title="Feature Distribution by Anomaly Status",
                template="plotly_dark",
                height=400,
                showlegend=True
            )
            figs['boxplot'] = fig_box
        
        return figs
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        return figs

def generate_pdf_report(insights, categorical_cols, numeric_cols):
    """Generate comprehensive PDF report"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
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
        
        # Top Categories Analysis for each categorical column
        if 'categorical_insights' in insights and insights['categorical_insights']:
            for cat_col, cat_data in insights['categorical_insights'].items():
                if cat_data['top_categories']:
                    elements.append(Paragraph(f"Top Categories by Anomaly Rate - {cat_col}", heading_style))
                    
                    table_data = [[cat_col, 'Anomaly Count', 'Total Count', 'Anomaly Rate (%)']]
                    for cat in cat_data['top_categories'][:10]:
                        table_data.append([
                            str(cat.get(cat_col, 'N/A'))[:30],  # Truncate long names
                            str(cat.get('anomaly_count', 0)),
                            str(cat.get('total_count', 0)),
                            f"{cat.get('anomaly_rate', 0):.2f}"
                        ])
                    
                    t = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ]))
                    elements.append(t)
                    elements.append(Spacer(1, 20))
        
        # Feature Importance
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
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            elements.append(t2)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Anomaly Hunter Pro</h1>
    <p>Enterprise-Grade Anomaly Detection Platform | Powered by DuckDB & Advanced ML</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Quick Guide
    
    1. **Upload Data** - CSV, Excel, TXT, or Parquet
    2. **Auto-converts** to Parquet for optimal performance
    3. **Select Features** - Choose columns for analysis
    4. **Pick Algorithm** - Statistical or ML-based
    5. **Analyze** - Get insights & visualizations
    6. **Export** - Download results & reports
    """)
    
    st.markdown("---")
    
    # Algorithm info
    with st.expander("🧠 Algorithm Guide"):
        st.markdown("""
        **Statistical Methods** (Fast, Scalable)
        - **Z-Score**: Detects outliers >3 std dev
        - **IQR**: Interquartile range method
        
        **ML Methods** (Advanced, Accurate)
        - **Isolation Forest**: Best for global anomalies
        - **LOF**: Local neighborhood outliers
        - **One-Class SVM**: Non-linear boundaries
        - **DBSCAN**: Density-based clustering
        """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
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
            help="Supported formats: CSV, Excel, Text, Parquet"
        )
    
    with col2:
        use_sample = st.checkbox("Use Sample Data", help="Load built-in sample dataset")
    
    if use_sample:
        with st.spinner("🎲 Generating sample data..."):
            # Create sample data
            np.random.seed(42)
            n_samples = 10000
            
            # Generate features
            feature1 = np.random.normal(100, 15, n_samples)
            feature2 = np.random.normal(200, 30, n_samples)
            feature3 = np.random.normal(50, 10, n_samples)
            
            # Inject anomalies
            anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            feature1[anomaly_indices] *= np.random.uniform(2, 4, len(anomaly_indices))
            feature2[anomaly_indices] *= np.random.uniform(0.2, 0.5, len(anomaly_indices))
            
            categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
            departments = np.random.choice(['Sales', 'Marketing', 'Engineering', 'Support'], n_samples)
            
            df_sample = pd.DataFrame({
                'Category': categories,
                'Department': departments,
                'Feature_1': feature1,
                'Feature_2': feature2,
                'Feature_3': feature3,
                'Date': pd.date_range('2024-01-01', periods=n_samples, freq='H')
            })
            
            # Convert to parquet
            parquet_path = os.path.join(st.session_state.temp_dir, 'sample_data.parquet')
            df_sample.to_parquet(parquet_path, compression='snappy')
            
            st.session_state.parquet_path = parquet_path
            st.session_state.data_loaded = True
            st.session_state.row_count = len(df_sample)
            
            # Initialize DuckDB
            con = initialize_duckdb()
            st.session_state.column_info = get_column_stats(con, parquet_path)
            
            st.success("✅ Sample data loaded successfully!")
    
    elif uploaded_file is not None:
        file_type = detect_file_type(uploaded_file)
        
        if file_type:
            st.info(f"📄 Detected file type: **{file_type.upper()}**")
            
            if st.button("🚀 Process File", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    parquet_path, row_count, col_count = convert_to_parquet(uploaded_file, file_type)
                    
                    if parquet_path:
                        st.session_state.parquet_path = parquet_path
                        st.session_state.data_loaded = True
                        st.session_state.row_count = row_count
                        
                        # Initialize DuckDB and get column stats
                        con = initialize_duckdb()
                        st.session_state.column_info = get_column_stats(con, parquet_path)
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ File Processed Successfully!</h4>
                            <p><b>Rows:</b> {row_count:,} | <b>Columns:</b> {col_count}</p>
                            <p><b>Storage:</b> Optimized Parquet format</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
        else:
            st.error("❌ Unsupported file format")
    
    # Display data info if loaded
    if st.session_state.data_loaded and st.session_state.parquet_path:
        st.markdown("---")
        st.markdown("### 📊 Data Overview")
        
        con = initialize_duckdb()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{st.session_state.row_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{len(st.session_state.column_info)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'INT']
            numeric_count = sum(1 for v in st.session_state.column_info.values() 
                              if any(ntype in v['type'].upper() for ntype in numeric_types))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Numeric Columns</div>
                <div class="metric-value">{numeric_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_count = sum(1 for v in st.session_state.column_info.values() 
                                  if 'VARCHAR' in v['type'].upper() or 'STRING' in v['type'].upper())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Categorical Columns</div>
                <div class="metric-value">{categorical_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Column details
        with st.expander("🔍 View Column Details", expanded=False):
            col_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': info['type'],
                    'Distinct Values': info['distinct'],
                    'Null Count': info['nulls'],
                    'Null %': f"{info['null_pct']:.2f}%"
                }
                for col, info in st.session_state.column_info.items()
            ])
            st.dataframe(col_df, use_container_width=True, height=400)
        
        # Data preview
        with st.expander("👀 Data Preview (First 1000 rows)", expanded=False):
            sample_df = get_sample_data(con, st.session_state.parquet_path, 1000)
            st.dataframe(sample_df, use_container_width=True, height=400)

# =============================================================================
# TAB 2: ANALYSIS CONFIGURATION
# =============================================================================

with tab2:
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ No Data Loaded</h4>
            <p>Please upload a file or use sample data in the <b>Data Upload</b> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### ⚙️ Configure Analysis")
        
        con = initialize_duckdb()
        
        # DEBUG SECTION - Show what we're detecting
        with st.expander("🔍 DEBUG: Column Detection", expanded=True):
            st.write("**All Columns and Types:**")
            debug_df = pd.DataFrame([
                {'Column': col, 'Type': info['type']}
                for col, info in st.session_state.column_info.items()
            ])
            st.dataframe(debug_df, use_container_width=True)
            
            # Show what will be categorized as numeric
            numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'INT']
            numeric_detected = [col for col, info in st.session_state.column_info.items() 
                               if any(ntype in info['type'].upper() for ntype in numeric_types)]
            st.write(f"**Numeric columns detected ({len(numeric_detected)}):** {numeric_detected}")
            
            # Show what will be categorized as categorical
            categorical_detected = [col for col, info in st.session_state.column_info.items() 
                                  if 'VARCHAR' in info['type'].upper() or 'STRING' in info['type'].upper()]
            st.write(f"**Categorical columns detected ({len(categorical_detected)}):** {categorical_detected}")
            
            # Show what will be categorized as date
            date_detected = [col for col, info in st.session_state.column_info.items() 
                           if 'DATE' in info['type'].upper() or 'TIMESTAMP' in info['type'].upper()]
            st.write(f"**Date columns detected ({len(date_detected)}):** {date_detected}")
        
        # Column selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📋 Select Categorical Columns")
            categorical_cols = [col for col, info in st.session_state.column_info.items() 
                              if 'VARCHAR' in info['type'].upper() or 'STRING' in info['type'].upper()]
            
            selected_categorical_cols = st.multiselect(
                "Choose categorical columns for grouping",
                categorical_cols,
                help="Used for category-level analysis (can select multiple)"
            )
        
        with col2:
            st.markdown("#### 🔢 Select Numeric Columns")
            # Check for various numeric types that DuckDB might return
            numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'INT']
            numeric_cols_available = [col for col, info in st.session_state.column_info.items() 
                                     if any(ntype in info['type'].upper() for ntype in numeric_types)]
            
            numeric_cols = st.multiselect(
                "Choose numeric columns for anomaly detection",
                numeric_cols_available,
                default=numeric_cols_available[:3] if len(numeric_cols_available) >= 3 else numeric_cols_available,
                help="Features used for anomaly detection (select multiple)"
            )
        
        with col3:
            st.markdown("#### 📅 Select Date Column (Optional)")
            # Detect date/timestamp columns
            date_cols = [col for col, info in st.session_state.column_info.items() 
                        if 'DATE' in info['type'].upper() or 'TIMESTAMP' in info['type'].upper()]
            
            date_col = st.selectbox(
                "Choose date column for time-based analysis",
                ['None'] + date_cols,
                help="View anomalies over time"
            )
            date_col = None if date_col == 'None' else date_col
        
        st.markdown("---")
        
        # Method selection
        st.markdown("#### 🎯 Select Detection Method")
        
        method_category = st.radio(
            "Choose method category",
            ['Statistical (Fast)', 'Machine Learning (Advanced)'],
            horizontal=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        if method_category == 'Statistical (Fast)':
            with col1:
                method = st.selectbox(
                    "Algorithm",
                    ['Z-Score', 'IQR Method'],
                    help="Statistical methods work on any dataset size"
                )
            
            with col2:
                if method == 'Z-Score':
                    threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)
                else:
                    threshold = None
            
            detection_method = 'zscore' if method == 'Z-Score' else 'iqr'
            use_ml = False
            
        else:
            with col1:
                method = st.selectbox(
                    "Algorithm",
                    ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'DBSCAN'],
                    help="ML methods provide more sophisticated detection"
                )
            
            with col2:
                contamination = st.slider(
                    "Expected Anomaly Rate",
                    0.01, 0.5, 0.1, 0.01,
                    help="Expected proportion of anomalies (0.1 = 10%)"
                )
            
            with col3:
                if st.session_state.row_count > 100000:
                    sample_size = st.number_input(
                        "Sample Size (for performance)",
                        1000, 100000, 50000,
                        help="ML methods will use a sample for large datasets"
                    )
                else:
                    sample_size = st.session_state.row_count
            
            method_map = {
                'Isolation Forest': 'isolation_forest',
                'Local Outlier Factor': 'lof',
                'One-Class SVM': 'one_class_svm',
                'DBSCAN': 'dbscan'
            }
            detection_method = method_map[method]
            use_ml = True
        
        st.markdown("---")
        
        # Run analysis
        if len(numeric_cols) == 0:
            st.warning("⚠️ Please select at least one numeric column")
        else:
            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                with st.spinner(f"🔍 Running {method} analysis..."):
                    start_time = time.time()
                    
                    try:
                        if use_ml:
                            # Sample data if needed
                            if st.session_state.row_count > sample_size:
                                query = f"""
                                SELECT * FROM parquet_scan('{st.session_state.parquet_path}')
                                USING SAMPLE {sample_size} ROWS
                                """
                                df = con.execute(query).df()
                                st.info(f"ℹ️ Using sample of {sample_size:,} rows for ML analysis")
                            else:
                                df = get_sample_data(con, st.session_state.parquet_path, st.session_state.row_count)
                            
                            results_df = detect_anomalies_ml(df, numeric_cols, detection_method, contamination)
                        else:
                            results_df = detect_anomalies_statistical(
                                con, st.session_state.parquet_path, numeric_cols, 
                                detection_method, threshold if detection_method == 'zscore' else 3
                            )
                        
                        if results_df is not None:
                            st.session_state.results_df = results_df
                            st.session_state.analysis_complete = True
                            st.session_state.selected_numeric_cols = numeric_cols
                            st.session_state.selected_categorical_cols = selected_categorical_cols
                            st.session_state.selected_date_col = date_col
                            st.session_state.selected_method = method
                            
                            elapsed_time = time.time() - start_time
                            
                            st.success(f"✅ Analysis complete in {elapsed_time:.2f} seconds!")
                            st.balloons()
                            
                            # Quick stats
                            anomaly_count = results_df['anomaly'].sum()
                            anomaly_rate = (anomaly_count / len(results_df) * 100)
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📊 Quick Results</h4>
                                <p><b>Anomalies Detected:</b> {anomaly_count:,} out of {len(results_df):,} records ({anomaly_rate:.2f}%)</p>
                                <p>👉 View detailed results in the <b>Results</b> tab</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

# =============================================================================
# TAB 3: RESULTS
# =============================================================================

with tab3:
    if not st.session_state.analysis_complete:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ No Analysis Results</h4>
            <p>Please run an analysis in the <b>Analysis</b> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_results = st.session_state.results_df
        numeric_cols = st.session_state.selected_numeric_cols
        categorical_cols = st.session_state.selected_categorical_cols
        date_col = st.session_state.selected_date_col
        method = st.session_state.selected_method
        
        st.markdown("### 📊 Analysis Results")
        
        # Generate insights
        insights = generate_insights(df_results, categorical_cols, numeric_cols, method)
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{insights['total_records']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Anomalies</div>
                <div class="metric-value">{insights['anomalies']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Anomaly Rate</div>
                <div class="metric-value">{insights['anomaly_rate']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Method Used</div>
                <div class="metric-value" style="font-size: 1.2rem;">{method}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        figs = create_visualizations(df_results, numeric_cols, categorical_cols, date_col)
        
        # Display charts in grid
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pie' in figs:
                st.plotly_chart(figs['pie'], use_container_width=True)
        
        with col2:
            if 'histogram' in figs:
                st.plotly_chart(figs['histogram'], use_container_width=True)
        
        # Time series chart (full width if exists)
        if 'timeseries' in figs:
            st.plotly_chart(figs['timeseries'], use_container_width=True)
        
        if 'scatter' in figs:
            st.plotly_chart(figs['scatter'], use_container_width=True)
        
        # Category bar charts for each categorical column
        for key in figs.keys():
            if key.startswith('category_bar_'):
                st.plotly_chart(figs[key], use_container_width=True)
        
        if 'parallel' in figs:
            with st.expander("🌐 Parallel Coordinates View", expanded=False):
                st.plotly_chart(figs['parallel'], use_container_width=True)
        
        if 'boxplot' in figs:
            with st.expander("📦 Box Plot Analysis", expanded=False):
                st.plotly_chart(figs['boxplot'], use_container_width=True)
        
        st.markdown("---")
        
        # Insights narrative
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Overall Summary")
            st.markdown(f"""
            - Analyzed **{insights['total_records']:,}** records using **{method}**
            - Detected **{insights['anomalies']:,}** anomalies (**{insights['anomaly_rate']:.2f}%**)
            - Features analyzed: **{', '.join(numeric_cols)}**
            """)
            
            if categorical_cols:
                st.markdown(f"- Categorical groupings: **{', '.join(categorical_cols)}**")
            
            if date_col:
                st.markdown(f"- Time analysis: **{date_col}**")
        
        with col2:
            if 'feature_importance' in insights and insights['feature_importance']:
                st.markdown("#### 📊 Feature Contributions")
                feature_diff = sorted(
                    insights['feature_importance'].items(),
                    key=lambda x: x[1]['difference'],
                    reverse=True
                )
                
                for feat, stats in feature_diff[:5]:
                    st.markdown(f"""
                    - **{feat}**: Anomaly avg = {stats['anomaly_mean']:.2f}, 
                      Normal avg = {stats['normal_mean']:.2f} 
                      (Δ = {stats['difference']:.2f})
                    """)
        
        # Categorical insights for each category
        if 'categorical_insights' in insights and insights['categorical_insights']:
            st.markdown("---")
            st.markdown("### 📋 Categorical Analysis")
            
            for cat_col, cat_data in insights['categorical_insights'].items():
                with st.expander(f"🏆 Top Categories by Anomaly Rate - {cat_col}", expanded=True):
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
                                st.info(f"🥇 Highest anomaly rate: **{highest[cat_col]}** at **{highest['anomaly_rate']:.2f}%**")
        
        # Top anomalies table
        if 'top_anomalies' in insights and insights['top_anomalies']:
            with st.expander("🔝 Top 5 Anomalies by Score", expanded=False):
                top_df = pd.DataFrame(insights['top_anomalies'])
                display_cols = (categorical_cols if categorical_cols else []) + numeric_cols + ['anomaly_score']
                display_cols = [col for col in display_cols if col in top_df.columns]
                if display_cols:
                    st.dataframe(top_df[display_cols], use_container_width=True)
        
        # Full results table
        with st.expander("📋 Full Results Table", expanded=False):
            st.dataframe(df_results, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Download options
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
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
            # Parquet download
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
            # PDF report
            pdf_buffer = generate_pdf_report(insights, categorical_cols, numeric_cols)
            if pdf_buffer:
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p style='font-size: 14px;'>
        🎯 <b>Anomaly Hunter Pro</b> | Developed with ❤️ by CE Innovation Team 2025<br>
        Powered by DuckDB, PyArrow, Scikit-learn & Plotly
    </p>
</div>
""", unsafe_allow_html=True)
