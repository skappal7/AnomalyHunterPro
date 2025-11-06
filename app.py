# app.py
# Anomaly Hunter Pro — Streamlit (explanatory UX + narratives, Streamlit Cloud–safe, NO PyArrow)

import os, io, tempfile, time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import streamlit as st
import pandas as pd
import numpy as np

# Keep pandas off Arrow-backed dtypes (extra guard on hosted envs)
pd.set_option("mode.dtype_backend", "numpy_nullable")

# Safe libs together on Streamlit Cloud (no pyarrow):
import duckdb
from fastparquet import write  # ensure fastparquet is present
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ---------------------------
# Page config + styling
# ---------------------------
st.set_page_config(page_title="Anomaly Hunter Pro", page_icon="🎯", layout="wide")

PRIMARY = "#6366f1"
ACCENT = "#8b5cf6"

st.markdown(f"""
<style>
* {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; }}
.main .block-container {{ max-width: 1300px; padding-top: 1.25rem; padding-bottom: 2rem; }}
h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
.card {{
  background: white; border-left: 4px solid {PRIMARY}; border-radius: 12px;
  padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}
.tag {{ display:inline-block; padding: 2px 8px; border-radius: 999px; background:#eef2ff; color:#1e293b; font-size:12px; }}
.note {{ font-size: 13px; color:#475569 }}
.stTabs [data-baseweb="tab"] {{
  background: white; border: 1px solid #e5e7eb; border-radius: 10px; color:#475569; font-weight:600;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, {PRIMARY}, {ACCENT}); color:white; border: 0;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:linear-gradient(135deg,{PRIMARY},{ACCENT});border-radius:16px;padding:22px;margin-bottom:18px;color:white">
  <h2 style="margin:0">🎯 Anomaly Hunter Pro</h2>
  <div style="opacity:.92;margin-top:6px">
    Upload large files → convert to Parquet → pick features & method → detect anomalies → visualize → export CSV/XLSX/Parquet/PDF.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session state & helpers
# ---------------------------
if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp(prefix="ahp_")
if "parquet_path" not in st.session_state:
    st.session_state.parquet_path = None
if "row_count" not in st.session_state:
    st.session_state.row_count = 0
if "col_stats" not in st.session_state:
    st.session_state.col_stats = {}
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "selected_date_col" not in st.session_state:
    st.session_state.selected_date_col = None

def tmp_path(name: str) -> str:
    return os.path.join(st.session_state.tmp_dir, name)

def init_duckdb() -> duckdb.DuckDBPyConnection:
    if "duck_con" not in st.session_state or st.session_state.duck_con is None:
        st.session_state.duck_con = duckdb.connect(database=":memory:")
    return st.session_state.duck_con

# ------------------------------------
# File handling (NO PyArrow dependency)
# ------------------------------------
def detect_file_type(filename: str) -> Optional[str]:
    suffix = Path(filename.lower()).suffix
    if suffix in [".csv", ".txt"]: return "csv_or_txt"
    if suffix in [".xlsx", ".xls"]: return "excel"
    if suffix in [".parquet"]: return "parquet"
    return None

def persist_upload(file) -> str:
    out = tmp_path(file.name)
    with open(out, "wb") as f:
        f.write(file.getbuffer())
    return out

def csv_to_parquet_via_duckdb(src_csv: str, dst_parquet: str) -> Tuple[int, int]:
    con = init_duckdb()
    con.execute(f"""
        CREATE OR REPLACE TABLE _src AS
        SELECT * FROM read_csv_auto('{src_csv}', SAMPLE_SIZE=200000, ALL_VARCHAR=FALSE);
    """)
    con.execute(f"""
        COPY _src TO '{dst_parquet}' (FORMAT 'parquet', COMPRESSION 'SNAPPY');
    """)
    nrows = con.execute("SELECT COUNT(*) FROM _src").fetchone()[0]
    ncols = len(con.execute("SELECT * FROM _src LIMIT 0").description)
    con.execute("DROP TABLE IF EXISTS _src;")
    return nrows, ncols

def excel_to_parquet_via_pandas(src_xl: str, dst_parquet: str) -> Tuple[int, int]:
    engine = "openpyxl" if src_xl.lower().endswith(".xlsx") else "xlrd"
    df = pd.read_excel(src_xl, engine=engine)
    df.to_parquet(dst_parquet, engine="fastparquet", compression="snappy")
    return len(df), df.shape[1]

def parquet_passthrough(src_parquet: str, dst_parquet: str) -> Tuple[int, int]:
    try:
        df = pd.read_parquet(src_parquet, engine="fastparquet")
        df.to_parquet(dst_parquet, engine="fastparquet", compression="snappy")
        return len(df), df.shape[1]
    except Exception:
        with open(src_parquet, "rb") as s, open(dst_parquet, "wb") as d:
            d.write(s.read())
        con = init_duckdb()
        nrows = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{dst_parquet}')").fetchone()[0]
        ncols = len(con.execute(f"SELECT * FROM parquet_scan('{dst_parquet}') LIMIT 0").description)
        return nrows, ncols

def convert_to_parquet(uploaded) -> Tuple[Optional[str], int, int]:
    ft = detect_file_type(uploaded.name)
    if ft is None:
        st.error("Unsupported file type. Please upload CSV, TXT, Excel or Parquet.")
        return None, 0, 0

    src = persist_upload(uploaded)
    dst = tmp_path("data.parquet")

    with st.spinner("Converting to Parquet..."):
        start = time.time()
        if ft == "csv_or_txt":
            rows, cols = csv_to_parquet_via_duckdb(src, dst)
        elif ft == "excel":
            rows, cols = excel_to_parquet_via_pandas(src, dst)
        else:
            rows, cols = parquet_passthrough(src, dst)
        elapsed = time.time() - start

    st.success(f"File converted in {elapsed:.2f}s → {rows:,} rows, {cols} columns")
    return dst, rows, cols

# --------------------
# Profiling & sampling
# --------------------
@st.cache_data(show_spinner=False, ttl=1800)
def get_column_stats(parquet_path: str) -> Dict[str, Dict[str, float]]:
    con = init_duckdb()
    schema = con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT 0").description
    stats: Dict[str, Dict[str, float]] = {}
    for col_name, col_type, *_ in schema:
        q = f"""
        SELECT
          COUNT(*) AS total,
          COUNT(DISTINCT "{col_name}") AS distinct_count,
          SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) AS nulls
        FROM parquet_scan('{parquet_path}')
        """
        total, distinct_count, nulls = con.execute(q).fetchone()
        stats[col_name] = {
            "type": str(col_type),
            "total": int(total),
            "distinct": int(distinct_count),
            "nulls": int(nulls),
            "null_pct": (float(nulls) / float(total) * 100.0) if total else 0.0,
        }
    return stats

def sample_df(parquet_path: str, limit: int = 10000) -> pd.DataFrame:
    con = init_duckdb()
    return con.execute(f"SELECT * FROM parquet_scan('{parquet_path}') LIMIT {int(limit)}").df()

# --------------------
# Detection engines
# --------------------
def detect_statistical(con: duckdb.DuckDBPyConnection, parquet_path: str,
                       numeric_cols: List[str], method: str, threshold: float = 3.0) -> pd.DataFrame:
    if not numeric_cols:
        raise ValueError("No numeric columns selected.")

    if method == "zscore":
        select_parts = [
            f'ABS(("{c}" - AVG("{c}") OVER ()) / NULLIF(STDDEV("{c}") OVER (), 0)) AS z_{c}'
            for c in numeric_cols
        ]
        q = f"""
        SELECT *,
               GREATEST({", ".join([f"z_{c}" for c in numeric_cols])}) AS max_z
        FROM (
            SELECT *, {", ".join(select_parts)}
            FROM parquet_scan('{parquet_path}')
        )
        """
        df = con.execute(q).df()
        df["anomaly_score"] = df["max_z"]
        df["anomaly"] = (df["anomaly_score"] > threshold).astype(int)
        return df

    # IQR
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in numeric_cols:
        q = f"""
        SELECT 
          percentile_cont(0.25) WITHIN GROUP (ORDER BY "{c}") AS q1,
          percentile_cont(0.75) WITHIN GROUP (ORDER BY "{c}") AS q3
        FROM parquet_scan('{parquet_path}')
        WHERE "{c}" IS NOT NULL
        """
        q1, q3 = con.execute(q).fetchone()
        iqr = (q3 - q1) if (q1 is not None and q3 is not None) else 0.0
        lower = (q1 - 1.5 * iqr) if q1 is not None else None
        upper = (q3 + 1.5 * iqr) if q3 is not None else None
        bounds[c] = (lower, upper)

    conds = []
    for c, (lo, hi) in bounds.items():
        if lo is None or hi is None:
            conds.append("FALSE")
        else:
            conds.append(f'("{c}" < {lo} OR "{c}" > {hi})')

    q = f"""
    SELECT *,
           CASE WHEN {" OR ".join(conds)} THEN 1 ELSE 0 END AS anomaly
    FROM parquet_scan('{parquet_path}')
    """
    df = con.execute(q).df()

    scores = []
    for c, (lo, hi) in bounds.items():
        if lo is None or hi is None:
            scores.append(np.zeros(len(df)))
        else:
            s = np.maximum(np.maximum(lo - df[c], 0), np.maximum(df[c] - hi, 0))
            scores.append(s.fillna(0))
    df["anomaly_score"] = np.max(scores, axis=0)
    return df

def detect_ml(df: pd.DataFrame, numeric_cols: List[str], method: str, contamination: float) -> pd.DataFrame:
    X = df[numeric_cols].copy()
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        pred = model.fit_predict(Xs)
        scores = -model.score_samples(Xs)
        anomaly = (pred == -1).astype(int)

    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
        pred = model.fit_predict(Xs)
        scores = -model.negative_outlier_factor_
        anomaly = (pred == -1).astype(int)

    else:  # One-Class SVM
        model = OneClassSVM(nu=min(max(contamination, 0.01), 0.5), kernel="rbf", gamma="auto")
        pred = model.fit_predict(Xs)
        scores = -model.decision_function(Xs)
        anomaly = (pred == -1).astype(int)

    out = df.copy()
    out["anomaly"] = anomaly
    out["anomaly_score"] = (scores - np.min(scores)) / (np.ptp(scores) + 1e-9)
    return out

# --------------------
# Narratives & PDF
# --------------------
def build_detailed_narrative(df: pd.DataFrame, numeric_cols: List[str], date_col: Optional[str]) -> str:
    total = len(df)
    anomalies = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    rate = (anomalies / total * 100.0) if total else 0.0

    parts = []
    parts.append(f"In a dataset of **{total:,}** records, **{anomalies:,}** were flagged as anomalies (**{rate:.2f}%**).")

    # Feature impact: mean shift between anomaly vs normal
    if "anomaly" in df.columns and numeric_cols:
        diffs = []
        g = df.groupby("anomaly")
        if 0 in g.groups and 1 in g.groups:
            base = g.get_group(0)[numeric_cols].mean(numeric_only=True)
            outl = g.get_group(1)[numeric_cols].mean(numeric_only=True)
            delta = (outl - base).abs().sort_values(ascending=False)
            top = delta.head(5)
            if not top.empty:
                ranked = ", ".join([f"**{c}** (Δ={top[c]:.3g})" for c in top.index])
                parts.append(f"The most influential features (by mean shift) appear to be: {ranked}.")

    # Distribution commentary on anomaly scores
    if "anomaly_score" in df.columns:
        q1, q5, q9 = df["anomaly_score"].quantile([0.25, 0.5, 0.9]).values
        parts.append(f"Anomaly scores show a central tendency around **{q5:.3f}** (25th: **{q1:.3f}**, 90th: **{q9:.3f}**).")

    # Time concentration if a date column exists
    if date_col and date_col in df.columns:
        try:
            s = pd.to_datetime(df[date_col], errors="coerce")
            mask = (df["anomaly"] == 1) if "anomaly" in df.columns else (pd.Series([False]*len(df)))
            by_hour = s[mask].dt.hour.value_counts().sort_index()
            if len(by_hour) > 0:
                peak_hour = int(by_hour.idxmax())
                parts.append(f"Anomalies concentrate around **{peak_hour:02d}:00** hours.")
        except Exception:
            pass

    # Categorical skew (top 1–2 categories where anomalies cluster)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]
    if "anomaly" in df.columns and cat_cols:
        try:
            skews = []
            normal = df[df["anomaly"] == 0]
            out = df[df["anomaly"] == 1]
            for c in cat_cols[:3]:  # limit for speed
                top_out = out[c].value_counts(normalize=True).head(1)
                top_norm = normal[c].value_counts(normalize=True).head(1)
                if not top_out.empty:
                    cat = top_out.index[0]
                    p_out = top_out.iloc[0]
                    p_norm = float(top_norm.get(cat, 0.0))
                    lift = (p_out / (p_norm + 1e-9))
                    skews.append((c, cat, p_out, lift))
            if skews:
                skews.sort(key=lambda x: x[3], reverse=True)
                c, cat, p, lift = skews[0]
                parts.append(f"Category **{c}={cat}** shows a strong anomaly lift (≈**{lift:.2f}×** relative to normals).")
        except Exception:
            pass

    parts.append("Use the feature selection and method choices to refine detection—statistical methods suit bell-curve data, while ML methods capture complex, non-linear outliers.")
    return " ".join(parts)

def generate_pdf_report(insights: Dict, features: List[str], method_label: str) -> io.BytesIO:
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("heading", parent=styles["Heading1"], fontSize=16, leading=20, textColor=colors.HexColor("#1e293b"))
    p = []

    p.append(Paragraph("Anomaly Hunter Pro — Report", heading))
    p.append(Spacer(1, 6))
    p.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Normal"]))
    p.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Total Records", f"{insights['total_records']:,}"],
        ["Anomalies", f"{insights['anomalies']:,}"],
        ["Anomaly Rate (%)", f"{insights['anomaly_rate']:.2f}"],
        ["Method", method_label],
        ["Features", ", ".join(features)]
    ]
    table = Table(data, colWidths=[2.2*inch, 4.6*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#8b5cf6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#eef2ff")]),
    ]))
    p.extend([table, Spacer(1, 12)])

    doc.build(p)
    buff.seek(0)
    return buff

# --------------------
# UI (with explanations)
# --------------------
tab1, tab2, tab3 = st.tabs(["📂 Data", "🔍 Analyze", "📊 Results"])

# ---- Tab 1: Data Upload ----
with tab1:
    st.subheader("Upload & Prepare Data")
    st.markdown(
        "<div class='note'>Upload CSV/TXT/Excel/Parquet. We standardize everything to Parquet for fast, columnar analytics.</div>",
        unsafe_allow_html=True
    )
    with st.expander("ℹ How this step works (deep dive)"):
        st.markdown("""
- **CSV/TXT** files are streamed into **DuckDB** and exported to **Parquet (Snappy)**. This is memory-friendly and fast.  
- **Excel** files are read by pandas and saved as Parquet using **fastparquet** (no PyArrow).  
- **Already-Parquet** files are normalized (resaved) for consistent metadata.  
- Parquet enables faster scans, column-level pruning, and lower memory use.
        """)

    c1, c2 = st.columns([2,1], gap="large")
    with c1:
        up = st.file_uploader("Choose CSV / TXT / Excel / Parquet", type=["csv","txt","xlsx","xls","parquet"])
    with c2:
        st.markdown("<div class='note'>Prefer CSV for large raw exports. XLS/XLSX are fine for smaller datasets.</div>", unsafe_allow_html=True)
        if st.button("Load sample (10k rows)"):
            np.random.seed(7)
            n = 10_000
            df = pd.DataFrame({
                "Category": np.random.choice(list("ABCDE"), n),
                "ScoreA": np.random.normal(100, 15, n),
                "ScoreB": np.random.normal(200, 30, n),
                "ScoreC": np.random.normal(50, 8, n),
                "Date": pd.date_range("2024-01-01", periods=n, freq="H")
            })
            idx = np.random.choice(n, size=int(0.05*n), replace=False)
            df.loc[idx, "ScoreA"] *= np.random.uniform(1.8, 3.2, len(idx))
            parquet_path = tmp_path("data.parquet")
            df.to_parquet(parquet_path, engine="fastparquet", compression="snappy")
            st.session_state.parquet_path = parquet_path
            st.session_state.row_count = len(df)
            st.session_state.col_stats = get_column_stats(parquet_path)
            st.success("Sample data loaded and standardized to Parquet.")

    if up and st.button("🚀 Process file"):
        pq, rows, cols = convert_to_parquet(up)
        if pq:
            st.session_state.parquet_path = pq
            st.session_state.row_count = rows
            st.session_state.col_stats = get_column_stats(pq)

    if st.session_state.parquet_path:
        st.markdown("---")
        st.subheader("Quick Overview")
        stats = st.session_state.col_stats
        num_types = ("INTEGER","BIGINT","SMALLINT","TINYINT","DOUBLE","FLOAT","DECIMAL","NUMERIC","REAL")
        numeric_count = sum(any(t in stats[c]["type"].upper() for t in num_types) for c in stats)
        categorical_count = sum(("VARCHAR" in stats[c]["type"].upper() or "STRING" in stats[c]["type"].upper())
                                for c in stats)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='card'><div class='tag'>Rows</div><h3>{st.session_state.row_count:,}</h3></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><div class='tag'>Columns</div><h3>{len(stats)}</h3></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><div class='tag'>Numeric</div><h3>{numeric_count}</h3></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='card'><div class='tag'>Categorical</div><h3>{categorical_count}</h3></div>", unsafe_allow_html=True)

        with st.expander("ℹ What do these counts mean?"):
            st.markdown("""
- **Numeric** columns are used to compute anomaly scores.  
- **Categorical** columns are used for context, grouping, and narratives.  
- If you have a **Date/Time** column, we can explain *when* anomalies occur.
            """)

        with st.expander("👀 Preview 1,000 rows"):
            st.dataframe(sample_df(st.session_state.parquet_path, 1000), use_container_width=True, height=360)

# ---- Tab 2: Analyze ----
with tab2:
    st.subheader("Select Features & Detection Method")
    st.markdown("<div class='note'>Pick the numeric features to score, optional categorical/date for context, and a detection method.</div>", unsafe_allow_html=True)
    with st.expander("ℹ Guidance on methods"):
        st.markdown("""
**Statistical**
- **Z-Score**: Finds values many standard deviations from the mean. Best for bell-curve (Gaussian) data.
- **IQR**: Flags values outside the interquartile range (robust to skew/outliers).

**Machine Learning**
- **Isolation Forest**: Great default. Isolates anomalies via random splits; handles non-linear patterns.
- **Local Outlier Factor (LOF)**: Detects local density deviations; good for clusters.
- **One-Class SVM**: Learns a boundary of “normal”; can be effective but sensitive to parameters.
        """)

    if not st.session_state.parquet_path:
        st.info("Upload or load data in the **Data** tab first.")
    else:
        stats = st.session_state.col_stats
        num_types = ("INTEGER","BIGINT","SMALLINT","TINYINT","DOUBLE","FLOAT","DECIMAL","NUMERIC","REAL")
        numeric_candidates = [c for c in stats if any(t in stats[c]["type"].upper() for t in num_types)]
        categorical_candidates = [c for c in stats if any(t in stats[c]["type"].upper() for t in ("VARCHAR","STRING"))]
        date_candidates = [c for c in stats if any(t in stats[c]["type"].upper() for t in ("DATE","TIMESTAMP"))]

        c1, c2, c3 = st.columns(3)
        with c1:
            numeric_cols = st.multiselect("Numeric features (required)", numeric_candidates,
                                          default=numeric_candidates[:3] if len(numeric_candidates)>=3 else numeric_candidates,
                                          help="These columns are scored to determine outliers.")
        with c2:
            categorical_cols = st.multiselect("Categorical (optional)", categorical_candidates,
                                              help="Used for grouping and narratives.")
        with c3:
            date_col = st.selectbox("Date/Time (optional)", ["None"] + date_candidates,
                                    help="If provided, we’ll analyze when anomalies occur.")
            date_col = None if date_col == "None" else date_col
            st.session_state.selected_date_col = date_col

        st.markdown("---")
        mode = st.radio("Method", ["Statistical (Z-Score/IQR)", "Machine Learning (IF/LOF/OC-SVM)"], horizontal=True)

        if mode.startswith("Statistical"):
            algo = st.selectbox("Algorithm", ["Z-Score", "IQR"])
            if algo == "Z-Score":
                threshold = st.slider("Z threshold", 2.0, 5.0, 3.0, 0.5, help="Higher = stricter (fewer anomalies).")
            else:
                threshold = None
        else:
            algo = st.selectbox("Algorithm", ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])
            contamination = st.slider("Expected anomaly rate", 0.01, 0.5, 0.10, 0.01,
                                      help="Rough share of anomalies in the data (used by IF/LOF/SVM).")

        st.markdown("---")
        if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                start = time.time()
                con = init_duckdb()
                try:
                    if not numeric_cols:
                        st.warning("Select at least one numeric column.")
                        st.stop()

                    if mode.startswith("Statistical"):
                        method_key = "zscore" if algo == "Z-Score" else "iqr"
                        df = detect_statistical(con, st.session_state.parquet_path, numeric_cols, method_key,
                                                threshold if method_key == "zscore" else 3.0)
                    else:
                        df_ml = con.execute(f"SELECT * FROM parquet_scan('{st.session_state.parquet_path}')").df()
                        df = detect_ml(df_ml, numeric_cols, algo, contamination)

                    # carry context columns forward (best-effort)
                    keep_cols = list(dict.fromkeys(["anomaly","anomaly_score"] + numeric_cols + categorical_cols +
                                                   ([date_col] if date_col else [])))
                    existing = [c for c in keep_cols if c in df.columns]
                    df = df[existing + [c for c in df.columns if c not in existing]]  # preferred first

                    st.session_state.results_df = df
                    st.success(f"Completed in {time.time() - start:.2f}s")

                except Exception as e:
                    st.error(f"Detection failed: {e}")

# ---- Tab 3: Results ----
with tab3:
    st.subheader("Results & Narratives")
    st.markdown("<div class='note'>Review the anomaly table, read the executive summary, and export results.</div>", unsafe_allow_html=True)
    with st.expander("ℹ How to read this panel"):
        st.markdown("""
- **Anomaly** = 1 indicates a row flagged as an outlier.  
- **Anomaly Score** (if present) shows *how strongly* a row stands out.  
- Use the **narrative** to understand which features and time windows are driving anomalies.
        """)

    if st.session_state.results_df is None:
        st.info("No results yet. Run the analysis first.")
    else:
        df = st.session_state.results_df
        total = len(df)
        anomalies = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
        rate = (anomalies / total * 100.0) if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='card'><div class='tag'>Total</div><h3>{total:,}</h3></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><div class='tag'>Anomalies</div><h3>{anomalies:,}</h3></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><div class='tag'>Anomaly Rate</div><h3>{rate:.2f}%</h3></div>", unsafe_allow_html=True)
        method_label = "Machine Learning" if "anomaly_score" in df.columns and df["anomaly"].dtype.kind in "iu" else "Statistical"
        c4.markdown(f"<div class='card'><div class='tag'>Method</div><h3>{method_label}</h3></div>", unsafe_allow_html=True)

        # Narrative (detailed executive summary)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("anomaly","anomaly_score")]
        narrative = build_detailed_narrative(df, numeric_cols=numeric_cols, date_col=st.session_state.selected_date_col)
        st.markdown("### Executive Narrative")
        st.markdown(narrative)

        # Table and visuals
        st.markdown("### Table (first 1,000 rows)")
        st.dataframe(df.head(1000), use_container_width=True, height=380)

        st.markdown("### Visuals")
        figs = {}
        if "anomaly" in df.columns:
            counts = df["anomaly"].value_counts().to_dict()
            figs["pie"] = go.Figure(data=[go.Pie(labels=["Normal","Anomaly"],
                                                 values=[counts.get(0,0), counts.get(1,0)], hole=0.45)])
            figs["pie"].update_layout(title="Anomaly Distribution", template="plotly_white", height=380)
        if "anomaly_score" in df.columns:
            figs["hist"] = px.histogram(df, x="anomaly_score", color="anomaly", nbins=50,
                                        title="Anomaly Score Distribution", template="plotly_white", height=380)
        cc1, cc2 = st.columns(2)
        with cc1:
            if "pie" in figs: st.plotly_chart(figs["pie"], use_container_width=True)
        with cc2:
            if "hist" in figs: st.plotly_chart(figs["hist"], use_container_width=True)

        st.markdown("### Export")
        c1, c2, c3, c4 = st.columns(4)

        # CSV
        with c1:
            buf_csv = io.BytesIO()
            df.to_csv(buf_csv, index=False)
            st.download_button("📥 CSV", data=buf_csv.getvalue(),
                               file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)

        # Excel (.xlsx)
        with c2:
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")
                # optional summary sheet
                pd.DataFrame({
                    "Metric": ["Total Records", "Anomalies", "Anomaly Rate (%)", "Method"],
                    "Value": [total, anomalies, f"{rate:.2f}", method_label]
                }).to_excel(writer, index=False, sheet_name="Summary")
            st.download_button("📥 Excel (.xlsx)", data=buf_xlsx.getvalue(),
                               file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        # Parquet (fastparquet)
        with c3:
            buf_pq = io.BytesIO()
            df.to_parquet(buf_pq, engine="fastparquet", compression="snappy")
            st.download_button("📥 Parquet", data=buf_pq.getvalue(),
                               file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                               mime="application/octet-stream", use_container_width=True)

        # PDF summary
        with c4:
            insights = {"total_records": total, "anomalies": anomalies, "anomaly_rate": rate}
            pdf = generate_pdf_report(insights, features=[c for c in df.columns if c not in ("anomaly","anomaly_score")],
                                      method_label=method_label)
            st.download_button("📥 PDF Report", data=pdf.getvalue(),
                               file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built for Streamlit Cloud • Parquet via FastParquet • CSV→Parquet via DuckDB • No PyArrow • Clear guidance & narratives")
