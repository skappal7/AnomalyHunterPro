# 🎯 Anomaly Hunter Pro - Anomaly Detection Tool

## 📊 Overview
Enterprise-grade anomaly detection platform with a minimalist design, Lottie animations, and optimized codebase.

## ✨ Key Improvements

### 🎨 Design Enhancements
- **Minimalist UI**: Clean white cards with subtle shadows
- **Modern Typography**: Inter font family with refined weights
- **Premium Color Scheme**: Indigo/violet gradients (#6366f1, #8b5cf6)
- **Smooth Animations**: Lottie animations for key interactions
- **Enhanced Spacing**: Better visual hierarchy and breathing room

### 🚀 Code Optimization
- **Line Reduction**: 1423 → 1323 lines (~100 lines saved)
- **Factory Pattern**: Unified detection method configuration
- **Python 3.11+**: Modern syntax with match/case statements
- **Type Hints**: Using `|` union operator for better type safety
- **DRY Principle**: Eliminated repetitive HTML/metric generation

### 🎬 Lottie Animations
1. **Upload State**: Animated icon when waiting for file upload
2. **Processing**: Loading animation during analysis
3. **Success**: Celebration animation on completion
4. **Empty State**: Friendly animation for no data/results

## 📦 Installation

### Required Dependencies
```bash
pip install streamlit pandas numpy duckdb pyarrow scikit-learn scipy plotly reportlab streamlit-lottie requests
```

### Full Requirements
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
duckdb>=0.9.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
plotly>=5.17.0
reportlab>=4.0.0
streamlit-lottie>=0.0.5
requests>=2.31.0
```

### Python Version
- **Minimum**: Python 3.11+
- **Recommended**: Python 3.12

## 🎯 Features

### Data Processing
- ✅ CSV, Excel, TXT, Parquet support
- ✅ Auto-optimization to Parquet format
- ✅ DuckDB-powered SQL analytics
- ✅ Smart type detection and conversion
- ✅ Memory-efficient processing

### Anomaly Detection Methods

**Statistical (Fast)**
- Z-Score: Standard deviation-based outlier detection
- IQR: Interquartile range method

**Machine Learning (Advanced)**
- Isolation Forest: Best for global anomalies
- Local Outlier Factor (LOF): Local density-based detection
- One-Class SVM: Non-linear boundary detection

### Visualizations
- 📊 Anomaly distribution pie chart
- 📈 Score distribution histogram
- ⏰ Time series analysis (if date column present)
- 🔍 Scatter plots for feature relationships
- 📦 Box plots by anomaly status
- 📋 Category-level bar charts

### Export Options
- CSV format
- Parquet format
- PDF report with insights

## 🚀 Usage

### 1. Start the Application
```bash
streamlit run anomalyhunter_optimized.py
```

### 2. Upload Data
- Click "Choose a file" or use sample data
- Supported formats: CSV, XLSX, XLS, TXT, Parquet
- Auto-converts to optimized Parquet format

### 3. Configure Analysis
- **Select Columns**:
  - Categorical: For grouping and category-level analysis
  - Numeric: Features for anomaly detection
  - Date: Optional time-based analysis
  
- **Choose Method**:
  - Statistical: Fast, works on any dataset size
  - ML: Advanced detection, best for < 50K rows

### 4. View Results
- Key metrics and insights
- Interactive visualizations
- Category-level analysis
- Export options (CSV, Parquet, PDF)

## 🎨 Design Philosophy

### Color Palette
- **Primary**: Indigo (#6366f1)
- **Secondary**: Violet (#8b5cf6)
- **Success**: Emerald (#10b981)
- **Warning**: Amber (#f59e0b)
- **Error**: Rose (#ef4444)
- **Background**: White with subtle shadows

### Typography
- **Font**: Inter (Google Fonts)
- **Weights**: 300 (light), 400 (regular), 600 (semibold), 700 (bold)
- **Hierarchy**: Clear size differences for scanning

### Layout
- Consistent spacing with rem units
- Card-based design with hover effects
- Responsive grid layouts
- Minimal borders, maximum content

## 🔧 Architecture Highlights

### Factory Pattern for Detection
```python
DETECTION_METHODS = {
    'isolation_forest': {
        'class': IsolationForest,
        'params': {'contamination': 'auto', 'random_state': 42},
        'type': 'ml'
    },
    # ... other methods
}
```

### Modern Python 3.11+ Features
```python
# Match/case for file type detection
match Path(filename).suffix:
    case '.csv': return 'csv'
    case '.xlsx': return 'xlsx'
    # ...

# Union type hints with |
def detect_file_type(file) -> str | None:
    # ...
```

### Reusable Components
```python
def create_metric_card(label: str, value: str | int) -> str:
    """Generate consistent metric card HTML"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
```

## 📊 Performance

### Optimizations
- **Parquet Format**: 2-5x faster queries vs CSV
- **DuckDB**: In-memory columnar processing
- **Sampling**: Automatic for large datasets (ML methods)
- **Type Optimization**: Downcasting to reduce memory

### Scalability
- **Statistical Methods**: No row limit (SQL-based)
- **Isolation Forest**: Works on full dataset
- **LOF/SVM**: Recommended < 50K rows, auto-sampling available

## 🐛 Troubleshooting

### Lottie Animations Not Showing
```bash
pip install streamlit-lottie
```
The app will work without it, showing fallback messages.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
- Use sampling for large datasets (>100K rows)
- Choose statistical methods for massive datasets
- Close other applications to free RAM

## 📝 Development Notes

### Code Structure
1. **Imports & Config** (lines 1-70)
2. **Styling** (lines 71-180)
3. **Utilities** (lines 181-350)
4. **Detection Engine** (lines 351-550)
5. **Visualization** (lines 551-750)
6. **Main App** (lines 751-1323)

## 🎯 Future Enhancements
- [ ] Add more detection algorithms (DBSCAN, etc.)
- [ ] Real-time streaming anomaly detection
- [ ] Custom model training interface
- [ ] Batch processing for multiple files
- [ ] API endpoint for programmatic access

## 📄 License
Developed by CE Innovation Team 2025

---

**Questions?** Check the code comments or open an issue!
