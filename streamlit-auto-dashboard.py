import subprocess
import sys
import os

def install_required_packages():
    """Checks and installs required packages from requirements.txt."""
    try:
        # Check if requirements.txt exists
        if os.path.exists('requirements.txt'):
            print("Installing required packages from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Installation complete!")
        else:
            print("requirements.txt not found. Installing core dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "streamlit==1.32.0",
                "pandas==2.2.0",
                "numpy==1.26.4",
                "plotly==5.20.0",
                "scipy==1.12.0",
                "matplotlib==3.8.3",
                "seaborn==0.13.1",
                "wordcloud==1.9.2"
            ])
            print("Core dependencies installed!")
    except Exception as e:
        print(f"Error installing packages: {e}")

# Call the function to install packages
install_required_packages()

# Now import the required packages
import streamlit as st
import pandas as pd
import numpy as np
import re
import uuid
import io
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    pass

# Set page configuration with custom theme
st.set_page_config(
    page_title="SmartDashPro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a color palette for the dashboard
THEME_COLORS = {
    'primary': '#2c3e50',      # Dark blue/slate for headers
    'secondary': '#3498db',    # Bright blue for highlights
    'tertiary': '#e74c3c',     # Red for alerts/negative trends
    'positive': '#2ecc71',     # Green for positive trends
    'neutral': '#f39c12',      # Orange for neutral/warning
    'background': '#f8f9fa',   # Light gray for background
    'card': '#ffffff',         # White for cards
    'text': '#333333',         # Dark gray for text
    'light_text': '#7f8c8d',   # Light gray for subtitles
    'border': '#ecf0f1',       # Very light gray for borders
}

# Enhanced custom CSS for better styling and professional appearance
st.markdown(f"""
<style>
    /* Global styles */
    .reportview-container .main .block-container {{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }}
    .main {{
        background-color: {THEME_COLORS['background']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {THEME_COLORS['primary']};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }}
    
    /* Dashboard header */
    .dashboard-header {{
        padding: 1rem;
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid {THEME_COLORS['secondary']};
    }}
    
    /* Dashboard title */
    .dashboard-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {THEME_COLORS['primary']};
        margin: 0;
        padding-bottom: 0.5rem;
    }}
    
    /* Dashboard subtitle */
    .dashboard-subtitle {{
        font-size: 1rem;
        color: {THEME_COLORS['light_text']};
        margin: 0;
    }}
    
    /* KPI section */
    .kpi-section {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }}
    
    /* KPI Card */
    .kpi-card {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        flex: 1;
        min-width: 220px;
        transition: transform 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
    }}
    
    /* KPI Card title */
    .kpi-title {{
        font-size: 0.85rem;
        font-weight: 500;
        color: {THEME_COLORS['light_text']};
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }}
    
    /* KPI Card value */
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {THEME_COLORS['primary']};
        margin: 0;
    }}
    
    /* KPI Card trend */
    .kpi-trend-positive {{
        color: {THEME_COLORS['positive']};
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}
    .kpi-trend-negative {{
        color: {THEME_COLORS['tertiary']};
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}
    .kpi-trend-neutral {{
        color: {THEME_COLORS['neutral']};
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}
    
    /* Filter section */
    .filter-section {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Chart card */
    .chart-card {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-top: 5px solid {THEME_COLORS['secondary']};
    }}
    
    /* Chart title */
    .chart-title {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {THEME_COLORS['primary']};
        margin-top: 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid {THEME_COLORS['border']};
        padding-bottom: 0.8rem;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: {THEME_COLORS['background']};
        border-radius: 10px;
        padding: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {THEME_COLORS['background']};
        border-radius: 8px;
        color: {THEME_COLORS['primary']};
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_COLORS['secondary']};
        color: white;
    }}
    
    /* Data quality cards */
    .quality-card {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    
    .quality-good {{
        border-left: 5px solid {THEME_COLORS['positive']};
    }}
    
    .quality-warning {{
        border-left: 5px solid {THEME_COLORS['neutral']};
    }}
    
    .quality-bad {{
        border-left: 5px solid {THEME_COLORS['tertiary']};
    }}
    
    /* Data quality score */
    .quality-score {{
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
    }}
    
    .quality-score-good {{
        color: {THEME_COLORS['positive']};
    }}
    
    .quality-score-warning {{
        color: {THEME_COLORS['neutral']};
    }}
    
    .quality-score-bad {{
        color: {THEME_COLORS['tertiary']};
    }}
    
    /* Recommendation card */
    .recommendation-card {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid {THEME_COLORS['secondary']};
    }}
    
    /* Footer */
    .dashboard-footer {{
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid {THEME_COLORS['border']};
        text-align: center;
        color: {THEME_COLORS['light_text']};
        font-size: 0.85rem;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {THEME_COLORS['secondary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: {THEME_COLORS['primary']};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    /* Custom stSelectbox */
    .stSelectbox label, .stMultiselect label {{
        color: {THEME_COLORS['primary']};
        font-weight: 500;
    }}
    
    /* Colored indicators */
    .indicator-positive {{
        color: {THEME_COLORS['positive']};
        font-weight: 600;
    }}
    
    .indicator-negative {{
        color: {THEME_COLORS['tertiary']};
        font-weight: 600;
    }}
    
    .indicator-neutral {{
        color: {THEME_COLORS['neutral']};
        font-weight: 600;
    }}
    
    /* Tooltip */
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: {THEME_COLORS['primary']};
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Info box */
    .info-box {{
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 5px solid {THEME_COLORS['secondary']};
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Warning box */
    .warning-box {{
        background-color: rgba(243, 156, 18, 0.1);
        border-left: 5px solid {THEME_COLORS['neutral']};
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Error box */
    .error-box {{
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 5px solid {THEME_COLORS['tertiary']};
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Success box */
    .success-box {{
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 5px solid {THEME_COLORS['positive']};
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Data tables */
    .styled-dataframe {{
        border-collapse: collapse;
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: white;
        border-radius: 5px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    
    .styled-dataframe thead th {{
        background-color: {THEME_COLORS['primary']};
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    
    .styled-dataframe tbody tr {{
        border-bottom: 1px solid {THEME_COLORS['border']};
    }}
    
    .styled-dataframe tbody tr:nth-of-type(even) {{
        background-color: {THEME_COLORS['background']};
    }}
    
    .styled-dataframe tbody tr:last-of-type {{
        border-bottom: 2px solid {THEME_COLORS['primary']};
    }}
    
    .styled-dataframe tbody td {{
        padding: 12px 15px;
        font-size: 0.9rem;
    }}
    
    /* Welcome screen container */
    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: {THEME_COLORS['card']};
        border-radius: 15px;
        padding: 3rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        margin: 3rem 0;
        text-align: center;
    }}
    
    /* Feature card */
    .feature-card {{
        background-color: {THEME_COLORS['card']};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
        height: 100%;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
    }}
    
    .feature-icon {{
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: {THEME_COLORS['secondary']};
    }}
    
    .feature-title {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {THEME_COLORS['primary']};
        margin-bottom: 0.8rem;
    }}
    
    .feature-description {{
        color: {THEME_COLORS['light_text']};
        font-size: 0.9rem;
    }}
    
    /* Divider */
    .divider {{
        border-top: 1px solid {THEME_COLORS['border']};
        margin: 2rem 0;
    }}
    
    /* Number input labels */
    .stNumberInput label {{
        color: {THEME_COLORS['primary']};
        font-weight: 500;
    }}
    
    /* Date input labels */
    .stDateInput label {{
        color: {THEME_COLORS['primary']};
        font-weight: 500;
    }}
    
    /* Text input labels */
    .stTextInput label {{
        color: {THEME_COLORS['primary']};
        font-weight: 500;
    }}
    
    /* Layout helper classes */
    .flex-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }}
    
    .flex-item {{
        flex: 1;
        min-width: 300px;
    }}
    
    /* Spacers */
    .spacer-sm {{
        height: 1rem;
    }}
    
    .spacer-md {{
        height: 2rem;
    }}
    
    .spacer-lg {{
        height: 3rem;
    }}
</style>
""", unsafe_allow_html=True)

# Check for required packages
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

# Function to detect column data types and semantics (enhanced version)
def analyze_column(df, column_name):
    """Analyzes a column to determine its properties and best visualization type"""
    col_data = df[column_name].dropna()
    
    if len(col_data) == 0:
        return {
            "type": "unknown",
            "viz_type": "none",
            "description": "Empty column"
        }
    
    # Check if column is numeric
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    
    # Check if column is datetime
    is_datetime = False
    if not is_numeric:
        try:
            # Try to convert to datetime
            pd.to_datetime(col_data, errors='raise')
            is_datetime = True
        except:
            is_datetime = False
    
    # Check if column contains geographical data
    geo_indicators = {
        "country": ["country", "nation", "land"],
        "city": ["city", "town", "municipality"],
        "state": ["state", "province", "territory", "region"],
        "address": ["address", "street", "avenue", "road"],
        "zip": ["zip", "postal", "postcode"],
        "continent": ["continent"]
    }
    
    is_geo = False
    geo_type = "none"
    col_lower = column_name.lower()
    
    for geo_category, indicators in geo_indicators.items():
        if any(indicator in col_lower for indicator in indicators):
            is_geo = True
            geo_type = geo_category
            break
    
    # Check if column is categorical/nominal
    unique_ratio = len(col_data.unique()) / len(col_data) if len(col_data) > 0 else 0
    
    # Adjust categorical threshold based on dataset size
    categorical_threshold = 0.05
    if len(col_data) > 10000:
        categorical_threshold = 0.01
    elif len(col_data) < 100:
        categorical_threshold = 0.3
    
    is_categorical = unique_ratio < categorical_threshold and len(col_data.unique()) <= 50
    
    # For text columns, check if they could be categorical even with higher cardinality
    if not is_numeric and not is_datetime and not is_categorical:
        # If value lengths are mostly short, it might be a categorical column with many values
        if col_data.astype(str).str.len().mean() < 15 and len(col_data.unique()) <= 200:
            is_categorical = True
    
    # Check if nominal column appears to be binary/boolean
    is_binary = is_categorical and len(col_data.unique()) <= 3
    if is_binary:
        # Check if column contains typical boolean values
        string_values = col_data.astype(str).str.lower()
        boolean_indicators = ['yes', 'no', 'true', 'false', 'y', 'n', 't', 'f', '1', '0']
        has_boolean_values = string_values.isin(boolean_indicators).any()
        is_binary = is_binary and has_boolean_values
    
    # Analyze column name for semantic meaning with enhanced patterns
    name_indicators = {
        "id": ["id", "identifier", "code", "key", "number", "no", "num", "_id"],
        "name": ["name", "title", "label", "description", "desc"],
        "gender": ["gender", "sex", "male", "female"],
        "age": ["age", "years", "year old", "days old", "months old", "year of birth", "yob", "birth year", "birth date", "birthdate", "dob"],
        "date": ["date", "time", "day", "month", "year", "created", "updated", "timestamp", "period", "dt_", "datetime"],
        "location": ["country", "city", "state", "region", "province", "location", "address", "place", "area", "territory", "zip", "postal"],
        "category": ["category", "type", "group", "class", "segment", "department", "section", "division", "kind", "genre", "tier"],
        "price": ["price", "cost", "fee", "amount", "payment", "revenue", "sales", "income", "expense", "spend", "budget", "money", "dollar", "euro", "pound", "yen", "currency"],
        "quantity": ["quantity", "count", "number", "amount", "total", "sum", "volume", "size", "capacity", "weight", "units"],
        "percentage": ["percent", "percentage", "ratio", "rate", "proportion", "share", "fraction", "pct", "portion"],
        "rating": ["rating", "score", "rank", "grade", "evaluation", "assessment", "review", "star", "level", "tier", "quality"],
        "status": ["status", "state", "condition", "stage", "phase", "tier", "level", "flag"],
        "email": ["email", "e-mail", "mail", "gmail", "outlook"],
        "phone": ["phone", "telephone", "mobile", "contact", "cell"],
        "url": ["url", "website", "link", "site", "web", "http", "www"],
        "social": ["facebook", "twitter", "instagram", "linkedin", "youtube", "social"],
        "duration": ["duration", "length", "period", "time", "interval", "span"],
        "frequency": ["frequency", "often", "repeat", "recur", "interval", "periodic", "times"]
    }
    
    col_semantic = "other"
    for semantic, indicators in name_indicators.items():
        if any(indicator in col_lower or col_lower.endswith(indicator) or col_lower.startswith(indicator) for indicator in indicators):
            col_semantic = semantic
            break
    
    # Look for patterns in values (enhanced)
    if not is_numeric and not is_datetime and not is_categorical:
        # Check for email pattern
        if col_data.astype(str).str.contains("@").mean() > 0.5:
            col_semantic = "email"
        
        # Check for URL pattern
        if col_data.astype(str).str.contains("http|www\.").mean() > 0.3:
            col_semantic = "url"
        
        # Check for phone pattern
        if col_data.astype(str).str.contains("\d{3}[-.\s]?\d{3}[-.\s]?\d{4}").mean() > 0.3:
            col_semantic = "phone"
    
    # Determine best visualization type with enhanced logic
    viz_type = "table"  # Default
    description = ""
    insight = ""
    
    # 1. Datetime columns - Time Series Analysis
    if is_datetime:
        viz_type = "time_series"
        description = "Datetime column, good for trend analysis over time"
        
        # Check for seasonality or periodicity if enough data
        if len(col_data) > 30:
            insight = "This time data could reveal seasonal patterns or trends."
    
    # 2. Geographic columns - Maps and Geo Visualizations
    elif is_geo:
        if geo_type in ["country", "state", "continent"]:
            viz_type = "choropleth_map"
            description = f"Geographic data at {geo_type} level, ideal for choropleth maps"
        else:  # city, address, zip
            viz_type = "point_map"
            description = f"Geographic data at {geo_type} level, ideal for point maps"
        
        insight = "Geographic patterns may reveal regional differences or clusters."
    
    # 3. Numeric columns - Statistical Visualizations
    elif is_numeric:
        # Check distribution characteristics
        if len(col_data) > 10:
            try:
                if SCIPY_AVAILABLE:
                    skewness = stats.skew(col_data)
                    kurtosis = stats.kurtosis(col_data)
                    
                    # Highly skewed distribution
                    if abs(skewness) > 2:
                        viz_type = "box_plot"
                        description = f"Highly skewed numeric data (skew: {skewness:.2f}), box plot shows distribution with outliers"
                        insight = "The data shows significant skew, suggesting presence of outliers or a non-normal distribution."
                    # Heavy-tailed distribution
                    elif kurtosis > 3:
                        viz_type = "violin_plot"
                        description = "Heavy-tailed distribution, violin plot shows distribution density"
                        insight = "The data has a heavy-tailed distribution, meaning extreme values occur more frequently than in a normal distribution."
                    # Bimodal distribution check
                    elif len(col_data) > 100:
                        # Simple bimodality check using histogram
                        hist, bin_edges = np.histogram(col_data, bins='auto')
                        peaks = [i for i in range(1, len(hist)-1) if hist[i-1] < hist[i] and hist[i] > hist[i+1]]
                        
                        if len(peaks) > 1:
                            viz_type = "kde_plot"
                            description = "Potentially multi-modal distribution, KDE plot shows multiple peaks"
                            insight = "The data may have multiple modes, suggesting distinct groups or clusters within this variable."
                        else:
                            viz_type = "histogram"
                            description = "Numeric data with approximately normal distribution"
                    else:
                        viz_type = "histogram"
                        description = "Numeric data, histogram shows distribution"
                else:
                    # Simple distribution analysis without scipy
                    viz_type = "histogram"
                    description = "Numeric data, histogram shows distribution"
            except:
                viz_type = "histogram"
                description = "Numeric data, histogram shows distribution"
        
        # Semantic-specific visualizations
        if "price" in col_semantic or "quantity" in col_semantic:
            if unique_ratio > 0.7:  # Many unique values
                viz_type = "histogram"
                description = f"Continuous {col_semantic} data, histogram shows distribution"
            else:
                viz_type = "bar_chart"
                description = f"Discrete {col_semantic} values, bar chart shows distribution"
            
            # Add insight based on semantic
            if "price" in col_semantic:
                insight = "Analyze price distribution to identify pricing tiers or anomalies."
            else:  # quantity
                insight = "Quantity distribution may reveal inventory patterns or usage trends."
        
        elif "age" in col_semantic:
            viz_type = "histogram"
            description = "Age distribution"
            insight = "Age distribution can reveal demographic patterns in your data."
        
        elif "percentage" in col_semantic or "rating" in col_semantic:
            if unique_ratio < 0.2 or len(col_data.unique()) < 20:
                viz_type = "bar_chart"
                description = f"{col_semantic.capitalize()} distribution as bar chart"
            else:
                viz_type = "histogram"
                description = f"{col_semantic.capitalize()} distribution as histogram"
            
            if "rating" in col_semantic:
                insight = "Ratings distribution may reveal customer satisfaction levels or product quality perception."
            else:  # percentage
                insight = "Percentage distributions can highlight efficiency metrics or completion rates."
        
        # For numeric columns with few unique values
        elif len(col_data.unique()) <= 15:
            viz_type = "bar_chart"
            description = "Discrete numeric values, bar chart shows distribution"
        
        # For numeric columns with high cardinality
        elif unique_ratio > 0.8:
            viz_type = "scatter_plot"
            description = "High-cardinality numeric data, might be continuous values"
    
    # 4. Categorical columns - Comparative Visualizations
    elif is_categorical:
        # Binary/Boolean data
        if is_binary:
            if col_semantic == "gender":
                viz_type = "donut_chart"
                description = "Gender distribution as donut chart"
                insight = "Gender distribution can reveal demographic balance."
            else:
                viz_type = "pie_chart"
                description = "Binary categorical data as pie chart"
                insight = "This binary split shows the proportion of different outcomes."
        
        # Location data
        elif col_semantic == "location":
            if len(col_data.unique()) <= 10:
                viz_type = "choropleth_map"
                description = "Location data suitable for choropleth map"
            else:
                viz_type = "bar_chart"
                description = "Location distribution as bar chart"
            
            insight = "Geographic distribution may reveal regional patterns or market concentrations."
        
        # Categorical data with few categories (suitable for pie/donut)
        elif len(col_data.unique()) <= 7:
            viz_type = "pie_chart"
            description = f"Categorical data with {len(col_data.unique())} categories as pie chart"
            insight = "The relative proportions of these categories show their distribution in the dataset."
        
        # Categorical data with moderate number of categories (bar chart)
        elif len(col_data.unique()) <= 20:
            viz_type = "bar_chart"
            description = f"Categorical data with {len(col_data.unique())} categories as bar chart"
            insight = "This distribution shows the frequency of each category."
        
        # Categorical data with many categories (treemap or packed bubbles)
        else:
            viz_type = "treemap"
            description = f"Categorical data with {len(col_data.unique())} categories as treemap"
            insight = "This visualization makes it easier to see the relative size of many categories at once."
    
    # 5. Text and other data types
    else:
        if col_semantic == "name" or col_semantic == "id":
            viz_type = "table"
            description = "Identifier column, best displayed in table"
        
        elif col_semantic == "email" or col_semantic == "phone" or col_semantic == "url":
            viz_type = "table"
            description = f"{col_semantic.capitalize()} data, best displayed in table"
        
        # Text data with high cardinality
        elif unique_ratio > 0.5:
            viz_type = "table"
            description = "High-cardinality text column, likely unique identifiers"
        
        # Text data with moderate cardinality
        else:
            if len(col_data.unique()) > 50:
                viz_type = "word_cloud"
                description = "Text data with high variability, word cloud shows frequency"
                insight = "Word frequency can reveal common themes or terms in your text data."
            else:
                viz_type = "bar_chart"
                description = "Text data with moderate cardinality as bar chart"
    
    # Calculate common statistics where appropriate
    statistics = {}
    
    if is_numeric:
        try:
            statistics = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "range": float(col_data.max() - col_data.min())
            }
            
            # Calculate quartiles and IQR for box plots
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            
            statistics.update({
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": q1 - 1.5 * iqr,
                "upper_fence": q3 + 1.5 * iqr
            })
            
            # Identify outliers
            outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
            statistics["outlier_count"] = len(outliers)
            statistics["outlier_pct"] = len(outliers) / len(col_data) * 100 if len(col_data) > 0 else 0
        except:
            # Fallback for errors in statistical calculation
            pass
    
    elif is_categorical or is_binary:
        # Get value counts and calculate entropy for categorical variables
        try:
            value_counts = col_data.value_counts()
            top_category = value_counts.index[0] if not value_counts.empty else None
            top_category_pct = value_counts.iloc[0] / value_counts.sum() * 100 if not value_counts.empty else 0
            
            statistics = {
                "unique_values": len(col_data.unique()),
                "most_common": str(top_category),
                "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "most_common_pct": float(top_category_pct)
            }
            
            # Calculate entropy (measure of dispersion)
            if len(value_counts) > 1:
                probabilities = value_counts / value_counts.sum()
                entropy = -sum(probabilities * np.log2(probabilities))
                statistics["entropy"] = float(entropy)
                
                # Normalized entropy (0 to 1)
                max_entropy = np.log2(len(value_counts))
                statistics["normalized_entropy"] = float(entropy / max_entropy) if max_entropy > 0 else 0
            
            # Calculate Gini impurity for binary/categorical variables
            if len(value_counts) > 1:
                probabilities = value_counts / value_counts.sum()
                gini = 1 - sum(probabilities ** 2)
                statistics["gini_impurity"] = float(gini)
        except:
            # Fallback for errors
            pass
    
    elif is_datetime:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(col_data):
                col_data = pd.to_datetime(col_data)
            
            statistics = {
                "min_date": col_data.min().strftime("%Y-%m-%d"),
                "max_date": col_data.max().strftime("%Y-%m-%d"),
                "range_days": (col_data.max() - col_data.min()).days,
                "most_common_year": int(col_data.dt.year.mode().iloc[0]) if not col_data.dt.year.mode().empty else None,
                "most_common_month": int(col_data.dt.month.mode().iloc[0]) if not col_data.dt.month.mode().empty else None,
                "most_common_day": int(col_data.dt.day.mode().iloc[0]) if not col_data.dt.day.mode().empty else None
            }
        except:
            # Fallback for errors in datetime processing
            pass
    
    return {
        "type": "numeric" if is_numeric else "datetime" if is_datetime else "categorical" if is_categorical else "text",
        "is_binary": is_binary,
        "is_geo": is_geo,
        "geo_type": geo_type if is_geo else "none",
        "unique_count": len(col_data.unique()),
        "unique_ratio": unique_ratio,
        "viz_type": viz_type,
        "semantic": col_semantic,
        "description": description,
        "insight": insight,
        "statistics": statistics
    }

# Enhanced function to create better visualizations based on column analysis
def create_enhanced_visualization(df, column_name, analysis, container=None, key_suffix="", height=400, width=None):
    """Creates a sophisticated visualization based on column analysis with better styling and interactivity"""
    
    # Set container for the visualization
    target_container = container if container else st
    
    if analysis["viz_type"] == "none":
        target_container.warning(f"No visualization available for empty column: {column_name}")
        return None
    
    col_data = df[column_name].dropna()
    
    # Common plot styling
    plot_layout = {
        "margin": dict(l=10, r=10, t=50, b=10),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif", size=12),
        "title": dict(
            text=f"<b>{column_name}</b>",
            font=dict(size=16, family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif", color="#2c3e50")
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        "height": height
    }
    
    if width:
        plot_layout["width"] = width
    
    # Initialize the visualization variable
    visualization = None
    
    if analysis["viz_type"] == "pie_chart":
        # Get value counts and calculate percentages
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
        
        # Use Plotly Express for better pie chart
        fig = px.pie(
            value_counts, 
            values='Count', 
            names=column_name,
            title=f"<b>Distribution of {column_name}</b>",
            hover_data=['Percentage'],
            color_discrete_sequence=px.colors.qualitative.G10,
            labels={column_name: column_name, 'Count': 'Count'}
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=1.5)),
            pull=[0.05 if i == 0 else 0 for i in range(len(value_counts))],  # Pull out the first slice
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>'
        )
        
        fig.update_layout(**plot_layout)
        visualization = fig
    
    elif analysis["viz_type"] == "donut_chart":
        # Get value counts and calculate percentages
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
        
        # Create donut chart with a hole
        fig = px.pie(
            value_counts, 
            values='Count', 
            names=column_name,
            title=f"<b>Distribution of {column_name}</b>",
            hover_data=['Percentage'],
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.6,  # This creates the donut effect
            labels={column_name: column_name, 'Count': 'Count'}
        )
        
        # Add the total count in the center
        total = value_counts['Count'].sum()
        fig.add_annotation(
            text=f"<b>Total</b><br>{total:,}",
            x=0.5, y=0.5,
            font_size=15,
            showarrow=False
        )
        
        fig.update_traces(
            textposition='outside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2)),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>'
        )
        
        fig.update_layout(**plot_layout)
        visualization = fig
    
    elif analysis["viz_type"] == "bar_chart":
        # Get value counts
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        
        # Sort by count descending
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 categories if there are many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f"<b>Top 20 {column_name} by Count</b>"
        else:
            title = f"<b>Distribution of {column_name}</b>"
        
        # Create color gradient based on values
        gradient_colors = px.colors.sequential.Blues[-len(value_counts):] if len(value_counts) <= 10 else px.colors.sequential.Blues_r
        
        # Enhanced bar chart with better styling
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=title,
            text='Count',
            color='Count',
            color_continuous_scale=gradient_colors,
            opacity=0.9,
            category_orders={column_name: value_counts[column_name].tolist()}
        )
        
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            marker_line_width=1,
            marker_line_color='white'
        )
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 5:
            fig.update_layout(xaxis_tickangle=-45)
            
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title="", tickfont=dict(size=11)),
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
            coloraxis_showscale=False
        )
        
        visualization = fig
    
    elif analysis["viz_type"] == "histogram":
        # Enhanced histogram with overlay elements
        if len(col_data.unique()) <= 10:
            # For few unique values, use bar chart instead
            distinct_values = sorted(col_data.unique())
            counts = [len(col_data[col_data == val]) for val in distinct_values]
            
            fig = px.bar(
                x=[str(val) for val in distinct_values],
                y=counts,
                labels={"x": column_name, "y": "Count"},
                title=f"<b>Distribution of {column_name}</b>"
            )
            
            fig.update_layout(**plot_layout)
            visualization = fig
        else:
            # Calculate optimal bin count based on data size
            bin_count = min(50, max(10, int(np.sqrt(len(col_data)))))
            
            # Calculate mean and median to show on the plot
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            # Create enhanced histogram
            fig = px.histogram(
                df, 
                x=column_name,
                title=f"<b>Distribution of {column_name}</b>",
                marginal="box",  # Add a box plot at the margin
                color_discrete_sequence=['#3498db'],
                opacity=0.8,
                nbins=bin_count,
                histnorm='percent'  # Use percentage for better interpretation
            )
            
            # Add mean and median lines
            fig.add_vline(
                x=mean_val, 
                line_dash="solid", 
                line_color="#e74c3c", 
                line_width=2,
                annotation_text=f"Mean: {mean_val:.2f}", 
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c")
            )
            
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="#2ecc71", 
                line_width=2,
                annotation_text=f"Median: {median_val:.2f}", 
                annotation_position="bottom right",
                annotation_font=dict(color="#2ecc71")
            )
            
            fig.update_layout(
                **plot_layout,
                xaxis=dict(
                    title=column_name, 
                    tickfont=dict(size=11),
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                ),
                yaxis=dict(
                    title="Percentage (%)", 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                )
            )
            
            # Adjust hover template
            fig.update_traces(
                hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"
            )
            
            visualization = fig
    
    elif analysis["viz_type"] == "box_plot":
        # Create enhanced box plot with violin overlay
        fig = go.Figure()
        
        # Add box plot
        fig.add_trace(go.Box(
            y=col_data,
            name=column_name,
            marker_color='#3498db',
            line=dict(color='#2c3e50'),
            boxmean=True,  # Show mean as a dashed line
            boxpoints='outliers',  # Only show outliers as individual points
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                size=4,
                color='#e74c3c',
                opacity=0.7
            )
        ))
        
        # Get statistics for annotations
        q1 = col_data.quantile(0.25)
        median = col_data.median()
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        # Add a violin plot for density visualization (semi-transparent)
        if len(col_data) > 30:
            fig.add_trace(go.Violin(
                y=col_data,
                name=column_name,
                side='right',
                line=dict(color='rgba(52, 152, 219, 0.2)'),
                fillcolor='rgba(52, 152, 219, 0.1)',
                marker=dict(opacity=0),
                showlegend=False,
                meanline=dict(visible=False),
                points=False
            ))
        
        fig.update_layout(
            **plot_layout,
            title=dict(text=f"<b>Distribution of {column_name}</b>"),
            yaxis=dict(
                title=column_name, 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)'
            ),
            xaxis=dict(
                title="", 
                showticklabels=False
            ),
            showlegend=False
        )
        
        visualization = fig
        
        # Show additional statistics below the chart
        with target_container.expander("Distribution Statistics", expanded=False):
            stats_cols = target_container.columns(4)
            stats_cols[0].metric("Median", f"{median:.2f}")
            stats_cols[1].metric("Q1 (25%)", f"{q1:.2f}")
            stats_cols[2].metric("Q3 (75%)", f"{q3:.2f}")
            stats_cols[3].metric("IQR", f"{iqr:.2f}")
            
            stats_cols2 = target_container.columns(3)
            stats_cols2[0].metric("Outliers", f"{len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)")
            stats_cols2[1].metric("Min", f"{col_data.min():.2f}")
            stats_cols2[2].metric("Max", f"{col_data.max():.2f}")
    
    elif analysis["viz_type"] == "violin_plot":
        # Create combined violin and box plot for detailed distribution
        fig = go.Figure()
        
        # Add violin plot
        fig.add_trace(go.Violin(
            y=col_data,
            name=column_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor='#3498db',
            opacity=0.6,
            line=dict(color='#2c3e50'),
            marker=dict(
                size=2,
                opacity=0.5
            ),
            side='both',
            points='all',
            pointpos=0,
            jitter=0.7,
            width=2
        ))
        
        fig.update_layout(
            **plot_layout,
            title=dict(text=f"<b>Distribution of {column_name}</b>"),
            yaxis=dict(
                title=column_name, 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)'
            ),
            xaxis=dict(
                title="", 
                showticklabels=False
            ),
            showlegend=False
        )
        
        visualization = fig
    
    elif analysis["viz_type"] == "kde_plot":
        # Create KDE plot with rug plot for detailed distribution
        try:
            # Use Seaborn for KDE calculation
            kde_data = sns.kdeplot(col_data).get_lines()[0].get_data()
            x_vals, y_vals = kde_data
            
            fig = go.Figure()
            
            # Add the KDE curve
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                fill='tozeroy',
                name='Density',
                line=dict(color='#3498db', width=2),
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            # Add rug plot (1D scatter) at the bottom
            fig.add_trace(go.Scatter(
                x=col_data,
                y=np.zeros(len(col_data)),
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=8,
                    color='#2c3e50',
                    opacity=0.5
                ),
                name='Observations'
            ))
            
            # Add mean and median lines
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="solid", 
                line_color="#e74c3c", 
                line_width=2,
                annotation_text=f"Mean: {mean_val:.2f}", 
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c")
            )
            
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="#2ecc71", 
                line_width=2,
                annotation_text=f"Median: {median_val:.2f}", 
                annotation_position="bottom right",
                annotation_font=dict(color="#2ecc71")
            )
            
            fig.update_layout(
                **plot_layout,
                title=dict(text=f"<b>Density Plot of {column_name}</b>"),
                xaxis=dict(
                    title=column_name, 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                ),
                yaxis=dict(
                    title="Density", 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                ),
                showlegend=False
            )
            
            visualization = fig
        except:
            # Fallback to histogram if KDE fails
            visualization = create_enhanced_visualization(
                df, column_name, {"viz_type": "histogram", "type": "numeric"}, 
                container=target_container, key_suffix=key_suffix, height=height, width=width
            )
    
    elif analysis["viz_type"] == "time_series":
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(col_data):
            try:
                col_data = pd.to_datetime(col_data)
                if isinstance(df, pd.DataFrame):
                    df[column_name] = col_data
            except:
                target_container.warning(f"Could not convert {column_name} to datetime format.")
                return None
        
        # Group by date components based on date range
        date_range = (col_data.max() - col_data.min()).days
        
        # Choose appropriate time grouping based on date range
        if date_range > 365 * 5:  # More than 5 years
            time_grouped = df.assign(date_group=col_data.dt.to_period('Y')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Year'
        elif date_range > 365 * 2:  # 2-5 years
            time_grouped = df.assign(date_group=col_data.dt.to_period('Q')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Quarter'
        elif date_range > 120:  # 4+ months
            time_grouped = df.assign(date_group=col_data.dt.to_period('M')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Month'
        elif date_range > 30:  # 1+ month
            time_grouped = df.assign(date_group=col_data.dt.to_period('W')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Week'
        else:  # Less than a month
            time_grouped = df.assign(date_group=col_data.dt.to_period('D')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Day'
        
        # Calculate trend using moving average
        window_size = max(2, min(5, len(time_df) // 5))
        if len(time_df) > window_size:
            time_df['Moving_Avg'] = time_df['Count'].rolling(window=window_size, min_periods=1).mean()
        
        # Create enhanced time series with multiple elements
        fig = go.Figure()
        
        # Add the main line with markers
        fig.add_trace(go.Scatter(
            x=time_df['Date'],
            y=time_df['Count'],
            mode='lines+markers',
            name='Count',
            line=dict(color='#3498db', width=2),
            marker=dict(size=7, color='#3498db'),
            hovertemplate='Date: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add moving average trend line if available
        if 'Moving_Avg' in time_df.columns:
            fig.add_trace(go.Scatter(
                x=time_df['Date'],
                y=time_df['Moving_Avg'],
                mode='lines',
                name=f'{window_size}-point Moving Average',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Moving Avg: %{y:.1f}<extra></extra>'
            ))
        
        # Add linear trend line
        try:
            import numpy as np
            from scipy import stats
            
            # Convert dates to ordinal numbers for linear regression
            x_numeric = np.array([(date - time_df['Date'].min()).total_seconds() for date in time_df['Date']])
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, time_df['Count'])
            
            trend_y = intercept + slope * x_numeric
            
            fig.add_trace(go.Scatter(
                x=time_df['Date'],
                y=trend_y,
                mode='lines',
                name='Linear Trend',
                line=dict(color='#2ecc71', width=2, dash='dot'),
                hoverinfo='skip'
            ))
            
            # Calculate trend direction and percentage
            first_count = trend_y[0]
            last_count = trend_y[-1]
            if first_count != 0:
                trend_pct = (last_count - first_count) / first_count * 100
            else:
                trend_pct = 0
                
            # Add annotation about trend
            trend_direction = "Increasing" if trend_pct > 0 else "Decreasing" if trend_pct < 0 else "Stable"
            trend_color = "#2ecc71" if trend_pct > 0 else "#e74c3c" if trend_pct < 0 else "#7f8c8d"
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"{trend_direction} trend: {abs(trend_pct):.1f}%",
                showarrow=False,
                font=dict(color=trend_color, size=12),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=trend_color,
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
        except:
            # Skip trend line if regression fails
            pass
        
        fig.update_layout(
            **plot_layout,
            title=dict(text=f"<b>Time Series of {column_name}</b>"),
            xaxis=dict(
                title=x_title, 
                tickfont=dict(size=11),
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)'
            ),
            yaxis=dict(
                title="Count", 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)'
            ),
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        visualization = fig
        
        # Show time period statistics
        with target_container.expander("Time Series Statistics", expanded=False):
            time_stats_cols = target_container.columns(4)
            time_stats_cols[0].metric("Total Records", time_df['Count'].sum())
            
            # Find peak period
            peak_idx = time_df['Count'].idxmax()
            peak_date = time_df.loc[peak_idx, 'Date']
            time_stats_cols[1].metric("Peak Period", peak_date.strftime('%Y-%m-%d'))
            time_stats_cols[2].metric("Peak Count", int(time_df.loc[peak_idx, 'Count']))
            
            # Calculate average per period
            avg_per_period = time_df['Count'].mean()
            time_stats_cols[3].metric("Avg per Period", f"{avg_per_period:.1f}")
    
    elif analysis["viz_type"] == "scatter_plot":
        # Find a good pair column for scatter plot
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != column_name]
        
        if numeric_cols:
            # Select best correlated column if possible
            best_corr = 0
            best_col = numeric_cols[0]
            
            for col in numeric_cols[:10]:  # Check only first 10 to save time
                try:
                    corr = abs(df[[column_name, col]].corr().iloc[0, 1])
                    if not np.isnan(corr) and corr > best_corr:
                        best_corr = corr
                        best_col = col
                except:
                    continue
            
            pair_col = best_col
            
            # Create enhanced scatter plot
            fig = px.scatter(
                df, 
                x=column_name, 
                y=pair_col,
                title=f"<b>Relationship between {column_name} and {pair_col}</b>",
                opacity=0.7,
                color_discrete_sequence=['#3498db'],
                trendline="ols",
                trendline_color_override="#e74c3c",
                marginal_x="histogram",
                marginal_y="histogram"
            )
            
            # Add correlation annotation
            corr = df[[column_name, pair_col]].corr().iloc[0, 1]
            corr_color = "#2ecc71" if abs(corr) > 0.7 else "#f39c12" if abs(corr) > 0.3 else "#e74c3c"
            
            fig.add_annotation(
                x=0.02,
                y=0.02,
                xref="paper",
                yref="paper",
                text=f"Correlation: {corr:.3f}",
                showarrow=False,
                font=dict(color=corr_color, size=12),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=corr_color,
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
            
            fig.update_layout(
                **plot_layout,
                xaxis=dict(
                    title=column_name, 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                ),
                yaxis=dict(
                    title=pair_col, 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(211,211,211,0.3)'
                ),
                hovermode='closest'
            )
            
            fig.update_traces(
                marker=dict(
                    size=8,
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                selector=dict(mode='markers')
            )
            
            visualization = fig
        else:
            # Fallback to histogram if no numeric columns to pair with
            visualization = create_enhanced_visualization(
                df, column_name, {"viz_type": "histogram", "type": "numeric"}, 
                container=target_container, key_suffix=key_suffix, height=height, width=width
            )
    
    elif analysis["viz_type"] == "choropleth_map":
        # For demonstrations, let's create a simple choropleth map fallback
        # In a real implementation, this would use geospatial data
        target_container.warning("Choropleth map visualization would typically require spatial data and libraries like pydeck or plotly's geo capabilities.")
        
        # Show bar chart as fallback
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 locations
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f"<b>Top 20 {column_name} by Count</b>"
        else:
            title = f"<b>Distribution of {column_name}</b>"
            
        # Create a geographic bar chart with different colors
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=title,
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title="", tickfont=dict(size=11), tickangle=-45),
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
            coloraxis_showscale=False
        )
        
        visualization = fig
    
    elif analysis["viz_type"] == "point_map":
        # Similar fallback as choropleth_map
        return create_enhanced_visualization(
            df, column_name, {"viz_type": "choropleth_map"}, 
            container=target_container, key_suffix=key_suffix, height=height, width=width
        )
    
    elif analysis["viz_type"] == "treemap":
        # Get value counts
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        
        # Create colorful treemap
        fig = px.treemap(
            value_counts,
            path=[column_name],
            values='Count',
            title=f"<b>Distribution of {column_name}</b>",
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=['Count']
        )
        
        fig.update_traces(
            textinfo="label+value+percent",
            textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
        )
        
        fig.update_layout(**plot_layout)
        visualization = fig
    
    elif analysis["viz_type"] == "word_cloud":
        # Create word cloud visualization
        try:
            # Generate word cloud from text data
            text_data = " ".join(col_data.astype(str).tolist())
            
            # Generate word frequencies
            from collections import Counter
            word_freq = Counter(text_data.split())
            top_words = dict(word_freq.most_common(50))
            
            # Create bar chart of top words as fallback
            top_words_df = pd.DataFrame({
                'Word': list(top_words.keys()),
                'Frequency': list(top_words.values())
            }).sort_values('Frequency', ascending=False)
            
            fig = px.bar(
                top_words_df,
                x='Word',
                y='Frequency',
                title=f"<b>Top 50 Words in {column_name}</b>",
                color='Frequency',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                **plot_layout,
                xaxis=dict(title="", tickfont=dict(size=11), tickangle=-45),
                yaxis=dict(title="Frequency", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
                coloraxis_showscale=False
            )
            
            visualization = fig
            
            # Display actual word cloud in expander
            with target_container.expander("View Word Cloud", expanded=False):
                if len(text_data) > 0:
                    try:
                        # Generate WordCloud
                        wordcloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            contour_width=1,
                            contour_color='steelblue'
                        ).generate(text_data)
                        
                        # Create matplotlib figure
                        fig_wc, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        ax.set_title(f"Word Cloud of {column_name}", fontsize=14)
                        
                        target_container.pyplot(fig_wc)
                    except:
                        target_container.warning("Could not generate word cloud visualization.")
        except:
            # Fallback to showing top values as enhanced bar chart
            value_counts = col_data.value_counts().reset_index()
            value_counts.columns = [column_name, 'Count']
            value_counts = value_counts.sort_values('Count', ascending=False).head(20)
            
            # Text data bar chart with color gradient
            fig = px.bar(
                value_counts, 
                x=column_name, 
                y='Count',
                title=f"<b>Top 20 values in {column_name}</b>",
                color='Count',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                **plot_layout,
                xaxis=dict(title="", tickfont=dict(size=11), tickangle=-45),
                yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
                coloraxis_showscale=False
            )
            
            visualization = fig
    
    elif analysis["viz_type"] == "table":
        # Show top values and their counts in a styled table
        value_counts = col_data.value_counts().nlargest(20).reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
        
        target_container.markdown(f"<div class='chart-title'>Top 20 values in {column_name}</div>", unsafe_allow_html=True)
        target_container.dataframe(value_counts, use_container_width=True, height=min(400, len(value_counts) * 35 + 38))
        return None  # Already displayed the data
    
    # Display the visualization
    if visualization:
        target_container.plotly_chart(visualization, use_container_width=True, config={'displayModeBar': True}, key=f"{column_name}_{key_suffix}")
    
    return visualization

# Function to create an enhanced correlation matrix
def create_enhanced_correlation_heatmap(df, container=None):
    """Creates a sophisticated correlation heatmap with annotations and insights"""
    target_container = container if container else st
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().round(2)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create enhanced heatmap with annotations
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu_r,
            title="<b>Correlation Heatmap for Numeric Variables</b>",
            color_continuous_midpoint=0,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
        )
        
        # Hide upper triangle
        for i, col in enumerate(corr_matrix.columns):
            for j, row in enumerate(corr_matrix.columns):
                if j > i:  # This is the upper triangle
                    fig.data[0].z[j][i] = None
                    fig.data[0].text[j][i] = None
        
        fig.update_layout(
            height=max(400, len(numeric_cols) * 25 + 100),
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(font=dict(size=18, family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif")),
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            xaxis=dict(tickangle=-45),
            coloraxis=dict(
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    tickmode="array",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0<br>(Perfect<br>Negative)", "-0.5", "0.0", "0.5", "1.0<br>(Perfect<br>Positive)"],
                    ticks="outside"
                )
            )
        )
        
        target_container.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Find and display strongest correlations
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
        
        if corr_pairs:
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Display top correlations with better styling
            top_n = min(10, len(corr_pairs))
            target_container.markdown("<div class='chart-title'>Top Correlations</div>", unsafe_allow_html=True)
            
            corr_df = pd.DataFrame(corr_pairs[:top_n], columns=['Variable 1', 'Variable 2', 'Correlation'])
            
            # Add correlation strength category
            def get_strength(val):
                abs_val = abs(val)
                if abs_val > 0.7:
                    return "Strong"
                elif abs_val > 0.3:
                    return "Moderate"
                else:
                    return "Weak"
            
            def get_color(val):
                abs_val = abs(val)
                if abs_val > 0.7:
                    return "#2ecc71" if val > 0 else "#e74c3c"
                elif abs_val > 0.3:
                    return "#27ae60" if val > 0 else "#c0392b"
                else:
                    return "#7f8c8d"
            
            corr_df['Strength'] = corr_df['Correlation'].apply(get_strength)
            corr_df['Direction'] = corr_df['Correlation'].apply(lambda x: "Positive" if x > 0 else "Negative")
            
            # Create styled dataframe
            corr_df_styled = corr_df.style.format({'Correlation': '{:.3f}'})
            
            # Apply color based on correlation value
            corr_df_styled = corr_df_styled.applymap(
                lambda val: f'color: {get_color(val)}; font-weight: bold',
                subset=['Correlation']
            )
            
            target_container.dataframe(corr_df_styled, use_container_width=True)
            
            # Add insights about strongest correlations
            if len(corr_pairs) > 0 and abs(corr_pairs[0][2]) > 0.5:
                strongest_pair = corr_pairs[0]
                corr_direction = "positive" if strongest_pair[2] > 0 else "negative"
                
                target_container.markdown(f"""
                <div class='info-box'>
                    <b>Insight:</b> The strongest relationship is between <b>{strongest_pair[0]}</b> and <b>{strongest_pair[1]}</b> 
                    with a {corr_direction} correlation of <b>{strongest_pair[2]:.2f}</b>. 
                    This suggests that they tend to {corr_direction == 'positive' and 'increase' or 'decrease'} together.
                </div>
                """, unsafe_allow_html=True)
    else:
        target_container.info("Not enough numeric columns to create a correlation heatmap. Upload data with at least two numeric columns to see correlations.")

# Function to create a dashboard for statistical overview
def create_statistical_overview(df, column_analyses, container=None):
    """Creates a statistical overview dashboard with key metrics and insights"""
    target_container = container if container else st
    
    # Get numerical and categorical columns
    numeric_cols = [col for col, analysis in column_analyses.items() 
                   if analysis["type"] == "numeric"]
    
    cat_cols = [col for col, analysis in column_analyses.items() 
               if analysis["type"] == "categorical"]
    
    # Create container with title
    target_container.markdown("<h3>Statistical Overview</h3>", unsafe_allow_html=True)
    
    # If we have numeric columns
    if numeric_cols:
        target_container.markdown("<div class='chart-title'>Numeric Columns Summary</div>", unsafe_allow_html=True)
        
        # Create a dataframe with key statistics
        numeric_stats = {
            'Column': [],
            'Min': [],
            'Max': [],
            'Mean': [],
            'Median': [],
            'Std Dev': [],
            'Missing %': []
        }
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            
            if len(col_data) > 0:
                numeric_stats['Column'].append(col)
                numeric_stats['Min'].append(round(col_data.min(), 2))
                numeric_stats['Max'].append(round(col_data.max(), 2))
                numeric_stats['Mean'].append(round(col_data.mean(), 2))
                numeric_stats['Median'].append(round(col_data.median(), 2))
                numeric_stats['Std Dev'].append(round(col_data.std(), 2))
                numeric_stats['Missing %'].append(round(missing_pct, 2))
        
        # Create styled dataframe
        numeric_df = pd.DataFrame(numeric_stats)
        numeric_styled = numeric_df.style.background_gradient(subset=['Missing %'], cmap='YlOrRd')
        
        # Display the table
        target_container.dataframe(numeric_styled, use_container_width=True)
    
    # If we have categorical columns
    if cat_cols:
        target_container.markdown("<div class='chart-title'>Categorical Columns Summary</div>", unsafe_allow_html=True)
        
        # Create a dataframe with key statistics
        cat_stats = {
            'Column': [],
            'Unique Values': [],
            'Most Common': [],
            'Most Common %': [],
            'Missing %': []
        }
        
        for col in cat_cols:
            col_data = df[col].dropna()
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            
            if len(col_data) > 0:
                # Get value counts
                val_counts = col_data.value_counts()
                top_val = val_counts.index[0] if len(val_counts) > 0 else "N/A"
                top_pct = (val_counts.iloc[0] / len(col_data) * 100) if len(val_counts) > 0 else 0
                
                cat_stats['Column'].append(col)
                cat_stats['Unique Values'].append(len(col_data.unique()))
                cat_stats['Most Common'].append(str(top_val)[:20])  # Truncate long values
                cat_stats['Most Common %'].append(round(top_pct, 2))
                cat_stats['Missing %'].append(round(missing_pct, 2))
        
        # Create styled dataframe
        cat_df = pd.DataFrame(cat_stats)
        cat_styled = cat_df.style.background_gradient(subset=['Missing %'], cmap='YlOrRd')
        
        # Display the table
        target_container.dataframe(cat_styled, use_container_width=True)
    
    # Add overall dataset statistics
    target_container.markdown("<div class='chart-title'>Dataset Overview</div>", unsafe_allow_html=True)
    
    # Create metrics in columns
    col1, col2, col3, col4 = target_container.columns(4)
    
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    
    # Calculate completeness - % of non-null values
    completeness = 100 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    col3.metric("Completeness", f"{completeness:.1f}%")
    
    # Calculate duplicate percentage
    duplicate_pct = df.duplicated().mean() * 100
    col4.metric("Unique Rows", f"{100-duplicate_pct:.1f}%")
    
    # Add data insights
    if df.shape[0] > 0:
        target_container.markdown("<div class='chart-title'>Quick Insights</div>", unsafe_allow_html=True)
        
        insights = []
        
        # Missing data insight
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            most_missing_col = df.isna().mean().idxmax()
            most_missing_pct = df[most_missing_col].isna().mean() * 100
            if most_missing_pct > 5:
                insights.append(f"<li><b>{most_missing_col}</b> has the most missing values ({most_missing_pct:.1f}% missing).</li>")
        
        # Skewed data insight
        if numeric_cols:
            most_skewed_col = None
            max_skew = 0
            for col in numeric_cols:
                try:
                    skew = abs(stats.skew(df[col].dropna()))
                    if skew > max_skew and not np.isnan(skew) and np.isfinite(skew):
                        max_skew = skew
                        most_skewed_col = col
                except:
                    continue
            
            if most_skewed_col and max_skew > 1:
                insights.append(f"<li><b>{most_skewed_col}</b> has a highly skewed distribution (skew: {max_skew:.2f}).</li>")
        
        # Highest correlation insight
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            highest_corr = 0
            corr_pair = None
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > highest_corr and not np.isnan(corr):
                        highest_corr = corr
                        corr_pair = (numeric_cols[i], numeric_cols[j])
            
            if corr_pair and highest_corr > 0.5:
                insights.append(f"<li><b>{corr_pair[0]}</b> and <b>{corr_pair[1]}</b> have the strongest correlation ({highest_corr:.2f}).</li>")
        
        # Categorical imbalance insight
        if cat_cols:
            for col in cat_cols[:3]:  # Only check first few columns
                try:
                    val_counts = df[col].value_counts()
                    top_val = val_counts.index[0]
                    top_pct = val_counts.iloc[0] / val_counts.sum() * 100
                    
                    if top_pct > 75 and len(val_counts) > 1:
                        insights.append(f"<li><b>{col}</b> is imbalanced with <b>{top_val}</b> representing {top_pct:.1f}% of values.</li>")
                except:
                    continue
        
        # Display insights
        if insights:
            target_container.markdown(f"""
            <div class="info-box">
                <ul>{''.join(insights)}</ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            target_container.info("No significant insights detected in the data.")

# Function to create enhanced data quality assessment
def create_enhanced_data_quality(df, column_analyses, container=None):
    """Creates a comprehensive data quality dashboard with metrics and recommendations"""
    target_container = container if container else st
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_values_pct = (missing_values / len(df)) * 100
    missing_values_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': missing_values_pct.values
    })
    missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    outlier_info = []
    
    if numeric_cols:
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if not outliers.empty:
                    outlier_info.append({
                        'Column': col,
                        'Outliers': len(outliers),
                        'Percentage': round(len(outliers) / len(df) * 100, 2)
                    })
            except:
                continue
    
    # Calculate data quality scores
    # 1. Completeness score (based on missing values)
    completeness_score = 100 - (missing_values.sum() / (df.shape[0] * df.shape[1]) * 100)
    
    # 2. Uniqueness score (based on duplicates)
    uniqueness_score = 100 - duplicate_pct
    
    # 3. Consistency score (based on outliers)
    outlier_count = sum(info['Outliers'] for info in outlier_info)
    if numeric_cols:
        consistency_score = 100 - (outlier_count / (len(df) * len(numeric_cols)) * 100)
    else:
        consistency_score = 100
    
    # Overall quality score (weighted average)
    quality_score = (completeness_score * 0.4 + uniqueness_score * 0.3 + consistency_score * 0.3)
    
    # Color coding based on score
    if quality_score >= 80:
        quality_class = "quality-score-good"
        card_class = "quality-good"
    elif quality_score >= 60:
        quality_class = "quality-score-warning"
        card_class = "quality-warning"
    else:
        quality_class = "quality-score-bad"
        card_class = "quality-bad"
    
    # Display quality dashboard
    target_container.markdown("<h2>Data Quality Assessment</h2>", unsafe_allow_html=True)
    
    # Create quality score card
    target_container.markdown(f"""
    <div class="quality-card {card_class}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3>Data Quality Score</h3>
                <p>Based on completeness, uniqueness, and consistency</p>
            </div>
            <div class="quality-score {quality_class}">
                {quality_score:.1f}%
            </div>
        </div>
        
        <div style="display: flex; margin-top: 20px; gap: 20px;">
            <div style="flex: 1; text-align: center;">
                <h4>Completeness</h4>
                <div style="font-size: 1.5rem; font-weight: 500; color: {completeness_score >= 90 and '#2ecc71' or completeness_score >= 70 and '#f39c12' or '#e74c3c'};">
                    {completeness_score:.1f}%
                </div>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <h4>Uniqueness</h4>
                <div style="font-size: 1.5rem; font-weight: 500; color: {uniqueness_score >= 90 and '#2ecc71' or uniqueness_score >= 70 and '#f39c12' or '#e74c3c'};">
                    {uniqueness_score:.1f}%
                </div>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <h4>Consistency</h4>
                <div style="font-size: 1.5rem; font-weight: 500; color: {consistency_score >= 90 and '#2ecc71' or consistency_score >= 70 and '#f39c12' or '#e74c3c'};">
                    {consistency_score:.1f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display quality details
    quality_tabs = target_container.tabs(["Missing Values", "Duplicates", "Outliers", "Recommendations"])
    
    # Missing Values Tab
    with quality_tabs[0]:
        if not missing_values_df.empty:
            target_container.markdown("<div class='chart-title'>Columns with Missing Values</div>", unsafe_allow_html=True)
            
            # Create styled dataframe with missing values
            missing_df_styled = missing_values_df.style.format({'Percentage': '{:.2f}%'})
            missing_df_styled = missing_df_styled.background_gradient(subset=['Percentage'], cmap='YlOrRd')
            
            target_container.dataframe(missing_df_styled, use_container_width=True)
            
            # Visualize missing values
            if len(missing_values_df) > 0:
                fig = px.bar(
                    missing_values_df,
                    x='Column',
                    y='Percentage',
                    title="<b>Missing Values by Column</b>",
                    color='Percentage',
                    color_continuous_scale='YlOrRd',
                    labels={'Percentage': 'Missing (%)'}
                )
                
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Missing Values (%)",
                    xaxis_tickangle=-45,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                target_container.plotly_chart(fig, use_container_width=True)
                
                # Add recommendations for handling missing values
                target_container.markdown("<div class='chart-title'>Recommendations for Missing Values</div>", unsafe_allow_html=True)
                
                recommendations = []
                
                # For columns with high missingness
                high_missing_cols = missing_values_df[missing_values_df['Percentage'] > 15]['Column'].tolist()
                if high_missing_cols:
                    high_missing_text = ", ".join([f"<b>{col}</b>" for col in high_missing_cols[:3]])
                    if len(high_missing_cols) > 3:
                        high_missing_text += f" and {len(high_missing_cols) - 3} more"
                    recommendations.append(f"<li>Consider dropping or imputing columns with high missing rates: {high_missing_text}</li>")
                
                # For columns with moderate missingness
                mod_missing_cols = missing_values_df[(missing_values_df['Percentage'] <= 15) & (missing_values_df['Percentage'] > 5)]['Column'].tolist()
                if mod_missing_cols:
                    # Generate imputation recommendations based on column type
                    for col in mod_missing_cols[:3]:  # Limit to 3 examples
                        col_type = column_analyses.get(col, {}).get("type", "unknown")
                        if col_type == "numeric":
                            recommendations.append(f"<li>For numeric column <b>{col}</b>, consider imputing with mean, median, or interpolation</li>")
                        elif col_type == "categorical":
                            recommendations.append(f"<li>For categorical column <b>{col}</b>, consider imputing with mode or a special 'Missing' category</li>")
                        elif col_type == "datetime":
                            recommendations.append(f"<li>For datetime column <b>{col}</b>, consider forward/backward fill or interpolation</li>")
                
                # General recommendation
                if len(missing_values_df) > 0:
                    recommendations.append("<li>Check if missing values are random or follow a pattern that could bias your analysis</li>")
                
                # Display recommendations
                if recommendations:
                    target_container.markdown(f"""
                    <div class="recommendation-card">
                        <ul>{''.join(recommendations)}</ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            target_container.success("✅ No missing values found in the dataset! Your data is complete.")
    
    # Duplicates Tab
    with quality_tabs[1]:
        target_container.markdown("<div class='chart-title'>Duplicate Row Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = target_container.columns(2)
        
        with col1:
            # Create a metric for duplicate count
            target_container.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Duplicate Rows</div>
                <div class="kpi-value">{duplicate_count:,}</div>
                <div class="kpi-title">({duplicate_pct:.2f}% of data)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a gauge chart for duplicate percentage
            if duplicate_pct > 0:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=duplicate_pct,
                    title={'text': "Duplicate Percentage"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#e74c3c" if duplicate_pct > 10 else "#f39c12" if duplicate_pct > 5 else "#2ecc71"},
                        'steps': [
                            {'range': [0, 5], 'color': 'rgba(46, 204, 113, 0.3)'},
                            {'range': [5, 10], 'color': 'rgba(243, 156, 18, 0.3)'},
                            {'range': [10, 100], 'color': 'rgba(231, 76, 60, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif")
                )
                
                target_container.plotly_chart(fig, use_container_width=True)
            else:
                target_container.success("No duplicates found in the dataset!")
        
        # Show duplicate rows if any
        if duplicate_count > 0:
            # Check for duplicates
            duplicates = df[df.duplicated(keep=False)]
            
            if len(duplicates) > 0:
                target_container.markdown("<div class='chart-title'>Sample Duplicate Rows</div>", unsafe_allow_html=True)
                
                # Count duplicates by group
                dup_counts = duplicates.groupby(list(duplicates.columns)).size().reset_index(name='Count')
                dup_counts = dup_counts.sort_values('Count', ascending=False)
                
                # Get sample duplicate groups
                sample_size = min(10, len(dup_counts))
                duplicates_sample = pd.DataFrame()
                
                for i in range(sample_size):
                    if i < len(dup_counts):
                        # Get filter conditions for the current group
                        conditions = True
                        for col in duplicates.columns:
                            conditions = conditions & (duplicates[col] == dup_counts.iloc[i][col])
                        
                        # Add one row from each duplicate group
                        group_sample = duplicates[conditions].head(1)
                        group_sample['Occurrences'] = dup_counts.iloc[i]['Count']
                        duplicates_sample = pd.concat([duplicates_sample, group_sample])
                
                # Display the sample with styling
                target_container.dataframe(duplicates_sample, use_container_width=True)
                
                # Recommendations for handling duplicates
                target_container.markdown("<div class='chart-title'>Recommendations for Duplicates</div>", unsafe_allow_html=True)
                
                target_container.markdown(f"""
                <div class="recommendation-card">
                    <ul>
                        <li>Consider removing duplicate rows to prevent bias in your analysis</li>
                        <li>Check if duplicates have legitimate reasons (e.g., repeated measurements)</li>
                        <li>Investigate the source of duplicates to prevent future data quality issues</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Add deduplicate button (in a real app this would do something)
                if target_container.button("Remove Duplicates", key="dedup_button"):
                    target_container.info("In a production app, this would remove the duplicates from the dataframe. For demonstration, no changes were made.")
        else:
            target_container.success("✅ No duplicate rows found in the dataset! Your data is clean.")
    
    # Outliers Tab
    with quality_tabs[2]:
        if numeric_cols:
            if outlier_info:
                # Create dataframe with outlier information
                outlier_df = pd.DataFrame(outlier_info)
                
                target_container.markdown("<div class='chart-title'>Columns with Outliers</div>", unsafe_allow_html=True)
                
                # Style the dataframe
                outlier_df_styled = outlier_df.style.format({'Percentage': '{:.2f}%'})
                outlier_df_styled = outlier_df_styled.background_gradient(subset=['Percentage'], cmap='YlOrRd')
                
                target_container.dataframe(outlier_df_styled, use_container_width=True)
                
                # Visualize outliers
                fig = px.bar(
                    outlier_df,
                    x='Column',
                    y='Percentage',
                    title="<b>Outliers by Column</b>",
                    color='Percentage',
                    text='Outliers',
                    color_continuous_scale='YlOrRd',
                    labels={'Percentage': 'Outliers (%)'}
                )
                
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Outliers (%)",
                    xaxis_tickangle=-45,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                target_container.plotly_chart(fig, use_container_width=True)
                
                # Create interactive outlier viewer
                target_container.markdown("<div class='chart-title'>Explore Outliers</div>", unsafe_allow_html=True)
                
                # Select column to explore
                selected_col = target_container.selectbox(
                    "Select column to explore outliers",
                    options=outlier_df['Column'].tolist(),
                    key="outlier_col_select"
                )
                
                # Create box plot for selected column
                if selected_col:
                    col_data = df[selected_col].dropna()
                    
                    # Calculate quartiles and IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Identify outliers
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    # Create figure with box plot and scatter plot overlay
                    fig = go.Figure()
                    
                    # Add box plot
                    fig.add_trace(go.Box(
                        y=col_data,
                        name='Distribution',
                        boxpoints=False,  # Hide all individual points
                        marker_color='#3498db',
                        line=dict(color='#2c3e50')
                    ))
                    
                    # Add scatter plot for outliers
                    if len(outliers) > 0:
                        fig.add_trace(go.Scatter(
                            y=outliers,
                            x=[0] * len(outliers),  # All points at x=0
                            mode='markers',
                            name='Outliers',
                            marker=dict(
                                color='#e74c3c',
                                size=8,
                                symbol='circle',
                                line=dict(color='#c0392b', width=1)
                            )
                        ))
                    
                    # Add labels for bounds
                    fig.add_annotation(
                        x=0, y=upper_bound,
                        text=f"Upper Bound: {upper_bound:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        font=dict(color="#2c3e50"),
                        bgcolor="white",
                        bordercolor="#3498db",
                        borderwidth=2,
                        borderpad=4
                    )
                    
                    fig.add_annotation(
                        x=0, y=lower_bound,
                        text=f"Lower Bound: {lower_bound:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        font=dict(color="#2c3e50"),
                        bgcolor="white",
                        bordercolor="#3498db",
                        borderwidth=2,
                        borderpad=4
                    )
                    
                    fig.update_layout(
                        title=f"<b>Outlier Analysis for {selected_col}</b>",
                        height=500,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(visible=False),
                        yaxis=dict(title=selected_col)
                    )
                    
                    target_container.plotly_chart(fig, use_container_width=True)
                    
                    # Display outlier statistics
                    col1, col2, col3 = target_container.columns(3)
                    col1.metric("Outlier Count", f"{len(outliers)}")
                    col2.metric("Outlier Percentage", f"{len(outliers)/len(col_data)*100:.2f}%")
                    col3.metric("Non-Outlier Range", f"{lower_bound:.2f} to {upper_bound:.2f}")
                    
                    # Recommendations for handling outliers
                    target_container.markdown("<div class='chart-title'>Recommendations for Outliers</div>", unsafe_allow_html=True)
                    
                    target_container.markdown(f"""
                    <div class="recommendation-card">
                        <ul>
                            <li>Verify if outliers represent actual anomalies or data errors</li>
                            <li>For genuine outliers, consider keeping them if they represent real phenomena</li>
                            <li>For analysis sensitive to outliers, consider techniques like:
                                <ul>
                                    <li>Winsorizing (capping values at certain percentiles)</li>
                                    <li>Transforming the data (e.g., log transformation)</li>
                                    <li>Using robust statistical methods</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                target_container.success("✅ No significant outliers detected in the numeric columns!")
        else:
            target_container.info("No numeric columns found in the dataset to check for outliers.")
    
    # Recommendations Tab
    with quality_tabs[3]:
        target_container.markdown("<div class='chart-title'>Data Quality Improvement Recommendations</div>", unsafe_allow_html=True)
        
        recommendations = []
        
        # Missing values recommendations
        if not missing_values_df.empty:
            missing_cols_high = missing_values_df[missing_values_df['Percentage'] > 20]['Column'].tolist()
            missing_cols_medium = missing_values_df[(missing_values_df['Percentage'] <= 20) & (missing_values_df['Percentage'] > 5)]['Column'].tolist()
            
            if missing_cols_high:
                high_missing_text = ", ".join([f"<b>{col}</b>" for col in missing_cols_high[:3]])
                if len(missing_cols_high) > 3:
                    high_missing_text += f" and {len(missing_cols_high) - 3} more"
                recommendations.append(f"<li><b>High Missing Values:</b> Consider removing or imputing columns with high missing rates: {high_missing_text}</li>")
            
            if missing_cols_medium:
                medium_missing_text = ", ".join([f"<b>{col}</b>" for col in missing_cols_medium[:3]])
                if len(missing_cols_medium) > 3:
                    medium_missing_text += f" and {len(missing_cols_medium) - 3} more"
                recommendations.append(f"<li><b>Moderate Missing Values:</b> Impute missing values in columns: {medium_missing_text}</li>")
        
        # Duplicate recommendations
        if duplicate_count > 0:
            recommendations.append(f"<li><b>Duplicates:</b> Remove {duplicate_count} duplicate rows ({duplicate_pct:.2f}% of data)</li>")
        
        # Outlier recommendations
        if outlier_info:
            outlier_cols_high = [info['Column'] for info in outlier_info if info['Percentage'] > 5]
            
            if outlier_cols_high:
                high_outlier_text = ", ".join([f"<b>{col}</b>" for col in outlier_cols_high[:3]])
                if len(outlier_cols_high) > 3:
                    high_outlier_text += f" and {len(outlier_cols_high) - 3} more"
                recommendations.append(f"<li><b>Outliers:</b> Examine and potentially handle outliers in: {high_outlier_text}</li>")
        
        # Data type recommendations
        incorrect_type_cols = []
        for col, analysis in column_analyses.items():
            # Check for potential date columns not recognized as dates
            if analysis["type"] != "datetime" and "date" in analysis["semantic"]:
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    incorrect_type_cols.append((col, "date"))
                except:
                    pass
            
            # Check for categorical columns stored as numeric
            elif analysis["type"] == "numeric" and analysis["unique_count"] <= 10 and analysis["unique_ratio"] < 0.05:
                incorrect_type_cols.append((col, "categorical"))
        
        if incorrect_type_cols:
            type_text = ", ".join([f"<b>{col}</b> (as {dtype})" for col, dtype in incorrect_type_cols[:3]])
            if len(incorrect_type_cols) > 3:
                type_text += f" and {len(incorrect_type_cols) - 3} more"
            recommendations.append(f"<li><b>Data Type Conversion:</b> Consider converting: {type_text}</li>")
        
        # Other general recommendations
        if completeness_score < 95:
            recommendations.append("<li><b>Improve Completeness:</b> Identify and address sources of missing data</li>")
        
        if consistency_score < 90:
            recommendations.append("<li><b>Improve Consistency:</b> Standardize formats and units across columns</li>")
        
        # Display recommendations
        if recommendations:
            target_container.markdown(f"""
            <div class="recommendation-card">
                <ol>{''.join(recommendations)}</ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            target_container.success("✅ Your data already has excellent quality! No major improvements needed.")
        
        # Add action buttons (in a real app these would do something)
        col1, col2 = target_container.columns(2)
        
        if col1.button("Apply Recommended Fixes", key="apply_fixes"):
            target_container.info("In a production app, this would apply automated fixes to improve data quality.")
        
        if col2.button("Generate Quality Report", key="quality_report"):
            target_container.info("In a production app, this would generate a downloadable data quality report.")

# Function to recommend related visualizations based on column relationships
def recommend_related_visualizations(df, column_analyses):
    """Recommends intelligent visualizations based on data relationships and popular visualization types"""
    recommendations = []
    
    # Get columns by type
    numeric_cols = [col for col, analysis in column_analyses.items() 
                   if analysis["type"] == "numeric"]
    
    cat_cols = [col for col, analysis in column_analyses.items() 
               if analysis["type"] == "categorical" and analysis["unique_count"] <= 15]
    
    date_cols = [col for col, analysis in column_analyses.items() 
                if analysis["viz_type"] == "time_series"]
    
    geo_cols = [col for col, analysis in column_analyses.items() 
               if analysis["is_geo"]]
    
    # 1. NUMERIC BY CATEGORICAL: Bar charts showing numeric values segmented by categories
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric cols
            # Prioritize categorical columns with lower cardinality
            sorted_cat_cols = sorted(
                [(col, column_analyses[col]["unique_count"]) for col in cat_cols[:5]], 
                key=lambda x: x[1]
            )
            
            for cat_col, _ in sorted_cat_cols[:2]:  # Take 2 best categorical columns
                recommendations.append({
                    "type": "grouped_bar",
                    "title": f"{num_col} by {cat_col}",
                    "columns": [num_col, cat_col],
                    "description": f"Compare how {num_col} varies across different {cat_col} categories",
                    "icon": "📊",
                    "priority": 1 if len(column_analyses[cat_col]["unique_count"]) <= 7 else 2  # Higher priority for fewer categories
                })
    
    # 2. TIME SERIES BY CATEGORY: Line charts showing trends broken down by category
    if len(date_cols) > 0 and len(cat_cols) > 0:
        for date_col in date_cols[:2]:  # Limit to first 2 date cols
            for cat_col in cat_cols[:2]:  # Limit to first 2 categorical cols with low cardinality
                if column_analyses[cat_col]["unique_count"] <= 7:  # Only for low cardinality
                    recommendations.append({
                        "type": "time_series_by_category",
                        "title": f"{date_col} trends by {cat_col}",
                        "columns": [date_col, cat_col],
                        "description": f"Analyze how trends in {date_col} differ across {cat_col} categories",
                        "icon": "📈",
                        "priority": 1
                    })
    
    # 3. CORRELATION MATRIX: For datasets with multiple numeric variables
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "correlation_heatmap",
            "title": "Correlation analysis",
            "columns": numeric_cols[:6],  # Limit to first 6 numeric columns
            "description": "Discover relationships between numeric variables",
            "icon": "🔄",
            "priority": 2
        })
    
    # 4. SCATTER PLOT MATRIX: For exploring relationships between multiple variables
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "scatter_matrix",
            "title": "Multi-variable relationships",
            "columns": numeric_cols[:4],  # Limit to first 4 numeric columns
            "description": "Explore relationships between multiple numeric variables simultaneously",
            "icon": "🔄",
            "priority": 3
        })
    
    # 5. DISTRIBUTION COMPARISON: Compare distributions across categories
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in cat_cols[:2]:
                if column_analyses[cat_col]["unique_count"] <= 6:  # Only for low cardinality
                    recommendations.append({
                        "type": "distribution_comparison",
                        "title": f"{num_col} distribution by {cat_col}",
                        "columns": [num_col, cat_col],
                        "description": f"Compare how {num_col} is distributed across different {cat_col} categories",
                        "icon": "📊",
                        "priority": 2
                    })
    
    # 6. GEOGRAPHIC VISUALIZATION: If geographic data is available
    if len(geo_cols) > 0 and len(numeric_cols) > 0:
        for geo_col in geo_cols[:1]:
            for num_col in numeric_cols[:2]:
                recommendations.append({
                    "type": "geo_visualization",
                    "title": f"{num_col} by {geo_col}",
                    "columns": [geo_col, num_col],
                    "description": f"Visualize geographic distribution of {num_col} across {geo_col}",
                    "icon": "🗺️",
                    "priority": 2
                })
    
    # 7. TIME SERIES DECOMPOSITION: For time series data
    if len(date_cols) > 0:
        for date_col in date_cols[:1]:
            recommendations.append({
                "type": "time_decomposition",
                "title": f"{date_col} decomposition",
                "columns": [date_col],
                "description": f"Break down {date_col} into trend, seasonal, and residual components",
                "icon": "📈",
                "priority": 3
            })
    
    # 8. TREEMAP: For hierarchical categorical data or high-cardinality categories
    high_cardinality_cats = [col for col, analysis in column_analyses.items() 
                            if analysis["type"] == "categorical" and 15 < analysis["unique_count"] <= 50]
    
    if high_cardinality_cats and numeric_cols:
        for cat_col in high_cardinality_cats[:1]:
            for num_col in numeric_cols[:1]:
                recommendations.append({
                    "type": "treemap",
                    "title": f"{num_col} across {cat_col} categories",
                    "columns": [cat_col, num_col],
                    "description": f"Hierarchical view of {num_col} distribution across {cat_col} categories",
                    "icon": "🔍",
                    "priority": 3
                })
    
    # 9. DUAL-AXIS CHART: For comparing trends of two different numeric variables
    if len(numeric_cols) >= 2 and len(date_cols) > 0:
        date_col = date_cols[0]
        recommendations.append({
            "type": "dual_axis",
            "title": f"Compare {numeric_cols[0]} and {numeric_cols[1]} over time",
            "columns": [date_col, numeric_cols[0], numeric_cols[1]],
            "description": f"Compare trends of {numeric_cols[0]} and {numeric_cols[1]} over {date_col}",
            "icon": "📉",
            "priority": 3
        })
    
    # 10. HEATMAP CALENDAR: For daily/weekly patterns in time series data
    if len(date_cols) > 0:
        for date_col in date_cols[:1]:
            recommendations.append({
                "type": "calendar_heatmap",
                "title": f"{date_col} calendar patterns",
                "columns": [date_col],
                "description": f"Visualize daily, weekly, or monthly patterns in {date_col}",
                "icon": "📅",
                "priority": 4
            })
    
    # Sort recommendations by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    return recommendations

# Function to create recommended visualizations
def create_recommended_visualization(df, recommendation, container=None, key_id=None):
    """Creates sophisticated visualizations based on recommendations"""
    # Set container for the visualization
    target_container = container if container else st
    
    # Generate a unique key for this recommendation
    rec_key = key_id or f"rec_{recommendation['type']}_{uuid.uuid4().hex[:8]}"
    
    # Initialize visualization variable
    visualization = None
    
    # 1. GROUPED BAR CHART: Comparing numeric values across categories
    if recommendation["type"] == "grouped_bar":
        num_col, cat_col = recommendation["columns"]
        
        # Group by category and calculate statistics
        grouped_data = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'count', 'std']).reset_index()
        grouped_data.columns = [cat_col, 'Mean', 'Median', 'Count', 'StdDev']
        
        # Sort by mean descending for better visualization
        grouped_data = grouped_data.sort_values('Mean', ascending=False)
        
        # Create enhanced grouped bar chart with error bars
        fig = go.Figure()
        
        # Add the main bars for mean
        fig.add_trace(go.Bar(
            x=grouped_data[cat_col],
            y=grouped_data['Mean'],
            name='Mean',
            marker_color='#3498db',
            error_y=dict(
                type='data',
                array=grouped_data['StdDev'],
                visible=True,
                color='#2c3e50',
                thickness=1.5,
                width=3
            ),
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<br>StdDev: %{error_y.array:.2f}<extra></extra>'
        ))
        
        # Add a line for median values
        fig.add_trace(go.Scatter(
            x=grouped_data[cat_col],
            y=grouped_data['Median'],
            mode='markers+lines',
            name='Median',
            marker=dict(color='#e74c3c', size=8, symbol='diamond'),
            line=dict(color='#e74c3c', width=2, dash='dot'),
            hovertemplate='<b>%{x}</b><br>Median: %{y:.2f}<extra></extra>'
        ))
        
        # Add sample size as annotations on each bar
        for i, row in enumerate(grouped_data.itertuples()):
            sample_size = getattr(row, 'Count')
            fig.add_annotation(
                x=getattr(row, cat_col),
                y=getattr(row, 'Mean'),
                text=f"n={sample_size}",
                showarrow=False,
                yshift=10,
                font=dict(size=9, color="#7f8c8d")
            )
        
        fig.update_layout(
            title=f"<b>{num_col} by {cat_col}</b>",
            xaxis=dict(
                title=cat_col,
                tickangle=-45 if len(grouped_data) > 5 else 0,
                tickfont=dict(size=10),
                categoryorder='total descending'
            ),
            yaxis=dict(
                title=f"{num_col}",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#bdc3c7',
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=40, r=40, t=80, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            hovermode='closest',
            bargap=0.15
        )
        
        # Add a statistical significance test if available
        if len(grouped_data) > 1 and SCIPY_AVAILABLE:
            try:
                # Run ANOVA to check for significant differences between groups
                groups = []
                for category in grouped_data[cat_col]:
                    group_data = df[df[cat_col] == category][num_col].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Add annotation about statistical significance
                    sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"
                    sig_color = "#2ecc71" if p_value < 0.05 else "#e74c3c"
                    
                    fig.add_annotation(
                        x=0.5,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text=f"Differences between groups are {sig_text} (p = {p_value:.4f})",
                        showarrow=False,
                        font=dict(size=12, color=sig_color),
                        bgcolor="white",
                        bordercolor=sig_color,
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )
            except:
                # Skip if test fails
                pass
        
        visualization = fig
    
    # 2. TIME SERIES BY CATEGORY: Line charts showing trends broken down by category
    elif recommendation["type"] == "time_series_by_category":
        time_col, cat_col = recommendation["columns"]
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                target_container.warning(f"Could not convert {time_col} to datetime format.")
                return None
        
        # Get unique categories (limit to top 6 for visibility)
        categories = df[cat_col].value_counts().nlargest(6).index.tolist()
        
        # Choose appropriate time period based on date range
        date_range = (df[time_col].max() - df[time_col].min()).days
        
        if date_range > 365 * 2:  # 2+ years
            period = 'M'
            period_name = 'Month'
            date_format = '%b %Y'
        elif date_range > 60:  # 2+ months
            period = 'W'
            period_name = 'Week'
            date_format = '%d %b %Y'
        else:  # Less than 2 months
            period = 'D'
            period_name = 'Day'
            date_format = '%d %b'
        
        # Create enhanced multi-line plot
        fig = go.Figure()
        
        # Color palette for multiple lines
        colors = px.colors.qualitative.Plotly
        
        # Create date groups for each category
        for i, category in enumerate(categories):
            # Filter data for this category
            cat_data = df[df[cat_col] == category]
            
            # Group by time period
            time_grouped = cat_data.assign(date_group=cat_data[time_col].dt.to_period(period)).groupby('date_group').size()
            
            # Skip if no data
            if len(time_grouped) == 0:
                continue
            
            # Convert period index to timestamps for plotting
            dates = time_grouped.index.to_timestamp()
            counts = time_grouped.values
            
            # Normalize data to see relative changes
            if len(counts) > 0 and counts[0] > 0:
                normalized = counts / counts[0] * 100
            else:
                normalized = counts
            
            # Add line to plot with custom hover template
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name=str(category),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f"{cat_col}: {category}<br>Date: %{{x|{date_format}}}<br>Count: %{{y}}<extra></extra>"
            ))
        
        # Calculate overall trend across all categories
        all_data = df.assign(date_group=df[time_col].dt.to_period(period)).groupby('date_group').size()
        all_dates = all_data.index.to_timestamp()
        all_counts = all_data.values
        
        # Add a faded total line
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=all_counts,
            mode='lines',
            name='Total (All Categories)',
            line=dict(color='rgba(0, 0, 0, 0.2)', width=3, dash='dot'),
            hovertemplate=f"All Categories<br>Date: %{{x|{date_format}}}<br>Count: %{{y}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"<b>Trends by {cat_col} over time</b>",
            xaxis=dict(
                title=period_name,
                tickformat=date_format,
                showgrid=True,
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            yaxis=dict(
                title="Count",
                showgrid=True,
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            hovermode='closest'
        )
        
        visualization = fig
        
        # Show insights about category trends
        with target_container.expander("Category Trend Insights", expanded=False):
            try:
                # Calculate growth rates for each category
                growth_data = []
                
                for category in categories:
                    cat_data = df[df[cat_col] == category]
                    time_grouped = cat_data.assign(date_group=cat_data[time_col].dt.to_period(period)).groupby('date_group').size()
                    
                    if len(time_grouped) >= 2:
                        first_value = time_grouped.iloc[0]
                        last_value = time_grouped.iloc[-1]
                        
                        if first_value > 0:
                            growth_pct = (last_value - first_value) / first_value * 100
                        else:
                            growth_pct = 0
                        
                        growth_data.append({
                            'Category': category,
                            'Growth': growth_pct,
                            'Direction': 'Increasing' if growth_pct > 0 else 'Decreasing' if growth_pct < 0 else 'Stable'
                        })
                
                # Sort and display growth rates
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    growth_df = growth_df.sort_values('Growth', ascending=False)
                    
                    # Display as colored table
                    def color_growth(val):
                        if val > 10:
                            return 'background-color: rgba(46, 204, 113, 0.2); color: #27ae60; font-weight: bold'
                        elif val < -10:
                            return 'background-color: rgba(231, 76, 60, 0.2); color: #c0392b; font-weight: bold'
                        else:
                            return 'background-color: rgba(243, 156, 18, 0.2); color: #d35400; font-weight: bold'
                    
                    growth_styled = growth_df.style.format({'Growth': '{:.1f}%'})
                    growth_styled = growth_styled.applymap(color_growth, subset=['Growth'])
                    
                    target_container.markdown("<div class='chart-title'>Growth Rate by Category</div>", unsafe_allow_html=True)
                    target_container.dataframe(growth_styled, use_container_width=True)
                    
                    # Highlight fastest growing and declining categories
                    if len(growth_df) > 1:
                        fastest_growing = growth_df.iloc[0]
                        fastest_declining = growth_df.iloc[-1]
                        
                        insights_cols = target_container.columns(2)
                        
                        if fastest_growing['Growth'] > 0:
                            insights_cols[0].markdown(f"""
                            <div class="success-box">
                                <b>Fastest Growing</b><br>
                                {fastest_growing['Category']}: {fastest_growing['Growth']:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if fastest_declining['Growth'] < 0:
                            insights_cols[1].markdown(f"""
                            <div class="error-box">
                                <b>Fastest Declining</b><br>
                                {fastest_declining['Category']}: {fastest_declining['Growth']:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
            except:
                target_container.info("Could not generate growth insights for categories.")
    
    # 3. CORRELATION HEATMAP: Matrix showing correlations between numeric variables
    elif recommendation["type"] == "correlation_heatmap":
        columns = recommendation["columns"]
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr().round(2)
        
        # Create enhanced heatmap with annotations
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',  # Show correlation values on the cells
            aspect="auto",  # Maintain aspect ratio
            color_continuous_scale=px.colors.diverging.RdBu_r,  # Red-Blue scale with white center
            color_continuous_midpoint=0,  # Center color scale at 0
            title="<b>Correlation Heatmap for Numeric Variables</b>",
            labels=dict(x="Variable", y="Variable", color="Correlation")
        )
        
        # Add correlation values as annotations
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                value = corr_matrix.iloc[i, j]
                
                # Make text dark for middle values, white for extreme values
                text_color = 'white' if abs(value) > 0.6 else 'black'
                
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=10)
                )
        
        # Enhance layout
        fig.update_layout(
            height=max(400, len(columns) * 40),
            width=max(600, len(columns) * 40),
            margin=dict(l=40, r=40, t=80, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            coloraxis=dict(
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300,
                    tickmode="array",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0<br>Perfect<br>Negative", "-0.5", "0.0", "0.5", "1.0<br>Perfect<br>Positive"],
                    ticks="outside"
                )
            )
        )
        
        # Hide the upper triangle
        for i in range(len(corr_matrix)):
            for j in range(i):
                fig.data[0].z[i][j] = None
        
        visualization = fig
        
        # Show insights about strongest correlations
        with target_container.expander("Correlation Insights", expanded=False):
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append((columns[i], columns[j], corr_value))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Create dataframe of top correlations
            if corr_pairs:
                top_n = min(5, len(corr_pairs))
                top_corrs = pd.DataFrame(corr_pairs[:top_n], columns=['Variable 1', 'Variable 2', 'Correlation'])
                
                # Add interpretation column
                def interpret_corr(val):
                    if abs(val) > 0.8:
                        strength = "Very Strong"
                    elif abs(val) > 0.6:
                        strength = "Strong"
                    elif abs(val) > 0.4:
                        strength = "Moderate"
                    elif abs(val) > 0.2:
                        strength = "Weak"
                    else:
                        strength = "Very Weak"
                    
                    direction = "Positive" if val > 0 else "Negative"
                    return f"{strength} {direction}"
                
                top_corrs['Interpretation'] = top_corrs['Correlation'].apply(interpret_corr)
                
                # Style the dataframe
                def color_corr(val):
                    if abs(val) > 0.8:
                        return f'background-color: {"rgba(46, 204, 113, 0.2)" if val > 0 else "rgba(231, 76, 60, 0.2)"}; font-weight: bold'
                    elif abs(val) > 0.6:
                        return f'background-color: {"rgba(46, 204, 113, 0.1)" if val > 0 else "rgba(231, 76, 60, 0.1)"}; font-weight: bold'
                    else:
                        return ''
                
                top_corrs_styled = top_corrs.style.format({'Correlation': '{:.3f}'})
                top_corrs_styled = top_corrs_styled.applymap(color_corr, subset=['Correlation'])
                
                target_container.markdown("<div class='chart-title'>Strongest Correlations</div>", unsafe_allow_html=True)
                target_container.dataframe(top_corrs_styled, use_container_width=True)
                
                # Add insight about strongest correlation
                if len(corr_pairs) > 0:
                    strongest = corr_pairs[0]
                    corr_dir = "positive" if strongest[2] > 0 else "negative"
                    target_container.markdown(f"""
                    <div class="info-box">
                        <b>Key Insight:</b> The strongest relationship is between <b>{strongest[0]}</b> and <b>{strongest[1]}</b> 
                        with a {corr_dir} correlation of <b>{strongest[2]:.2f}</b>. 
                        This means they tend to {corr_dir == 'positive' and 'increase' or 'decrease'} together.
                    </div>
                    """, unsafe_allow_html=True)
    
    # 4. SCATTER MATRIX: Multiple scatter plots showing relationships between variables
    elif recommendation["type"] == "scatter_matrix":
        columns = recommendation["columns"]
        
        # Create enhanced scatter plot matrix
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            color_discrete_sequence=['#3498db'],
            opacity=0.6,
            title="<b>Multi-variable Relationship Matrix</b>",
            labels={col: col for col in columns}  # Used for axis titles
        )
        
        # Enhance the scatter plot
        fig.update_traces(
            diagonal_visible=False,  # Hide diagonal
            showupperhalf=False,     # Show only lower half to avoid redundancy
            marker=dict(
                size=5,
                line=dict(width=0.5, color='white')
            )
        )
        
        # Calculate correlations to add to the plot
        corr_matrix = df[columns].corr().round(2)
        
        # Add correlation values as annotations
        for i, row_var in enumerate(columns):
            for j, col_var in enumerate(columns):
                if j < i:  # Lower triangle only
                    corr = corr_matrix.loc[row_var, col_var]
                    
                    # Determine text color based on background
                    text_color = 'white' if abs(corr) > 0.7 else 'black'
                    bg_color = 'rgba(46, 204, 113, 0.7)' if corr > 0.7 else \
                              'rgba(231, 76, 60, 0.7)' if corr < -0.7 else \
                              'rgba(243, 156, 18, 0.7)' if abs(corr) > 0.3 else \
                              'rgba(189, 195, 199, 0.7)'
                    
                    # Add annotation for correlation
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"r = {corr:.2f}",
                        showarrow=False,
                        font=dict(color=text_color, size=9),
                        bgcolor=bg_color,
                        borderpad=3,
                        opacity=0.8,
                        xref=f"x{j+1}",
                        yref=f"y{i+1}"
                    )
        
        fig.update_layout(
            height=250 * len(columns),
            width=250 * len(columns),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        visualization = fig
    
    # 5. DISTRIBUTION COMPARISON: Comparing distributions across categories
    elif recommendation["type"] == "distribution_comparison":
        num_col, cat_col = recommendation["columns"]
        
        # Get unique categories (limit to preserve readability)
        categories = df[cat_col].value_counts().nlargest(6).index.tolist()
        
        # Create distribution comparison figure
        fig = go.Figure()
        
        # Color palette for multiple distributions
        colors = px.colors.qualitative.Plotly
        
        # Choose appropriate distribution visualization based on data size
        use_violins = df[df[cat_col].isin(categories)].shape[0] > 100
        
        if use_violins:
            # Use violin plots for larger datasets
            for i, category in enumerate(categories):
                # Get data for this category
                cat_data = df[df[cat_col] == category][num_col].dropna()
                
                # Skip if no data
                if len(cat_data) == 0:
                    continue
                
                # Add violin plot
                fig.add_trace(go.Violin(
                    y=cat_data,
                    name=str(category),
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[i % len(colors)],
                    marker_color=colors[i % len(colors)],
                    line_color='white',
                    opacity=0.6,
                    side='positive',  # All violins on positive side
                    points=False,  # Don't show individual points
                    jitter=0,
                    bandwidth=None,  # Let plotly determine optimal bandwidth
                    spanmode='hard',  # Hard boundary at min/max
                    hoverinfo='name+y',
                    hovertemplate=f"{cat_col}: %{{name}}<br>{num_col}: %{{y}}<extra></extra>"
                ))
        else:
            # Use box plots for smaller datasets
            for i, category in enumerate(categories):
                # Get data for this category
                cat_data = df[df[cat_col] == category][num_col].dropna()
                
                # Skip if no data
                if len(cat_data) == 0:
                    continue
                
                # Add box plot
                fig.add_trace(go.Box(
                    y=cat_data,
                    name=str(category),
                    marker_color=colors[i % len(colors)],
                    boxmean=True,  # Show mean
                    boxpoints='all',  # Show all points for small datasets
                    jitter=0.5,
                    pointpos=-1.8,
                    line=dict(width=2),
                    hovertemplate=f"{cat_col}: %{{name}}<br>{num_col}: %{{y}}<extra></extra>"
                ))
        
        # Add a statistics table below the chart
        stats_table = []
        for category in categories:
            cat_data = df[df[cat_col] == category][num_col].dropna()
            if len(cat_data) > 0:
                stats_table.append({
                    'Category': str(category),
                    'Count': len(cat_data),
                    'Mean': cat_data.mean(),
                    'Median': cat_data.median(),
                    'StdDev': cat_data.std(),
                    'Min': cat_data.min(),
                    'Max': cat_data.max()
                })
        
        # Run statistical test for significant differences if scipy is available
        p_value = None
        test_name = None
        if len(categories) > 1 and SCIPY_AVAILABLE:
            try:
                # Collect groups for testing
                groups = []
                for category in categories:
                    cat_data = df[df[cat_col] == category][num_col].dropna()
                    if len(cat_data) > 0:
                        groups.append(cat_data)
                
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    if len(groups) == 2:
                        # Use t-test for two groups
                        t_stat, p_value = stats.ttest_ind(*groups, equal_var=False)
                        test_name = "Two-sample t-test"
                    else:
                        # Use ANOVA for multiple groups
                        f_stat, p_value = stats.f_oneway(*groups)
                        test_name = "ANOVA"
            except:
                # Skip if test fails
                pass
        
        # Add statistical test result as annotation
        if p_value is not None:
            sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"
            sig_color = "#2ecc71" if p_value < 0.05 else "#e74c3c"
            
            fig.add_annotation(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"{test_name}: Differences are {sig_text} (p = {p_value:.4f})",
                showarrow=False,
                font=dict(size=12, color=sig_color),
                bgcolor="white",
                bordercolor=sig_color,
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )
        
        fig.update_layout(
            title=f"<b>Distribution of {num_col} by {cat_col}</b>",
            yaxis=dict(
                title=num_col,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#bdc3c7',
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            xaxis=dict(
                title=cat_col,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#bdc3c7'
            ),
            height=600,
            margin=dict(l=40, r=40, t=80, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            showlegend=False
        )
        
        visualization = fig
        
        # Show statistics table in expander
        if stats_table:
            with target_container.expander("Distribution Statistics", expanded=False):
                stats_df = pd.DataFrame(stats_table)
                stats_df_styled = stats_df.style.format({
                    'Mean': '{:.2f}',
                    'Median': '{:.2f}',
                    'StdDev': '{:.2f}',
                    'Min': '{:.2f}',
                    'Max': '{:.2f}'
                })
                
                target_container.dataframe(stats_df_styled, use_container_width=True)
                
                # Highlight min and max category
                if len(stats_df) > 1:
                    max_cat = stats_df.loc[stats_df['Mean'].idxmax(), 'Category']
                    min_cat = stats_df.loc[stats_df['Mean'].idxmin(), 'Category']
                    
                    stat_cols = target_container.columns(2)
                    
                    stat_cols[0].markdown(f"""
                    <div class="success-box">
                        <b>Highest Average</b><br>
                        {max_cat}: {stats_df.loc[stats_df['Mean'].idxmax(), 'Mean']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stat_cols[1].markdown(f"""
                    <div class="warning-box">
                        <b>Lowest Average</b><br>
                        {min_cat}: {stats_df.loc[stats_df['Mean'].idxmin(), 'Mean']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
    
    # 6. GEOGRAPHIC VISUALIZATION: Showing data on maps
    elif recommendation["type"] == "geo_visualization":
        # This would typically use geo libraries like folium, pydeck, or plotly's geo capabilities
        # For this demo, we'll show a warning and a fallback visualization
        geo_col, num_col = recommendation["columns"]
        
        target_container.warning("Interactive geographic visualization would typically require specialized mapping libraries. Showing simplified view.")
        
        # Create a simple bar chart by region as fallback
        geo_grouped = df.groupby(geo_col)[num_col].mean().reset_index()
        geo_grouped = geo_grouped.sort_values(num_col, ascending=False)
        
        # Limit to top 20 regions for readability
        if len(geo_grouped) > 20:
            geo_grouped = geo_grouped.head(20)
        
        # Create enhanced bar chart
        fig = px.bar(
            geo_grouped,
            x=geo_col,
            y=num_col,
            title=f"<b>{num_col} by {geo_col} (Geographic Data)</b>",
            color=num_col,
            color_continuous_scale=px.colors.sequential.Blues,
            text=round(geo_grouped[num_col], 2)
        )
        
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside',
            hovertemplate=f'{geo_col}: %{{x}}<br>{num_col}: %{{y:.2f}}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis=dict(
                title=geo_col,
                tickangle=-45 if len(geo_grouped) > 5 else 0,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=num_col,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#bdc3c7',
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            height=500,
            margin=dict(l=40, r=40, t=80, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            coloraxis_showscale=False
        )
        
        visualization = fig
    
    # 7. TIME SERIES DECOMPOSITION: Breaking down a time series into components
    elif recommendation["type"] == "time_decomposition":
        time_col = recommendation["columns"][0]
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                target_container.warning(f"Could not convert {time_col} to datetime format.")
                return None
        
        # Group by appropriate time period
        date_range = (df[time_col].max() - df[time_col].min()).days
        
        if date_range > 365 * 2:  # 2+ years
            period = 'M'
            period_name = 'Month'
        elif date_range > 60:  # 2+ months
            period = 'W'
            period_name = 'Week'
        else:  # Less than 2 months
            period = 'D'
            period_name = 'Day'
        
        # Group data by time period
        time_grouped = df.assign(date_group=df[time_col].dt.to_period(period)).groupby('date_group').size()
        
        # Convert to time series for potential decomposition
        time_df = pd.DataFrame({
            'Date': time_grouped.index.to_timestamp(),
            'Count': time_grouped.values
        })
        
        # Create a figure with multiple subplots for components
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Original Time Series", "Trend Component", "Seasonality/Residual Component"),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Add original time series
        fig.add_trace(
            go.Scatter(
                x=time_df['Date'],
                y=time_df['Count'],
                mode='lines+markers',
                name='Original',
                marker=dict(color='#3498db', size=6),
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )
        
        # Calculate trend using moving average
        window_size = max(2, min(7, len(time_df) // 5))
        if len(time_df) > window_size:
            time_df['Trend'] = time_df['Count'].rolling(window=window_size, center=True).mean()
            
            # Add trend component
            fig.add_trace(
                go.Scatter(
                    x=time_df['Date'],
                    y=time_df['Trend'],
                    mode='lines',
                    name=f'Trend ({window_size}-point MA)',
                    line=dict(color='#2ecc71', width=3)
                ),
                row=2, col=1
            )
            
            # Calculate residual (original - trend)
            time_df['Residual'] = time_df['Count'] - time_df['Trend']
            
            # Add residual component (simple version of seasonality/residual)
            fig.add_trace(
                go.Bar(
                    x=time_df['Date'],
                    y=time_df['Residual'],
                    name='Residual',
                    marker_color='#e74c3c'
                ),
                row=3, col=1
            )
            
            # Add zero line on residual plot
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="#7f8c8d",
                row=3, col=1
            )
        else:
            # If not enough data for decomposition, show a message
            target_container.info("Not enough time periods for full decomposition. Showing original time series only.")
        
        fig.update_layout(
            title=f"<b>Time Series Decomposition of {time_col}</b>",
            height=800,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Residual", row=3, col=1)
        
        visualization = fig
    
    # 8. TREEMAP: Hierarchical visualization of categories
    elif recommendation["type"] == "treemap":
        cat_col, num_col = recommendation["columns"]
        
        # Aggregate data by category
        cat_grouped = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).reset_index()
        cat_grouped.columns = [cat_col, 'Sum', 'Mean', 'Count']
        
        # Sort by sum for better visualization
        cat_grouped = cat_grouped.sort_values('Sum', ascending=False)
        
        # Create colorful treemap
        fig = px.treemap(
            cat_grouped,
            path=[cat_col],
            values='Sum',
            color='Mean',
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=['Count', 'Mean'],
            title=f"<b>Hierarchical View of {num_col} by {cat_col}</b>"
        )
        
        # Enhance treemap with better labels and hover info
        fig.update_traces(
            textinfo="label+value+percent",
            hovertemplate=f"<b>%{{label}}</b><br>Sum: %{{value:,.2f}}<br>Mean: %{{customdata[1]:.2f}}<br>Count: %{{customdata[0]}}<extra></extra>"
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            coloraxis=dict(
                colorbar=dict(
                    title="Mean Value",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300
                )
            )
        )
        
        visualization = fig
    
    # 9. DUAL-AXIS CHART: Comparing two metrics over time
    elif recommendation["type"] == "dual_axis":
        time_col, y1_col, y2_col = recommendation["columns"]
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                target_container.warning(f"Could not convert {time_col} to datetime format.")
                return None
        
        # Group by appropriate time period
        date_range = (df[time_col].max() - df[time_col].min()).days
        
        if date_range > 365 * 2:  # 2+ years
            period = 'M'
            period_name = 'Month'
        elif date_range > 60:  # 2+ months
            period = 'W'
            period_name = 'Week'
        else:  # Less than 2 months
            period = 'D'
            period_name = 'Day'
        
        # Aggregate data by time period
        time_df = df.assign(date_group=df[time_col].dt.to_period(period)).groupby('date_group').agg({
            y1_col: 'mean',
            y2_col: 'mean'
        }).reset_index()
        
        time_df['date_group'] = time_df['date_group'].dt.to_timestamp()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add trace for y1 on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=time_df['date_group'],
                y=time_df[y1_col],
                name=y1_col,
                mode='lines+markers',
                line=dict(color='#3498db', width=2),
                marker=dict(color='#3498db', size=8)
            ),
            secondary_y=False
        )
        
        # Add trace for y2 on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=time_df['date_group'],
                y=time_df[y2_col],
                name=y2_col,
                mode='lines+markers',
                line=dict(color='#e74c3c', width=2, dash='dot'),
                marker=dict(color='#e74c3c', size=8, symbol='diamond')
            ),
            secondary_y=True
        )
        
        # Calculate correlation between the two series
        correlation = time_df[y1_col].corr(time_df[y2_col])
        
        # Add correlation annotation
        corr_color = "#2ecc71" if correlation > 0.7 else \
                    "#e74c3c" if correlation < -0.7 else \
                    "#f39c12"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Correlation: {correlation:.2f}",
            showarrow=False,
            font=dict(color=corr_color, size=12),
            bgcolor="white",
            bordercolor=corr_color,
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        # Customize layout
        fig.update_layout(
            title=f"<b>Comparing {y1_col} and {y2_col} over Time</b>",
            xaxis=dict(
                title=period_name,
                showgrid=True,
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text=y1_col, secondary_y=False)
        fig.update_yaxes(title_text=y2_col, secondary_y=True)
        
        visualization = fig
        
        # Show insights about the relationship
        with target_container.expander("Relationship Insights", expanded=False):
            try:
                # Calculate if the two series move together or oppositely
                y1_trend = (time_df[y1_col].iloc[-1] - time_df[y1_col].iloc[0]) / time_df[y1_col].iloc[0] if time_df[y1_col].iloc[0] != 0 else 0
                y2_trend = (time_df[y2_col].iloc[-1] - time_df[y2_col].iloc[0]) / time_df[y2_col].iloc[0] if time_df[y2_col].iloc[0] != 0 else 0
                
                relationship = "move together" if (y1_trend > 0 and y2_trend > 0) or (y1_trend < 0 and y2_trend < 0) else "move in opposite directions"
                
                target_container.markdown(f"""
                <div class="info-box">
                    <b>Key Insight:</b> {y1_col} and {y2_col} appear to {relationship} over time 
                    (correlation: {correlation:.2f}). 
                    {y1_col} has changed by {y1_trend*100:.1f}% while {y2_col} has changed by {y2_trend*100:.1f}%.
                </div>
                """, unsafe_allow_html=True)
                
                # Show peaks and valleys
                y1_peak_idx = time_df[y1_col].idxmax()
                y1_valley_idx = time_df[y1_col].idxmin()
                y2_peak_idx = time_df[y2_col].idxmax()
                y2_valley_idx = time_df[y2_col].idxmin()
                
                col1, col2 = target_container.columns(2)
                
                col1.markdown(f"<div class='chart-title'>{y1_col} Extremes</div>", unsafe_allow_html=True)
                col1.markdown(f"""
                - Peak: {time_df.loc[y1_peak_idx, y1_col]:.2f} on {time_df.loc[y1_peak_idx, 'date_group'].strftime('%Y-%m-%d')}
                - Valley: {time_df.loc[y1_valley_idx, y1_col]:.2f} on {time_df.loc[y1_valley_idx, 'date_group'].strftime('%Y-%m-%d')}
                """)
                
                col2.markdown(f"<div class='chart-title'>{y2_col} Extremes</div>", unsafe_allow_html=True)
                col2.markdown(f"""
                - Peak: {time_df.loc[y2_peak_idx, y2_col]:.2f} on {time_df.loc[y2_peak_idx, 'date_group'].strftime('%Y-%m-%d')}
                - Valley: {time_df.loc[y2_valley_idx, y2_col]:.2f} on {time_df.loc[y2_valley_idx, 'date_group'].strftime('%Y-%m-%d')}
                """)
            except:
                target_container.info("Could not generate trend insights for the time series.")
    
    # 10. CALENDAR HEATMAP: Showing daily/weekly patterns
    elif recommendation["type"] == "calendar_heatmap":
        time_col = recommendation["columns"][0]
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                target_container.warning(f"Could not convert {time_col} to datetime format.")
                return None
        
        # Extract day of week and hour for temporal patterns
        df_temp = df.copy()
        df_temp['day_of_week'] = df_temp[time_col].dt.day_name()
        df_temp['hour'] = df_temp[time_col].dt.hour
        
        # Count events by day of week and hour
        heatmap_data = df_temp.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        
        # Define day of week order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Filter to days present in the data
        days_present = [day for day in days_order if day in heatmap_data['day_of_week'].unique()]
        
        # Create heatmap
        fig = px.density_heatmap(
            heatmap_data,
            x='hour',
            y='day_of_week',
            z='count',
            color_continuous_scale=px.colors.sequential.Blues,
            title="<b>Activity by Day of Week and Hour</b>",
            category_orders={"day_of_week": days_present}
        )
        
        # Customize layout
        fig.update_layout(
            xaxis=dict(
                title="Hour of Day",
                tickmode='linear',
                tick0=0,
                dtick=2,
                showgrid=True,
                gridcolor='rgba(189, 195, 199, 0.5)',
                gridwidth=1
            ),
            yaxis=dict(
                title="Day of Week",
                autorange="reversed"  # Put Monday at the top
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            coloraxis=dict(
                colorbar=dict(
                    title="Count",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300
                )
            )
        )
        
        visualization = fig
        
        # Show insights about temporal patterns
        with target_container.expander("Temporal Pattern Insights", expanded=False):
            try:
                # Find busiest day and hour
                busiest_row = heatmap_data.loc[heatmap_data['count'].idxmax()]
                busiest_day = busiest_row['day_of_week']
                busiest_hour = int(busiest_row['hour'])
                busiest_count = int(busiest_row['count'])
                
                # Format busiest hour in 12-hour format
                busiest_hour_12 = f"{busiest_hour if busiest_hour < 12 else busiest_hour - 12}{' AM' if busiest_hour < 12 else ' PM'}"
                
                # Find quietest day and hour (with at least some data)
                non_zero_data = heatmap_data[heatmap_data['count'] > 0]
                if len(non_zero_data) > 0:
                    quietest_row = non_zero_data.loc[non_zero_data['count'].idxmin()]
                    quietest_day = quietest_row['day_of_week']
                    quietest_hour = int(quietest_row['hour'])
                    quietest_count = int(quietest_row['count'])
                    
                    # Format quietest hour in 12-hour format
                    quietest_hour_12 = f"{quietest_hour if quietest_hour < 12 else quietest_hour - 12}{' AM' if quietest_hour < 12 else ' PM'}"
                    
                    target_container.markdown(f"""
                    <div class="info-box">
                        <b>Key Insights:</b>
                        <ul>
                            <li>Busiest time: <b>{busiest_day} at {busiest_hour_12}</b> ({busiest_count} events)</li>
                            <li>Quietest time: <b>{quietest_day} at {quietest_hour_12}</b> ({quietest_count} events)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Calculate daily patterns
                daily_counts = heatmap_data.groupby('day_of_week')['count'].sum().reindex(days_present)
                busiest_day_overall = daily_counts.idxmax()
                quietest_day_overall = daily_counts.idxmin()
                
                # Calculate hourly patterns
                hourly_counts = heatmap_data.groupby('hour')['count'].sum()
                busiest_hour_overall = hourly_counts.idxmax()
                busiest_hour_overall_12 = f"{busiest_hour_overall if busiest_hour_overall < 12 else busiest_hour_overall - 12}{' AM' if busiest_hour_overall < 12 else ' PM'}"
                
                col1, col2 = target_container.columns(2)
                
                # Create daily pattern chart
                fig1 = px.bar(
                    daily_counts.reset_index(),
                    x='day_of_week',
                    y='count',
                    title="Activity by Day",
                    color='count',
                    color_continuous_scale=px.colors.sequential.Blues,
                    category_orders={"day_of_week": days_present}
                )
                
                fig1.update_layout(
                    xaxis_title="",
                    yaxis_title="Count",
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=300,
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                col1.plotly_chart(fig1, use_container_width=True)
                
                # Create hourly pattern chart
                fig2 = px.line(
                    hourly_counts.reset_index(),
                    x='hour',
                    y='count',
                    title="Activity by Hour",
                    markers=True
                )
                
                fig2.update_layout(
                    xaxis=dict(
                        title="Hour",
                        tickmode='linear',
                        tick0=0,
                        dtick=3
                    ),
                    yaxis_title="Count",
                    showlegend=False,
                    height=300,
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                col2.plotly_chart(fig2, use_container_width=True)
                
                target_container.markdown(f"""
                <div class="info-box">
                    <b>Summary:</b> Activity is highest on <b>{busiest_day_overall}</b> and peaks at around <b>{busiest_hour_overall_12}</b>.
                </div>
                """, unsafe_allow_html=True)
                
            except:
                target_container.info("Could not generate detailed temporal insights.")
    
    # Display the visualization
    if visualization:
        target_container.plotly_chart(visualization, use_container_width=True, config={'displayModeBar': True}, key=rec_key)
    
    return visualization

# Function to create enhanced data download options
def create_download_options(df, filename_prefix="data", container=None):
    """Creates a section with various download options for the data"""
    target_container = container if container else st
    
    target_container.markdown("<div class='chart-title'>Download Options</div>", unsafe_allow_html=True)
    
    # Create download buttons
    col1, col2, col3 = target_container.columns(3)
    
    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"{filename_prefix}_export.csv",
        mime="text/csv",
        key="download_csv"
    )
    
    # Excel Download
    try:
        # Create in-memory Excel file
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Data', index=False)
        writer.save()
        excel_data = output.getvalue()
        
        col2.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{filename_prefix}_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
    except:
        col2.warning("Excel export not available. Install xlsxwriter package.")
    
    # JSON Download
    json = df.to_json(orient='records', date_format='iso')
    col3.download_button(
        label="Download as JSON",
        data=json,
        file_name=f"{filename_prefix}_export.json",
        mime="application/json",
        key="download_json"
    )

# Function to create key metrics dashboard
def create_key_metrics(df, column_analyses, container=None):
    """Creates a dashboard section with key metrics from the dataset"""
    target_container = container if container else st
    
    # Get columns by type
    numeric_cols = [col for col, analysis in column_analyses.items() 
                   if analysis["type"] == "numeric"]
    
    cat_cols = [col for col, analysis in column_analyses.items() 
               if analysis["type"] == "categorical"]
    
    date_cols = [col for col, analysis in column_analyses.items() 
                if analysis["viz_type"] == "time_series"]
    
    # Create KPI metrics based on data types
    target_container.markdown("""
    <div class="kpi-section">
    """, unsafe_allow_html=True)
    
    # 1. Select and display numeric KPIs
    kpi_candidates = []
    
    # Prioritize certain columns for KPIs based on semantics
    priority_semantics = ["price", "quantity", "percentage", "rating", "age"]
    
    for semantic in priority_semantics:
        for col, analysis in column_analyses.items():
            if analysis["semantic"] == semantic and analysis["type"] == "numeric":
                # Get statistics from the analysis
                stats = analysis.get("statistics", {})
                
                if stats:
                    kpi_candidates.append({
                        "column": col,
                        "semantic": semantic,
                        "mean": stats.get("mean", df[col].mean()),
                        "median": stats.get("median", df[col].median()),
                        "min": stats.get("min", df[col].min()),
                        "max": stats.get("max", df[col].max()),
                        "priority": priority_semantics.index(semantic)
                    })
    
    # Add any remaining numeric columns if needed
    for col in numeric_cols:
        if not any(kpi["column"] == col for kpi in kpi_candidates):
            kpi_candidates.append({
                "column": col,
                "semantic": column_analyses[col]["semantic"],
                "mean": df[col].mean(),
                "median": df[col].median(),
                "min": df[col].min(),
                "max": df[col].max(),
                "priority": 99  # Lower priority
            })
    
    # Sort by priority and select top 4
    kpi_candidates.sort(key=lambda x: x["priority"])
    kpi_metrics = kpi_candidates[:4]
    
    # Create KPI cards
    for kpi in kpi_metrics:
        col_name = kpi["column"]
        semantic = kpi["semantic"]
        
        # Determine display value and trend
        if semantic in ["price", "quantity", "amount"]:
            display_val = f"${kpi['mean']:,.2f}" if kpi['mean'] >= 1000 else f"${kpi['mean']:.2f}"
            display_metric = "Average"
        elif semantic in ["percentage", "ratio", "rate"]:
            display_val = f"{kpi['mean']:.1f}%"
            display_metric = "Average"
        else:
            if abs(kpi['mean']) >= 1000:
                display_val = f"{kpi['mean']:,.0f}"
            elif abs(kpi['mean']) >= 1:
                display_val = f"{kpi['mean']:.2f}"
            else:
                display_val = f"{kpi['mean']:.4f}"
            display_metric = "Average"
        
        # Calculate a simple trend if possible (compare to median)
        trend_val = ((kpi['mean'] - kpi['median']) / kpi['median'] * 100) if kpi['median'] != 0 else 0
        
        # Generate trend text and class
        if abs(trend_val) < 1:
            trend_class = "kpi-trend-neutral"
            trend_text = "Stable"
        elif trend_val > 0:
            trend_class = "kpi-trend-positive"
            trend_text = f"↑ {abs(trend_val):.1f}% vs median"
        else:
            trend_class = "kpi-trend-negative"
            trend_text = f"↓ {abs(trend_val):.1f}% vs median"
        
        # Create HTML for the KPI card
        target_container.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{display_metric} {col_name}</div>
            <div class="kpi-value">{display_val}</div>
            <div class="{trend_class}">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 5. Add record count KPI
    target_container.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Records</div>
        <div class="kpi-value">{df.shape[0]:,}</div>
        <div class="kpi-trend-neutral">Row Count</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Continuing the create_key_metrics function
    # Add date range KPI if available
    if date_cols:
        date_col = date_cols[0]
        date_min = df[date_col].min()
        date_max = df[date_col].max()
        date_range = (date_max - date_min).days
        
        # Format range based on the span
        if date_range > 365:
            range_text = f"{date_range // 365} year{'s' if date_range // 365 > 1 else ''}"
        elif date_range > 30:
            range_text = f"{date_range // 30} month{'s' if date_range // 30 > 1 else ''}"
        else:
            range_text = f"{date_range} day{'s' if date_range > 1 else ''}"
        
        target_container.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Date Range</div>
            <div class="kpi-value">{range_text}</div>
            <div class="kpi-trend-neutral">{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close the KPI section
    target_container.markdown("""
    </div>
    """, unsafe_allow_html=True)
    
    # Show top categories if categorical data available
    if cat_cols and len(cat_cols) > 0:
        # Select the categorical column with the lowest cardinality
        best_cat_col = min(
            [(col, column_analyses[col]["unique_count"]) for col in cat_cols], 
            key=lambda x: x[1]
        )[0]
        
        if column_analyses[best_cat_col]["unique_count"] <= 15:  # Only for reasonable cardinality
            target_container.markdown("<div class='chart-title'>Category Distribution</div>", unsafe_allow_html=True)
            
            col1, col2 = target_container.columns([1, 2])
            
            # Create distribution summary
            value_counts = df[best_cat_col].value_counts().reset_index()
            value_counts.columns = [best_cat_col, 'Count']
            value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
            
            # Show top categories table
            with col1:
                # Style the dataframe
                value_counts_styled = value_counts.head(5).style.format({'Percentage': '{:.1f}%'})
                value_counts_styled = value_counts_styled.bar(subset=['Percentage'], color='#3498db', vmin=0, vmax=100)
                
                target_container.dataframe(value_counts_styled, use_container_width=True, height=200)
            
            # Show pie chart
            with col2:
                create_enhanced_visualization(df, best_cat_col, column_analyses[best_cat_col], container=col2, key_suffix="kpi", height=250)

# Function to create the filter section for interactive dashboard
def create_filter_section(df, column_analyses, container=None):
    """Creates an interactive filter section for the dashboard"""
    target_container = container if container else st
    
    # Create a filter section with expandable UI
    with target_container.expander("Data Filters", expanded=False):
        # Select filter columns based on data types
        datetime_cols = [col for col, analysis in column_analyses.items() 
                        if analysis["viz_type"] == "time_series"]
        
        categorical_cols = [col for col, analysis in column_analyses.items() 
                           if analysis["type"] == "categorical" and analysis["unique_count"] <= 20]
        
        numeric_cols = [col for col, analysis in column_analyses.items() 
                       if analysis["type"] == "numeric"]
        
        # Store filters in session state if not already present
        if 'filters' not in st.session_state:
            st.session_state.filters = {
                'datetime': {},
                'categorical': {},
                'numeric': {}
            }
        
        # 1. Date/Time filters
        if datetime_cols:
            target_container.markdown("<div class='chart-title'>Date Range Filters</div>", unsafe_allow_html=True)
            
            col1, col2 = target_container.columns(2)
            
            # For each datetime column, add a date range filter
            for date_col in datetime_cols[:1]:  # Limit to 1 date column for simplicity
                date_min = df[date_col].min().date()
                date_max = df[date_col].max().date()
                
                # Create date range selector
                with col1:
                    start_date = target_container.date_input(
                        f"Start date for {date_col}",
                        value=date_min,
                        min_value=date_min,
                        max_value=date_max,
                        key=f"date_start_{date_col}"
                    )
                
                with col2:
                    end_date = target_container.date_input(
                        f"End date for {date_col}",
                        value=date_max,
                        min_value=date_min,
                        max_value=date_max,
                        key=f"date_end_{date_col}"
                    )
                
                # Store in session state
                st.session_state.filters['datetime'][date_col] = (start_date, end_date)
        
        # 2. Categorical filters
        if categorical_cols:
            target_container.markdown("<div class='chart-title'>Category Filters</div>", unsafe_allow_html=True)
            
            # Display up to 3 categorical filters in columns
            cols = target_container.columns(min(3, len(categorical_cols)))
            
            for i, cat_col in enumerate(categorical_cols[:3]):  # Limit to 3 categorical columns
                with cols[i % len(cols)]:
                    # Get unique values
                    unique_vals = sorted(df[cat_col].dropna().unique())
                    
                    # Create multi-select
                    selected_vals = target_container.multiselect(
                        f"Select {cat_col}",
                        options=unique_vals,
                        default=[],
                        key=f"cat_select_{cat_col}"
                    )
                    
                    # Store in session state
                    st.session_state.filters['categorical'][cat_col] = selected_vals
        
        # 3. Numeric range filters
        if numeric_cols:
            target_container.markdown("<div class='chart-title'>Numeric Range Filters</div>", unsafe_allow_html=True)
            
            # Display up to 3 numeric filters in columns
            cols = target_container.columns(min(3, len(numeric_cols)))
            
            for i, num_col in enumerate(numeric_cols[:3]):  # Limit to 3 numeric columns
                with cols[i % len(cols)]:
                    # Get min/max values
                    min_val = float(df[num_col].min())
                    max_val = float(df[num_col].max())
                    
                    # Create range slider
                    num_range = target_container.slider(
                        f"Range for {num_col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"num_range_{num_col}"
                    )
                    
                    # Store in session state
                    st.session_state.filters['numeric'][num_col] = num_range
        
        # Add reset button
        if target_container.button("Reset Filters", key="reset_filters"):
            # Reset all filters
            st.session_state.filters = {
                'datetime': {},
                'categorical': {},
                'numeric': {}
            }
            # Force rerun to update UI
            st.experimental_rerun()
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    
    # Apply datetime filters
    for date_col, (start_date, end_date) in st.session_state.filters['datetime'].items():
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df[date_col].dt.date >= start_date) & 
                                       (filtered_df[date_col].dt.date <= end_date)]
    
    # Apply categorical filters
    for cat_col, selected_vals in st.session_state.filters['categorical'].items():
        if selected_vals:  # Only apply if values are selected
            filtered_df = filtered_df[filtered_df[cat_col].isin(selected_vals)]
    
    # Apply numeric filters
    for num_col, (min_val, max_val) in st.session_state.filters['numeric'].items():
        filtered_df = filtered_df[(filtered_df[num_col] >= min_val) & 
                                   (filtered_df[num_col] <= max_val)]
    
    # Show filter summary
    if len(filtered_df) != len(df):
        target_container.markdown(f"""
        <div class="info-box">
            Showing {len(filtered_df):,} out of {len(df):,} records ({len(filtered_df)/len(df)*100:.1f}%) based on applied filters.
        </div>
        """, unsafe_allow_html=True)
    
    return filtered_df

# Function to create sample data for demo purposes
def create_sample_data(dataset_type):
    """Creates sample datasets for demonstration"""
    
    if dataset_type == "Sales Data":
        # Create a sample sales dataset
        import datetime
        import random
        
        # Define parameters
        num_rows = 1000
        products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
        regions = ["North", "South", "East", "West", "Central"]
        channels = ["Online", "Store", "Distributor"]
        
        # Start date - 2 years ago
        start_date = datetime.datetime.now() - datetime.timedelta(days=730)
        
        # Generate sample data
        data = {
            "Date": [start_date + datetime.timedelta(days=random.randint(0, 729)) for _ in range(num_rows)],
            "Product": [random.choice(products) for _ in range(num_rows)],
            "Region": [random.choice(regions) for _ in range(num_rows)],
            "Channel": [random.choice(channels) for _ in range(num_rows)],
            "Units": [random.randint(1, 100) for _ in range(num_rows)],
            "Price": [round(random.uniform(10, 1000), 2) for _ in range(num_rows)]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add calculated columns
        df["Revenue"] = df["Units"] * df["Price"]
        df["Cost"] = df["Price"] * random.uniform(0.4, 0.7)  # Random cost factor
        df["Profit"] = df["Revenue"] - (df["Cost"] * df["Units"])
        df["Profit_Margin"] = (df["Profit"] / df["Revenue"]) * 100
        
        # Add some seasonality for more interesting time series
        for idx, row in df.iterrows():
            month = row["Date"].month
            # Higher sales in November-December (holiday season)
            if month in [11, 12]:
                df.at[idx, "Units"] = int(df.at[idx, "Units"] * random.uniform(1.2, 1.5))
            # Lower sales in January-February
            elif month in [1, 2]:
                df.at[idx, "Units"] = int(df.at[idx, "Units"] * random.uniform(0.7, 0.9))
            
            # Recalculate revenue and profit
            df.at[idx, "Revenue"] = df.at[idx, "Units"] * df.at[idx, "Price"]
            df.at[idx, "Profit"] = df.at[idx, "Revenue"] - (df.at[idx, "Cost"] * df.at[idx, "Units"])
            df.at[idx, "Profit_Margin"] = (df.at[idx, "Profit"] / df.at[idx, "Revenue"]) * 100
        
        return df
    
    elif dataset_type == "Customer Survey":
        # Create a sample customer survey dataset
        import random
        
        # Generate sample data
        num_rows = 500
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
        genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
        satisfaction_levels = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
        product_categories = ["Electronics", "Clothing", "Home Goods", "Food & Beverage", "Health & Beauty"]
        countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Brazil", "India", "China"]
        
        # Create data dictionary
        data = {
            "Age_Group": [random.choice(age_groups) for _ in range(num_rows)],
            "Gender": [random.choice(genders) for _ in range(num_rows)],
            "Satisfaction": [random.choice(satisfaction_levels) for _ in range(num_rows)],
            "NPS_Score": [random.randint(0, 10) for _ in range(num_rows)],
            "Purchase_Frequency": [random.randint(1, 20) for _ in range(num_rows)],
            "Spending_Amount": [round(random.uniform(50, 500), 2) for _ in range(num_rows)],
            "Would_Recommend": [random.choice(["Yes", "No", "Maybe"]) for _ in range(num_rows)],
            "Product_Category": [random.choice(product_categories) for _ in range(num_rows)],
            "Country": [random.choice(countries) for _ in range(num_rows)]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add a survey date within the last year
        import datetime
        start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        df["Survey_Date"] = [start_date + datetime.timedelta(days=random.randint(0, 364)) for _ in range(num_rows)]
        
        # Add some missing values to make it more realistic
        for col in df.columns:
            mask = np.random.random(size=len(df)) < 0.02  # 2% missing rate
            df.loc[mask, col] = None
        
        # Add some relationships between variables
        # Satisfaction influences NPS and recommendation
        for idx, row in df.iterrows():
            if row["Satisfaction"] == "Very Satisfied":
                df.at[idx, "NPS_Score"] = random.randint(8, 10)
                df.at[idx, "Would_Recommend"] = random.choices(["Yes", "Maybe", "No"], weights=[0.9, 0.09, 0.01])[0]
            elif row["Satisfaction"] == "Satisfied":
                df.at[idx, "NPS_Score"] = random.randint(7, 9)
                df.at[idx, "Would_Recommend"] = random.choices(["Yes", "Maybe", "No"], weights=[0.7, 0.25, 0.05])[0]
            elif row["Satisfaction"] == "Neutral":
                df.at[idx, "NPS_Score"] = random.randint(4, 7)
                df.at[idx, "Would_Recommend"] = random.choices(["Yes", "Maybe", "No"], weights=[0.3, 0.5, 0.2])[0]
            elif row["Satisfaction"] == "Dissatisfied":
                df.at[idx, "NPS_Score"] = random.randint(1, 5)
                df.at[idx, "Would_Recommend"] = random.choices(["Yes", "Maybe", "No"], weights=[0.05, 0.15, 0.8])[0]
            elif row["Satisfaction"] == "Very Dissatisfied":
                df.at[idx, "NPS_Score"] = random.randint(0, 3)
                df.at[idx, "Would_Recommend"] = random.choices(["Yes", "Maybe", "No"], weights=[0.01, 0.09, 0.9])[0]
        
        return df
    
    elif dataset_type == "Stock Prices":
        # Create a sample stock price dataset
        import datetime
        import numpy as np
        import random
        
        # Generate sample data for stock prices
        start_date = datetime.datetime(2022, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
        
        # Create several stocks with different trends
        stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
        
        # Generate random walk prices with trends
        data = {"Date": dates}
        for stock in stocks:
            # Start with a random base price
            base_price = random.uniform(100, 1000)
            
            # Generate trend component (some stocks up, some down)
            trend_direction = random.choice([-1, 1])
            trend_magnitude = random.uniform(0, 200)
            trend = np.linspace(0, trend_direction * trend_magnitude, len(dates))
            
            # Generate random fluctuations
            noise = np.random.normal(0, base_price * 0.02, len(dates))
            
            # Calculate price series with cumulative noise
            prices = base_price + trend + noise.cumsum()
            
            # Ensure no negative prices
            prices = np.maximum(prices, 1)
            
            # Add to data
            data[stock] = prices
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add trading volume
        for stock in stocks:
            # Base volume
            base_volume = random.randint(1000000, 10000000)
            
            # Daily random variation
            volume_variation = np.random.lognormal(mean=0, sigma=0.3, size=len(dates))
            
            # Calculate volumes
            volumes = np.round(base_volume * volume_variation)
            
            # Add to dataframe
            df[f"{stock}_Volume"] = volumes
        
        # Add market index
        market_start = 10000
        market_trend = np.linspace(0, random.uniform(500, 1500), len(dates))
        market_noise = np.random.normal(0, 100, len(dates))
        market_index = market_start + market_trend + market_noise.cumsum()
        df["Market_Index"] = market_index
        
        return df
    
    # Fallback to empty dataframe
    return pd.DataFrame()

# Generate a dashboard title dynamically
def generate_dashboard_title(df, dataset_name=None):
    """Creates a descriptive title for the dashboard based on the data"""
    if dataset_name:
        return f"{dataset_name} Dashboard"
    
    # Try to infer a title from column names
    cols = df.columns.tolist()
    col_lower = [col.lower() for col in cols]
    
    if any(col in col_lower for col in ["sales", "revenue", "profit"]):
        return "Sales Performance Dashboard"
    elif any(col in col_lower for col in ["customer", "satisfaction", "nps", "survey"]):
        return "Customer Insights Dashboard"
    elif any(col in col_lower for col in ["stock", "price", "market", "index"]):
        return "Financial Market Dashboard"
    elif any(col in col_lower for col in ["product", "inventory", "stock", "quantity"]):
        return "Product Analytics Dashboard"
    elif any(col in col_lower for col in ["employee", "hr", "salary", "performance"]):
        return "HR Analytics Dashboard"
    else:
        # Generic title
        return "Data Analytics Dashboard"

# Function to get current date/time formatted nicely
def get_formatted_datetime():
    """Returns the current date and time formatted for display"""
    now = datetime.now()
    return now.strftime("%B %d, %Y at %I:%M %p")

# Function to create download report as PDF (simulated)
def create_download_report(df, report_type="PDF"):
    """Creates a simulated report download button"""
    # This would typically generate a PDF report
    # For this demo, we'll create a markdown string as a placeholder
    report_content = f"""# Data Analytics Report
Generated on {get_formatted_datetime()}

## Dataset Summary
- Total Records: {len(df):,}
- Number of Columns: {df.shape[1]:,}
- Column Types: {len(df.select_dtypes(include=['number']).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical

## Data Preview
```
{df.head(5).to_string()}
```

## Key Statistics
```
{df.describe().to_string()}
```
"""
    
    # Encode the report content
    report_bytes = report_content.encode()
    
    # Return a download button
    return st.download_button(
        label=f"Download {report_type} Report",
        data=report_bytes,
        file_name=f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="download_report"
    )

# Main application logic
def main():
    # Sidebar controls
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>Dashboard Controls</h3>", unsafe_allow_html=True)
        
        # Add logo/branding
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 28px; font-weight: 700; color: #3498db;">📊 SmartDashPro</div>
            <div style="font-size: 14px; color: #7f8c8d;">Intelligent Data Visualization</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme selection (for future implementation)
        # theme_options = ["Default", "Blue", "Green", "Red", "Purple", "Dark"]
        # selected_theme = st.selectbox("Select color theme:", theme_options, label_visibility="collapsed")
        
        # Upload section
        st.markdown("<div style='margin-top: 20px; font-weight: 500;'>Load Data</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
        
        # Sample dataset selection
        st.markdown("<div style='margin-top: 10px;'>Or try a sample dataset:</div>", unsafe_allow_html=True)
        sample_option = st.selectbox(
            "Sample Dataset",
            ["None", "Sales Data", "Customer Survey", "Stock Prices"],
            label_visibility="collapsed"
        )
        
        # Advanced options
        st.markdown("<div style='margin-top: 20px; font-weight: 500;'>Display Options</div>", unsafe_allow_html=True)
        show_data_quality = st.checkbox("Show Data Quality Score", value=True)
        show_statistics = st.checkbox("Show Advanced Statistics", value=False)
        show_recommendations = st.checkbox("Show Smart Recommendations", value=True)
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
        
        # Export options
        st.markdown("<div style='margin-top: 20px; font-weight: 500;'>Export Options</div>", unsafe_allow_html=True)
        
        # These would do something in a full implementation
        if st.button("Save Dashboard", key="save_dashboard"):
            st.success("Dashboard state saved successfully!")
        
        if st.button("Share Dashboard", key="share_dashboard"):
            st.info("Dashboard sharing options would appear here.")
        
        # About section
        st.markdown("""
        <div style='margin-top: 50px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
            <div style='font-size: 14px; font-weight: 500; text-align: center; margin-bottom: 10px;'>About</div>
            <p style='font-size: 12px; color: #7f8c8d; text-align: center;'>
                SmartDashPro automatically analyzes your data and generates intelligent visualizations based on content.
            </p>
            <p style='font-size: 12px; color: #7f8c8d; text-align: center;'>
                Version 2.0
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process the uploaded file or sample dataset
    df = None
    file_source = None
    
    if uploaded_file is not None:
        # Load the uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                file_source = uploaded_file.name
            else:
                df = pd.read_excel(uploaded_file)
                file_source = uploaded_file.name
                
            # Convert date columns to datetime
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass  # Skip if conversion fails
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    elif sample_option != "None":
        # Load sample dataset
        df = create_sample_data(sample_option)
        file_source = f"Sample: {sample_option}"
    
    # If no data is loaded, show the welcome screen
    if df is None:
        # Show welcome screen
        st.markdown("""
        <div class="welcome-container">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
            <h1>Welcome to SmartDashPro</h1>
            <p style="font-size: 1.2rem; max-width: 600px; margin: 1rem auto;">
                Upload your CSV or Excel file to get started, or select a sample dataset from the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show features
        st.markdown("<h2 style='text-align: center; margin: 2rem 0 1rem 0;'>Features</h2>", unsafe_allow_html=True)
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Smart Visualizations</div>
                <div class="feature-description">
                    Automatically detects the best chart type for your data and creates beautiful, interactive visualizations.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with feature_cols[1]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Intelligent Insights</div>
                <div class="feature-description">
                    Uncovers hidden patterns and relationships in your data with advanced statistical analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with feature_cols[2]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Interactive Filters</div>
                <div class="feature-description">
                    Drill down into your data with powerful filtering capabilities to find exactly what you need.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        feature_cols2 = st.columns(3)
        
        with feature_cols2[0]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📱</div>
                <div class="feature-title">Responsive Design</div>
                <div class="feature-description">
                    Beautiful dashboards that look great on any device, from desktop to mobile.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with feature_cols2[1]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📋</div>
                <div class="feature-title">Data Quality Analysis</div>
                <div class="feature-description">
                    Automatically identifies missing values, outliers, and other data quality issues.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with feature_cols2[2]:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔄</div>
                <div class="feature-title">Multiple Export Options</div>
                <div class="feature-description">
                    Export your visualizations and reports in various formats for sharing and presentation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show how it works
        st.markdown("<h2 style='text-align: center; margin: 3rem 0 1rem 0;'>How It Works</h2>", unsafe_allow_html=True)
        
        steps_cols = st.columns(4)
        
        with steps_cols[0]:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">1️⃣</div>
                <div style="font-weight: 600; margin: 0.5rem 0;">Upload Data</div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">Upload your CSV or Excel file through the sidebar</div>
            </div>
            """, unsafe_allow_html=True)
        
        with steps_cols[1]:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">2️⃣</div>
                <div style="font-weight: 600; margin: 0.5rem 0;">Analyze</div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">SmartDashPro automatically analyzes your data structure</div>
            </div>
            """, unsafe_allow_html=True)
        
        with steps_cols[2]:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">3️⃣</div>
                <div style="font-weight: 600; margin: 0.5rem 0;">Visualize</div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">Intelligent visualizations are generated based on data</div>
            </div>
            """, unsafe_allow_html=True)
        
        with steps_cols[3]:
            st.markdown("""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">4️⃣</div>
                <div style="font-weight: 600; margin: 0.5rem 0;">Interact</div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">Explore your data through filters and interactive charts</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample screenshot
        st.markdown("<div style='margin: 3rem 0; text-align: center;'>", unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2020/07/08/04/12/work-5382501_960_720.jpg", caption="Sample Dashboard View", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # End welcome screen
        st.markdown("<div class='dashboard-footer'>Get started by uploading your data file using the sidebar!</div>", unsafe_allow_html=True)
        return
    
    # If data is loaded, proceed with dashboard creation
    # Start with a progress animation
    with st.spinner("Creating interactive dashboard..."):
        # Analyze all columns
        column_analyses = {}
        
        for col in df.columns:
            column_analyses[col] = analyze_column(df, col)
        
        # Apply filters if any
        filtered_df = create_filter_section(df, column_analyses)
        
        # Create dashboard header with title and summary
        dashboard_title = generate_dashboard_title(df, file_source)
        
        st.markdown(f"""
        <div class="dashboard-header">
            <div class="dashboard-title">{dashboard_title}</div>
            <div class="dashboard-subtitle">
                {len(filtered_df):,} records • {df.shape[1]} columns • Generated on {get_formatted_datetime()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create key metrics section
        create_key_metrics(filtered_df, column_analyses)
        
        # Main dashboard tabs
        tabs = st.tabs(["📊 Dashboard", "📈 Advanced Insights", "🔍 Data Explorer", "📋 Data Quality"])
        
        # Tab 1: Main Dashboard
        with tabs[0]:
            # Select most interesting columns for visualization
            # Prioritize visualization types
            viz_priority = ["time_series", "pie_chart", "bar_chart", "histogram", "box_plot", "scatter_plot"]
            priority_cols = []
            
            # Add one column of each priority type if available
            for viz_type in viz_priority:
                cols = [col for col, analysis in column_analyses.items() if analysis["viz_type"] == viz_type]
                if cols:
                    priority_cols.append(cols[0])
                    if len(priority_cols) >= 4:  # Limit to 4 visualizations
                        break
            
            # Create two-column layout for visualizations
            if priority_cols:
                st.markdown("<h3>Key Visualizations</h3>", unsafe_allow_html=True)
                
                viz_cols = st.columns(2)
                
                for i, col_name in enumerate(priority_cols[:4]):  # Limit to 4 visualizations
                    with viz_cols[i % 2]:
                        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
                        create_enhanced_visualization(filtered_df, col_name, column_analyses[col_name], key_suffix=f"main_{i}")
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Add recommended visualizations if enabled
            if show_recommendations:
                recommendations = recommend_related_visualizations(filtered_df, column_analyses)
                
                if recommendations:
                    st.markdown("<h3>Recommended Insights</h3>", unsafe_allow_html=True)
                    
                    # Show first recommendation in full width
                    if len(recommendations) > 0:
                        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="chart-title">
                            <span style="margin-right: 8px;">{recommendations[0]['icon']}</span>
                            {recommendations[0]['title']}
                        </div>
                        <p style="color: #7f8c8d; margin-bottom: 1rem;">{recommendations[0]['description']}</p>
                        """, unsafe_allow_html=True)
                        
                        create_recommended_visualization(filtered_df, recommendations[0], key_id=f"rec_main_{0}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show next 2 recommendations in columns if available
                    if len(recommendations) > 1:
                        rec_cols = st.columns(2)
                        
                        for i, rec in enumerate(recommendations[1:3]):  # Show at most 2 more
                            with rec_cols[i % 2]:
                                st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="chart-title">
                                    <span style="margin-right: 8px;">{rec['icon']}</span>
                                    {rec['title']}
                                </div>
                                <p style="color: #7f8c8d; margin-bottom: 1rem;">{rec['description']}</p>
                                """, unsafe_allow_html=True)
                                
                                create_recommended_visualization(filtered_df, rec, key_id=f"rec_main_{i+1}")
                                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add download options
            st.markdown("<h3>Export Options</h3>", unsafe_allow_html=True)
            download_cols = st.columns([2, 1])
            
            with download_cols[0]:
                create_download_options(filtered_df, filename_prefix="dashboard_data")
            
            with download_cols[1]:
                create_download_report(filtered_df)
        
        # Tab 2: Advanced Insights
        with tabs[1]:
            # Show statistical overview
            if show_statistics:
                create_statistical_overview(filtered_df, column_analyses)
            
            # Correlation analysis for numeric columns
            numeric_cols = [col for col, analysis in column_analyses.items() 
                           if analysis["type"] == "numeric"]
            
            if len(numeric_cols) > 1:
                st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                create_enhanced_correlation_heatmap(filtered_df)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show all recommendations
            recommendations = recommend_related_visualizations(filtered_df, column_analyses)
            
            if recommendations:
                st.markdown("<h3>All Recommended Visualizations</h3>", unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations[:6]):  # Limit to 6 recommendations
                    with st.expander(f"{rec['icon']} {rec['title']}", expanded=(i == 0)):
                        st.markdown(f"<p style='color: #7f8c8d;'>{rec['description']}</p>", unsafe_allow_html=True)
                        create_recommended_visualization(filtered_df, rec, key_id=f"rec_all_{i}")
        
        # Tab 3: Data Explorer
        with tabs[2]:
            # Create column selector
            st.markdown("<h3>Column Explorer</h3>", unsafe_allow_html=True)
            
            explorer_cols = st.columns([1, 3])
            
            with explorer_cols[0]:
                st.markdown("<div class='chart-card' style='height: 100%;'>", unsafe_allow_html=True)
                # Group columns by type
                numeric_cols = [col for col, analysis in column_analyses.items() if analysis["type"] == "numeric"]
                categorical_cols = [col for col, analysis in column_analyses.items() if analysis["type"] == "categorical"]
                datetime_cols = [col for col, analysis in column_analyses.items() if analysis["viz_type"] == "time_series"]
                other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
                
                # Create column selector organized by type
                st.markdown("<h4>Select Column</h4>", unsafe_allow_html=True)
                
                # Use expanders for each column type
                with st.expander("Numeric Columns", expanded=True):
                    selected_column = None
                    for col in numeric_cols:
                        if st.button(col, key=f"select_{col}", use_container_width=True):
                            selected_column = col
                
                with st.expander("Categorical Columns", expanded=True):
                    for col in categorical_cols:
                        if st.button(col, key=f"select_{col}", use_container_width=True):
                            selected_column = col
                
                with st.expander("Date/Time Columns", expanded=True):
                    for col in datetime_cols:
                        if st.button(col, key=f"select_{col}", use_container_width=True):
                            selected_column = col
                
                with st.expander("Other Columns", expanded=False):
                    for col in other_cols:
                        if st.button(col, key=f"select_{col}", use_container_width=True):
                            selected_column = col
                            
                st.markdown("</div>", unsafe_allow_html=True)
            
            with explorer_cols[1]:
                # If a column is selected, show its visualization and details
                if 'selected_column' in locals() and selected_column is not None:
                    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                    
                    # Display column metadata
                    analysis = column_analyses[selected_column]
                    
                    st.markdown(f"<h4>{selected_column}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #7f8c8d; margin-bottom: 1rem;'>{analysis['description']}</p>", unsafe_allow_html=True)
                    
                    # Display column properties
                    prop_cols = st.columns(3)
                    prop_cols[0].metric("Type", analysis["type"].capitalize())
                    prop_cols[1].metric("Unique Values", analysis["unique_count"])
                    
                    missing_pct = (filtered_df[selected_column].isna().sum() / len(filtered_df)) * 100
                    prop_cols[2].metric("Missing", f"{missing_pct:.1f}%")
                    
                    # Display column visualization
                    create_enhanced_visualization(filtered_df, selected_column, analysis, key_suffix="explorer")
                    
                    # Display column insight if available
                    if analysis["insight"]:
                        st.markdown(f"""
                        <div class="info-box">
                            <b>Insight:</b> {analysis["insight"]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("👈 Select a column from the left panel to explore")
            
            # Add data preview section
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            
            preview_tabs = st.tabs(["Table View", "Column Summary", "Statistics"])
            
            with preview_tabs[0]:
                # Add row selector
                row_count = len(filtered_df)
                preview_rows = st.slider("Number of rows to display", min_value=5, max_value=min(100, row_count), value=10)
                st.dataframe(filtered_df.head(preview_rows), use_container_width=True)
                
                # Add pagination controls (simulated)
                pagination_cols = st.columns([1, 3, 1])
                with pagination_cols[1]:
                    st.markdown("""
                    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                        <button disabled style="background-color: #f0f2f6; color: #7f8c8d; border: none; padding: 5px 15px; border-radius: 5px;">Previous</button>
                        <span style="padding: 5px 10px;">Page 1 of 10</span>
                        <button style="background-color: #3498db; color: white; border: none; padding: 5px 15px; border-radius: 5px;">Next</button>
                    </div>
                    """, unsafe_allow_html=True)
            
            with preview_tabs[1]:
                # Show column info
                column_info = []
                
                for col in filtered_df.columns:
                    info = {
                        'Column': col,
                        'Type': column_analyses[col]["type"].capitalize(),
                        'Unique Values': column_analyses[col]["unique_count"],
                        'Missing Values': filtered_df[col].isna().sum(),
                        'Missing %': round(filtered_df[col].isna().sum() / len(filtered_df) * 100, 2),
                        'Visualization': column_analyses[col]["viz_type"].replace("_", " ").title()
                    }
                    column_info.append(info)
                
                column_df = pd.DataFrame(column_info)
                
                # Style the dataframe
                st.dataframe(column_df.style.background_gradient(subset=['Missing %'], cmap='YlOrRd'), use_container_width=True)
            
            with preview_tabs[2]:
                if numeric_cols:
                    st.dataframe(filtered_df[numeric_cols].describe().T, use_container_width=True)
                else:
                    st.info("No numeric columns available for statistics")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 4: Data Quality
        with tabs[3]:
            # Show data quality assessment
            create_enhanced_data_quality(filtered_df, column_analyses)
            
            # Display data summary and metadata
            st.markdown("<h3>Dataset Information</h3>", unsafe_allow_html=True)
            
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.markdown("<div class='chart-title'>Dataset Structure</div>", unsafe_allow_html=True)
                
                # Create dataset info table
                info_data = {
                    "Metric": [
                        "Number of Rows", 
                        "Number of Columns", 
                        "Numeric Columns", 
                        "Categorical Columns", 
                        "DateTime Columns",
                        "Other Columns",
                        "Memory Usage",
                        "Duplicate Rows"
                    ],
                    "Value": [
                        f"{len(filtered_df):,}",
                        f"{filtered_df.shape[1]}",
                        f"{len([col for col, analysis in column_analyses.items() if analysis['type'] == 'numeric'])}",
                        f"{len([col for col, analysis in column_analyses.items() if analysis['type'] == 'categorical'])}",
                        f"{len([col for col, analysis in column_analyses.items() if analysis['viz_type'] == 'time_series'])}",
                        f"{len([col for col, analysis in column_analyses.items() if analysis['type'] not in ['numeric', 'categorical'] and analysis['viz_type'] != 'time_series'])}",
                        f"{filtered_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
                        f"{filtered_df.duplicated().sum():,} ({filtered_df.duplicated().sum() / len(filtered_df) * 100:.2f}%)"
                    ]
                }
                
                info_df = pd.DataFrame(info_data)
                st.table(info_df)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with info_cols[1]:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.markdown("<div class='chart-title'>Data Completeness</div>", unsafe_allow_html=True)
                
                # Calculate completeness by column
                completeness_data = {
                    'Column': [],
                    'Complete': [],
                    'Missing': [],
                    'Percent Complete': []
                }
                
                for col in filtered_df.columns:
                    missing = filtered_df[col].isna().sum()
                    complete = len(filtered_df) - missing
                    complete_pct = complete / len(filtered_df) * 100
                    
                    completeness_data['Column'].append(col)
                    completeness_data['Complete'].append(complete)
                    completeness_data['Missing'].append(missing)
                    completeness_data['Percent Complete'].append(complete_pct)
                
                completeness_df = pd.DataFrame(completeness_data)
                
                # Create bar chart of completeness
                fig = px.bar(
                    completeness_df.sort_values('Percent Complete'),
                    y='Column',
                    x='Percent Complete',
                    orientation='h',
                    color='Percent Complete',
                    color_continuous_scale=px.colors.sequential.Blues,
                    title="Data Completeness by Column",
                    text='Percent Complete'
                )
                
                fig.update_traces(
                    texttemplate='%{x:.1f}%',
                    textposition='outside'
                )
                
                fig.update_layout(
                    xaxis=dict(
                        title="Percent Complete",
                        range=[0, 100],
                        ticksuffix="%"
                    ),
                    yaxis=dict(
                        title=""
                    ),
                    coloraxis_showscale=False,
                    height=max(400, len(filtered_df.columns) * 30),
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Data cleaning recommendations section
            if show_data_quality:
                st.markdown("<h3>Data Improvement Suggestions</h3>", unsafe_allow_html=True)
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                
                # Generate automatic recommendations
                suggestions = []
                
                # Check for missing values
                high_missing_cols = []
                for col in filtered_df.columns:
                    missing_pct = filtered_df[col].isna().sum() / len(filtered_df) * 100
                    if missing_pct > 15:
                        high_missing_cols.append((col, missing_pct))
                
                if high_missing_cols:
                    cols_text = ", ".join([f"<b>{col}</b> ({pct:.1f}%)" for col, pct in high_missing_cols[:3]])
                    if len(high_missing_cols) > 3:
                        cols_text += f" and {len(high_missing_cols) - 3} more"
                    
                    suggestions.append({
                        "title": "Handle Missing Values",
                        "description": f"Consider handling missing values in columns: {cols_text}",
                        "icon": "🔍",
                        "severity": "high" if any(pct > 30 for _, pct in high_missing_cols) else "medium"
                    })
                
                # Check for duplicates
                duplicate_pct = filtered_df.duplicated().sum() / len(filtered_df) * 100
                if duplicate_pct > 1:
                    suggestions.append({
                        "title": "Remove Duplicate Rows",
                        "description": f"Found {filtered_df.duplicated().sum():,} duplicate rows ({duplicate_pct:.1f}% of data)",
                        "icon": "🔄",
                        "severity": "high" if duplicate_pct > 10 else "medium" if duplicate_pct > 5 else "low"
                    })
                
                # Check for outliers in numeric columns
                outlier_cols = []
                for col in numeric_cols:
                    try:
                        q1 = filtered_df[col].quantile(0.25)
                        q3 = filtered_df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = filtered_df[(filtered_df[col] < lower_bound) | (filtered_df[col] > upper_bound)][col]
                        outlier_pct = len(outliers) / len(filtered_df) * 100
                        
                        if outlier_pct > 5:
                            outlier_cols.append((col, outlier_pct))
                    except:
                        continue
                
                if outlier_cols:
                    cols_text = ", ".join([f"<b>{col}</b> ({pct:.1f}%)" for col, pct in outlier_cols[:3]])
                    if len(outlier_cols) > 3:
                        cols_text += f" and {len(outlier_cols) - 3} more"
                    
                    suggestions.append({
                        "title": "Handle Outliers",
                        "description": f"Consider addressing outliers in columns: {cols_text}",
                        "icon": "📉",
                        "severity": "medium"
                    })
                
                # Check for potential date columns
                date_candidates = []
                for col, analysis in column_analyses.items():
                    if analysis["type"] != "datetime" and analysis["semantic"] == "date":
                        date_candidates.append(col)
                
                if date_candidates:
                    cols_text = ", ".join([f"<b>{col}</b>" for col in date_candidates[:3]])
                    if len(date_candidates) > 3:
                        cols_text += f" and {len(date_candidates) - 3} more"
                    
                    suggestions.append({
                        "title": "Convert Date Columns",
                        "description": f"These columns might contain dates that could be converted: {cols_text}",
                        "icon": "📅",
                        "severity": "low"
                    })
                
                # Display suggestions
                if suggestions:
                    for suggestion in suggestions:
                        severity_color = "#e74c3c" if suggestion["severity"] == "high" else \
                                        "#f39c12" if suggestion["severity"] == "medium" else \
                                        "#3498db"
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid {severity_color}; padding: 1rem; margin-bottom: 1rem; background-color: white; border-radius: 4px;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{suggestion["icon"]}</span>
                                <span style="font-size: 1.1rem; font-weight: 600;">{suggestion["title"]}</span>
                            </div>
                            <p style="margin: 0; color: #555;">{suggestion["description"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No significant data quality issues detected. Your data looks good!")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        # Add dashboard footer
        st.markdown(f"""
        <div class="dashboard-footer">
            Dashboard generated by SmartDashPro • Data source: {file_source} • 
            Updated: {get_formatted_datetime()}
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
