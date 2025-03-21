import streamlit as st
import pandas as pd
import numpy as np
import re
import sys
import importlib.util
import os
import uuid

# Set page configuration
st.set_page_config(
    page_title="Smart Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding:5%;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #777;
    }
    .card {
        border-radius: 10px;
        background-color: white;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    .plotly-chart {
        border-radius: 8px;
        padding: 8px;
        background-color: white;
    }
    .recommendation-card {
        border-left: 4px solid #4e8df5;
        background-color: white;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    }
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        color: #888;
        font-size: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Check for required packages
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Dashboard header
st.markdown('<h1 style="text-align: center; color: #2e4057;">Smart Data Dashboard</h1>', unsafe_allow_html=True)

# App description
st.markdown("""
<div style="text-align: center; padding: 10px; margin-bottom: 30px;">
    Upload your CSV or Excel file and get an automatically generated beautiful dashboard with smart visualizations tailored to your data!
</div>
""", unsafe_allow_html=True)

# Function to detect column data types and semantics
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
    is_datetime = pd.api.types.is_datetime64_dtype(col_data)
    
    # Check if column is categorical/nominal
    unique_ratio = len(col_data.unique()) / len(col_data)
    is_categorical = unique_ratio < 0.1 and len(col_data.unique()) <= 20
    
    # Check if nominal column appears to be binary/boolean
    is_binary = is_categorical and len(col_data.unique()) <= 3
    
    # Check for common column name patterns
    col_lower = column_name.lower()
    
    # Analyze column name for semantic meaning
    name_indicators = {
        "gender": ["gender", "sex"],
        "date": ["date", "time", "day", "month", "year", "created", "updated"],
        "location": ["country", "city", "state", "region", "province", "location", "address"],
        "category": ["category", "type", "group", "class", "segment"],
        "name": ["name", "title", "label"],
        "id": ["id", "code", "identifier", "key"],
        "price": ["price", "cost", "fee", "amount", "payment", "revenue", "sales"],
        "quantity": ["quantity", "count", "number", "amount", "total"],
        "percentage": ["percent", "percentage", "ratio", "rate"],
        "age": ["age", "years"],
        "rating": ["rating", "score", "rank", "grade"]
    }
    
    col_semantic = "other"
    for semantic, indicators in name_indicators.items():
        if any(indicator in col_lower or col_lower.endswith(indicator) or col_lower.startswith(indicator) for indicator in indicators):
            col_semantic = semantic
            break
    
    # Determine best visualization type
    viz_type = "table"  # Default
    description = ""
    
    if is_datetime:
        viz_type = "time_series"
        description = "Datetime column, good for trend analysis over time"
    elif is_numeric:
        if "age" in col_semantic:
            viz_type = "histogram"
            description = "Age distribution"
        elif "price" in col_semantic or "quantity" in col_semantic:
            # Check distribution skew
            if len(col_data) > 10:
                if SCIPY_AVAILABLE:
                    skewness = stats.skew(col_data)
                    if abs(skewness) > 1.5:
                        viz_type = "box_plot"
                        description = f"Skewed numeric data (skew: {skewness:.2f}), box plot shows distribution with outliers"
                    else:
                        viz_type = "histogram"
                        description = "Numeric data with relatively normal distribution"
                else:
                    # Simple skewness approximation without scipy
                    median = col_data.median()
                    mean = col_data.mean()
                    std = col_data.std()
                    if std == 0:
                        simple_skew = 0
                    else:
                        simple_skew = 3 * (mean - median) / std
                    
                    if abs(simple_skew) > 1:
                        viz_type = "box_plot"
                        description = "Likely skewed numeric data, box plot shows distribution with outliers"
                    else:
                        viz_type = "histogram"
                        description = "Numeric data with relatively normal distribution"
            else:
                viz_type = "bar_chart"
                description = "Small set of numeric values"
        elif "percentage" in col_semantic or "rating" in col_semantic:
            viz_type = "bar_chart"
            description = "Percentage or rating data"
        else:
            # Check for high cardinality
            if unique_ratio > 0.8:
                viz_type = "scatter_plot"
                description = "High-cardinality numeric data, might be continuous values"
            else:
                viz_type = "histogram"
                description = "Numeric data, histogram shows distribution"
    elif is_binary:
        if col_semantic == "gender":
            viz_type = "pie_chart"
            description = "Gender distribution"
        else:
            viz_type = "bar_chart"
            description = "Binary categorical data"
    elif is_categorical:
        if col_semantic == "location":
            viz_type = "map" if "country" in col_lower else "bar_chart"
            description = "Location data" if "country" in col_lower else "Location data (non-country)"
        elif len(col_data.unique()) <= 10:
            viz_type = "pie_chart"
            description = f"Categorical data with {len(col_data.unique())} categories"
        else:
            viz_type = "bar_chart"
            description = f"Categorical data with {len(col_data.unique())} categories"
    else:
        if col_semantic == "name" or col_semantic == "id":
            viz_type = "table"
            description = "Identifier column, best displayed in table"
        else:
            if unique_ratio > 0.5:
                viz_type = "table"
                description = "High-cardinality text column, likely unique identifiers"
            else:
                viz_type = "word_cloud" if len(col_data.unique()) > 20 else "bar_chart"
                description = "Text data with moderate cardinality"
    
    return {
        "type": "numeric" if is_numeric else "datetime" if is_datetime else "categorical" if is_categorical else "text",
        "is_binary": is_binary,
        "unique_count": len(col_data.unique()),
        "unique_ratio": unique_ratio,
        "viz_type": viz_type,
        "semantic": col_semantic,
        "description": description
    }

# Function to create enhanced visualizations based on column analysis
def create_enhanced_visualization(df, column_name, analysis, key_suffix=""):
    """Creates a better styled visualization based on column analysis"""
    
    if analysis["viz_type"] == "none":
        st.write(f"No visualization available for empty column: {column_name}")
        return
    
    col_data = df[column_name].dropna()
    
    # Common plot styling
    plot_layout = {
        "margin": dict(l=10, r=10, t=50, b=10),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="Arial, sans-serif", size=12),
        "title": dict(font=dict(size=16, family="Arial, sans-serif", color="#2e4057"))
    }
    
    if analysis["viz_type"] == "pie_chart":
        # Get value counts and calculate percentages
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
        
        # Create enhanced pie chart with plotly
        fig = px.pie(
            value_counts, 
            values='Count', 
            names=column_name,
            title=f'<b>Distribution of {column_name}</b>',
            hover_data=['Percentage'],
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={column_name: column_name, 'Count': 'Count'}
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]:.1f}%<extra></extra>'
        )
        fig.update_layout(**plot_layout)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                      key=f"pie_{column_name}_{key_suffix}")
        
        # Show data table with counts and percentages
        with st.expander("See detailed data"):
            st.dataframe(value_counts.style.format({'Percentage': '{:.1f}%'}), use_container_width=True)
    
    elif analysis["viz_type"] == "bar_chart":
        # Get value counts
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        
        # Sort by count descending
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 categories if there are many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f'<b>Top 20 {column_name} by Count</b>'
        else:
            title = f'<b>Distribution of {column_name}</b>'
        
        # Choose colors based on whether the chart is showing a ranking
        colors = None
        if len(value_counts) > 3:
            colors = ['#1f77b4' if i < 3 else '#a5b7ce' for i in range(len(value_counts))]
        
        # Create enhanced bar chart
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=title,
            text='Count',
            color_discrete_sequence=['#1f77b4'] if colors is None else None,
            color=column_name if colors is not None else None,
            opacity=0.8
        )
        
        # If custom colors are defined, apply them
        if colors:
            fig.update_traces(marker_color=colors)
            
        fig.update_traces(
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 5:
            fig.update_layout(xaxis_tickangle=-45)
            
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title="", tickfont=dict(size=11)),
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                      key=f"bar_{column_name}_{key_suffix}")
        
        with st.expander("See detailed data"):
            st.dataframe(value_counts, use_container_width=True)
    
    elif analysis["viz_type"] == "histogram":
        # Calculate mean and median to show on the plot
        mean_val = col_data.mean()
        median_val = col_data.median()
        
        # Create enhanced histogram
        fig = px.histogram(
            df, 
            x=column_name,
            title=f'<b>Distribution of {column_name}</b>',
            marginal="box",
            color_discrete_sequence=['#4e8df5'],
            opacity=0.7,
            nbins=min(50, int(len(col_data) / 5)) if len(col_data) > 50 else 10
        )
        
        # Add mean and median lines
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#ff7f0e", 
                      annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#2ca02c", 
                      annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom right")
        
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title=column_name, tickfont=dict(size=11)),
            yaxis=dict(title="Frequency", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                      key=f"hist_{column_name}_{key_suffix}")
        
        # Show summary statistics
        with st.expander("See statistics"):
            stats_cols = st.columns(4)
            stats_cols[0].metric("Mean", f"{mean_val:.2f}")
            stats_cols[1].metric("Median", f"{median_val:.2f}")
            stats_cols[2].metric("Std Dev", f"{col_data.std():.2f}")
            stats_cols[3].metric("Range", f"{col_data.min():.2f} - {col_data.max():.2f}")
    
    elif analysis["viz_type"] == "box_plot":
        # Create enhanced box plot
        fig = px.box(
            df, 
            y=column_name,
            title=f'<b>Distribution of {column_name}</b>',
            points="outliers",
            color_discrete_sequence=['#4e8df5']
        )
        
        fig.update_layout(
            **plot_layout,
            yaxis=dict(title=column_name, showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
            xaxis=dict(title="", showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                      key=f"box_{column_name}_{key_suffix}")
        
        # Show outlier statistics
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        stat_cols = st.columns(4)
        stat_cols[0].metric("Q1 (25%)", f"{Q1:.2f}")
        stat_cols[1].metric("Median", f"{col_data.median():.2f}")
        stat_cols[2].metric("Q3 (75%)", f"{Q3:.2f}")
        stat_cols[3].metric("Outliers", f"{len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)")
    
    elif analysis["viz_type"] == "time_series":
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(col_data):
            try:
                col_data = pd.to_datetime(col_data)
            except:
                st.warning(f"Could not convert {column_name} to datetime format.")
                return
        
        # Group by date components based on date range
        date_range = (col_data.max() - col_data.min()).days
        
        if date_range > 365 * 2:
            # Group by month for multi-year data
            time_grouped = df.assign(date_group=col_data.dt.to_period('M')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Month'
        elif date_range > 60:
            # Group by week for multi-month data
            time_grouped = df.assign(date_group=col_data.dt.to_period('W')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Week'
        else:
            # Group by day for shorter periods
            time_grouped = df.assign(date_group=col_data.dt.to_period('D')).groupby('date_group').size()
            time_df = pd.DataFrame({
                'Date': time_grouped.index.to_timestamp(),
                'Count': time_grouped.values
            })
            x_title = 'Day'
        
        # Create enhanced time series plot
        fig = px.line(
            time_df, 
            x='Date', 
            y='Count',
            title=f'<b>Time Series of {column_name}</b>',
            labels={'Count': 'Frequency', 'Date': x_title},
            markers=True
        )
        
        # Add trend line
        trend_data = time_df.copy()
        trend_data['DateNum'] = range(len(trend_data))
        import numpy as np
        z = np.polyfit(trend_data['DateNum'], trend_data['Count'], 1)
        trend_data['Trend'] = np.poly1d(z)(trend_data['DateNum'])
        
        fig.add_trace(go.Scatter(
            x=trend_data['Date'],
            y=trend_data['Trend'],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash'),
            name='Trend'
        ))
        
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title=x_title, tickfont=dict(size=11)),
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                       key=f"time_{column_name}_{key_suffix}")
        
        # Show time period statistics
        time_stats_cols = st.columns(3)
        time_stats_cols[0].metric("Total Records", time_df['Count'].sum())
        time_stats_cols[1].metric("Peak Period", 
                                time_df.loc[time_df['Count'].idxmax(), 'Date'].strftime('%Y-%m-%d'))
        
        # Calculate trend direction
        trend_direction = "Increasing" if z[0] > 0 else "Decreasing" if z[0] < 0 else "Stable"
        trend_color = "green" if z[0] > 0 else "red" if z[0] < 0 else "gray"
        time_stats_cols[2].markdown(f"<h3>Trend</h3><p style='color:{trend_color}'>{trend_direction}</p>", unsafe_allow_html=True)
    
    elif analysis["viz_type"] == "scatter_plot":
        # Find another numeric column to pair with if available
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != column_name]
        
        if numeric_cols:
            # Use the first numeric column as default y-axis
            pair_col = numeric_cols[0]
            
            # Add dropdown to select different y variables if there are multiple options
            if len(numeric_cols) > 1:
                pair_col = st.selectbox(
                    f"Select variable to plot against {column_name}:",
                    numeric_cols,
                    key=f"scatter_select_{key_suffix}"
                )
            
            # Create enhanced scatter plot
            fig = px.scatter(
                df, 
                x=column_name, 
                y=pair_col,
                title=f'<b>Relationship between {column_name} and {pair_col}</b>',
                opacity=0.7,
                color_discrete_sequence=['#4e8df5'],
                trendline="ols",
                trendline_color_override="#ff7f0e"
            )
            
            fig.update_layout(
                **plot_layout,
                xaxis=dict(title=column_name, showgrid=True, gridcolor='rgba(211,211,211,0.3)'),
                yaxis=dict(title=pair_col, showgrid=True, gridcolor='rgba(211,211,211,0.3)')
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                          key=f"scatter_{column_name}_{pair_col}_{key_suffix}")
            
            # Calculate correlation
            corr = df[[column_name, pair_col]].corr().iloc[0, 1]
            
            # Show correlation metric
            correlation_color = "green" if abs(corr) > 0.7 else "orange" if abs(corr) > 0.3 else "red"
            correlation_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
            
            corr_cols = st.columns(2)
            corr_cols[0].metric("Correlation Coefficient", f"{corr:.3f}")
            corr_cols[1].markdown(f"<h3>Strength</h3><p style='color:{correlation_color}'>{correlation_strength}</p>", unsafe_allow_html=True)
        else:
            # If no other numeric column is available, show histogram
            st.info(f"No other numeric columns found to pair with {column_name}. Showing distribution instead.")
            fig = px.histogram(
                df, 
                x=column_name,
                title=f'<b>Distribution of {column_name}</b>',
                color_discrete_sequence=['#4e8df5']
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                          key=f"fallback_hist_{column_name}_{key_suffix}")
    
    elif analysis["viz_type"] == "map":
        st.write(f"Map visualization for {column_name} would be shown here. This requires geospatial data processing.")
        # Simplified implementation - show as enhanced bar chart instead
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 locations
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f'<b>Top 20 {column_name} by Count</b>'
        else:
            title = f'<b>Distribution of {column_name}</b>'
            
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
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                       key=f"map_{column_name}_{key_suffix}")
    
    elif analysis["viz_type"] == "word_cloud":
        st.write(f"Word cloud visualization for {column_name} would be shown here.")
        # Fallback to showing top values as enhanced bar chart
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False).head(20)
        
        # Text data bar chart with color gradient
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=f'<b>Top 20 values in {column_name}</b>',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            **plot_layout,
            xaxis=dict(title="", tickfont=dict(size=11), tickangle=-45),
            yaxis=dict(title="Count", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                       key=f"wordcloud_{column_name}_{key_suffix}")
    
    elif analysis["viz_type"] == "table":
        # Show top values and their counts in a styled table
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False).head(20)
        
        st.markdown(f"<div class='section-header'>Top 20 values in {column_name}</div>", unsafe_allow_html=True)
        st.dataframe(value_counts, use_container_width=True, height=min(400, len(value_counts) * 35 + 38))

# Function to create correlation heatmap for numeric columns
def create_enhanced_correlation_heatmap(df):
    """Creates an enhanced correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Create enhanced heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu_r,
            title="<b>Correlation Heatmap for Numeric Variables</b>",
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(font=dict(size=18, family="Arial, sans-serif")),
            font=dict(family="Arial, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                       key="correlation_heatmap")
        
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
            st.markdown(f"<div class='section-header'>Top {top_n} strongest correlations</div>", unsafe_allow_html=True)
            
            corr_df = pd.DataFrame(corr_pairs[:top_n], columns=['Variable 1', 'Variable 2', 'Correlation'])
            
            # Style the correlation values with colored backgrounds
            def style_corr(val):
                color = "rgba(0, 128, 0, 0.2)" if val > 0.5 else \
                       "rgba(255, 165, 0, 0.2)" if val > 0 else \
                       "rgba(255, 0, 0, 0.2)"
                return f'background-color: {color}'
            
            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}).applymap(style_corr, subset=['Correlation']), 
                       use_container_width=True)
    else:
        st.info("Not enough numeric columns to create a correlation heatmap. Upload data with at least two numeric columns to see correlations.")

# Function to create enhanced recommendations based on column relationships
def recommend_related_visualizations(df, column_analyses):
    """Recommends related visualizations based on column relationships"""
    recommendations = []
    
    # Get all numeric columns
    numeric_cols = [col for col, analysis in column_analyses.items() 
                   if analysis["type"] == "numeric"]
    
    # Get all categorical columns with reasonable cardinality
    cat_cols = [col for col, analysis in column_analyses.items() 
               if analysis["type"] == "categorical" and analysis["unique_count"] <= 10]
    
    # Get datetime columns
    time_cols = [col for col, analysis in column_analyses.items() 
                if analysis["viz_type"] == "time_series"]
    
    # Recommend numeric by categorical breakdowns
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric cols
            for cat_col in cat_cols[:3]:  # Limit to first 3 categorical cols
                recommendations.append({
                    "type": "grouped_bar",
                    "title": f"{num_col} by {cat_col}",
                    "columns": [num_col, cat_col],
                    "description": f"Analyze how {num_col} varies across different {cat_col} categories",
                    "icon": "📊"
                })
    
    # Recommend time series breakdown by category
    if len(time_cols) > 0 and len(cat_cols) > 0:
        for time_col in time_cols[:2]:  # Limit to first 2 time cols
            for cat_col in cat_cols[:2]:  # Limit to first 2 categorical cols
                recommendations.append({
                    "type": "time_series_by_category",
                    "title": f"{time_col} trends by {cat_col}",
                    "columns": [time_col, cat_col],
                    "description": f"Analyze how trends in {time_col} differ across {cat_col} categories",
                    "icon": "📈"
                })
    
    # Recommend scatter plot matrix for numeric columns
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "scatter_matrix",
            "title": "Relationships between numeric variables",
            "columns": numeric_cols[:4],  # Limit to first 4 numeric columns
            "description": "Explore relationships between multiple numeric variables simultaneously",
            "icon": "🔄"
        })
    
    return recommendations

# Function to create enhanced recommended visualizations
def create_enhanced_recommendation(df, recommendation, key_id=None):
    """Creates an enhanced visualization based on a recommendation"""
    viz_type = recommendation["type"]
    columns = recommendation["columns"]
    
    # Generate a unique key for this recommendation
    rec_key = key_id or f"rec_{viz_type}_{uuid.uuid4().hex[:8]}"

    
    if viz_type == "grouped_bar":
        num_col, cat_col = columns
        
        # Group by category and calculate mean, count, and std
        grouped_data = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
        grouped_data.columns = [cat_col, 'Mean', 'Count', 'StdDev']
        
        # Sort by mean descending
        grouped_data = grouped_data.sort_values('Mean', ascending=False)
        
        # Color each bar based on its relative value
        max_mean = grouped_data['Mean'].max()
        colors = px.colors.sequential.Blues[2:]  # Using a blue color scale
        
        # Create enhanced grouped bar chart
        fig = px.bar(
            grouped_data,
            x=cat_col,
            y='Mean',
            title=f"<b>Average {num_col} by {cat_col}</b>",
            error_y='StdDev',
            hover_data=['Count'],
            labels={'Mean': f'Average {num_col}'},
            color='Mean',
            color_continuous_scale=px.colors.sequential.Blues,
            text='Mean'
        )
        
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(title="", tickfont=dict(size=11), tickangle=-45 if len(grouped_data) > 5 else 0),
            yaxis=dict(title=f"Average {num_col}", showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)'),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(font=dict(size=16, family="Arial, sans-serif", color="#2e4057"))
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=rec_key)
        
        with st.expander("See detailed data"):
            st.dataframe(grouped_data.style.format({'Mean': '{:.2f}', 'StdDev': '{:.2f}'}), use_container_width=True)
            
        # Show statistical insights
        if len(grouped_data) > 1 and SCIPY_AVAILABLE:
            insight_cols = st.columns(2)
            
            # Calculate statistical significance
            try:
                # Create groups for ANOVA
                groups = [df[df[cat_col] == category][num_col].dropna() for category in grouped_data[cat_col]]
                
                # Run ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Display ANOVA results with styling
                is_significant = p_value < 0.05
                insight_cols[0].metric("F-statistic", f"{f_stat:.3f}")
                
                if is_significant:
                    insight_cols[1].markdown(
                        f"<div style='background-color: rgba(0, 128, 0, 0.1); padding: 10px; border-radius: 5px;'>"
                        f"<strong>Statistically Significant</strong><br>p-value: {p_value:.5f}"
                        f"<br>The differences between groups are statistically significant (p < 0.05).</div>",
                        unsafe_allow_html=True
                    )
                else:
                    insight_cols[1].markdown(
                        f"<div style='background-color: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 5px;'>"
                        f"<strong>Not Statistically Significant</strong><br>p-value: {p_value:.5f}"
                        f"<br>The differences between groups are not statistically significant (p ≥ 0.05).</div>",
                        unsafe_allow_html=True
                    )
            except:
                st.info("Could not perform statistical test on this data. Check for adequate sample sizes in each group.")
    
    elif viz_type == "time_series_by_category":
        time_col, cat_col = columns
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                st.warning(f"Could not convert {time_col} to datetime format.")
                return
        
        # Get unique categories (limit to top 6 for visibility)
        categories = df[cat_col].value_counts().nlargest(6).index.tolist()
        
        # Group by time period and category
        date_range = (df[time_col].max() - df[time_col].min()).days
        
        if date_range > 365 * 2:
            period = 'M'
            period_name = 'Month'
        elif date_range > 60:
            period = 'W'
            period_name = 'Week'
        else:
            period = 'D'
            period_name = 'Day'
        
        # Create enhanced line plot for each category
        fig = go.Figure()
        
        # Color palette for multiple lines
        colors = px.colors.qualitative.Safe
        
        # Add line for each category
        for i, category in enumerate(categories):
            # Filter data for this category
            cat_data = df[df[cat_col] == category]
            
            # Group by time period
            time_grouped = cat_data.assign(date_group=cat_data[time_col].dt.to_period(period)).groupby('date_group').size()
            
            # Convert period index to timestamps for plotting
            dates = time_grouped.index.to_timestamp()
            counts = time_grouped.values
            
            # Add line to plot with custom hover template
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name=str(category),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f"{cat_col}: {category}<br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"<b>Trends by {cat_col} over time</b>",
            xaxis_title=period_name,
            yaxis_title="Count",
            legend_title=f"<b>{cat_col}</b>",
            hovermode="closest",
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16, family="Arial, sans-serif", color="#2e4057"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=rec_key)
        
        # Show key insights
        insights_cols = st.columns(len(categories) if len(categories) <= 3 else 3)
        
        # Display top category metrics
        for i, category in enumerate(categories[:3]):
            cat_data = df[df[cat_col] == category]
            time_grouped = cat_data.assign(date_group=cat_data[time_col].dt.to_period(period)).groupby('date_group').size()
            
            # Calculate trend (using last 3 vs first 3 periods if enough data)
            if len(time_grouped) >= 6:
                first_periods = time_grouped.iloc[:3].mean()
                last_periods = time_grouped.iloc[-3:].mean()
                change_pct = ((last_periods - first_periods) / first_periods * 100) if first_periods > 0 else 0
                
                # Display metric with trend indicator
                insights_cols[i % 3].metric(
                    f"{category}",
                    f"Total: {time_grouped.sum()}",
                    f"{change_pct:.1f}%" if change_pct != 0 else "0%"
                )
            else:
                insights_cols[i % 3].metric(f"{category}", f"Total: {time_grouped.sum()}")
    
    elif viz_type == "scatter_matrix":
        # Create enhanced scatter plot matrix with custom styling
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            title="<b>Relationships between numeric variables</b>",
            color_discrete_sequence=['#4e8df5'],
            opacity=0.7
        )
        
        # Enhance the matrix display
        fig.update_traces(
            diagonal_visible=False,  # Hide diagonal
            showupperhalf=False,     # Show only lower half to avoid redundancy
        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(font=dict(size=16, family="Arial, sans-serif", color="#2e4057")),
            height=600 if len(columns) > 2 else 500,
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=rec_key)
        
        # Show correlation matrix as table
        st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)
        corr_matrix = df[columns].corr()
        
        # Style the correlation matrix
        def style_corr_matrix(val):
            color = "rgba(0, 128, 0, 0.2)" if val > 0.5 else \
                    "rgba(255, 165, 0, 0.2)" if val > 0 else \
                    "rgba(255, 0, 0, 0.2)" if val < -0.5 else \
                    "rgba(0, 0, 0, 0.1)"
            return f'background-color: {color}'
        
        # Display the styled correlation matrix
        st.dataframe(
            corr_matrix.style.format('{:.2f}').applymap(style_corr_matrix),
            use_container_width=True
        )

# Function to create a data quality assessment card
def create_data_quality_card(df):
    """Creates a card with data quality metrics and issues"""
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    outlier_info = []
    
    if numeric_cols:
        for col in numeric_cols:
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
    
    # Create quality score (simple metric based on missing values, duplicates, and outliers)
    missing_score = 100 - (missing_values.sum() / (df.shape[0] * df.shape[1]) * 100)
    duplicate_score = 100 - (duplicate_count / df.shape[0] * 100)
    outlier_score = 100 - (sum(info['Outliers'] for info in outlier_info) / (len(df) * len(numeric_cols)) * 100 if numeric_cols else 0)
    
    # Overall quality score (weighted average)
    quality_score = (missing_score * 0.4 + duplicate_score * 0.3 + outlier_score * 0.3)
    
    # Display quality score with color coding
    score_color = "green" if quality_score >= 80 else "orange" if quality_score >= 60 else "red"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="background-color: white; border-radius: 10px; padding: 15px; text-align: center; width: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #444;">Data Quality Score</h3>
                <div style="font-size: 42px; font-weight: bold; color: {score_color}; margin: 10px 0;">
                    {quality_score:.1f}%
                </div>
                <div style="font-size: 12px; color: #777;">
                    Based on completeness, uniqueness, and consistency
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display quality metrics in columns
    quality_cols = st.columns(3)
    
    with quality_cols[0]:
        st.markdown("""
        <div class="card">
            <div class="section-header">Missing Values</div>
        """, unsafe_allow_html=True)
        
        if not missing_values.empty:
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Values': missing_values.values,
                'Percentage': [round(val / len(df) * 100, 2) for val in missing_values.values]
            })
            st.dataframe(missing_df.style.format({'Percentage': '{:.2f}%'}), use_container_width=True, height=min(400, len(missing_df) * 35 + 38))
        else:
            st.success("✅ No missing values found in the dataset.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with quality_cols[1]:
        st.markdown("""
        <div class="card">
            <div class="section-header">Duplicates</div>
        """, unsafe_allow_html=True)
        
        if duplicate_count > 0:
            st.warning(f"⚠️ Found {duplicate_count} duplicate rows ({round(duplicate_count / len(df) * 100, 2)}% of data)")
            
            # Show duplicate detection button
            if st.button("Show sample duplicates"):
                duplicates = df[df.duplicated(keep=False)].head(10)
                st.dataframe(duplicates, use_container_width=True)
        else:
            st.success("✅ No duplicate rows found in the dataset.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with quality_cols[2]:
        st.markdown("""
        <div class="card">
            <div class="section-header">Outliers</div>
        """, unsafe_allow_html=True)
        
        if outlier_info:
            outlier_df = pd.DataFrame(outlier_info)
            st.dataframe(outlier_df.style.format({'Percentage': '{:.2f}%'}), use_container_width=True, height=min(400, len(outlier_df) * 35 + 38))
        else:
            st.success("✅ No significant outliers detected in numeric columns.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Function to create key metrics dashboard
def create_key_metrics(df, column_analyses):
    """Creates a dashboard section with key metrics from the dataset"""
    
    # Identify key numeric columns based on semantic meaning
    numeric_cols = [col for col, analysis in column_analyses.items() 
                   if analysis["type"] == "numeric"]
    
    # Identify key categorical columns
    cat_cols = [col for col, analysis in column_analyses.items() 
               if analysis["type"] == "categorical" and analysis["unique_count"] <= 15]
    
    # If we have numeric columns, show some metrics
    if numeric_cols:
        # Select up to 4 numeric columns to highlight
        highlight_metrics = []
        
        # First prioritize columns with semantic meaning
        priority_semantics = ["price", "quantity", "percentage", "rating", "age"]
        for semantic in priority_semantics:
            semantic_cols = [col for col, analysis in column_analyses.items() 
                            if analysis["semantic"] == semantic and analysis["type"] == "numeric"]
            highlight_metrics.extend(semantic_cols)
        
        # Add any remaining numeric columns
        remaining_cols = [col for col in numeric_cols if col not in highlight_metrics]
        highlight_metrics.extend(remaining_cols)
        
        # Limit to 4 metrics
        highlight_metrics = highlight_metrics[:4]
        
        # Create metric cards
        if highlight_metrics:
            metric_cols = st.columns(len(highlight_metrics))
            
            for i, col in enumerate(highlight_metrics):
                # Calculate key statistics
                mean_val = df[col].mean()
                median_val = df[col].median()
                
                # Format the display values based on magnitude
                if abs(mean_val) >= 1000:
                    display_val = f"{mean_val:,.0f}"
                elif abs(mean_val) >= 1:
                    display_val = f"{mean_val:.2f}"
                else:
                    display_val = f"{mean_val:.4f}"
                
                # Create a metric card with additional info
                metric_cols[i].markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 14px; color: #555; margin-bottom: 8px;">{col}</div>
                    <div class="metric-value">{display_val}</div>
                    <div class="metric-label">Average Value</div>
                    <div style="margin-top: 15px; font-size: 12px; color: #777;">
                        <span>Median: {median_val:.2f}</span> | 
                        <span>Range: {df[col].min():.2f} - {df[col].max():.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # If we have categorical columns, show top category distribution for one column
    if cat_cols:
        # Prioritize gender, location or category columns
        priority_semantics = ["gender", "location", "category"]
        selected_cat_col = None
        
        for semantic in priority_semantics:
            semantic_cols = [col for col, analysis in column_analyses.items() 
                           if analysis["semantic"] == semantic and analysis["type"] == "categorical"]
            if semantic_cols:
                selected_cat_col = semantic_cols[0]
                break
        
        # If no semantic match, take the first categorical column
        if selected_cat_col is None and cat_cols:
            selected_cat_col = cat_cols[0]
        
        # Create a pie chart for the selected categorical column
        if selected_cat_col:
            st.markdown(f"<h3 style='margin-top: 40px;'>Distribution of {selected_cat_col}</h3>", unsafe_allow_html=True)
            
            # Get value counts
            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = [selected_cat_col, 'Count']
            value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
            
            # Limit to top 8 categories for readability
            if len(value_counts) > 8:
                # Keep top 7 and group the rest as "Other"
                top_values = value_counts.iloc[:7].copy()
                other_values = value_counts.iloc[7:].copy()
                other_row = pd.DataFrame({
                    selected_cat_col: ['Other'],
                    'Count': [other_values['Count'].sum()],
                    'Percentage': [other_values['Percentage'].sum()]
                })
                value_counts = pd.concat([top_values, other_row], ignore_index=True)
            
            # Create enhanced pie chart
            fig = px.pie(
                value_counts,
                values='Count',
                names=selected_cat_col,
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4,
                height=400
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.1),
                annotations=[dict(
                    text=f'Total<br>{value_counts["Count"].sum():,}',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, 
                           key=f"key_metric_{selected_cat_col}")

# Main app logic
def main():
    # Sidebar controls
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>Dashboard Controls</h3>", unsafe_allow_html=True)
        
        # Theme selection
        #st.markdown("<div style='margin-top: 20px;'><b>Visual Theme</b></div>", unsafe_allow_html=True)
        #theme_options = ["Default", "Blue", "Green", "Red", "Purple", "Dark"]
        #selected_theme = st.selectbox("Select color theme:", theme_options, label_visibility="collapsed")
        
        # Auto refresh toggle
        st.markdown("<div style='margin-top: 20px;'><b>Auto-Refresh</b></div>", unsafe_allow_html=True)
        auto_refresh = st.toggle("Enable auto-refresh", value=False, label_visibility="collapsed")
        
        # Show advanced options
        st.markdown("<div style='margin-top: 20px;'><b>Advanced Options</b></div>", unsafe_allow_html=True)
        show_details = st.checkbox("Show detailed analysis", value=False)
        
        # About section
        st.markdown("""
        <div style='margin-top: 50px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
            <h4 style='text-align: center;'>About</h4>
            <p style='font-size: 12px; color: #666;'>
                This dashboard automatically analyzes your data and generates intelligent visualizations based on content.
            </p>
            <p style='font-size: 12px; color: #666; text-align: center;'>
                Version 1.0
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload area with enhanced styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'])
        
    with col2:
        # Sample dataset selection
        st.markdown("<div style='padding-top: 23px;'></div>", unsafe_allow_html=True)
        sample_option = st.selectbox(
            "Or try a sample dataset:",
            ["None", "Sales Data", "Customer Survey", "Stock Prices"]
        )
    
    # Process the uploaded file or sample dataset
    if uploaded_file is not None or sample_option != "None":
        # Simulate a loading spinner for better UX
        with st.spinner("Analyzing your data..."):
            # Load data
            try:
                # Handle sample datasets
                if sample_option != "None" and uploaded_file is None:
                    if sample_option == "Sales Data":
                        # Create a sample sales dataset
                        import datetime
                        import random
                        
                        # Generate sample data
                        num_rows = 1000
                        products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
                        regions = ["North", "South", "East", "West", "Central"]
                        channels = ["Online", "Store", "Distributor"]
                        
                        # Create sample dataframe
                        data = {
                            "Date": [datetime.datetime(2023, random.randint(1, 12), random.randint(1, 28)) for _ in range(num_rows)],
                            "Product": [random.choice(products) for _ in range(num_rows)],
                            "Region": [random.choice(regions) for _ in range(num_rows)],
                            "Channel": [random.choice(channels) for _ in range(num_rows)],
                            "Units": [random.randint(1, 100) for _ in range(num_rows)],
                            "Price": [random.uniform(10, 1000) for _ in range(num_rows)]
                        }
                        
                        df = pd.DataFrame(data)
                        df["Revenue"] = df["Units"] * df["Price"]
                        
                    elif sample_option == "Customer Survey":
                        # Create a sample customer survey dataset
                        import random
                        
                        # Generate sample data
                        num_rows = 500
                        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
                        genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
                        satisfaction_levels = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
                        
                        # Create sample dataframe
                        data = {
                            "Age Group": [random.choice(age_groups) for _ in range(num_rows)],
                            "Gender": [random.choice(genders) for _ in range(num_rows)],
                            "Satisfaction": [random.choice(satisfaction_levels) for _ in range(num_rows)],
                            "NPS Score": [random.randint(0, 10) for _ in range(num_rows)],
                            "Purchase Frequency": [random.randint(1, 20) for _ in range(num_rows)],
                            "Spending Amount": [random.uniform(50, 500) for _ in range(num_rows)],
                            "Would Recommend": [random.choice(["Yes", "No", "Maybe"]) for _ in range(num_rows)]
                        }
                        
                        df = pd.DataFrame(data)
                        
                    elif sample_option == "Stock Prices":
                        # Create a sample stock price dataset
                        import datetime
                        import random
                        import numpy as np
                        
                        # Generate sample data
                        start_date = datetime.datetime(2022, 1, 1)
                        dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
                        
                        # Create several stocks with different trends
                        stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
                        
                        # Generate random walk prices
                        data = {"Date": []}
                        for stock in stocks:
                            # Start with a random price
                            base_price = random.uniform(100, 1000)
                            # Generate a trend component
                            trend = np.linspace(0, random.uniform(-200, 200), len(dates))
                            # Generate random fluctuations
                            noise = np.random.normal(0, base_price * 0.02, len(dates))
                            # Calculate price series
                            prices = base_price + trend + noise.cumsum()
                            prices = np.maximum(prices, 1)  # Ensure no negative prices
                            
                            # Add to data dictionary
                            if len(data["Date"]) == 0:
                                data["Date"] = dates
                            data[stock] = prices
                        
                        df = pd.DataFrame(data)
                        
                    else:
                        st.error("Sample dataset not available. Please select another option.")
                        return
                else:
                    # Load uploaded file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                
                # Display dashboard header with data summary
                st.markdown("<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
                # Display dataset title and source
                if uploaded_file is not None:
                    st.markdown(f"<h2 style='text-align: center;'>{uploaded_file.name}</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>{sample_option}</h2>", unsafe_allow_html=True)
                
                # Display basic dataset metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with metric_cols[1]:
                    st.metric("Columns", f"{df.shape[1]:,}")
                with metric_cols[2]:
                    completeness_pct = 100 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                    st.metric("Completeness", f"{completeness_pct:.1f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Analyze columns
                column_analyses = {}
                for col in df.columns:
                    column_analyses[col] = analyze_column(df, col)
                
                # Create main dashboard tabs
                tabs = st.tabs(["📊 Dashboard", "📈 Insights", "🔍 Data Explorer", "📋 Data Quality"])
                
                with tabs[0]:  # Dashboard tab
                    # Add key metrics section
                    st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
                    create_key_metrics(df, column_analyses)
                    
                    # Main visualizations - select the most insightful columns
                    st.markdown("<h3 style='margin-top: 40px;'>Main Visualizations</h3>", unsafe_allow_html=True)
                    
                    # Prioritize columns for visualization
                    priority_columns = []
                    
                    # First add time series columns (dates)
                    time_cols = [col for col, analysis in column_analyses.items() 
                                if analysis["viz_type"] == "time_series"]
                    priority_columns.extend(time_cols[:1])  # Add up to 1 time column
                    
                    # Then add categorical columns with semantic meaning
                    semantic_categories = ['gender', 'location', 'category']
                    for semantic in semantic_categories:
                        semantic_cols = [col for col, analysis in column_analyses.items() 
                                        if analysis["semantic"] == semantic]
                        priority_columns.extend(semantic_cols[:1])  # Add 1 of each semantic type
                    
                    # Then add important numeric columns
                    numeric_cols = [col for col, analysis in column_analyses.items() 
                                   if analysis["type"] == "numeric" and col not in priority_columns]
                    priority_columns.extend(numeric_cols[:2])  # Add up to 2 numeric columns
                    
                    # Ensure we have some columns to display
                    if not priority_columns:
                        priority_columns = list(df.columns)[:4]  # Take first 4 columns
                    
                    # Display main visualizations in 2 columns
                    viz_columns = st.columns(2)
                    
                    # Add visualizations to each column
                    for i, column_name in enumerate(priority_columns[:4]):  # Limit to 4 visualizations
                        with viz_columns[i % 2]:
                            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                            analysis = column_analyses[column_name]
                            create_enhanced_visualization(df, column_name, analysis, key_suffix=f"main_{i}")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Get recommendations
                    recommendations = recommend_related_visualizations(df, column_analyses)
                    
                    # If we have recommendations, show the first one
                    if recommendations:
                        st.markdown("<h3 style='margin-top: 40px;'>Key Insight</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        recommendation = recommendations[0]
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <div style='display: flex; align-items: center;'>
                                <span style='font-size: 24px; margin-right: 10px;'>{recommendation['icon']}</span>
                                <h4 style='margin: 0;'>{recommendation['title']}</h4>
                            </div>
                            <p style='color: #666; margin-top: 5px;'>{recommendation['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        create_enhanced_recommendation(df, recommendation)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[1]:  # Insights tab
                    # Correlation analysis
                    st.markdown("<h3>Relationship Analysis</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    create_enhanced_correlation_heatmap(df)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Recommended visualizations
                    if recommendations:
                        st.markdown("<h3 style='margin-top: 40px;'>Recommended Insights</h3>", unsafe_allow_html=True)
                        
                        # Display recommendations in a grid
                        for i, recommendation in enumerate(recommendations[:6]):  # Limit to 6 recommendations
                            # Create a unique key for each recommendation
                            rec_id = f"rec_{i}_{uuid.uuid4().hex[:6]}"
                            create_enhanced_recommendation(df, recommendation, key_id=rec_id)
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class='recommendation-card'>
                                <div style='display: flex; align-items: center;'>
                                    <span style='font-size: 24px; margin-right: 10px;'>{recommendation['icon']}</span>
                                    <h4 style='margin: 0;'>{recommendation['title']}</h4>
                                </div>
                                <p style='color: #666; margin-top: 5px;'>{recommendation['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            create_enhanced_recommendation(df, recommendation)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No specific insights could be recommended for this dataset. Try uploading data with more variables.")
                
                with tabs[2]:  # Data Explorer tab
                    # Create column selector
                    st.markdown("<h3>Column Explorer</h3>", unsafe_allow_html=True)
                    
                    explorer_cols = st.columns([1, 3])
                    
                    with explorer_cols[0]:
                        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
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
                        if 'selected_column' in locals() and selected_column:
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            analysis = column_analyses[selected_column]
                            
                            # Show column statistics
                            st.markdown(f"<h3>{selected_column}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #666;'>{analysis['description']}</p>", unsafe_allow_html=True)
                            
                            # Display column visualization
                            create_enhanced_visualization(df, selected_column, analysis, key_suffix="explorer")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("👈 Select a column from the left panel to explore")
                    
                    # Add a data preview section
                    st.markdown("<h3 style='margin-top: 40px;'>Data Preview</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Add a button to show more data
                    if st.button("Show more rows"):
                        st.dataframe(df.head(50), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[3]:  # Data Quality tab
                    st.markdown("<h3>Data Quality Assessment</h3>", unsafe_allow_html=True)
                    create_data_quality_card(df)
                    
                    # Add column statistics section
                    st.markdown("<h3 style='margin-top: 40px;'>Column Statistics</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # Get descriptive statistics
                    numeric_stats = df.describe().transpose().reset_index()
                    numeric_stats.columns = ['Column'] + list(numeric_stats.columns[1:])
                    
                    # Add column analyses info
                    column_meta = []
                    for col, analysis in column_analyses.items():
                        column_meta.append({
                            'Column': col,
                            'Type': analysis['type'].capitalize(),
                            'Unique Values': analysis.get('unique_count', 'N/A'),
                            'Missing Values': df[col].isna().sum(),
                            'Missing %': round(df[col].isna().sum() / len(df) * 100, 2)
                        })
                    
                    column_meta_df = pd.DataFrame(column_meta)
                    
                    # Display in tabs
                    meta_tabs = st.tabs(["Column Metadata", "Numeric Statistics"])
                    
                    with meta_tabs[0]:
                        st.dataframe(column_meta_df, use_container_width=True)
                    
                    with meta_tabs[1]:
                        if len(numeric_stats) > 0:
                            st.dataframe(numeric_stats, use_container_width=True)
                        else:
                            st.info("No numeric columns found in the dataset.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Add dashboard footer
                st.markdown("""
                <div class="footer">
                    <p>Smart Data Dashboard | Generated on {}</p>
                    <p>This dashboard automatically analyzes your data and creates intelligent visualizations based on content.</p>
                </div>
                """.format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
    else:
        # Show welcome message
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;">
            <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100">
            <h2 style="margin-top: 20px; color: #333;">Welcome to Smart Data Dashboard</h2>
            <p style="color: #666; max-width: 600px; margin: 20px auto;">
                Upload your CSV or Excel file to get started, or select a sample dataset to explore.
                The dashboard will automatically analyze your data and create intelligent visualizations.
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 30px;">
                <div style="text-align: center; max-width: 200px;">
                    <div style="font-size: 36px; margin-bottom: 10px;">📊</div>
                    <h4>Smart Visualizations</h4>
                    <p style="font-size: 14px; color: #777;">Automatically detects the best chart type for your data</p>
                </div>
                <div style="text-align: center; max-width: 200px;">
                    <div style="font-size: 36px; margin-bottom: 10px;">🔍</div>
                    <h4>Insight Analysis</h4>
                    <p style="font-size: 14px; color: #777;">Uncovers relationships and patterns in your data</p>
                </div>
                <div style="text-align: center; max-width: 200px;">
                    <div style="font-size: 36px; margin-bottom: 10px;">📝</div>
                    <h4>Data Quality</h4>
                    <p style="font-size: 14px; color: #777;">Highlights missing values, outliers, and other data issues</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display feature highlights
        st.markdown("""
        <div style="margin-top: 40px;">
            <h3 style="text-align: center; margin-bottom: 30px;">Features</h3>
            
            <div style="display: flex; margin-bottom: 30px;">
                <div style="flex: 0 0 60px; font-size: 36px; color: #4e8df5; text-align: center;">⚡</div>
                <div>
                    <h4 style="margin: 0;">Automatic Data Analysis</h4>
                    <p style="color: #666;">
                        The dashboard analyzes your data to identify column types, distributions, and relationships, 
                        then selects the most appropriate visualizations for each element.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 30px;">
                <div style="flex: 0 0 60px; font-size: 36px; color: #4e8df5; text-align: center;">📈</div>
                <div>
                    <h4 style="margin: 0;">Interactive Visualizations</h4>
                    <p style="color: #666;">
                        Explore your data through beautiful, interactive charts and graphs. Hover, zoom, and filter 
                        to uncover insights that might be hidden in tables.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 30px;">
                <div style="flex: 0 0 60px; font-size: 36px; color: #4e8df5; text-align: center;">🧩</div>
                <div>
                    <h4 style="margin: 0;">Relationship Discovery</h4>
                    <p style="color: #666;">
                        Automatically identifies and visualizes relationships between variables, helping you discover 
                        correlations and patterns in your data.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 30px;">
                <div style="flex: 0 0 60px; font-size: 36px; color: #4e8df5; text-align: center;">🛠️</div>
                <div>
                    <h4 style="margin: 0;">Data Quality Assessment</h4>
                    <p style="color: #666;">
                        Identifies data quality issues such as missing values, outliers, and duplicates, helping you 
                        understand potential limitations in your analysis.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
