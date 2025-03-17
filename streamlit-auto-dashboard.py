import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re

# Set page configuration
st.set_page_config(
    page_title="Auto Dashboard Generator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("🎛️ Auto Dashboard Generator")
st.markdown("""
Upload your CSV or Excel file and get an automatically generated dashboard with appropriate visualizations for your data!
The app intelligently analyzes your data and selects the best visualization types based on the content.
""")

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
                skewness = stats.skew(col_data)
                if abs(skewness) > 1.5:
                    viz_type = "box_plot"
                    description = f"Skewed numeric data (skew: {skewness:.2f}), box plot shows distribution with outliers"
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

# Function to create visualizations based on column analysis
def create_visualization(df, column_name, analysis, key_suffix=""):
    """Creates the appropriate visualization based on column analysis"""
    
    if analysis["viz_type"] == "none":
        st.write(f"No visualization available for empty column: {column_name}")
        return
    
    col_data = df[column_name].dropna()
    
    if analysis["viz_type"] == "pie_chart":
        # Get value counts and calculate percentages
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts['Percentage'] = value_counts['Count'] / value_counts['Count'].sum() * 100
        
        # Create pie chart with plotly
        fig = px.pie(
            value_counts, 
            values='Count', 
            names=column_name,
            title=f'Distribution of {column_name}',
            hover_data=['Percentage'],
            labels={column_name: column_name, 'Count': 'Count'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table with counts and percentages
        with st.expander("See data table"):
            st.dataframe(value_counts.style.format({'Percentage': '{:.1f}%'}))
    
    elif analysis["viz_type"] == "bar_chart":
        # Get value counts
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        
        # Sort by count descending
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 categories if there are many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f'Top 20 {column_name} by Count'
        else:
            title = f'Distribution of {column_name}'
            
        # Create bar chart
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=title,
            text='Count'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 5:
            fig.update_layout(xaxis_tickangle=-45)
            
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("See data table"):
            st.dataframe(value_counts)
    
    elif analysis["viz_type"] == "histogram":
        fig = px.histogram(
            df, 
            x=column_name,
            title=f'Distribution of {column_name}',
            marginal="box"  # adds a boxplot to the histogram
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show basic statistics
        with st.expander("See statistics"):
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    col_data.mean(),
                    col_data.median(),
                    col_data.std(),
                    col_data.min(),
                    col_data.quantile(0.25),
                    col_data.quantile(0.5),
                    col_data.quantile(0.75),
                    col_data.max()
                ]
            })
            st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))
    
    elif analysis["viz_type"] == "box_plot":
        fig = px.box(
            df, 
            y=column_name,
            title=f'Distribution of {column_name}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show basic statistics
        with st.expander("See statistics with outliers"):
            # Calculate IQR and outlier boundaries
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'IQR', 'Lower Outlier Bound', 'Upper Outlier Bound', 'Outlier Count'],
                'Value': [
                    col_data.mean(),
                    col_data.median(),
                    col_data.std(),
                    col_data.min(),
                    col_data.max(),
                    IQR,
                    lower_bound,
                    upper_bound,
                    len(outliers)
                ]
            })
            st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))
    
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
        
        fig = px.line(
            time_df, 
            x='Date', 
            y='Count',
            title=f'Time Series of {column_name}',
            labels={'Count': 'Frequency', 'Date': x_title}
        )
        
        # Add markers to the line
        fig.update_traces(mode='lines+markers')
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("See data by time period"):
            st.dataframe(time_df)
    
    elif analysis["viz_type"] == "scatter_plot":
        # Find another numeric column to pair with if available
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != column_name]
        
        if numeric_cols:
            # Use the first numeric column as y-axis
            pair_col = numeric_cols[0]
            
            # Create scatter plot
            fig = px.scatter(
                df, 
                x=column_name, 
                y=pair_col,
                title=f'Relationship between {column_name} and {pair_col}',
                opacity=0.7
            )
            
            # Add trend line
            fig.update_layout(showlegend=False)
            
            # Add dropdown to select different y variables
            if len(numeric_cols) > 1:
                selected_y = st.selectbox(
                    f"Select variable to plot against {column_name}:",
                    numeric_cols,
                    key=f"scatter_{key_suffix}"
                )
                
                fig = px.scatter(
                    df, 
                    x=column_name, 
                    y=selected_y,
                    title=f'Relationship between {column_name} and {selected_y}',
                    opacity=0.7,
                    trendline="ols"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            with st.expander("See correlation statistics"):
                y_col = selected_y if len(numeric_cols) > 1 else pair_col
                corr = df[[column_name, y_col]].corr().iloc[0, 1]
                st.write(f"Correlation coefficient: {corr:.3f}")
                
                # Interpret correlation strength
                if abs(corr) < 0.3:
                    st.write("Weak correlation")
                elif abs(corr) < 0.7:
                    st.write("Moderate correlation")
                else:
                    st.write("Strong correlation")
        else:
            # If no other numeric column is available, show histogram
            fig = px.histogram(
                df, 
                x=column_name,
                title=f'Distribution of {column_name}'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis["viz_type"] == "map":
        st.write(f"Map visualization for {column_name} would be shown here. This requires geospatial data processing.")
        # Simplified implementation - show as bar chart instead
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False)
        
        # Limit to top 20 locations
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title = f'Top 20 {column_name} by Count'
        else:
            title = f'Distribution of {column_name}'
            
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=title
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis["viz_type"] == "word_cloud":
        st.write(f"Word cloud visualization for {column_name} would be shown here.")
        # Fallback to showing top values
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False).head(20)
        
        fig = px.bar(
            value_counts, 
            x=column_name, 
            y='Count',
            title=f'Top 20 values in {column_name}'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis["viz_type"] == "table":
        # Just show top values and their counts
        value_counts = col_data.value_counts().reset_index()
        value_counts.columns = [column_name, 'Count']
        value_counts = value_counts.sort_values('Count', ascending=False).head(20)
        
        st.write(f"Top 20 values in {column_name}:")
        st.dataframe(value_counts)

# Function to create correlation heatmap for numeric columns
def create_correlation_heatmap(df):
    """Creates a correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap for Numeric Columns"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find and display strongest correlations
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
        
        if corr_pairs:
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Display top correlations
            top_n = min(10, len(corr_pairs))
            st.write(f"Top {top_n} strongest correlations:")
            
            corr_df = pd.DataFrame(corr_pairs[:top_n], columns=['Variable 1', 'Variable 2', 'Correlation'])
            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}))
    else:
        st.write("Not enough numeric columns to create a correlation heatmap.")

# Function to recommend related visualizations based on column relationships
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
                    "description": f"Analyze how {num_col} varies across different {cat_col} categories"
                })
    
    # Recommend time series breakdown by category
    if len(time_cols) > 0 and len(cat_cols) > 0:
        for time_col in time_cols[:2]:  # Limit to first 2 time cols
            for cat_col in cat_cols[:2]:  # Limit to first 2 categorical cols
                recommendations.append({
                    "type": "time_series_by_category",
                    "title": f"{time_col} trends by {cat_col}",
                    "columns": [time_col, cat_col],
                    "description": f"Analyze how trends in {time_col} differ across {cat_col} categories"
                })
    
    # Recommend scatter plot matrix for numeric columns
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "scatter_matrix",
            "title": "Relationships between numeric variables",
            "columns": numeric_cols[:4],  # Limit to first 4 numeric columns
            "description": "Explore relationships between multiple numeric variables simultaneously"
        })
    
    return recommendations

# Function to create recommended visualizations
def create_recommended_visualization(df, recommendation):
    """Creates a visualization based on a recommendation"""
    viz_type = recommendation["type"]
    columns = recommendation["columns"]
    
    if viz_type == "grouped_bar":
        num_col, cat_col = columns
        
        # Group by category and calculate mean, count, and std
        grouped_data = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
        grouped_data.columns = [cat_col, 'Mean', 'Count', 'StdDev']
        
        # Sort by mean descending
        grouped_data = grouped_data.sort_values('Mean', ascending=False)
        
        # Create grouped bar chart
        fig = px.bar(
            grouped_data,
            x=cat_col,
            y='Mean',
            title=f"Average {num_col} by {cat_col}",
            error_y='StdDev',
            hover_data=['Count'],
            labels={'Mean': f'Average {num_col}'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("See data table"):
            st.dataframe(grouped_data)
            
        # Show ANOVA test results if there are multiple categories
        if len(grouped_data) > 1:
            with st.expander("Statistical significance"):
                try:
                    # Create groups for ANOVA
                    groups = [df[df[cat_col] == category][num_col].dropna() 
                             for category in grouped_data[cat_col]]
                    
                    # Run ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    st.write(f"ANOVA F-statistic: {f_stat:.3f}")
                    st.write(f"p-value: {p_value:.5f}")
                    
                    if p_value < 0.05:
                        st.write("The differences between groups are statistically significant (p < 0.05).")
                    else:
                        st.write("The differences between groups are not statistically significant (p ≥ 0.05).")
                except:
                    st.write("Could not perform statistical test on this data.")
    
    elif viz_type == "time_series_by_category":
        time_col, cat_col = columns
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                st.warning(f"Could not convert {time_col} to datetime format.")
                return
        
        # Get unique categories (limit to top 8 for visibility)
        categories = df[cat_col].value_counts().nlargest(8).index.tolist()
        
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
        
        # Create line plot for each category
        fig = go.Figure()
        
        for category in categories:
            # Filter data for this category
            cat_data = df[df[cat_col] == category]
            
            # Group by time period
            time_grouped = cat_data.assign(date_group=cat_data[time_col].dt.to_period(period)).groupby('date_group').size()
            
            # Convert period index to timestamps for plotting
            dates = time_grouped.index.to_timestamp()
            counts = time_grouped.values
            
            # Add line to plot
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name=str(category),
                hovertemplate=f"{cat_col}: {category}<br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"Trends by {cat_col} over time",
            xaxis_title=period_name,
            yaxis_title="Count",
            legend_title=cat_col
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "scatter_matrix":
        # Create scatter plot matrix for selected numeric columns
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            title="Scatter Plot Matrix"
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# Main app logic
def main():
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        try:
            # Check file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display basic dataset info
            st.subheader("📋 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Display data sample
            with st.expander("Preview Dataset"):
                st.dataframe(df.head(10))
            
            # Analyze columns
            st.subheader("🔬 Column Analysis")
            column_analyses = {}
            
            for col in df.columns:
                column_analyses[col] = analyze_column(df, col)
            
            # Display column analysis
            with st.expander("See column analysis details"):
                analysis_data = []
                for col, analysis in column_analyses.items():
                    analysis_data.append({
                        "Column": col,
                        "Data Type": analysis["type"].capitalize(),
                        "Best Visualization": analysis["viz_type"].replace("_", " ").title(),
                        "Unique Values": analysis.get("unique_count", "N/A"),
                        "Description": analysis["description"]
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df)
            
            # Create correlation heatmap for numeric columns
            st.subheader("🔗 Correlation Analysis")
            create_correlation_heatmap(df)
            
            # Create dashboard visualizations
            st.subheader("📊 Auto-Generated Dashboard")
            
            # Display 3 columns with the most important visualizations
            viz_count = min(9, len(df.columns))
            cols_per_row = 3
            
            # First, get recommendations for related visualizations
            recommendations = recommend_related_visualizations(df, column_analyses)
            
            # Prioritize columns for visualization
            # 1. Time series (dates)
            # 2. Key categorical columns (like gender, location)
            # 3. Important numeric columns
            priority_columns = []
            
            # Add time series columns first
            time_cols = [col for col, analysis in column_analyses.items() 
                        if analysis["viz_type"] == "time_series"]
            priority_columns.extend(time_cols[:2])  # Add up to 2 time columns
            
            # Add categorical columns with semantic meaning
            semantic_categories = ['gender', 'location', 'category']
            for semantic in semantic_categories:
                semantic_cols = [col for col, analysis in column_analyses.items() 
                                if analysis["semantic"] == semantic]
                priority_columns.extend(semantic_cols[:1])  # Add 1 of each semantic type
            
            # Add numeric columns
            numeric_cols = [col for col, analysis in column_analyses.items() 
                           if analysis["type"] == "numeric" and col not in priority_columns]
            priority_columns.extend(numeric_cols[:3])  # Add up to 3 numeric columns
            
            # Fill in with other columns
            other_cols = [col for col in df.columns if col not in priority_columns]
            priority_columns.extend(other_cols)
            
            # Limit to the visualization count
            display_columns = priority_columns[:viz_count]
            
            # Create tabs for individual visualizations and recommendations
            tab1, tab2 = st.tabs(["Individual Column Visualizations", "Recommended Insights"])
            
            with tab1:
                # Display visualizations in rows with 3 columns each
                for i in range(0, len(display_columns), cols_per_row):
                    # Create a row with columns
                    cols = st.columns(cols_per_row)
                    
                    # Add visualizations to each column
                    for j in range(cols_per_row):
                        if i + j < len(display_columns):
                            column_name = display_columns[i + j]
                            analysis = column_analyses[column_name]
                            
                            with cols[j]:
                                st.subheader(f"{column_name}")
                                st.caption(analysis["description"])
                                create_visualization(df, column_name, analysis, key_suffix=f"{i}_{j}")
            
            with tab2:
                if recommendations:
                    st.write("Based on your data, here are some recommended insights:")
                    
                    # Display recommended visualizations
                    for i, recommendation in enumerate(recommendations[:6]):  # Limit to 6 recommendations
                        st.subheader(recommendation["title"])
                        st.caption(recommendation["description"])
                        create_recommended_visualization(df, recommendation)
                        
                        # Add separator except for the last item
                        if i < len(recommendations[:6]) - 1:
                            st.markdown("---")
                else:
                    st.write("No specific insights recommended for this dataset.")
            
            # Data quality assessment
            st.subheader("📝 Data Quality Assessment")
            
            # Calculate missing values
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not missing_values.empty:
                    st.write("Columns with missing values:")
                    missing_df = pd.DataFrame({
                        'Column': missing_values.index,
                        'Missing Values': missing_values.values,
                        'Percentage': (missing_values.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df)
                else:
                    st.write("No missing values found in the dataset.")
            
            with col2:
                # Check for duplicates
                duplicate_count = df.duplicated().sum()
                if duplicate_count > 0:
                    st.write(f"Found {duplicate_count} duplicate rows ({(duplicate_count / len(df) * 100):.2f}% of data)")
                else:
                    st.write("No duplicate rows found in the dataset.")
                
                # Check for outliers in numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    outlier_info = []
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
                                'Percentage': (len(outliers) / len(df) * 100).round(2)
                            })
                    
                    if outlier_info:
                        st.write("Columns with potential outliers:")
                        st.dataframe(pd.DataFrame(outlier_info))
                    else:
                        st.write("No significant outliers detected in numeric columns.")
            
            # Export options
            st.subheader("📤 Export Options")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.write("Export dashboard to HTML (coming soon)")
                st.button("Export Dashboard", disabled=True)
            
            with export_col2:
                st.write("Export processed data (coming soon)")
                st.button("Export Processed Data", disabled=True)
            
            # Dashboard settings
            st.sidebar.header("Dashboard Settings")
            
            # Color theme selection
            st.sidebar.subheader("Visual Theme")
            theme_options = ["Default", "Blue", "Green", "Red", "Purple", "Dark"]
            selected_theme = st.sidebar.selectbox("Select color theme:", theme_options)
            st.sidebar.caption("Theme customization will be applied in a future update.")
            
            # Chart type overrides
            st.sidebar.subheader("Chart Type Overrides")
            st.sidebar.caption("Override automatic visualization choices (coming soon)")
            
            # Footer
            st.markdown("---")
            st.caption("Auto Dashboard Generator | Data analysis and visualization tool")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        # Show welcome message and instructions when no file is uploaded
        st.subheader("How to use this tool:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            1. **Upload your data file** (CSV or Excel)
            2. The app will automatically analyze your data
            3. A dashboard with appropriate visualizations is generated
            4. Explore individual column visualizations and recommended insights
            5. Review data quality assessment
            """)
        
        with col2:
            st.markdown("""
            **Features:**
            - Automatic data type detection
            - Smart visualization selection based on data content
            - Column relationship analysis
            - Data quality assessment
            - Customizable dashboard layout
            """)
        
        # Sample dataset options
        st.subheader("Try with a sample dataset:")
        sample_option = st.selectbox(
            "Select a sample dataset:",
            ["None", "Sales Data", "Customer Survey", "Stock Prices"]
        )
        
        if sample_option != "None":
            st.info(f"Sample dataset '{sample_option}' selected. This feature will be available in a future update.")

if __name__ == "__main__":
    main()
