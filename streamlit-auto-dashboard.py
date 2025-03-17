                                f_stat, p_value = stats.f_oneway(*groups)
                                
                                st.write(f"**ANOVA Test Results:**")
                                st.write(f"F-statistic: {f_stat:.4f}")
                                st.write(f"p-value: {p_value:.4f}")
                                
                                if p_value < 0.05:
                                    st.success(f"The differences between groups are statistically significant (p < 0.05)")
                                else:
                                    st.info(f"The differences between groups are not statistically significant (p >= 0.05)")
                            else:
                                st.error("Need at least two groups with data for ANOVA test")
                    else:
                        st.info("No categorical columns available for comparison")
                
                elif relationship_type == "Numerical over Time":
                    # Time series analysis
                    time_cols = [col for col in df.columns if data_types.get(col, "") == 'datetime']
                    
                    if time_cols:
                        time_col = st.selectbox("Select time column", time_cols)
                        
                        # Try to convert to datetime if not already
                        if not pd.api.types.is_datetime64_dtype(df[time_col]):
                            try:
                                date_series = pd.to_datetime(df[time_col])
                                df_copy = df.copy()
                                df_copy[time_col] = date_series
                            except:
                                st.error("Unable to convert to datetime for visualization")
                                df_copy = None
                        else:
                            df_copy = df
                        
                        if df_copy is not None:
                            # Determine appropriate aggregation based on date range
                            date_range = (df_copy[time_col].max() - df_copy[time_col].min()).days
                            
                            agg_options = ["Day", "Week", "Month", "Quarter", "Year"]
                            default_agg = "Month"
                            
                            if date_range <= 30:
                                default_agg = "Day"
                            elif date_range <= 180:
                                default_agg = "Week"
                            elif date_range <= 730:
                                default_agg = "Month"
                            else:
                                default_agg = "Quarter"
                            
                            agg_period = st.selectbox(
                                "Select time aggregation",
                                agg_options,
                                index=agg_options.index(default_agg)
                            )
                            
                            # Map to pandas frequency strings
                            freq_map = {
                                "Day": "D",
                                "Week": "W",
                                "Month": "M",
                                "Quarter": "Q",
                                "Year": "Y"
                            }
                            
                            # Group by time and calculate statistics
                            try:
                                agg_df = df_copy.groupby(pd.Grouper(key=time_col, freq=freq_map[agg_period]))[selected_column].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
                                
                                # Create a line chart with multiple metrics
                                metrics = st.multiselect(
                                    "Select metrics to show",
                                    ['mean', 'median', 'min', 'max'],
                                    default=['mean']
                                )
                                
                                if not metrics:
                                    metrics = ['mean']
                                
                                fig = go.Figure()
                                
                                for metric in metrics:
                                    fig.add_trace(go.Scatter(
                                        x=agg_df[time_col],
                                        y=agg_df[metric],
                                        mode='lines+markers',
                                        name=f"{metric.capitalize()}"
                                    ))
                                
                                # Add trend line
                                if st.checkbox("Show trend line"):
                                    # Simple moving average for trend
                                    window = max(3, len(agg_df) // 5)
                                    if len(agg_df) > window:
                                        agg_df['trend'] = agg_df['mean'].rolling(window=window, min_periods=1).mean()
                                        
                                        fig.add_trace(go.Scatter(
                                            x=agg_df[time_col],
                                            y=agg_df['trend'],
                                            mode='lines',
                                            line=dict(color='red', width=2, dash='dash'),
                                            name='Trend'
                                        ))
                                
                                fig.update_layout(
                                    title=f"{selected_column} over Time (by {agg_period})",
                                    template='plotly_white',
                                    height=500,
                                    xaxis_title="Time",
                                    yaxis_title=selected_column,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show sample of the data
                                if st.checkbox("Show aggregated data"):
                                    st.dataframe(agg_df)
                                    
                                # Time series decomposition
                                if st.checkbox("Show time series decomposition"):
                                    try:
                                        from statsmodels.tsa.seasonal import seasonal_decompose
                                        
                                        # Need sufficient data points for decomposition
                                        if len(agg_df) >= 12:
                                            # Fill any missing values for decomposition
                                            ts_data = agg_df.set_index(time_col)['mean'].fillna(method='ffill').fillna(method='bfill')
                                            
                                            # Determine period for seasonal decomposition
                                            if agg_period == "Day":
                                                period = 7  # Weekly seasonality
                                            elif agg_period == "Week":
                                                period = 4  # Monthly seasonality
                                            elif agg_period == "Month":
                                                period = 12  # Yearly seasonality
                                            else:
                                                period = 4  # Quarterly seasonality
                                            
                                            # Only do seasonal decomposition if we have at least 2 * period observations
                                            if len(ts_data) >= 2 * period:
                                                # Perform decomposition
                                                result = seasonal_decompose(ts_data, model='additive', period=period)
                                                
                                                # Create subplots
                                                fig = make_subplots(
                                                    rows=4, cols=1,
                                                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                                                )
                                                
                                                # Add traces
                                                fig.add_trace(go.Scatter(x=ts_data.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
                                                fig.add_trace(go.Scatter(x=ts_data.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
                                                fig.add_trace(go.Scatter(x=ts_data.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                                                fig.add_trace(go.Scatter(x=ts_data.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)
                                                
                                                fig.update_layout(height=800, title_text="Time Series Decomposition", showlegend=False)
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.warning(f"Need at least {2 * period} data points for seasonal decomposition with period {period}")
                                        else:
                                            st.warning("Need at least 12 data points for time series decomposition")
                                    except Exception as e:
                                        st.error(f"Error in time series decomposition: {e}")
                            
                            except Exception as e:
                                st.error(f"Error in time aggregation: {e}")
                    else:
                        st.info("No datetime columns available for time series analysis")
            
            elif col_type in ['categorical', 'gender', 'binary']:
                # For categorical columns
                st.subheader("Explore categorical relationships")
                
                # Options for exploration
                exploration_type = st.radio(
                    "Select analysis type",
                    ["Compare with another categorical column", "Analyze with numerical column"],
                    horizontal=True
                )
                
                if exploration_type == "Compare with another categorical column":
                    # Chi-square test and cross-tabulation
                    other_cat_cols = [col for col in df.columns if col != selected_column and 
                                     data_types.get(col, "") in ['categorical', 'gender', 'binary']]
                    
                    if other_cat_cols:
                        other_col = st.selectbox("Select column to compare with", other_cat_cols)
                        
                        # Create cross-tabulation
                        cross_tab = pd.crosstab(df[selected_column], df[other_col])
                        
                        # Display options
                        display_option = st.radio(
                            "Select display type",
                            ["Counts", "Percentages by Row", "Percentages by Column", "Percentages of Total"],
                            horizontal=True
                        )
                        
                        if display_option == "Counts":
                            display_table = cross_tab
                        elif display_option == "Percentages by Row":
                            display_table = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
                        elif display_option == "Percentages by Column":
                            display_table = cross_tab.div(cross_tab.sum(axis=0), axis=1) * 100
                        else:  # Percentages of Total
                            display_table = cross_tab / cross_tab.sum().sum() * 100
                        
                        # Format for display
                        if display_option != "Counts":
                            display_table = display_table.round(1).astype(str) + '%'
                        
                        # Show table
                        st.dataframe(display_table, use_container_width=True)
                        
                        # Visualization options
                        viz_type = st.radio(
                            "Select visualization",
                            ["Heatmap", "Bar Chart", "Mosaic Plot"],
                            horizontal=True
                        )
                        
                        if viz_type == "Heatmap":
                            # Heatmap of the crosstab
                            fig = px.imshow(
                                cross_tab,
                                labels=dict(x=other_col, y=selected_column, color="Count"),
                                title=f"Heatmap of {selected_column} vs {other_col}",
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="Blues"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Bar Chart":
                            # Stacked or grouped bar chart
                            stack_mode = st.radio("Bar chart type", ["Stacked", "Grouped"], horizontal=True)
                            
                            # Prepare data
                            cross_tab_reset = cross_tab.reset_index().melt(id_vars=selected_column)
                            
                            if stack_mode == "Stacked":
                                barmode = "stack"
                            else:
                                barmode = "group"
                            
                            fig = px.bar(
                                cross_tab_reset,
                                x=selected_column,
                                y="value",
                                color=other_col,
                                title=f"{selected_column} vs {other_col}",
                                barmode=barmode,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:  # Mosaic Plot
                            try:
                                from statsmodels.graphics.mosaicplot import mosaic
                                import matplotlib.pyplot as plt
                                
                                # Create a crosstab for mosaic plot
                                contingency_data = df[[selected_column, other_col]].dropna()
                                
                                # Create the mosaic plot
                                fig, _ = plt.subplots(figsize=(10, 6))
                                mosaic(contingency_data, [selected_column, other_col], ax=_)
                                plt.title(f"Mosaic Plot of {selected_column} vs {other_col}")
                                
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error creating mosaic plot: {e}")
                                st.info("Falling back to heatmap")
                                
                                # Fallback to heatmap
                                fig = px.imshow(
                                    cross_tab,
                                    labels=dict(x=other_col, y=selected_column, color="Count"),
                                    title=f"Heatmap of {selected_column} vs {other_col}",
                                    text_auto=True,
                                    aspect="auto",
                                    color_continuous_scale="Blues"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical test
                        if st.checkbox("Run Chi-square test for independence"):
                            from scipy.stats import chi2_contingency
                            
                            # Perform chi-square test
                            chi2, p, dof, expected = chi2_contingency(cross_tab)
                            
                            st.write(f"**Chi-square Test Results:**")
                            st.write(f"Chi-square value: {chi2:.4f}")
                            st.write(f"p-value: {p:.4f}")
                            st.write(f"Degrees of freedom: {dof}")
                            
                            if p < 0.05:
                                st.success(f"The variables are dependent (reject null hypothesis of independence, p < 0.05)")
                            else:
                                st.info(f"The variables are independent (fail to reject null hypothesis, p >= 0.05)")
                            
                            # Check expected counts
                            min_expected = expected.min()
                            if min_expected < 5:
                                st.warning(f"Caution: Some expected counts are less than 5 (minimum: {min_expected:.2f}). Chi-square test may not be reliable.")
                    else:
                        st.info("No other categorical columns available for comparison")
                
                else:  # Analyze with numerical column
                    num_cols = [col for col in df.columns if 
                              data_types.get(col, "") in ['numerical', 'financial']]
                    
                    if num_cols:
                        num_col = st.selectbox("Select numerical column", num_cols)
                        
                        # Group by the categorical column and calculate statistics
                        agg_df = df.groupby(selected_column)[num_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
                        
                        # Display aggregated statistics
                        st.dataframe(agg_df.round(2), use_container_width=True)
                        
                        # Visualization options
                        viz_type = st.radio(
                            "Select visualization",
                            ["Bar Chart", "Box Plot", "Violin Plot"],
                            horizontal=True
                        )
                        
                        if viz_type == "Bar Chart":
                            # Bar chart of means
                            agg_metric = st.selectbox(
                                "Select metric to visualize",
                                ['mean', 'median', 'min', 'max', 'std']
                            )
                            
                            fig = px.bar(
                                agg_df,
                                x=selected_column,
                                y=agg_metric,
                                title=f"{agg_metric.capitalize()} of {num_col} by {selected_column}",
                                template="plotly_white",
                                color=selected_column
                            )
                            
                            # Add reference line for overall metric
                            overall_value = df[num_col].agg(agg_metric)
                            fig.add_hline(y=overall_value, line_dash="dash", line_color="red",
                                        annotation_text=f"Overall {agg_metric}: {overall_value:.2f}",
                                        annotation_position="top right")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Box Plot":
                            # Box plot
                            fig = px.box(
                                df,
                                x=selected_column,
                                y=num_col,
                                title=f"Distribution of {num_col} by {selected_column}",
                                template="plotly_white",
                                color=selected_column
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:  # Violin Plot
                            # Violin plot
                            fig = px.violin(
                                df,
                                x=selected_column,
                                y=num_col,
                                title=f"Distribution of {num_col} by {selected_column}",
                                template="plotly_white",
                                color=selected_column,
                                box=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical test
                        if st.checkbox("Run statistical test for differences"):
                            from scipy import stats
                            
                            # Group data by category
                            groups = []
                            group_names = []
                            for category in df[selected_column].unique():
                                group_data = df[df[selected_column] == category][num_col].dropna()
                                if len(group_data) > 0:
                                    groups.append(group_data)
                                    group_names.append(category)
                            
                            if len(groups) >= 2:
                                if len(groups) == 2:
                                    # T-test for two groups
                                    t_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                                    
                                    st.write(f"**T-test Results between {group_names[0]} and {group_names[1]}:**")
                                    st.write(f"t-statistic: {t_stat:.4f}")
                                    st.write(f"p-value: {p_value:.4f}")
                                    
                                    if p_value < 0.05:
                                        st.success(f"The difference is statistically significant (p < 0.05)")
                                    else:
                                        st.info(f"The difference is not statistically significant (p >= 0.05)")
                                else:
                                    # ANOVA for more than two groups
                                    f_stat, p_value = stats.f_oneway(*groups)
                                    
                                    st.write(f"**ANOVA Test Results:**")
                                    st.write(f"F-statistic: {f_stat:.4f}")
                                    st.write(f"p-value: {p_value:.4f}")
                                    
                                    if p_value < 0.05:
                                        st.success(f"The differences between groups are statistically significant (p < 0.05)")
                                    else:
                                        st.info(f"The differences between groups are not statistically significant (p >= 0.05)")
                            else:
                                st.error("Need at least two groups with data for statistical testing")
                    else:
                        st.info("No numerical columns available for analysis")
            
            elif col_type == 'datetime':
                # For datetime columns
                st.subheader("Explore date/time patterns")
                
                # Try to convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(df[selected_column]):
                    try:
                        date_series = pd.to_datetime(df[selected_column])
                        df_copy = df.copy()
                        df_copy[selected_column] = date_series
                    except:
                        st.error("Unable to convert to datetime for analysis")
                        df_copy = None
                else:
                    df_copy = df
                
                if df_copy is not None:
                    # Options for datetime analysis
                    analysis_type = st.radio(
                        "Select analysis type",
                        ["Time Distribution", "Compare with Numerical Column", "Events Over Time"],
                        horizontal=True
                    )
                    
                    if analysis_type == "Time Distribution":
                        # Analyze distribution by different time components
                        time_component = st.selectbox(
                            "Analyze distribution by",
                            ["Year", "Month", "Day of Week", "Hour of Day", "Month-Year"]
                        )
                        
                        if time_component == "Year":
                            # Distribution by year
                            year_counts = df_copy[selected_column].dt.year.value_counts().sort_index()
                            
                            fig = px.bar(
                                x=year_counts.index,
                                y=year_counts.values,
                                title="Distribution by Year",
                                template="plotly_white",
                                labels={'x': 'Year', 'y': 'Count'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif time_component == "Month":
                            # Distribution by month
                            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            month_counts = df_copy[selected_column].dt.month.value_counts().sort_index()
                            month_counts.index = [months[i-1] for i in month_counts.index]
                            
                            fig = px.bar(
                                x=month_counts.index,
                                y=month_counts.values,
                                title="Distribution by Month",
                                template="plotly_white",
                                labels={'x': 'Month', 'y': 'Count'},
                                color=month_counts.index
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif time_component == "Day of Week":
                            # Distribution by day of week
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                  'Friday', 'Saturday', 'Sunday']
                            day_counts = df_copy[selected_column].dt.dayofweek.value_counts().sort_index()
                            day_counts.index = [days[i] for i in day_counts.index]
                            
                            fig = px.bar(
                                x=day_counts.index,
                                y=day_counts.values,
                                title="Distribution by Day of Week",
                                template="plotly_white",
                                labels={'x': 'Day', 'y': 'Count'},
                                color=day_counts.index,
                                color_discrete_map={
                                    'Saturday': 'rgba(255, 127, 14, 0.7)',
                                    'Sunday': 'rgba(255, 127, 14, 0.7)'
                                }
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif time_component == "Hour of Day":
                            # Check if time component exists
                            has_time = (df_copy[selected_column].dt.hour != 0).any()
                            
                            if has_time:
                                # Distribution by hour
                                hour_counts = df_copy[selected_column].dt.hour.value_counts().sort_index()
                                
                                fig = px.bar(
                                    x=hour_counts.index,
                                    y=hour_counts.values,
                                    title="Distribution by Hour of Day",
                                    template="plotly_white",
                                    labels={'x': 'Hour', 'y': 'Count'}
                                )
                                
                                # Highlight business hours
                                for hour in range(9, 18):
                                    if hour in hour_counts.index:
                                        fig.add_vrect(
                                            x0=hour-0.5, x1=hour+0.5,
                                            fillcolor="rgba(0, 255, 0, 0.1)",
                                            layer="below", line_width=0
                                        )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No time component found in the datetime column, only dates are present")
                        
                        else:  # Month-Year
                            # Distribution by month and year
                            df_copy['month_year'] = df_copy[selected_column].dt.to_period('M')
                            month_year_counts = df_copy['month_year'].value_counts().sort_index()
                            
                            # Convert period to string for plotting
                            month_year_counts = month_year_counts.reset_index()
                            month_year_counts.columns = ['month_year', 'count']
                            month_year_counts['month_year'] = month_year_counts['month_year'].astype(str)
                            
                            fig = px.bar(
                                month_year_counts,
                                x='month_year',
                                y='count',
                                title="Distribution by Month-Year",
                                template="plotly_white"
                            )
                            
                            # Rotate x-axis labels for better readability
                            fig.update_layout(xaxis_tickangle=-45)
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "Compare with Numerical Column":
                        # Time series analysis with numerical column
                        num_cols = [col for col in df.columns if 
                                  data_types.get(col, "") in ['numerical', 'financial']]
                        
                        if num_cols:
                            num_col = st.selectbox("Select numerical column", num_cols)
                            
                            # Determine appropriate aggregation based on date range
                            date_range = (df_copy[selected_column].max() - df_copy[selected_column].min()).days
                            
                            agg_options = ["Day", "Week", "Month", "Quarter", "Year"]
                            default_agg = "Month"
                            
                            if date_range <= 30:
                                default_agg = "Day"
                            elif date_range <= 180:
                                default_agg = "Week"
                            elif date_range <= 730:
                                default_agg = "Month"
                            else:
                                default_agg = "Quarter"
                            
                            agg_period = st.selectbox(
                                "Select time aggregation",
                                agg_options,
                                index=agg_options.index(default_agg)
                            )
                            
                            # Map to pandas frequency strings
                            freq_map = {
                                "Day": "D",
                                "Week": "W",
                                "Month": "M",
                                "Quarter": "Q",
                                "Year": "Y"
                            }
                            
                            # Aggregation function
                            agg_func = st.selectbox(
                                "Select aggregation function",
                                ["Mean", "Sum", "Median", "Min", "Max", "Count"]
                            )
                            
                            agg_func_map = {
                                "Mean": "mean",
                                "Sum": "sum",
                                "Median": "median",
                                "Min": "min",
                                "Max": "max",
                                "Count": "count"
                            }
                            
                            # Group by time and calculate statistics
                            agg_df = df_copy.groupby(pd.Grouper(key=selected_column, freq=freq_map[agg_period]))[num_col].agg(agg_func_map[agg_func.lower()]).reset_index()
                            
                            # Create line chart
                            fig = px.line(
                                agg_df,
                                x=selected_column,
                                y=num_col,
                                title=f"{agg_func} of {num_col} by {agg_period}",
                                template="plotly_white",
                                markers=True
                            )
                            
                            # Add trend line
                            if st.checkbox("Show trend line"):
                                # Simple moving average for trend
                                window = max(3, len(agg_df) // 5)
                                if len(agg_df) > window:
                                    agg_df['trend'] = agg_df[num_col].rolling(window=window, min_periods=1).mean()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=agg_df[selected_column],
                                        y=agg_df['trend'],
                                        mode='lines',
                                        line=dict(color='red', width=2, dash='dash'),
                                        name='Trend'
                                    ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Seasonal analysis
                            if st.checkbox("Show seasonal patterns"):
                                seasonal_component = st.radio(
                                    "Select seasonal component",
                                    ["Month", "Day of Week", "Hour of Day"],
                                    horizontal=True
                                )
                                
                                if seasonal_component == "Month":
                                    # Monthly patterns
                                    df_copy['month'] = df_copy[selected_column].dt.month
                                    monthly_agg = df_copy.groupby('month')[num_col].agg(agg_func_map[agg_func.lower()]).reset_index()
                                    
                                    # Map month numbers to names
                                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                    monthly_agg['month_name'] = monthly_agg['month'].apply(lambda x: months[x-1])
                                    
                                    fig = px.bar(
                                        monthly_agg,
                                        x='month_name',
                                        y=num_col,
                                        title=f"{agg_func} of {num_col} by Month",
                                        template="plotly_white",
                                        color='month_name'
                                    )
                                    
                                    # Set x-axis order
                                    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':months})
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif seasonal_component == "Day of Week":
                                    # Day of week patterns
                                    df_copy['day_of_week'] = df='1/1/2018', periods=n, freq='M'),
                        'Purchases': np.random.randint(1, 50, size=n),
                        'CLV': np.random.normal(500, 200, size=n)
                    })
                
                elif dataset_type == "Financial Performance":
                    np.random.seed(42)
                    quarters = pd.date_range(start='1/1/2018', periods=16, freq='Q')
                    departments = ['Sales', 'Marketing', 'R&D', 'Operations', 'IT', 'HR']
                    
                    rows = []
                    for quarter in quarters:
                        for dept in departments:
                            revenue = np.random.normal(100000, 20000)
                            expenses = np.random.normal(revenue * 0.7, 10000)
                            rows.append({
                                'Quarter': quarter,
                                'Department': dept,
                                'Revenue': revenue,
                                'Expenses': expenses,
                                'Profit': revenue - expenses,
                                'Employees': np.random.randint(5, 50),
                                'Projects_Completed': np.random.randint(1, 10)
                            })
                    
                    df = pd.DataFrame(rows)
                
                elif dataset_type == "Product Ratings":
                    np.random.seed(42)
                    n = 300
                    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
                    categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']
                    
                    df = pd.DataFrame({
                        'Product': np.random.choice(products, size=n),
                        'Category': np.random.choice(categories, size=n),
                        'Rating': np.random.uniform(1, 5, size=n),
                        'Price': np.random.uniform(10, 200, size=n),
                        'Review_Date': pd.date_range(start='1/1/2022', periods=n),
                        'Customer_Age': np.random.randint(18, 70, size=n),
                        'Customer_Gender': np.random.choice(['Male', 'Female', 'Other'], size=n),
                        'Verified_Purchase': np.random.choice([True, False], size=n, p=[0.8, 0.2])
                    })
                
                st.success(f"Sample {dataset_type} dataset loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns with semantic understanding
        data_types, semantic_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Advanced Analysis", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "graph-up", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Display a sample of the data
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df.head(5))
            
            # Summary statistics
            stats = generate_summary_stats(df, data_types)
            
            # Create a two-column layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Dataset Statistics</h3>', unsafe_allow_html=True)
                st.write(f"**Rows:** {stats['rows']}")
                st.write(f"**Columns:** {stats['columns']}")
                st.write(f"**Numerical Columns:** {stats['numerical_columns']}")
                st.write(f"**Categorical Columns:** {stats['categorical_columns']}")
                st.write(f"**Datetime Columns:** {stats['datetime_columns']}")
                st.write(f"**Missing Values:** {stats['missing_values']} ({stats['missing_percentage']:.2f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Column Types</h3>', unsafe_allow_html=True)
                
                # Create a pie chart of column types
                fig = go.Figure(
                    data=[go.Pie(
                        labels=['Numerical', 'Categorical', 'Datetime', 'Other'],
                        values=[
                            stats['numerical_columns'],
                            stats['categorical_columns'],
                            stats['datetime_columns'],
                            stats['columns'] - stats['numerical_columns'] - stats['categorical_columns'] - stats['datetime_columns']
                        ],
                        hole=.4,
                        marker_colors=px.colors.qualitative.Pastel
                    )]
                )
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Data Completeness</h3>', unsafe_allow_html=True)
                
                # Create a gauge chart for data completeness
                completeness = 100 - stats['missing_percentage']
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = completeness,
                    title = {'text': "Completeness"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "red"},
                            {'range': [60, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 95
                        }
                    }
                ))
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data quality insights
            st.markdown('<h3 class="sub-header">Data Insights</h3>', unsafe_allow_html=True)
            
            # Get data insights
            insights = analyze_dataset(df)
            
            if insights:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                for insight in insights:
                    st.markdown(insight)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No significant insights detected in the data.")
            
            # Column details
            st.markdown('<h3 class="sub-header">Column Details</h3>', unsafe_allow_html=True)
            
            with st.expander("Column Information", expanded=True):
                # Create a table with column information
                col_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    missing = df[col].isna().sum()
                    missing_pct = (missing / len(df)) * 100
                    unique = df[col].nunique()
                    unique_pct = (unique / len(df)) * 100
                    
                    col_info.append({
                        "Column": col,
                        "Type": data_types.get(col, "unknown"),
                        "Data Type": dtype,
                        "Missing Values": f"{missing} ({missing_pct:.1f}%)",
                        "Unique Values": f"{unique} ({unique_pct:.1f}%)"
                    })
                
                st.table(pd.DataFrame(col_info))
        
        elif selected == "Data Explorer":
            st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
            
            # Column selector
            selected_column = st.selectbox("Select a column to explore", df.columns)
            
            # Get the data type of the selected column
            col_type = data_types.get(selected_column, "unknown")
            
            # Create a two-column layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h3>Summary: {selected_column}</h3>', unsafe_allow_html=True)
                
                # Display different statistics based on column type
                if col_type in ['numerical', 'financial']:
                    # Numerical summary
                    stats = df[selected_column].describe()
                    
                    # Format nicely
                    st.write(f"**Mean:** {stats['mean']:.2f}")
                    st.write(f"**Median:** {df[selected_column].median():.2f}")
                    st.write(f"**Std Dev:** {stats['std']:.2f}")
                    st.write(f"**Min:** {stats['min']:.2f}")
                    st.write(f"**Max:** {stats['max']:.2f}")
                    st.write(f"**Range:** {stats['max'] - stats['min']:.2f}")
                    
                    # Additional stats
                    st.write(f"**Skewness:** {df[selected_column].skew():.2f}")
                    st.write(f"**Kurtosis:** {df[selected_column].kurtosis():.2f}")
                    
                    # Missing values
                    missing = df[selected_column].isna().sum()
                    st.write(f"**Missing Values:** {missing} ({missing/len(df)*100:.1f}%)")
                    
                    # Outliers
                    q1 = df[selected_column].quantile(0.25)
                    q3 = df[selected_column].quantile(0.75)
                    iqr = q3 - q1
                    outliers = df[(df[selected_column] < q1 - 1.5 * iqr) | (df[selected_column] > q3 + 1.5 * iqr)][selected_column]
                    st.write(f"**Outliers:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
                
                elif col_type in ['categorical', 'gender', 'binary']:
                    # Categorical summary
                    value_counts = df[selected_column].value_counts()
                    top_value = value_counts.index[0]
                    top_count = value_counts.iloc[0]
                    
                    st.write(f"**Unique Values:** {df[selected_column].nunique()}")
                    st.write(f"**Most Common:** {top_value} ({top_count} times, {top_count/len(df)*100:.1f}%)")
                    st.write(f"**Least Common:** {value_counts.index[-1]} ({value_counts.iloc[-1]} times, {value_counts.iloc[-1]/len(df)*100:.1f}%)")
                    
                    # Missing values
                    missing = df[selected_column].isna().sum()
                    st.write(f"**Missing Values:** {missing} ({missing/len(df)*100:.1f}%)")
                    
                    # Mode
                    st.write(f"**Mode:** {df[selected_column].mode()[0]}")
                    
                    # Entropy (measure of randomness)
                    probabilities = value_counts / len(df)
                    entropy = -sum(p * np.log2(p) for p in probabilities)
                    max_entropy = np.log2(df[selected_column].nunique())
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    st.write(f"**Entropy (Randomness):** {normalized_entropy:.2f} (0=uniform, 1=random)")
                
                elif col_type == 'datetime':
                    # Datetime summary
                    min_date = df[selected_column].min()
                    max_date = df[selected_column].max()
                    
                    # Handle string conversion to datetime if needed
                    if not pd.api.types.is_datetime64_dtype(df[selected_column]):
                        try:
                            min_date = pd.to_datetime(min_date)
                            max_date = pd.to_datetime(max_date)
                        except:
                            pass
                    
                    date_range = max_date - min_date
                    
                    st.write(f"**Earliest Date:** {min_date}")
                    st.write(f"**Latest Date:** {max_date}")
                    st.write(f"**Range:** {date_range}")
                    
                    # Missing values
                    missing = df[selected_column].isna().sum()
                    st.write(f"**Missing Values:** {missing} ({missing/len(df)*100:.1f}%)")
                    
                    # If we can convert to datetime, show day of week distribution
                    try:
                        if not pd.api.types.is_datetime64_dtype(df[selected_column]):
                            date_series = pd.to_datetime(df[selected_column])
                        else:
                            date_series = df[selected_column]
                            
                        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = date_series.dt.dayofweek.value_counts().sort_index()
                        day_counts.index = [days[i] for i in day_counts.index]
                        
                        st.write("**Day of Week Distribution:**")
                        st.write(day_counts)
                    except:
                        pass
                
                else:
                    # Text or unknown type
                    st.write(f"**Type:** {df[selected_column].dtype}")
                    st.write(f"**Unique Values:** {df[selected_column].nunique()}")
                    
                    # Missing values
                    missing = df[selected_column].isna().sum()
                    st.write(f"**Missing Values:** {missing} ({missing/len(df)*100:.1f}%)")
                    
                    # Length statistics if it's text
                    if df[selected_column].dtype == 'object':
                        try:
                            lengths = df[selected_column].str.len()
                            st.write(f"**Average Length:** {lengths.mean():.1f} characters")
                            st.write(f"**Min Length:** {lengths.min()} characters")
                            st.write(f"**Max Length:** {lengths.max()} characters")
                        except:
                            pass
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Visualization</h3>', unsafe_allow_html=True)
                
                # Create visualizations based on column type
                if col_type in ['numerical', 'financial']:
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
                    
                    with viz_tabs[0]:
                        # Histogram
                        fig = px.histogram(
                            df, x=selected_column,
                            nbins=20,
                            title=f"Distribution of {selected_column}",
                            template='plotly_white'
                        )
                        
                        # Add mean line
                        mean_value = df[selected_column].mean()
                        fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                                    annotation_text=f"Mean: {mean_value:.2f}",
                                    annotation_position="top right")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[1]:
                        # Box plot
                        fig = px.box(
                            df, y=selected_column,
                            title=f"Box Plot of {selected_column}",
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[2]:
                        # Violin plot
                        fig = px.violin(
                            df, y=selected_column,
                            title=f"Violin Plot of {selected_column}",
                            template='plotly_white',
                            box=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif col_type in ['categorical', 'gender', 'binary']:
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Bar Chart", "Pie Chart", "Tree Map"])
                    
                    with viz_tabs[0]:
                        # Bar chart
                        value_counts = df[selected_column].value_counts().reset_index()
                        value_counts.columns = [selected_column, 'count']
                        
                        # Sort by count
                        value_counts = value_counts.sort_values('count', ascending=False)
                        
                        # Limit to top 10
                        if len(value_counts) > 10:
                            display_counts = value_counts.head(10)
                            other_count = value_counts['count'][10:].sum()
                            other_row = pd.DataFrame({selected_column: ['Other'], 'count': [other_count]})
                            display_counts = pd.concat([display_counts, other_row])
                        else:
                            display_counts = value_counts
                        
                        fig = px.bar(
                            display_counts,
                            x=selected_column,
                            y='count',
                            title=f"Count by {selected_column}",
                            template='plotly_white'
                        )
                        
                        # Add percentage labels
                        total = display_counts['count'].sum()
                        percentages = [f"{(val/total)*100:.1f}%" for val in display_counts['count']]
                        fig.update_traces(text=percentages, textposition='outside')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[1]:
                        # Pie chart
                        value_counts = df[selected_column].value_counts()
                        
                        # Limit to top 10
                        if len(value_counts) > 10:
                            display_counts = value_counts.head(10)
                            other_count = value_counts[10:].sum()
                            display_counts['Other'] = other_count
                        else:
                            display_counts = value_counts
                        
                        fig = px.pie(
                            values=display_counts.values,
                            names=display_counts.index,
                            title=f"Distribution of {selected_column}",
                            template='plotly_white',
                            hole=0.4
                        )
                        
                        fig.update_traces(textinfo='percent+label')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[2]:
                        # Tree map
                        value_counts = df[selected_column].value_counts().reset_index()
                        value_counts.columns = [selected_column, 'count']
                        
                        # Limit to top 15
                        if len(value_counts) > 15:
                            display_counts = value_counts.head(15)
                            other_count = value_counts['count'][15:].sum()
                            other_row = pd.DataFrame({selected_column: ['Other'], 'count': [other_count]})
                            display_counts = pd.concat([display_counts, other_row])
                        else:
                            display_counts = value_counts
                        
                        fig = px.treemap(
                            display_counts,
                            path=[selected_column],
                            values='count',
                            title=f"Tree Map of {selected_column}",
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif col_type == 'datetime':
                    # Try to convert to datetime if not already
                    if not pd.api.types.is_datetime64_dtype(df[selected_column]):
                        try:
                            date_series = pd.to_datetime(df[selected_column])
                        except:
                            st.error("Unable to convert to datetime for visualization")
                            date_series = None
                    else:
                        date_series = df[selected_column]
                    
                    if date_series is not None:
                        # Create tabs for different time visualizations
                        viz_tabs = st.tabs(["Timeline", "By Month", "By Day of Week"])
                        
                        with viz_tabs[0]:
                            # Timeline of counts
                            df_copy = df.copy()
                            df_copy['date_col'] = date_series
                            
                            # Extract the appropriate time component based on date range
                            date_range = (date_series.max() - date_series.min()).days
                            
                            if date_range > 365*2:  # More than 2 years
                                freq = 'M'
                                title = "Monthly Distribution"
                                df_copy['time_group'] = df_copy['date_col'].dt.to_period('M')
                            elif date_range > 90:  # More than 3 months
                                freq = 'W'
                                title = "Weekly Distribution"
                                df_copy['time_group'] = df_copy['date_col'].dt.to_period('W')
                            else:
                                freq = 'D'
                                title = "Daily Distribution"
                                df_copy['time_group'] = df_copy['date_col'].dt.to_period('D')
                            
                            # Count by time period
                            time_counts = df_copy['time_group'].value_counts().sort_index()
                            time_counts = time_counts.reset_index()
                            time_counts.columns = ['date', 'count']
                            
                            # Convert period to datetime for plotting
                            time_counts['date'] = time_counts['date'].astype(str)
                            time_counts['date'] = pd.to_datetime(time_counts['date'])
                            
                            fig = px.line(
                                time_counts,
                                x='date',
                                y='count',
                                title=title,
                                template='plotly_white'
                            )
                            
                            # Add markers
                            fig.update_traces(mode='lines+markers')
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_tabs[1]:
                            # Distribution by month
                            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            month_counts = date_series.dt.month.value_counts().sort_index()
                            month_counts.index = [months[i-1] for i in month_counts.index]
                            
                            fig = px.bar(
                                x=month_counts.index,
                                y=month_counts.values,
                                title="Distribution by Month",
                                template='plotly_white',
                                labels={'x': 'Month', 'y': 'Count'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_tabs[2]:
                            # Distribution by day of week
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                  'Friday', 'Saturday', 'Sunday']
                            day_counts = date_series.dt.dayofweek.value_counts().sort_index()
                            day_counts.index = [days[i] for i in day_counts.index]
                            
                            fig = px.bar(
                                x=day_counts.index,
                                y=day_counts.values,
                                title="Distribution by Day of Week",
                                template='plotly_white',
                                labels={'x': 'Day', 'y': 'Count'},
                                color=day_counts.index,
                                color_discrete_map={
                                    'Saturday': 'rgba(255, 127, 14, 0.7)',
                                    'Sunday': 'rgba(255, 127, 14, 0.7)'
                                }
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Text or unknown type
                    st.info(f"Visualization not available for this column type: {df[selected_column].dtype}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Relationships section
            st.markdown('<h3 class="sub-header">Explore Relationships</h3>', unsafe_allow_html=True)
            
            # Find columns to compare with
            if col_type in ['numerical', 'financial']:
                # For numerical columns, show relationship with other columns
                relationship_types = ["Numerical vs Numerical", "Numerical vs Categorical", "Numerical over Time"]
                relationship_type = st.radio("Select relationship type to explore", relationship_types, horizontal=True)
                
                if relationship_type == "Numerical vs Numerical":
                    # Scatter plot with another numerical column
                    other_num_cols = [col for col in df.columns if col != selected_column and 
                                    data_types.get(col, "") in ['numerical', 'financial']]
                    
                    if other_num_cols:
                        other_col = st.selectbox("Select column to compare with", other_num_cols)
                        
                        fig = px.scatter(
                            df,
                            x=selected_column,
                            y=other_col,
                            title=f"{selected_column} vs {other_col}",
                            template='plotly_white',
                            trendline="ols"
                        )
                        
                        # Calculate correlation
                        corr = df[[selected_column, other_col]].corr().iloc[0, 1]
                        
                        fig.add_annotation(
                            x=0.95, y=0.05,
                            xref="paper", yref="paper",
                            text=f"Correlation: {corr:.2f}",
                            showarrow=False,
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add regression details
                        if st.checkbox("Show regression details"):
                            import statsmodels.api as sm
                            
                            X = df[selected_column]
                            y = df[other_col]
                            
                            # Add constant for intercept
                            X = sm.add_constant(X)
                            
                            # Create and fit the model
                            model = sm.OLS(y, X).fit()
                            
                            # Display model summary
                            st.text(model.summary().as_text())
                    else:
                        st.info("No other numerical columns available for comparison")
                
                elif relationship_type == "Numerical vs Categorical":
                    # Box plot by categorical column
                    cat_cols = [col for col in df.columns if 
                              data_types.get(col, "") in ['categorical', 'gender', 'binary']]
                    
                    if cat_cols:
                        cat_col = st.selectbox("Select categorical column", cat_cols)
                        
                        # Check if we need to limit categories
                        unique_cats = df[cat_col].nunique()
                        if unique_cats > 10:
                            top_n = st.slider("Number of top categories to show", 3, 10, 5)
                            top_cats = df[cat_col].value_counts().head(top_n).index
                            filtered_df = df[df[cat_col].isin(top_cats)]
                            st.info(f"Showing top {top_n} categories out of {unique_cats}")
                        else:
                            filtered_df = df
                        
                        viz_type = st.radio("Select visualization", ["Box Plot", "Violin Plot", "Bar Chart"], horizontal=True)
                        
                        if viz_type == "Box Plot":
                            fig = px.box(
                                filtered_df,
                                x=cat_col,
                                y=selected_column,
                                title=f"{selected_column} by {cat_col}",
                                template='plotly_white',
                                color=cat_col
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Violin Plot":
                            fig = px.violin(
                                filtered_df,
                                x=cat_col,
                                y=selected_column,
                                title=f"{selected_column} by {cat_col}",
                                template='plotly_white',
                                color=cat_col,
                                box=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:  # Bar Chart
                            # Group by category and calculate mean
                            agg_df = filtered_df.groupby(cat_col)[selected_column].mean().reset_index()
                            
                            fig = px.bar(
                                agg_df,
                                x=cat_col,
                                y=selected_column,
                                title=f"Average {selected_column} by {cat_col}",
                                template='plotly_white',
                                color=cat_col
                            )
                            
                            # Add mean line
                            mean_value = filtered_df[selected_column].mean()
                            fig.add_hline(y=mean_value, line_dash="dash", line_color="red",
                                        annotation_text=f"Overall Mean: {mean_value:.2f}",
                                        annotation_position="top right")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # ANOVA test for significance
                        if st.checkbox("Test significance (ANOVA)"):
                            from scipy import stats
                            
                            # Group data by category
                            groups = []
                            for category in filtered_df[cat_col].unique():
                                group_data = filtered_df[filtered_df[cat_col] == category][selected_column].dropna()
                                if len(group_data) > 0:
                                    groups.append(group_data)
                            
                            if len(groups) >= 2:
                                # Perform ANOVA
                                f_stat, p_value =import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Auto Dashboard Generator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def detect_data_types(df):
    """Detect data types for each column in the DataFrame with semantic understanding."""
    data_types = {}
    semantic_types = {}
    
    # Common semantic patterns
    demographic_patterns = {
        'gender': ['gender', 'sex', 'male', 'female'],
        'age': ['age', 'years', 'yrs', 'year old'],
        'location': ['country', 'state', 'city', 'region', 'province', 'address', 'zip', 'postal'],
        'income': ['income', 'salary', 'earnings', 'wage'],
        'education': ['education', 'degree', 'qualification', 'academic', 'school', 'college', 'university']
    }
    
    financial_patterns = {
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'turnover'],
        'cost': ['cost', 'expense', 'expenditure', 'spending', 'price'],
        'profit': ['profit', 'margin', 'gain', 'earnings', 'net income'],
        'currency': ['currency', 'usd', 'eur', 'gbp', '

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types, semantic_types=None):
    """
    Intelligently recommend charts based on data types, semantic understanding, 
    and visualization best practices.
    """
    if semantic_types is None:
        semantic_types = {}
    
    # Get columns by type
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    financial_cols = [col for col, type_ in data_types.items() if type_ == 'financial']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    gender_cols = [col for col, type_ in data_types.items() if type_ == 'gender']
    binary_cols = [col for col, type_ in data_types.items() if type_ == 'binary']
    rating_cols = [col for col, type_ in data_types.items() if type_ == 'rating']
    
    # Combine financial with numeric for certain operations
    all_numeric_cols = numeric_cols + financial_cols
    
    charts = []
    
    # DEMOGRAPHIC VISUALIZATIONS
    
    # Gender distribution (pie chart is typically best practice)
    for col in gender_cols:
        charts.append({
            'type': 'pie',
            'title': f'Gender Distribution',
            'column': col,
            'priority': 10  # High priority
        })
    
    # General demographics for categorical columns (prioritize by column name relevance)
    for col in categ_cols:
        priority = 5  # Default priority
        title = f'Distribution of {col}'
        
        # Increase priority for important demographic columns
        if col.lower() in ['country', 'region', 'state', 'city', 'location']:
            priority = 9
            title = f'Geographic Distribution: {col}'
        elif col.lower() in ['age group', 'age_group', 'age range', 'age_range']:
            priority = 9
            title = f'Age Distribution'
        elif col.lower() in ['education', 'qualification', 'degree']:
            priority = 8
            title = f'Education Level Distribution'
        elif col.lower() in ['occupation', 'job', 'profession', 'role', 'position']:
            priority = 8
            title = f'Occupation Distribution'
            
        # Choose visualization type based on number of unique values
        if df[col].nunique() <= 6:
            charts.append({
                'type': 'pie',
                'title': title,
                'column': col,
                'priority': priority
            })
        else:
            # Get top categories by count
            value_counts = df[col].value_counts().head(10)
            charts.append({
                'type': 'bar',
                'title': title,
                'x': col,
                'values': value_counts.index.tolist(),
                'priority': priority
            })
    
    # FINANCIAL VISUALIZATIONS
    
    # Financial columns should get special treatment
    for col in financial_cols:
        # Distribution of financial data
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col,
            'priority': 7
        })
        
        # If we have time data, financial data over time is very valuable
        if datetime_cols:
            charts.append({
                'type': 'line',
                'title': f'{col} Trend Over Time',
                'x': datetime_cols[0],
                'y': col,
                'priority': 10
            })
    
    # TIME-BASED VISUALIZATIONS
    
    # Time series for numeric columns (very important)
    if datetime_cols:
        for num_col in all_numeric_cols[:3]:
            priority = 8
            # Financial metrics over time get highest priority
            if num_col in financial_cols or num_col.lower() in ['revenue', 'sales', 'profit', 'income', 'cost', 'expense']:
                priority = 10
            # Rating metrics over time get high priority
            elif num_col in rating_cols or 'satisfaction' in num_col.lower() or 'rating' in num_col.lower():
                priority = 9
                
            charts.append({
                'type': 'line',
                'title': f'{num_col} Trend Over Time',
                'x': datetime_cols[0],
                'y': num_col,
                'priority': priority
            })
    
    # NUMERICAL DISTRIBUTIONS
    
    # Distribution charts for numeric columns
    for col in all_numeric_cols[:5]:
        priority = 5
        if col in rating_cols or 'score' in col.lower() or 'rating' in col.lower():
            priority = 8  # Higher priority for ratings
            
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col,
            'priority': priority
        })
    
    # CORRELATION AND RELATIONSHIP CHARTS
    
    # Scatter plots for potentially related numerical columns
    if len(all_numeric_cols) >= 2:
        # Try to find semantically related pairs first
        related_pairs = []
        
        # Common related pairs to check for
        pair_patterns = [
            (['revenue', 'sales', 'income'], ['profit', 'margin']),
            (['cost', 'expense'], ['revenue', 'sales', 'income']),
            (['price'], ['quantity', 'volume', 'units']),
            (['age'], ['income', 'salary', 'earnings']),
            (['rating', 'satisfaction', 'score'], ['price', 'cost'])
        ]
        
        # Check for semantically related pairs
        for pattern1, pattern2 in pair_patterns:
            cols1 = [col for col in all_numeric_cols if any(p in col.lower() for p in pattern1)]
            cols2 = [col for col in all_numeric_cols if any(p in col.lower() for p in pattern2)]
            
            for col1 in cols1:
                for col2 in cols2:
                    if col1 != col2:
                        related_pairs.append((col1, col2, 9))  # High priority for related pairs
        
        # Add some standard pairs if we don't have enough related ones
        if len(related_pairs) < 2:
            # Find columns that might be correlated
            if len(all_numeric_cols) >= 2:
                for i in range(min(3, len(all_numeric_cols))):
                    for j in range(i+1, min(4, len(all_numeric_cols))):
                        # Skip if this pair is already in related_pairs
                        if not any((all_numeric_cols[i] == x[0] and all_numeric_cols[j] == x[1]) or 
                                   (all_numeric_cols[j] == x[0] and all_numeric_cols[i] == x[1]) 
                                   for x in related_pairs):
                            related_pairs.append((all_numeric_cols[i], all_numeric_cols[j], 6))
        
        # Add scatter plots for the pairs we found
        for col1, col2, priority in related_pairs:
            charts.append({
                'type': 'scatter',
                'title': f'Relationship: {col1} vs {col2}',
                'x': col1,
                'y': col2,
                'priority': priority
            })
    
    # CATEGORICAL BREAKDOWNS (numerical value by category)
    
    # Numeric by category
    if all_numeric_cols and categ_cols:
        # Try to find meaningful combinations first
        for num_col in all_numeric_cols[:3]:
            for cat_col in categ_cols[:3]:
                priority = 6
                
                # Increase priority for certain combinations
                if num_col in financial_cols and cat_col.lower() in ['region', 'country', 'state', 'product', 'category']:
                    priority = 9
                    charts.append({
                        'type': 'bar',
                        'title': f'{num_col} by {cat_col}',
                        'x': cat_col,
                        'y': num_col,
                        'aggregation': 'sum',
                        'priority': priority
                    })
                
                # Box plots for showing distribution by category
                charts.append({
                    'type': 'box',
                    'title': f'Distribution of {num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col,
                    'priority': priority - 1  # Slightly lower priority than bar charts
                })
    
    # BINARY DATA VISUALIZATIONS
    
    # Binary columns are best shown as pie charts
    for col in binary_cols:
        charts.append({
            'type': 'pie',
            'title': f'{col} Distribution',
            'column': col,
            'priority': 8
        })
    
    # RATING DATA VISUALIZATIONS
    
    # Rating data often works well with specialized visualizations
    for col in rating_cols:
        # Basic distribution as a bar chart
        charts.append({
            'type': 'bar',
            'title': f'Rating Distribution: {col}',
            'x': col,
            'priority': 8
        })
        
        # If we have categories, show ratings by category
        if categ_cols:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{col} by {cat_col}',
                    'x': cat_col,
                    'y': col,
                    'priority': 7
                })
    
    # Sort charts by priority (highest first)
    charts.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    # Take top N unique chart configurations to avoid redundancy
    unique_charts = []
    unique_configs = set()
    
    for chart in charts:
        # Create a simplified representation of the chart for uniqueness checking
        if 'x' in chart and 'y' in chart:
            chart_key = (chart['type'], chart['x'], chart.get('y', ''))
        elif 'column' in chart:
            chart_key = (chart['type'], chart['column'], '')
        else:
            chart_key = (chart['type'], chart.get('x', ''), '')
        
        if chart_key not in unique_configs:
            unique_configs.add(chart_key)
            # Remove priority field before returning
            if 'priority' in chart:
                del chart['priority']
            unique_charts.append(chart)
    
    return unique_charts[:15]  # Limit to top 15 unique charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration with intelligent formatting."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        # Determine optimal number of bins based on data
        x_col = chart_config['x']
        n_bins = min(max(10, int(df[x_col].nunique() / 5)), 50)
        
        # Choose color based on the column name
        if any(term in x_col.lower() for term in ['revenue', 'sales', 'income', 'profit']):
            color = '#2E8B57'  # Green for positive financial metrics
        elif any(term in x_col.lower() for term in ['cost', 'expense', 'loss']):
            color = '#CD5C5C'  # Red for negative financial metrics
        elif any(term in x_col.lower() for term in ['rating', 'score', 'satisfaction']):
            color = '#4682B4'  # Blue for ratings
        else:
            color = '#6495ED'  # Default blue
        
        fig = px.histogram(
            df, x=x_col,
            title=title,
            height=chart_height,
            template='plotly_white',
            nbins=n_bins,
            color_discrete_sequence=[color]
        )
        
        # Add mean line
        mean_value = df[x_col].mean()
        fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_value:.2f}",
                     annotation_position="top right")
    
    elif chart_type == 'bar':
        # Determine if this is a categorical count or a numerical aggregation
        if 'y' in chart_config:
            # This is a numerical aggregation by category
            x_col = chart_config['x']
            y_col = chart_config['y']
            agg_func = chart_config.get('aggregation', 'mean')
            
            # Limit to top categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            # Group and aggregate
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
            
            # Choose color based on the column name
            if any(term in y_col.lower() for term in ['revenue', 'sales', 'income', 'profit']):
                color = '#2E8B57'  # Green for positive financial metrics
            elif any(term in y_col.lower() for term in ['cost', 'expense', 'loss']):
                color = '#CD5C5C'  # Red for negative financial metrics
            else:
                color = '#6495ED'  # Default blue
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                hole=0.4,
                color_discrete_sequence=color_sequence
            )
            
            # Add percentage labels
            fig.update_traces(textinfo='percent+label')
        else:
            # Handle legacy configuration
            x_col = chart_config.get('x', '')
            value_counts = df[x_col].value_counts().head(10)
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=chart_height,
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_traces(textinfo='percent+label')
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Choose color based on the column name
        if any(term in y_col.lower() for term in ['revenue', 'sales', 'income', 'profit']):
            color = '#2E8B57'  # Green for positive financial metrics
        elif any(term in y_col.lower() for term in ['cost', 'expense', 'loss']):
            color = '#CD5C5C'  # Red for negative financial metrics
        elif any(term in y_col.lower() for term in ['rating', 'score', 'satisfaction']):
            color = '#4682B4'  # Blue for ratings
        else:
            color = '#6495ED'  # Default blue
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Determine best aggregation period based on date range
        try:
            date_range = (df[x_col].max() - df[x_col].min()).days
            
            if date_range > 365*2:  # More than 2 years
                freq = 'M'  # Monthly
                title = f"{y_col} by Month"
            elif date_range > 90:  # More than 3 months
                freq = 'W'  # Weekly
                title = f"{y_col} by Week"
            else:
                freq = 'D'  # Daily
                title = f"{y_col} by Day"
            
            # Group by date and aggregate
            agg_df = df.groupby(pd.Grouper(key=x_col, freq=freq))[y_col].agg(['mean', 'sum']).reset_index()
            
            # Choose aggregation based on the column name
            if any(term in y_col.lower() for term in ['revenue', 'sales', 'cost', 'profit', 'expense', 'price', 'total']):
                y_agg = 'sum'
                agg_label = 'Sum'
            else:
                y_agg = 'mean'
                agg_label = 'Average'
                
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_agg,
                title=f"{agg_label} {title}",
                height=chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add markers for better visibility
            fig.update_traces(mode='lines+markers')
            
            # Properly format y-axis for financial data
            if any(term in y_col.lower() for term in ['revenue', 'sales', 'cost', 'expense', 'price', 'profit']):
                fig.update_layout(yaxis=dict(tickprefix='chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add average line for reference
            avg_value = agg_df[y_col].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_value:.2f}",
                         annotation_position="top right")
        else:
            # This is a categorical count
            x_col = chart_config['x']
            
            # Handle case when there are too many categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            value_counts = filtered_df[x_col].value_counts().reset_index()
            
            # Choose color based on the column name
            if any(term in x_col.lower() for term in ['gender', 'sex']):
                color_map = {'Male': '#ADD8E6', 'Female': '#FFB6C1', 'Other': '#98FB98'}
                
                # Create custom color sequence
                colors = [color_map.get(str(val), '#6495ED') for val in value_counts['index']]
                
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=colors
                )
            elif any(term in x_col.lower() for term in ['country', 'nation', 'region', 'state', 'province']):
                # Geographic columns look good with a map color scheme
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color_discrete_sequence=['#6495ED']
                )
            
            # Add percentage labels to the bars
            total = value_counts[x_col].sum()
            percentages = [f"{(val/total)*100:.1f}%" for val in value_counts[x_col]]
            
            fig.update_traces(text=percentages, textposition='outside')
    
    elif chart_type == 'pie':
        # This handles both traditional pie charts and categorical distributions
        if 'column' in chart_config:
            column = chart_config['column']
            
            # Get counts and calculate percentages
            value_counts = df[column].value_counts().head(10)
            
            # Check if we need to handle "Other" category
            if df[column].nunique() > 10:
                other_count = df[column].value_counts().iloc[10:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Choose colors based on column name
            if any(term in column.lower() for term in ['gender', 'sex']):
                # Use gender-specific colors if values match common gender terms
                values = value_counts.index.tolist()
                color_sequence = []
                
                for val in values:
                    val_str = str(val).lower()
                    if 'male' in val_str or 'm' == val_str:
                        color_sequence.append('#ADD8E6')  # Light blue for male
                    elif 'female' in val_str or 'f' == val_str:
                        color_sequence.append('#FFB6C1')  # Light pink for female
                    elif 'other' in val_str or 'non-binary' in val_str or 'nonbinary' in val_str:
                        color_sequence.append('#98FB98')  # Light green for other
                    else:
                        color_sequence.append('#D3D3D3')  # Light gray for unknown
            else:
                color_sequence = px.colors.qualitative.Pastel
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def analyze_dataset(df):
    """Perform advanced analysis on the dataset to extract insights."""
    insights = []
    
    # Detect data types
    data_types, semantic_types = detect_data_types(df)
    
    # Get numeric columns
    numeric_cols = [col for col, type_ in data_types.items() if type_ in ['numerical', 'financial']]
    categorical_cols = [col for col, type_ in data_types.items() if type_ in ['categorical', 'gender', 'binary']]
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    # 1. Check for highly correlated features
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated pairs (absolute correlation > 0.7)
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    relationship = "positively" if corr_value > 0 else "negatively"
                    insights.append(f"**Strong Correlation:** {col1} and {col2} are {relationship} correlated (r={corr_value:.2f}).")
    
    # 2. Check for outliers in numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        if outlier_pct > 5:
            insights.append(f"**Outliers:** {col} has significant outliers ({outlier_pct:.1f}% of data).")
    
    # 3. Check for skewed distributions
    for col in numeric_cols[:5]:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"**Skewed Distribution:** {col} is {direction}-skewed (skewness={skewness:.2f}).")
    
    # 4. Check for imbalanced categories
    for col in categorical_cols[:5]:
        value_counts = df[col].value_counts(normalize=True)
        if value_counts.iloc[0] > 0.8:  # If dominant category > 80%
            insights.append(f"**Imbalanced Category:** {col} is dominated by '{value_counts.index[0]}' ({value_counts.iloc[0]*100:.1f}%).")
    
    # 5. Time series insights
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        
        for num_col in numeric_cols[:3]:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Check for trend
                df_sorted = df.sort_values(date_col)
                df_sorted = df_sorted.dropna(subset=[date_col, num_col])
                
                if len(df_sorted) >= 10:  # Only if we have enough data points
                    earliest_value = df_sorted[num_col].iloc[0]
                    latest_value = df_sorted[num_col].iloc[-1]
                    pct_change = ((latest_value - earliest_value) / earliest_value) * 100
                    
                    if abs(pct_change) > 20:  # Significant change
                        direction = "increased" if pct_change > 0 else "decreased"
                        insights.append(f"**Trend Detected:** {num_col} has {direction} by {abs(pct_change):.1f}% over the time period.")
            except:
                pass
    
    # 6. Missing value patterns
    missing_cols = df.columns[df.isnull().sum() > 0]
    if len(missing_cols) > 0:
        missing_percentages = df[missing_cols].isnull().mean() * 100
        for col, pct in missing_percentages.items():
            if pct > 5:  # More than 5% missing
                insights.append(f"**Missing Data:** {col} is missing {pct:.1f}% of values.")
    
    # Limit to top 10 insights
    return insights[:10]

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read
                if uploaded_file.name.endswith('.csv'):
                    # Try to infer encoding and delimiter
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except:
                        try:
                            df = pd.read_csv(uploaded_file, encoding='latin1')
                        except:
                            # Try to detect delimiter
                            data = uploaded_file.getvalue().decode('utf-8', errors='replace')
                            sniffer = csv.Sniffer()
                            dialect = sniffer.sniff(data[:1024])
                            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=dialect.delimiter)
                else:
                    # For Excel files
                    try:
                        df = pd.read_excel(uploaded_file, engine='openpyxl')
                    except:
                        df = pd.read_excel(uploaded_file, engine='xlrd')
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Display basic dataset info
                st.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Sample data option
                if len(df) > 1000:
                    sample_size = st.slider("Sample size for large datasets", 
                                         min_value=1000, 
                                         max_value=min(50000, len(df)), 
                                         value=min(len(df), 5000),
                                         step=1000)
                    
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Data cleaning options
                st.subheader("Data Cleaning Options")
                
                # Missing value handling
                missing_values = df.isnull().sum().sum()
                if missing_values > 0:
                    st.write(f"Missing Values: {missing_values} ({missing_values/(df.size)*100:.1f}%)")
                    missing_strategy = st.radio(
                        "How to handle missing values?",
                        ["Fill numerical with mean", 
                         "Fill categorical with mode",
                         "Remove rows with missing values", 
                         "No handling"]
                    )
                    
                    if missing_strategy == "Remove rows with missing values":
                        df = df.dropna()
                        st.success(f"Removed rows with missing values. New shape: {df.shape}")
                    elif missing_strategy == "Fill numerical with mean":
                        for col in df.select_dtypes(include=['number']).columns:
                            df[col] = df[col].fillna(df[col].mean())
                        st.success("Filled numerical missing values with mean")
                    elif missing_strategy == "Fill categorical with mode":
                        for col in df.select_dtypes(exclude=['number']).columns:
                            if df[col].mode().shape[0] > 0:
                                df[col] = df[col].fillna(df[col].mode()[0])
                        st.success("Filled categorical missing values with mode")
                
                # Duplicate handling
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.write(f"Duplicate Rows: {duplicates} ({duplicates/len(df)*100:.1f}%)")
                    if st.checkbox("Remove duplicate rows"):
                        df = df.drop_duplicates()
                        st.success(f"Removed duplicate rows. New shape: {df.shape}")
                
                # Outlier handling
                if st.checkbox("Check for outliers"):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    outlier_cols = []
                    
                    for col in numeric_cols:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)].shape[0]
                        if outliers > 0:
                            outlier_pct = outliers / len(df) * 100
                            outlier_cols.append((col, outlier_pct))
                    
                    if outlier_cols:
                        st.write("Columns with outliers:")
                        for col, pct in outlier_cols:
                            st.write(f"- {col}: {pct:.1f}% outliers")
                        
                        outlier_strategy = st.radio(
                            "How to handle outliers?",
                            ["No handling", "Cap outliers", "Remove outlier rows"]
                        )
                        
                        if outlier_strategy == "Cap outliers":
                            for col, _ in outlier_cols:
                                q1 = df[col].quantile(0.25)
                                q3 = df[col].quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                df[col] = df[col].clip(lower_bound, upper_bound)
                            st.success("Capped outliers to IQR boundaries")
                        
                        elif outlier_strategy == "Remove outlier rows":
                            original_len = len(df)
                            for col, _ in outlier_cols:
                                q1 = df[col].quantile(0.25)
                                q3 = df[col].quantile(0.75)
                                iqr = q3 - q1
                                df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
                            st.success(f"Removed {original_len - len(df)} outlier rows. New shape: {df.shape}")
                
                # Data transformation options
                st.subheader("Data Transformation")
                transform_options = st.multiselect(
                    "Select transformations to apply",
                    ["Normalize numeric columns", "Log transform skewed numeric columns", 
                     "Convert categorical to one-hot", "Extract date components"]
                )
                
                if "Normalize numeric columns" in transform_options:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        df[f"{col}_normalized"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    st.success("Added normalized versions of numeric columns")
                
                if "Log transform skewed numeric columns" in transform_options:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        # Only transform if positive and skewed
                        if df[col].min() > 0 and abs(df[col].skew()) > 1:
                            df[f"{col}_log"] = np.log1p(df[col])
                    st.success("Added log-transformed versions of skewed numeric columns")
                
                if "Convert categorical to one-hot" in transform_options:
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    
                    if categorical_cols.any():
                        selected_cols = st.multiselect(
                            "Select categorical columns to convert",
                            categorical_cols
                        )
                        
                        if selected_cols:
                            df = pd.get_dummies(df, columns=selected_cols, drop_first=True)
                            st.success(f"Converted {len(selected_cols)} categorical columns to one-hot encoding")
                
                if "Extract date components" in transform_options:
                    # Identify datetime columns
                    datetime_cols = []
                    for col in df.columns:
                        try:
                            if not pd.api.types.is_datetime64_dtype(df[col]):
                                pd.to_datetime(df[col], errors='raise')
                            datetime_cols.append(col)
                        except:
                            pass
                    
                    if datetime_cols:
                        selected_cols = st.multiselect(
                            "Select date columns to extract components from",
                            datetime_cols
                        )
                        
                        for col in selected_cols:
                            # Convert to datetime if not already
                            if not pd.api.types.is_datetime64_dtype(df[col]):
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            
                            # Extract components
                            df[f"{col}_year"] = df[col].dt.year
                            df[f"{col}_month"] = df[col].dt.month
                            df[f"{col}_day"] = df[col].dt.day
                            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                            df[f"{col}_quarter"] = df[col].dt.quarter
                        
                        st.success(f"Extracted date components from {len(selected_cols)} columns")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                st.subheader("Sample Data Options")
                dataset_type = st.selectbox(
                    "Choose sample dataset type",
                    ["Sales Data", "Customer Demographics", "Financial Performance", "Product Ratings"]
                )
                
                # Create different sample datasets based on selection
                if dataset_type == "Sales Data":
                    dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
                    regions = ['North', 'South', 'East', 'West']
                    
                    np.random.seed(42)
                    df = pd.DataFrame({
                        'Date': dates,
                        'Product': np.random.choice(products, size=100),
                        'Region': np.random.choice(regions, size=100),
                        'Sales': np.random.normal(1000, 200, size=100),
                        'Units': np.random.randint(10, 100, size=100),
                        'Profit': np.random.normal(200, 50, size=100),
                        'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                    })
                
                elif dataset_type == "Customer Demographics":
                    np.random.seed(42)
                    n = 200
                    genders = ['Male', 'Female', 'Other']
                    gender_probabilities = [0.48, 0.48, 0.04]
                    education = ['High School', 'Bachelor', 'Master', 'PhD', 'Other']
                    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Other']
                    country_probabilities = [0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1]
                    
                    df = pd.DataFrame({
                        'Age': np.random.randint(18, 85, size=n),
                        'Gender': np.random.choice(genders, size=n, p=gender_probabilities),
                        'Income': np.random.normal(60000, 20000, size=n),
                        'Education': np.random.choice(education, size=n),
                        'Country': np.random.choice(countries, size=n, p=country_probabilities),
                        'Customer_Since': pd.date_range(start
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, '€', '£']
    }
    
    temporal_patterns = {
        'date': ['date', 'day', 'time', 'timestamp'],
        'year': ['year', 'yr', 'annual'],
        'month': ['month', 'mon'],
        'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']
    }
    
    rating_patterns = {
        'rating': ['rating', 'score', 'satisfaction', 'review', 'stars', 'grade', 'rank']
    }
    
    binary_patterns = {
        'binary': ['yes/no', 'true/false', 'y/n', 't/f', 'pass/fail', 'approved/rejected', 'success/failure']
    }
    
    # Combine all patterns
    all_patterns = {}
    all_patterns.update({k: v for d in [demographic_patterns, financial_patterns, temporal_patterns, 
                                       rating_patterns, binary_patterns] for k, v in d.items()})
    
    for col in df.columns:
        col_lower = col.lower()
        
        # First detect semantic type based on column name
        for semantic_type, keywords in all_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                semantic_types[col] = semantic_type
                break
        
        # Then detect data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a binary numeric column (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                data_types[col] = 'binary'
            # Check if it's a rating or score (limited range)
            elif col in semantic_types and semantic_types[col] == 'rating':
                data_types[col] = 'rating'
            # Check if it's a small set of integers that should be treated as categories
            elif df[col].nunique() <= 10 or (df[col].dtype == 'int64' and df[col].max() <= 10):
                data_types[col] = 'categorical'
            # Otherwise it's a regular numeric column
            else:
                data_types[col] = 'numerical'
                
                # Additional financial type detection
                if col in semantic_types and semantic_types[col] in ['revenue', 'cost', 'profit', 'income']:
                    data_types[col] = 'financial'
        
        # Date/time detection
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            data_types[col] = 'datetime'
        
        # Categorical data detection
        elif df[col].nunique() <= 20 or df[col].nunique() / len(df) < 0.05:
            data_types[col] = 'categorical'
            
            # Check if it's specifically a gender column
            if col in semantic_types and semantic_types[col] == 'gender':
                data_types[col] = 'gender'
            
            # Check if it's a yes/no or true/false column
            elif df[col].nunique() <= 2:
                # Look at the values to determine if it's binary
                values = set(str(x).lower() for x in df[col].dropna().unique())
                if values.issubset({'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 
                                   'pass', 'fail', 'success', 'failure'}):
                    data_types[col] = 'binary'
        
        # Text data
        else:
            data_types[col] = 'text'
    
    return data_types, semantic_types

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types):
    """Recommend charts based on the data types."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    charts = []
    
    # Distribution charts for numeric columns
    for col in numeric_cols[:5]:  # Limit to 5 to avoid too many charts
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col
        })
    
    # Bar charts for categorical columns
    for col in categ_cols[:5]:
        # Get top categories by count
        value_counts = df[col].value_counts().head(10)
        charts.append({
            'type': 'bar',
            'title': f'Count by {col}',
            'x': col,
            'values': value_counts.index.tolist()
        })
    
    # Time series for datetime columns
    for datetime_col in datetime_cols[:1]:
        for num_col in numeric_cols[:3]:
            charts.append({
                'type': 'line',
                'title': f'{num_col} over {datetime_col}',
                'x': datetime_col,
                'y': num_col
            })
    
    # Scatter plots for pairs of numerical columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                charts.append({
                    'type': 'scatter',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'x': numeric_cols[i],
                    'y': numeric_cols[j]
                })
    
    # Numeric by category
    if len(numeric_cols) > 0 and len(categ_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col
                })
    
    return charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        fig = px.histogram(
            df, x=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'bar':
        # Handle case when there are too many categories
        if 'values' in chart_config:
            filtered_df = df[df[chart_config['x']].isin(chart_config['values'])]
        else:
            top_values = df[chart_config['x']].value_counts().head(10).index
            filtered_df = df[df[chart_config['x']].isin(top_values)]
        
        fig = px.bar(
            filtered_df[chart_config['x']].value_counts().reset_index(),
            x='index', y=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white',
            labels={'index': chart_config['x'], chart_config['x']: 'Count'}
        )
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Group by date and aggregate
        try:
            agg_df = df.groupby(pd.Grouper(key=x_col, freq='D')).agg({y_col: 'mean'}).reset_index()
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
        except:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'box':
        fig = px.box(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, separatethousands=True))
            
            # Add trend line for longer time series
            if len(agg_df) > 5:
                # Create simple moving average for trend
                agg_df['trend'] = agg_df[y_agg].rolling(window=min(3, len(agg_df)), min_periods=1).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=agg_df[x_col], 
                        y=agg_df['trend'],
                        mode='lines',
                        line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash'),
                        name='Trend'
                    )
                )
        except Exception as e:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add markers for better visibility
            fig.update_traces(mode='lines+markers')
    
    elif chart_type == 'scatter':
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Check if we should add a trend line
        add_trend = True
        
        # Choose color based on the column names
        if any(term in y_col.lower() for term in ['revenue', 'sales', 'income', 'profit']):
            color = '#2E8B57'  # Green for positive financial metrics
        elif any(term in y_col.lower() for term in ['cost', 'expense', 'loss']):
            color = '#CD5C5C'  # Red for negative financial metrics
        else:
            color = '#6495ED'  # Default blue
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            height=chart_height,
            template='plotly_white',
            color_discrete_sequence=[color],
            opacity=0.7
        )
        
        # Add trendline for numerical relationships
        if add_trend:
            fig.update_layout(
                shapes=[
                    dict(
                        type='line',
                        xref='x', yref='y',
                        x0=df[x_col].min(), y0=df[y_col].min(),
                        x1=df[x_col].max(), y1=df[y_col].max(),
                        line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash')
                    )
                ]
            )
            
            # Add correlation coefficient as annotation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            fig.add_annotation(
                x=0.95, y=0.05,
                xref="paper", yref="paper",
                text=f"Correlation: {corr:.2f}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Properly format axes for financial data
        if any(term in x_col.lower() for term in ['revenue', 'sales', 'cost', 'expense', 'price', 'profit']):
            fig.update_layout(xaxis=dict(tickprefix='chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add average line for reference
            avg_value = agg_df[y_col].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_value:.2f}",
                         annotation_position="top right")
        else:
            # This is a categorical count
            x_col = chart_config['x']
            
            # Handle case when there are too many categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            value_counts = filtered_df[x_col].value_counts().reset_index()
            
            # Choose color based on the column name
            if any(term in x_col.lower() for term in ['gender', 'sex']):
                color_map = {'Male': '#ADD8E6', 'Female': '#FFB6C1', 'Other': '#98FB98'}
                
                # Create custom color sequence
                colors = [color_map.get(str(val), '#6495ED') for val in value_counts['index']]
                
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=colors
                )
            elif any(term in x_col.lower() for term in ['country', 'nation', 'region', 'state', 'province']):
                # Geographic columns look good with a map color scheme
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color_discrete_sequence=['#6495ED']
                )
            
            # Add percentage labels to the bars
            total = value_counts[x_col].sum()
            percentages = [f"{(val/total)*100:.1f}%" for val in value_counts[x_col]]
            
            fig.update_traces(text=percentages, textposition='outside')
    
    elif chart_type == 'pie':
        # This handles both traditional pie charts and categorical distributions
        if 'column' in chart_config:
            column = chart_config['column']
            
            # Get counts and calculate percentages
            value_counts = df[column].value_counts().head(10)
            
            # Check if we need to handle "Other" category
            if df[column].nunique() > 10:
                other_count = df[column].value_counts().iloc[10:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Choose colors based on column name
            if any(term in column.lower() for term in ['gender', 'sex']):
                # Use gender-specific colors if values match common gender terms
                values = value_counts.index.tolist()
                color_sequence = []
                
                for val in values:
                    val_str = str(val).lower()
                    if 'male' in val_str or 'm' == val_str:
                        color_sequence.append('#ADD8E6')  # Light blue for male
                    elif 'female' in val_str or 'f' == val_str:
                        color_sequence.append('#FFB6C1')  # Light pink for female
                    elif 'other' in val_str or 'non-binary' in val_str or 'nonbinary' in val_str:
                        color_sequence.append('#98FB98')  # Light green for other
                    else:
                        color_sequence.append('#D3D3D3')  # Light gray for unknown
            else:
                color_sequence = px.colors.qualitative.Pastel
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, '€', '£']
    }
    
    temporal_patterns = {
        'date': ['date', 'day', 'time', 'timestamp'],
        'year': ['year', 'yr', 'annual'],
        'month': ['month', 'mon'],
        'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']
    }
    
    rating_patterns = {
        'rating': ['rating', 'score', 'satisfaction', 'review', 'stars', 'grade', 'rank']
    }
    
    binary_patterns = {
        'binary': ['yes/no', 'true/false', 'y/n', 't/f', 'pass/fail', 'approved/rejected', 'success/failure']
    }
    
    # Combine all patterns
    all_patterns = {}
    all_patterns.update({k: v for d in [demographic_patterns, financial_patterns, temporal_patterns, 
                                       rating_patterns, binary_patterns] for k, v in d.items()})
    
    for col in df.columns:
        col_lower = col.lower()
        
        # First detect semantic type based on column name
        for semantic_type, keywords in all_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                semantic_types[col] = semantic_type
                break
        
        # Then detect data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a binary numeric column (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                data_types[col] = 'binary'
            # Check if it's a rating or score (limited range)
            elif col in semantic_types and semantic_types[col] == 'rating':
                data_types[col] = 'rating'
            # Check if it's a small set of integers that should be treated as categories
            elif df[col].nunique() <= 10 or (df[col].dtype == 'int64' and df[col].max() <= 10):
                data_types[col] = 'categorical'
            # Otherwise it's a regular numeric column
            else:
                data_types[col] = 'numerical'
                
                # Additional financial type detection
                if col in semantic_types and semantic_types[col] in ['revenue', 'cost', 'profit', 'income']:
                    data_types[col] = 'financial'
        
        # Date/time detection
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            data_types[col] = 'datetime'
        
        # Categorical data detection
        elif df[col].nunique() <= 20 or df[col].nunique() / len(df) < 0.05:
            data_types[col] = 'categorical'
            
            # Check if it's specifically a gender column
            if col in semantic_types and semantic_types[col] == 'gender':
                data_types[col] = 'gender'
            
            # Check if it's a yes/no or true/false column
            elif df[col].nunique() <= 2:
                # Look at the values to determine if it's binary
                values = set(str(x).lower() for x in df[col].dropna().unique())
                if values.issubset({'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 
                                   'pass', 'fail', 'success', 'failure'}):
                    data_types[col] = 'binary'
        
        # Text data
        else:
            data_types[col] = 'text'
    
    return data_types, semantic_types

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types):
    """Recommend charts based on the data types."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    charts = []
    
    # Distribution charts for numeric columns
    for col in numeric_cols[:5]:  # Limit to 5 to avoid too many charts
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col
        })
    
    # Bar charts for categorical columns
    for col in categ_cols[:5]:
        # Get top categories by count
        value_counts = df[col].value_counts().head(10)
        charts.append({
            'type': 'bar',
            'title': f'Count by {col}',
            'x': col,
            'values': value_counts.index.tolist()
        })
    
    # Time series for datetime columns
    for datetime_col in datetime_cols[:1]:
        for num_col in numeric_cols[:3]:
            charts.append({
                'type': 'line',
                'title': f'{num_col} over {datetime_col}',
                'x': datetime_col,
                'y': num_col
            })
    
    # Scatter plots for pairs of numerical columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                charts.append({
                    'type': 'scatter',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'x': numeric_cols[i],
                    'y': numeric_cols[j]
                })
    
    # Numeric by category
    if len(numeric_cols) > 0 and len(categ_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col
                })
    
    return charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        fig = px.histogram(
            df, x=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'bar':
        # Handle case when there are too many categories
        if 'values' in chart_config:
            filtered_df = df[df[chart_config['x']].isin(chart_config['values'])]
        else:
            top_values = df[chart_config['x']].value_counts().head(10).index
            filtered_df = df[df[chart_config['x']].isin(top_values)]
        
        fig = px.bar(
            filtered_df[chart_config['x']].value_counts().reset_index(),
            x='index', y=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white',
            labels={'index': chart_config['x'], chart_config['x']: 'Count'}
        )
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Group by date and aggregate
        try:
            agg_df = df.groupby(pd.Grouper(key=x_col, freq='D')).agg({y_col: 'mean'}).reset_index()
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
        except:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'box':
        fig = px.box(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, separatethousands=True))
        if any(term in y_col.lower() for term in ['revenue', 'sales', 'cost', 'expense', 'price', 'profit']):
            fig.update_layout(yaxis=dict(tickprefix='chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add average line for reference
            avg_value = agg_df[y_col].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_value:.2f}",
                         annotation_position="top right")
        else:
            # This is a categorical count
            x_col = chart_config['x']
            
            # Handle case when there are too many categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            value_counts = filtered_df[x_col].value_counts().reset_index()
            
            # Choose color based on the column name
            if any(term in x_col.lower() for term in ['gender', 'sex']):
                color_map = {'Male': '#ADD8E6', 'Female': '#FFB6C1', 'Other': '#98FB98'}
                
                # Create custom color sequence
                colors = [color_map.get(str(val), '#6495ED') for val in value_counts['index']]
                
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=colors
                )
            elif any(term in x_col.lower() for term in ['country', 'nation', 'region', 'state', 'province']):
                # Geographic columns look good with a map color scheme
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color_discrete_sequence=['#6495ED']
                )
            
            # Add percentage labels to the bars
            total = value_counts[x_col].sum()
            percentages = [f"{(val/total)*100:.1f}%" for val in value_counts[x_col]]
            
            fig.update_traces(text=percentages, textposition='outside')
    
    elif chart_type == 'pie':
        # This handles both traditional pie charts and categorical distributions
        if 'column' in chart_config:
            column = chart_config['column']
            
            # Get counts and calculate percentages
            value_counts = df[column].value_counts().head(10)
            
            # Check if we need to handle "Other" category
            if df[column].nunique() > 10:
                other_count = df[column].value_counts().iloc[10:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Choose colors based on column name
            if any(term in column.lower() for term in ['gender', 'sex']):
                # Use gender-specific colors if values match common gender terms
                values = value_counts.index.tolist()
                color_sequence = []
                
                for val in values:
                    val_str = str(val).lower()
                    if 'male' in val_str or 'm' == val_str:
                        color_sequence.append('#ADD8E6')  # Light blue for male
                    elif 'female' in val_str or 'f' == val_str:
                        color_sequence.append('#FFB6C1')  # Light pink for female
                    elif 'other' in val_str or 'non-binary' in val_str or 'nonbinary' in val_str:
                        color_sequence.append('#98FB98')  # Light green for other
                    else:
                        color_sequence.append('#D3D3D3')  # Light gray for unknown
            else:
                color_sequence = px.colors.qualitative.Pastel
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, '€', '£']
    }
    
    temporal_patterns = {
        'date': ['date', 'day', 'time', 'timestamp'],
        'year': ['year', 'yr', 'annual'],
        'month': ['month', 'mon'],
        'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']
    }
    
    rating_patterns = {
        'rating': ['rating', 'score', 'satisfaction', 'review', 'stars', 'grade', 'rank']
    }
    
    binary_patterns = {
        'binary': ['yes/no', 'true/false', 'y/n', 't/f', 'pass/fail', 'approved/rejected', 'success/failure']
    }
    
    # Combine all patterns
    all_patterns = {}
    all_patterns.update({k: v for d in [demographic_patterns, financial_patterns, temporal_patterns, 
                                       rating_patterns, binary_patterns] for k, v in d.items()})
    
    for col in df.columns:
        col_lower = col.lower()
        
        # First detect semantic type based on column name
        for semantic_type, keywords in all_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                semantic_types[col] = semantic_type
                break
        
        # Then detect data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a binary numeric column (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                data_types[col] = 'binary'
            # Check if it's a rating or score (limited range)
            elif col in semantic_types and semantic_types[col] == 'rating':
                data_types[col] = 'rating'
            # Check if it's a small set of integers that should be treated as categories
            elif df[col].nunique() <= 10 or (df[col].dtype == 'int64' and df[col].max() <= 10):
                data_types[col] = 'categorical'
            # Otherwise it's a regular numeric column
            else:
                data_types[col] = 'numerical'
                
                # Additional financial type detection
                if col in semantic_types and semantic_types[col] in ['revenue', 'cost', 'profit', 'income']:
                    data_types[col] = 'financial'
        
        # Date/time detection
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            data_types[col] = 'datetime'
        
        # Categorical data detection
        elif df[col].nunique() <= 20 or df[col].nunique() / len(df) < 0.05:
            data_types[col] = 'categorical'
            
            # Check if it's specifically a gender column
            if col in semantic_types and semantic_types[col] == 'gender':
                data_types[col] = 'gender'
            
            # Check if it's a yes/no or true/false column
            elif df[col].nunique() <= 2:
                # Look at the values to determine if it's binary
                values = set(str(x).lower() for x in df[col].dropna().unique())
                if values.issubset({'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 
                                   'pass', 'fail', 'success', 'failure'}):
                    data_types[col] = 'binary'
        
        # Text data
        else:
            data_types[col] = 'text'
    
    return data_types, semantic_types

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types):
    """Recommend charts based on the data types."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    charts = []
    
    # Distribution charts for numeric columns
    for col in numeric_cols[:5]:  # Limit to 5 to avoid too many charts
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col
        })
    
    # Bar charts for categorical columns
    for col in categ_cols[:5]:
        # Get top categories by count
        value_counts = df[col].value_counts().head(10)
        charts.append({
            'type': 'bar',
            'title': f'Count by {col}',
            'x': col,
            'values': value_counts.index.tolist()
        })
    
    # Time series for datetime columns
    for datetime_col in datetime_cols[:1]:
        for num_col in numeric_cols[:3]:
            charts.append({
                'type': 'line',
                'title': f'{num_col} over {datetime_col}',
                'x': datetime_col,
                'y': num_col
            })
    
    # Scatter plots for pairs of numerical columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                charts.append({
                    'type': 'scatter',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'x': numeric_cols[i],
                    'y': numeric_cols[j]
                })
    
    # Numeric by category
    if len(numeric_cols) > 0 and len(categ_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col
                })
    
    return charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        fig = px.histogram(
            df, x=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'bar':
        # Handle case when there are too many categories
        if 'values' in chart_config:
            filtered_df = df[df[chart_config['x']].isin(chart_config['values'])]
        else:
            top_values = df[chart_config['x']].value_counts().head(10).index
            filtered_df = df[df[chart_config['x']].isin(top_values)]
        
        fig = px.bar(
            filtered_df[chart_config['x']].value_counts().reset_index(),
            x='index', y=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white',
            labels={'index': chart_config['x'], chart_config['x']: 'Count'}
        )
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Group by date and aggregate
        try:
            agg_df = df.groupby(pd.Grouper(key=x_col, freq='D')).agg({y_col: 'mean'}).reset_index()
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
        except:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'box':
        fig = px.box(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, separatethousands=True))
    
    elif chart_type == 'box':
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Limit to top categories if there are too many
        if df[x_col].nunique() > 10:
            top_values = df[x_col].value_counts().head(10).index
            filtered_df = df[df[x_col].isin(top_values)]
        else:
            filtered_df = df
        
        # Choose color based on the column name
        if any(term in y_col.lower() for term in ['revenue', 'sales', 'income', 'profit']):
            color_discrete_sequence = px.colors.sequential.Greens
        elif any(term in y_col.lower() for term in ['cost', 'expense', 'loss']):
            color_discrete_sequence = px.colors.sequential.Reds
        elif any(term in y_col.lower() for term in ['rating', 'score', 'satisfaction']):
            color_discrete_sequence = px.colors.sequential.Blues
        else:
            color_discrete_sequence = px.colors.qualitative.Pastel
        
        fig = px.box(
            filtered_df,
            x=x_col,
            y=y_col,
            title=title,
            height=chart_height,
            template='plotly_white',
            color=x_col,
            color_discrete_sequence=color_discrete_sequence
        )
        
        # Add mean markers
        fig.update_traces(boxmean=True)
        
        # Properly format y-axis for financial data
        if any(term in y_col.lower() for term in ['revenue', 'sales', 'cost', 'expense', 'price', 'profit']):
            fig.update_layout(yaxis=dict(tickprefix='chart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add average line for reference
            avg_value = agg_df[y_col].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_value:.2f}",
                         annotation_position="top right")
        else:
            # This is a categorical count
            x_col = chart_config['x']
            
            # Handle case when there are too many categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            value_counts = filtered_df[x_col].value_counts().reset_index()
            
            # Choose color based on the column name
            if any(term in x_col.lower() for term in ['gender', 'sex']):
                color_map = {'Male': '#ADD8E6', 'Female': '#FFB6C1', 'Other': '#98FB98'}
                
                # Create custom color sequence
                colors = [color_map.get(str(val), '#6495ED') for val in value_counts['index']]
                
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=colors
                )
            elif any(term in x_col.lower() for term in ['country', 'nation', 'region', 'state', 'province']):
                # Geographic columns look good with a map color scheme
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color_discrete_sequence=['#6495ED']
                )
            
            # Add percentage labels to the bars
            total = value_counts[x_col].sum()
            percentages = [f"{(val/total)*100:.1f}%" for val in value_counts[x_col]]
            
            fig.update_traces(text=percentages, textposition='outside')
    
    elif chart_type == 'pie':
        # This handles both traditional pie charts and categorical distributions
        if 'column' in chart_config:
            column = chart_config['column']
            
            # Get counts and calculate percentages
            value_counts = df[column].value_counts().head(10)
            
            # Check if we need to handle "Other" category
            if df[column].nunique() > 10:
                other_count = df[column].value_counts().iloc[10:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Choose colors based on column name
            if any(term in column.lower() for term in ['gender', 'sex']):
                # Use gender-specific colors if values match common gender terms
                values = value_counts.index.tolist()
                color_sequence = []
                
                for val in values:
                    val_str = str(val).lower()
                    if 'male' in val_str or 'm' == val_str:
                        color_sequence.append('#ADD8E6')  # Light blue for male
                    elif 'female' in val_str or 'f' == val_str:
                        color_sequence.append('#FFB6C1')  # Light pink for female
                    elif 'other' in val_str or 'non-binary' in val_str or 'nonbinary' in val_str:
                        color_sequence.append('#98FB98')  # Light green for other
                    else:
                        color_sequence.append('#D3D3D3')  # Light gray for unknown
            else:
                color_sequence = px.colors.qualitative.Pastel
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, '€', '£']
    }
    
    temporal_patterns = {
        'date': ['date', 'day', 'time', 'timestamp'],
        'year': ['year', 'yr', 'annual'],
        'month': ['month', 'mon'],
        'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']
    }
    
    rating_patterns = {
        'rating': ['rating', 'score', 'satisfaction', 'review', 'stars', 'grade', 'rank']
    }
    
    binary_patterns = {
        'binary': ['yes/no', 'true/false', 'y/n', 't/f', 'pass/fail', 'approved/rejected', 'success/failure']
    }
    
    # Combine all patterns
    all_patterns = {}
    all_patterns.update({k: v for d in [demographic_patterns, financial_patterns, temporal_patterns, 
                                       rating_patterns, binary_patterns] for k, v in d.items()})
    
    for col in df.columns:
        col_lower = col.lower()
        
        # First detect semantic type based on column name
        for semantic_type, keywords in all_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                semantic_types[col] = semantic_type
                break
        
        # Then detect data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a binary numeric column (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                data_types[col] = 'binary'
            # Check if it's a rating or score (limited range)
            elif col in semantic_types and semantic_types[col] == 'rating':
                data_types[col] = 'rating'
            # Check if it's a small set of integers that should be treated as categories
            elif df[col].nunique() <= 10 or (df[col].dtype == 'int64' and df[col].max() <= 10):
                data_types[col] = 'categorical'
            # Otherwise it's a regular numeric column
            else:
                data_types[col] = 'numerical'
                
                # Additional financial type detection
                if col in semantic_types and semantic_types[col] in ['revenue', 'cost', 'profit', 'income']:
                    data_types[col] = 'financial'
        
        # Date/time detection
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            data_types[col] = 'datetime'
        
        # Categorical data detection
        elif df[col].nunique() <= 20 or df[col].nunique() / len(df) < 0.05:
            data_types[col] = 'categorical'
            
            # Check if it's specifically a gender column
            if col in semantic_types and semantic_types[col] == 'gender':
                data_types[col] = 'gender'
            
            # Check if it's a yes/no or true/false column
            elif df[col].nunique() <= 2:
                # Look at the values to determine if it's binary
                values = set(str(x).lower() for x in df[col].dropna().unique())
                if values.issubset({'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 
                                   'pass', 'fail', 'success', 'failure'}):
                    data_types[col] = 'binary'
        
        # Text data
        else:
            data_types[col] = 'text'
    
    return data_types, semantic_types

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types):
    """Recommend charts based on the data types."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    charts = []
    
    # Distribution charts for numeric columns
    for col in numeric_cols[:5]:  # Limit to 5 to avoid too many charts
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col
        })
    
    # Bar charts for categorical columns
    for col in categ_cols[:5]:
        # Get top categories by count
        value_counts = df[col].value_counts().head(10)
        charts.append({
            'type': 'bar',
            'title': f'Count by {col}',
            'x': col,
            'values': value_counts.index.tolist()
        })
    
    # Time series for datetime columns
    for datetime_col in datetime_cols[:1]:
        for num_col in numeric_cols[:3]:
            charts.append({
                'type': 'line',
                'title': f'{num_col} over {datetime_col}',
                'x': datetime_col,
                'y': num_col
            })
    
    # Scatter plots for pairs of numerical columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                charts.append({
                    'type': 'scatter',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'x': numeric_cols[i],
                    'y': numeric_cols[j]
                })
    
    # Numeric by category
    if len(numeric_cols) > 0 and len(categ_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col
                })
    
    return charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        fig = px.histogram(
            df, x=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'bar':
        # Handle case when there are too many categories
        if 'values' in chart_config:
            filtered_df = df[df[chart_config['x']].isin(chart_config['values'])]
        else:
            top_values = df[chart_config['x']].value_counts().head(10).index
            filtered_df = df[df[chart_config['x']].isin(top_values)]
        
        fig = px.bar(
            filtered_df[chart_config['x']].value_counts().reset_index(),
            x='index', y=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white',
            labels={'index': chart_config['x'], chart_config['x']: 'Count'}
        )
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Group by date and aggregate
        try:
            agg_df = df.groupby(pd.Grouper(key=x_col, freq='D')).agg({y_col: 'mean'}).reset_index()
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
        except:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'box':
        fig = px.box(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, separatethousands=True))
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return figchart_height,
                template='plotly_white',
                color_discrete_sequence=[color]
            )
            
            # Add average line for reference
            avg_value = agg_df[y_col].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_value:.2f}",
                         annotation_position="top right")
        else:
            # This is a categorical count
            x_col = chart_config['x']
            
            # Handle case when there are too many categories
            if 'values' in chart_config:
                filtered_df = df[df[x_col].isin(chart_config['values'])]
            else:
                top_values = df[x_col].value_counts().head(10).index
                filtered_df = df[df[x_col].isin(top_values)]
            
            value_counts = filtered_df[x_col].value_counts().reset_index()
            
            # Choose color based on the column name
            if any(term in x_col.lower() for term in ['gender', 'sex']):
                color_map = {'Male': '#ADD8E6', 'Female': '#FFB6C1', 'Other': '#98FB98'}
                
                # Create custom color sequence
                colors = [color_map.get(str(val), '#6495ED') for val in value_counts['index']]
                
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=colors
                )
            elif any(term in x_col.lower() for term in ['country', 'nation', 'region', 'state', 'province']):
                # Geographic columns look good with a map color scheme
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color='index',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                fig = px.bar(
                    value_counts,
                    x='index', y=x_col,
                    title=title,
                    height=chart_height,
                    template='plotly_white',
                    labels={'index': x_col, x_col: 'Count'},
                    color_discrete_sequence=['#6495ED']
                )
            
            # Add percentage labels to the bars
            total = value_counts[x_col].sum()
            percentages = [f"{(val/total)*100:.1f}%" for val in value_counts[x_col]]
            
            fig.update_traces(text=percentages, textposition='outside')
    
    elif chart_type == 'pie':
        # This handles both traditional pie charts and categorical distributions
        if 'column' in chart_config:
            column = chart_config['column']
            
            # Get counts and calculate percentages
            value_counts = df[column].value_counts().head(10)
            
            # Check if we need to handle "Other" category
            if df[column].nunique() > 10:
                other_count = df[column].value_counts().iloc[10:].sum()
                value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Choose colors based on column name
            if any(term in column.lower() for term in ['gender', 'sex']):
                # Use gender-specific colors if values match common gender terms
                values = value_counts.index.tolist()
                color_sequence = []
                
                for val in values:
                    val_str = str(val).lower()
                    if 'male' in val_str or 'm' == val_str:
                        color_sequence.append('#ADD8E6')  # Light blue for male
                    elif 'female' in val_str or 'f' == val_str:
                        color_sequence.append('#FFB6C1')  # Light pink for female
                    elif 'other' in val_str or 'non-binary' in val_str or 'nonbinary' in val_str:
                        color_sequence.append('#98FB98')  # Light green for other
                    else:
                        color_sequence.append('#D3D3D3')  # Light gray for unknown
            else:
                color_sequence = px.colors.qualitative.Pastel
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                height=

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(, '€', '£']
    }
    
    temporal_patterns = {
        'date': ['date', 'day', 'time', 'timestamp'],
        'year': ['year', 'yr', 'annual'],
        'month': ['month', 'mon'],
        'quarter': ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']
    }
    
    rating_patterns = {
        'rating': ['rating', 'score', 'satisfaction', 'review', 'stars', 'grade', 'rank']
    }
    
    binary_patterns = {
        'binary': ['yes/no', 'true/false', 'y/n', 't/f', 'pass/fail', 'approved/rejected', 'success/failure']
    }
    
    # Combine all patterns
    all_patterns = {}
    all_patterns.update({k: v for d in [demographic_patterns, financial_patterns, temporal_patterns, 
                                       rating_patterns, binary_patterns] for k, v in d.items()})
    
    for col in df.columns:
        col_lower = col.lower()
        
        # First detect semantic type based on column name
        for semantic_type, keywords in all_patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                semantic_types[col] = semantic_type
                break
        
        # Then detect data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a binary numeric column (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                data_types[col] = 'binary'
            # Check if it's a rating or score (limited range)
            elif col in semantic_types and semantic_types[col] == 'rating':
                data_types[col] = 'rating'
            # Check if it's a small set of integers that should be treated as categories
            elif df[col].nunique() <= 10 or (df[col].dtype == 'int64' and df[col].max() <= 10):
                data_types[col] = 'categorical'
            # Otherwise it's a regular numeric column
            else:
                data_types[col] = 'numerical'
                
                # Additional financial type detection
                if col in semantic_types and semantic_types[col] in ['revenue', 'cost', 'profit', 'income']:
                    data_types[col] = 'financial'
        
        # Date/time detection
        elif pd.api.types.is_datetime64_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
            data_types[col] = 'datetime'
        
        # Categorical data detection
        elif df[col].nunique() <= 20 or df[col].nunique() / len(df) < 0.05:
            data_types[col] = 'categorical'
            
            # Check if it's specifically a gender column
            if col in semantic_types and semantic_types[col] == 'gender':
                data_types[col] = 'gender'
            
            # Check if it's a yes/no or true/false column
            elif df[col].nunique() <= 2:
                # Look at the values to determine if it's binary
                values = set(str(x).lower() for x in df[col].dropna().unique())
                if values.issubset({'yes', 'no', 'y', 'n', 'true', 'false', 't', 'f', '1', '0', 
                                   'pass', 'fail', 'success', 'failure'}):
                    data_types[col] = 'binary'
        
        # Text data
        else:
            data_types[col] = 'text'
    
    return data_types, semantic_types

def generate_summary_stats(df, data_types):
    """Generate summary statistics for the DataFrame."""
    stats = {}
    
    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['numerical_columns'] = sum(1 for t in data_types.values() if t == 'numerical')
    stats['categorical_columns'] = sum(1 for t in data_types.values() if t == 'categorical')
    stats['datetime_columns'] = sum(1 for t in data_types.values() if t == 'datetime')
    stats['missing_values'] = df.isna().sum().sum()
    stats['missing_percentage'] = (stats['missing_values'] / (stats['rows'] * stats['columns'])) * 100
    
    return stats

def get_recommended_charts(df, data_types):
    """Recommend charts based on the data types."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    
    charts = []
    
    # Distribution charts for numeric columns
    for col in numeric_cols[:5]:  # Limit to 5 to avoid too many charts
        charts.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'x': col
        })
    
    # Bar charts for categorical columns
    for col in categ_cols[:5]:
        # Get top categories by count
        value_counts = df[col].value_counts().head(10)
        charts.append({
            'type': 'bar',
            'title': f'Count by {col}',
            'x': col,
            'values': value_counts.index.tolist()
        })
    
    # Time series for datetime columns
    for datetime_col in datetime_cols[:1]:
        for num_col in numeric_cols[:3]:
            charts.append({
                'type': 'line',
                'title': f'{num_col} over {datetime_col}',
                'x': datetime_col,
                'y': num_col
            })
    
    # Scatter plots for pairs of numerical columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                charts.append({
                    'type': 'scatter',
                    'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                    'x': numeric_cols[i],
                    'y': numeric_cols[j]
                })
    
    # Numeric by category
    if len(numeric_cols) > 0 and len(categ_cols) > 0:
        for num_col in numeric_cols[:2]:
            for cat_col in categ_cols[:2]:
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'x': cat_col,
                    'y': num_col
                })
    
    return charts

def create_chart(df, chart_config, chart_height=400):
    """Create a chart based on the chart configuration."""
    chart_type = chart_config['type']
    title = chart_config['title']
    
    if chart_type == 'histogram':
        fig = px.histogram(
            df, x=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'bar':
        # Handle case when there are too many categories
        if 'values' in chart_config:
            filtered_df = df[df[chart_config['x']].isin(chart_config['values'])]
        else:
            top_values = df[chart_config['x']].value_counts().head(10).index
            filtered_df = df[df[chart_config['x']].isin(top_values)]
        
        fig = px.bar(
            filtered_df[chart_config['x']].value_counts().reset_index(),
            x='index', y=chart_config['x'],
            title=title,
            height=chart_height,
            template='plotly_white',
            labels={'index': chart_config['x'], chart_config['x']: 'Count'}
        )
    
    elif chart_type == 'line':
        # For time series, we need to ensure the x-axis is datetime and aggregate if necessary
        x_col = chart_config['x']
        y_col = chart_config['y']
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        
        # Group by date and aggregate
        try:
            agg_df = df.groupby(pd.Grouper(key=x_col, freq='D')).agg({y_col: 'mean'}).reset_index()
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
        except:
            # If grouping fails, just use the data as is
            fig = px.line(
                df.sort_values(x_col),
                x=x_col,
                y=y_col,
                title=title,
                height=chart_height,
                template='plotly_white'
            )
    
    elif chart_type == 'scatter':
        fig = px.scatter(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    elif chart_type == 'box':
        fig = px.box(
            df,
            x=chart_config['x'],
            y=chart_config['y'],
            title=title,
            height=chart_height,
            template='plotly_white'
        )
    
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'font': {'size': 16, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def generate_correlation_heatmap(df, data_types):
    """Generate correlation heatmap for numerical columns."""
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    
    if len(numeric_cols) <= 1:
        return None
    
    corr_df = df[numeric_cols].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_pie_charts(df, data_types):
    """Generate pie charts for categorical columns with few unique values."""
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    pie_charts = []
    
    for col in categ_cols:
        if df[col].nunique() <= 10:  # Only for columns with few categories
            value_counts = df[col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {col}",
                hole=0.4,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=400,
                title={
                    'font': {'size': 16, 'color': '#424242'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            pie_charts.append(fig)
    
    return pie_charts

def create_custom_chart(df, data_types):
    """Create a custom chart based on user selections."""
    st.subheader("Create Custom Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
        )
    
    numeric_cols = [col for col, type_ in data_types.items() if type_ == 'numerical']
    categ_cols = [col for col, type_ in data_types.items() if type_ == 'categorical']
    datetime_cols = [col for col, type_ in data_types.items() if type_ == 'datetime']
    all_cols = list(df.columns)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("Select X-axis (Category)", categ_cols, key="bar_x")
            agg_func = st.selectbox("Select Aggregation", ["Count", "Sum", "Mean", "Median", "Min", "Max"], key="bar_agg")
            
            if agg_func != "Count":
                y_col = st.selectbox("Select Y-axis (Value)", numeric_cols, key="bar_y")
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 10)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if agg_func == "Count":
            value_counts = df[x_col].value_counts().head(top_n)
            if show_others and len(df[x_col].unique()) > top_n:
                others_sum = df[x_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.bar(
                value_counts,
                title=f"Count by {x_col}",
                labels={'index': x_col, 'value': 'Count'}
            )
        else:
            # Group by category and apply aggregation
            agg_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Get top categories
            top_categories = df[x_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[x_col].isin(top_categories)].copy()
            
            if show_others and len(df[x_col].unique()) > top_n:
                df_others = df[~df[x_col].isin(top_categories)].copy()
                df_others[x_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=f"{agg_func} of {y_col} by {x_col}",
                text_auto='.2s'
            )
    
    elif chart_type == "Line Chart":
        with col1:
            x_options = datetime_cols + numeric_cols
            x_col = st.selectbox("Select X-axis", x_options, key="line_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
        
        with col2:
            if len(categ_cols) > 0:
                group_by = st.selectbox("Group by (optional)", ["None"] + categ_cols, key="line_group")
            else:
                group_by = "None"
            
            if x_col in datetime_cols:
                agg_period = st.selectbox("Aggregation Period", ["Day", "Week", "Month", "Quarter", "Year"], key="line_period")
        
        # Generate chart
        if x_col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            
            period_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }
            
            if group_by != "None":
                agg_df = df.groupby([pd.Grouper(key=x_col, freq=period_map[agg_period]), group_by])[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} over Time by {group_by}"
                )
            else:
                agg_df = df.groupby(pd.Grouper(key=x_col, freq=period_map[agg_period]))[y_col].mean().reset_index()
                fig = px.line(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over Time"
                )
        else:
            # For numeric x-axis
            if group_by != "None":
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=group_by,
                    title=f"{y_col} vs {x_col} by {group_by}"
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="scatter_color")
            else:
                color_by = "None"
            
            size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        # Generate chart
        if color_by != "None" and size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=size_col,
                title=f"{y_col} vs {x_col} by {color_by}, sized by {size_col}"
            )
        elif color_by != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"{y_col} vs {x_col} by {color_by}"
            )
        elif size_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                title=f"{y_col} vs {x_col}, sized by {size_col}"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}"
            )
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
        
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 20)
            if len(categ_cols) > 0:
                color_by = st.selectbox("Color by (optional)", ["None"] + categ_cols, key="hist_color")
            else:
                color_by = "None"
        
        # Generate chart
        if color_by != "None":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_by,
                nbins=n_bins,
                title=f"Distribution of {x_col} by {color_by}"
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=n_bins,
                title=f"Distribution of {x_col}"
            )
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Select Value Column", numeric_cols, key="box_y")
        
        with col2:
            if len(categ_cols) > 0:
                x_col = st.selectbox("Group by Category", categ_cols, key="box_x")
                if len(categ_cols) > 1:
                    color_by = st.selectbox("Color by (optional)", ["None"] + [c for c in categ_cols if c != x_col], key="box_color")
                else:
                    color_by = "None"
            else:
                st.error("No categorical columns available for grouping")
                return None
        
        # Generate chart
        if 'color_by' in locals() and color_by != "None":
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=f"Distribution of {y_col} by {x_col} and {color_by}"
            )
        else:
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=f"Distribution of {y_col} by {x_col}"
            )
    
    elif chart_type == "Pie Chart":
        with col1:
            category_col = st.selectbox("Select Category Column", categ_cols, key="pie_cat")
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column (optional)", ["Count"] + numeric_cols, key="pie_val")
            else:
                value_col = "Count"
        
        with col2:
            top_n = st.slider("Top N Categories", 1, 20, 5)
            show_others = st.checkbox("Group Others as 'Other'", True)
        
        # Generate chart
        if value_col == "Count":
            value_counts = df[category_col].value_counts().head(top_n)
            if show_others and len(df[category_col].unique()) > top_n:
                others_sum = df[category_col].value_counts().iloc[top_n:].sum()
                value_counts = pd.concat([value_counts, pd.Series([others_sum], index=["Other"])])
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col}",
                hole=0.4
            )
        else:
            # Group by category and sum values
            top_categories = df[category_col].value_counts().head(top_n).index.tolist()
            filtered_df = df[df[category_col].isin(top_categories)].copy()
            
            if show_others and len(df[category_col].unique()) > top_n:
                df_others = df[~df[category_col].isin(top_categories)].copy()
                df_others[category_col] = "Other"
                filtered_df = pd.concat([filtered_df, df_others])
            
            agg_df = filtered_df.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                agg_df,
                values=value_col,
                names=category_col,
                title=f"Sum of {value_col} by {category_col}",
                hole=0.4
            )
    
    # Common layout updates
    fig.update_layout(
        height=500,
        title={
            'font': {'size': 18, 'color': '#424242'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def generate_custom_dashboard(df, data_types):
    """Generate a custom dashboard with user-selected metrics and charts."""
    st.subheader("Customize Your Dashboard")
    
    # 1. Select metrics to display
    st.write("Select Metrics to Display")
    metric_cols = st.multiselect(
        "Choose columns for summary metrics",
        options=[col for col, type_ in data_types.items() if type_ == 'numerical'],
        default=[col for col, type_ in data_types.items() if type_ == 'numerical'][:3]
    )
    
    # 2. Select charts to display
    st.write("Select Charts to Include")
    recommended_charts = get_recommended_charts(df, data_types)
    
    selected_charts = []
    for i, chart in enumerate(recommended_charts):
        selected = st.checkbox(f"{chart['title']}", value=i < 4)  # Default select first 4
        if selected:
            selected_charts.append(chart)
    
    # 3. Add custom chart
    add_custom = st.checkbox("Add Custom Chart", True)
    
    # Generate dashboard
    if st.button("Generate Custom Dashboard"):
        st.header("Your Custom Dashboard")
        
        # Display metrics
        if metric_cols:
            st.subheader("Key Metrics")
            cols = st.columns(len(metric_cols))
            
            for i, col_name in enumerate(metric_cols):
                with cols[i]:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{df[col_name].mean():.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Avg {col_name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected charts
        if selected_charts:
            st.subheader("Data Visualizations")
            
            # Create a 2-column layout for charts
            for i in range(0, len(selected_charts), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(selected_charts):
                        with cols[j]:
                            fig = create_chart(df, selected_charts[i + j])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
        
        # Add custom chart if selected
        if add_custom:
            st.subheader("Custom Chart")
            custom_fig = create_custom_chart(df, data_types)
            if custom_fig:
                st.plotly_chart(custom_fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">Auto Dashboard Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded: {uploaded_file.name}")
                
                # Sample data option
                sample_size = st.slider("Sample size for large datasets", 
                                     min_value=1000, 
                                     max_value=max(10000, len(df)), 
                                     value=min(len(df), 5000),
                                     step=1000)
                
                if len(df) > sample_size:
                    use_sample = st.checkbox("Use data sample for faster processing", True)
                    if use_sample:
                        df = df.sample(sample_size, random_state=42)
                        st.info(f"Using a sample of {sample_size} rows")
                
                # Missing value handling
                st.write("Missing Value Handling")
                missing_strategy = st.radio(
                    "How to handle missing values?",
                    ["Remove rows with missing values", 
                     "Fill numerical with mean", 
                     "Fill categorical with mode",
                     "No handling"]
                )
                
                if missing_strategy == "Remove rows with missing values":
                    df = df.dropna()
                elif missing_strategy == "Fill numerical with mean":
                    for col in df.select_dtypes(include=['number']).columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill categorical with mode":
                    for col in df.select_dtypes(exclude=['number']).columns:
                        if df[col].mode().shape[0] > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None
        else:
            st.info("Please upload a CSV or Excel file to get started")
            
            # Option to use sample data
            use_sample_data = st.checkbox("Use sample data instead", True)
            if use_sample_data:
                # Create sample data
                dates = pd.date_range(start='1/1/2022', periods=100, freq='D')
                categories = ['A', 'B', 'C', 'D', 'E']
                regions = ['North', 'South', 'East', 'West']
                
                np.random.seed(42)
                df = pd.DataFrame({
                    'Date': dates,
                    'Category': np.random.choice(categories, size=100),
                    'Region': np.random.choice(regions, size=100),
                    'Sales': np.random.normal(1000, 200, size=100),
                    'Units': np.random.randint(10, 100, size=100),
                    'Profit': np.random.normal(200, 50, size=100),
                    'Customer_Satisfaction': np.random.uniform(3, 5, size=100)
                })
                
                st.success("Sample data loaded")
            else:
                df = None
    
    # Main content
    if df is not None:
        # Detect data types for columns
        data_types = detect_data_types(df)
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Data Explorer", "Auto-Generated Dashboard", "Custom Dashboard", "Export"],
            icons=["clipboard-data", "table", "bar-chart", "gear", "download"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected == "Overview":
            # Data overview
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            stats = generate_summary_stats(