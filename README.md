# SmartDashPro

![SmartDashPro Logo](https://cdn-icons-png.flaticon.com/512/6295/6295417.png)

SmartDashPro is an intelligent data dashboard that automatically analyzes your data and generates beautiful, insightful visualizations with zero configuration required. Simply upload your CSV or Excel file and get an auto-generated dashboard tailored to your data.

## 🌟 Features

### ⚡ Automatic Data Intelligence
- **Smart Column Analysis**: Automatically detects column types, semantics, and best visualization types
- **Visualization Selection**: Chooses optimal chart types based on data characteristics
- **Insight Generation**: Identifies correlations and relationships between variables

### 📊 Rich Visualizations
- **Interactive Charts**: Beautifully styled charts with hover tooltips and interactive elements
- **Dashboard Layout**: Well-organized layout with key metrics, main visualizations, and detailed insights
- **Multi-tab Interface**: Separate tabs for dashboard overview, insights, data exploration, and quality assessment

### 🔍 Data Quality Assessment
- **Quality Scoring**: Overall data quality score based on completeness, uniqueness, and consistency
- **Issue Detection**: Identifies missing values, duplicates, and outliers
- **Detailed Reports**: In-depth analysis of data quality issues with recommendations

### 📈 Advanced Analytics
- **Correlation Analysis**: Heatmaps showing relationships between numeric variables
- **Distribution Analysis**: Histograms, box plots, and statistical metrics for numeric data
- **Time Series Analysis**: Trend detection and time-based visualizations for date columns

## 📋 Dashboard Sections

1. **Dashboard** - Key metrics and main visualizations
   - Summary metrics (rows, columns, completeness)
   - Key metric cards highlighting important statistics
   - Auto-selected visualizations for the most insightful columns
   - Featured insight recommendation

2. **Insights** - Deeper analytical views
   - Correlation analysis with heatmap
   - Recommended visualizations based on data relationships
   - Multi-variable analyses showing relationships between columns

3. **Data Explorer** - Interactive data examination
   - Column explorer organized by data types
   - Detailed column statistics and visualizations
   - Data preview with expandable view

4. **Data Quality** - Quality assessment and statistics
   - Overall data quality score
   - Missing value analysis
   - Duplicate detection
   - Outlier identification
   - Column-level statistics

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Streamlit
- Pandas
- Plotly
- SciPy (optional, for advanced statistical analysis)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sanalpillai/SmartDashPro.git
   cd SmartDashPro
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run streamlit-auto-dashboard.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

### Using the Dashboard

1. Upload your CSV or Excel file using the file uploader
2. Alternatively, select one of the built-in sample datasets
3. Navigate between different tabs to explore your data:
   - **Dashboard**: Overview of key metrics and visualizations
   - **Insights**: Deeper analysis and relationships
   - **Data Explorer**: Explore individual columns
   - **Data Quality**: Assess data completeness and issues

## 💡 How It Works

SmartDashPro analyzes your data using these intelligent processes:

1. **Data Loading & Initial Analysis**
   - Detects file type and loads data
   - Analyzes basic statistics like row count, column types, and completeness

2. **Column Analysis**
   - Examines each column to determine data type, semantics, and cardinality
   - Analyzes column names to infer meaning (e.g., identifying gender, location, etc.)
   - Determines the most appropriate visualization type for each column

3. **Insight Generation**
   - Identifies correlations between numeric columns
   - Detects categorical relationships and creates cross-tabulations
   - Examines time-based patterns if date columns are present

4. **Visualization Creation**
   - Builds appropriately styled charts for each column based on its characteristics
   - Creates combinations of columns for relationship analysis
   - Generates summary metrics and key insights

5. **Quality Assessment**
   - Calculates data quality metrics like completeness and consistency
   - Identifies outliers, missing values, and duplicates
   - Provides a quality score with detailed component analysis

## 🛠️ Customization

You can modify the dashboard's appearance and behavior by:

- Editing the custom CSS in the `st.markdown()` section at the top of the script
- Adjusting the visualization parameters in the `create_enhanced_visualization()` function
- Adding new sample datasets in the main app logic

## 📚 Use Cases

- **Data Exploration**: Quickly understand new datasets without manual analysis
- **Data Quality Assessment**: Identify issues in your data before deeper analysis
- **Business Intelligence**: Generate shareable dashboards from business data
- **Research Visualization**: Create publication-ready charts for research data
- **Educational Tool**: Help students understand data visualization principles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Plotly](https://plotly.com/) for the interactive visualization library
- [Pandas](https://pandas.pydata.org/) for powerful data manipulation

---

Made with ❤️ by Sanal Pillai
