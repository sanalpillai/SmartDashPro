# SmartDashPro

<img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="64" height="64">

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

## Screenshots

![Smartdash-vid-gif](https://github.com/user-attachments/assets/cc1d3f4c-4941-4b9f-9193-a320c1c64238)

<img width="1508" alt="Screenshot 2025-03-23 at 8 26 12 PM" src="https://github.com/user-attachments/assets/76940d45-e19b-442a-9b4c-82631ae05f33" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 26 28 PM" src="https://github.com/user-attachments/assets/39f00823-bbe0-435c-8929-2812708ed8fe" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 19 03 PM" src="https://github.com/user-attachments/assets/f8fd71c5-a8de-4529-9aa1-b5af9d4dae11" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 19 19 PM" src="https://github.com/user-attachments/assets/e0e48dcd-f1cf-482b-8c47-c4ccc2c3b83f" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 19 34 PM" src="https://github.com/user-attachments/assets/61e4caf1-5664-43ed-91d3-45d3bb4ba031" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 19 51 PM" src="https://github.com/user-attachments/assets/b5db59c8-2d53-4fd2-9fbb-4da867ad6473" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 20 10 PM" src="https://github.com/user-attachments/assets/2ec6b3e3-acad-4db2-8055-9257ce61820a" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 20 23 PM" src="https://github.com/user-attachments/assets/068a1166-230c-447d-9d9e-07a86a4fd65b" />
<img width="1508" alt="Screenshot 2025-03-23 at 8 20 36 PM" src="https://github.com/user-attachments/assets/20a082fa-da38-4f0d-a1c7-5b6467fbc9c0" />



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
