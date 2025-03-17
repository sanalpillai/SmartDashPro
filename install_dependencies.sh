#!/bin/bash

# This script handles installing dependencies for the Streamlit Auto Dashboard Generator
# It creates a virtual environment and installs the required packages

# Exit on error
set -e

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

echo "====== Streamlit Dashboard Dependencies Installer ======"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Using existing virtual environment."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Creating requirements.txt file..."
    cat > requirements.txt << EOL
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.4
plotly==5.20.0
scipy==1.12.0
EOL
    pip install -r requirements.txt
fi

# Check if installations were successful
echo "Verifying installations..."
python -c "import streamlit, pandas, numpy, plotly, scipy; print('All packages installed successfully!')"

# Inform user that installation is complete
echo ""
echo "====== Installation Complete! ======"
echo "To run the Streamlit app, use the following commands:"
echo ""
echo "source venv/bin/activate"
echo "streamlit run streamlit_app.py"
echo ""
