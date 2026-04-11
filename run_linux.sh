#!/bin/bash
# -------------------------------------------------------------
# Paper Review System - Linux Runner
# -------------------------------------------------------------

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting Streamlit Application..."
# Run the app and stream output
streamlit run app.py
