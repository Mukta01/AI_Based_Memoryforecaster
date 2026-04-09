#!/bin/bash

# AI Memory Forecaster - Production Runner (Linux/Mac)
echo "========================================================"
echo "Checking and installing dependencies for Production"
echo "========================================================"

# Try pip3 then pip
if command -v pip3 &> /dev/null
then
    pip3 install -r ../requirements.txt
else
    pip install -r ../requirements.txt
fi

echo ""
echo "========================================================"
echo "Starting AI Memory Forecaster Dashboard via Waitress"
echo "========================================================"
echo ""

# Run the production server
python3 serve.py || python serve.py
