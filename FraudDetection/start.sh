#!/bin/bash
# Quick Start Script for Automotive Fraud Detection System

echo "=========================================="
echo "Automotive Fraud Detection System"
echo "Quick Start Setup"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your GEMINI_API_KEY"
    echo "Get your key from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter after you've added your API key to .env file..."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Starting Fraud Detection System..."
echo "The web interface will open at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the application
python input.py
