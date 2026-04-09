@echo off
title AI Memory Forecaster - Production
echo ========================================================
echo Checking and installing dependencies for Production
echo ========================================================
pip install -r ..\requirements.txt

echo.
echo ========================================================
echo Starting AI Memory Forecaster Dashboard via Waitress
echo ========================================================
echo.
python serve.py
pause
