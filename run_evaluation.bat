@echo off
chcp 65001 >nul
title Tesla Roadster 2D - Independent Evaluator
cd /d "%~dp0"
echo ======================================================================
echo 📊 Đánh giá độc lập trên 100 Episodes chưa từng thấy (Seeds 30000..30099)...
echo ======================================================================
python main.py evaluate --suite stochastic --episodes 100
pause
