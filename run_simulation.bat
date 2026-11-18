@echo off
chcp 65001 >nul
title Tesla Roadster 2D - Simulation
cd /d "%~dp0"
echo ======================================================================
echo 🚀 Chạy mô phỏng hạ cánh vật lý 2D với Model tối ưu (Trained DDQN)...
echo ======================================================================
python main.py simulate --episodes 3 --verbose
pause
