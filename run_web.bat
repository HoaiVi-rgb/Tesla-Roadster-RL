@echo off
chcp 65001 >nul
title Tesla Roadster 2D - Web Visualizer
cd /d "%~dp0"
echo ======================================================================
echo 🚀 Khởi chạy Web Visualizer (Giao diện mô phỏng trực quan 2D)...
echo    Đang mở trình duyệt tại: http://localhost:8000
echo ======================================================================
python main.py web
pause
