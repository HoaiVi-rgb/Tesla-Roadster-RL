@echo off
chcp 65001 >nul
title Tesla Roadster 2D - Test Suite
cd /d "%~dp0"
echo ======================================================================
echo 🧪 Chạy toàn bộ 117 bài kiểm thử tự động (PyTest)...
echo ======================================================================
pytest -v
pause
