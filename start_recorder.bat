@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo [START] Starting GPT-SoVITS Recording + Training WebUI...
.\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe sentence_webui.py
pause
