@echo off
echo ========================================================
echo Starting SynthAI Milestone 4 - FastAPI REST API Backend
echo ========================================================
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
pause
