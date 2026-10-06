@echo off
echo ========================================================
echo Launching SynthAI Milestone 4 Full-Stack Application
echo ========================================================
start "SynthAI API Backend" cmd /c "python -m uvicorn api:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "SynthAI Streamlit Dashboard" cmd /c "python -m streamlit run app.py --server.port 8501"
echo Both Backend (http://127.0.0.1:8000/docs) and Dashboard (http://localhost:8501) launched!
pause
