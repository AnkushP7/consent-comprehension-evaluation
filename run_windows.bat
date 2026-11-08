@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment...
  py -m venv .venv
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 (
    echo Could not create a virtual environment. Install Python 3.10+ and try again.
    pause
    exit /b 1
  )
)
call .venv\Scripts\activate.bat
python -c "import streamlit, pandas, textstat, pypdf, docx, sklearn, joblib" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages for the first run...
  python -m pip install --upgrade pip
  python -m pip install -r requirements-ml.txt
  if errorlevel 1 (
    echo.
    echo Package installation failed. Check the error above and ensure internet access is available.
    pause
    exit /b 1
  )
)
python -m streamlit run app.py
pause
