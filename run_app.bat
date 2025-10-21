@echo off
echo Starting Mental Health in Tech Predictor...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Start the Streamlit app
echo.
echo Starting Streamlit application...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py

pause