@echo off
echo Stopping Streamlit application...
echo.

REM Find and kill Streamlit processes
tasklist /FI "IMAGENAME eq streamlit.exe" 2>NUL | find /I /N "streamlit.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Found Streamlit process. Stopping...
    taskkill /F /IM streamlit.exe >NUL 2>&1
    echo Streamlit application stopped.
) else (
    echo No Streamlit application running.
)

REM Also check for Python processes running Streamlit
tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH | findstr "streamlit" >NUL
if "%ERRORLEVEL%"=="0" (
    echo Stopping Python Streamlit processes...
    for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH ^| findstr "streamlit"') do (
        taskkill /F /PID %%i >NUL 2>&1
    )
    echo Python Streamlit processes stopped.
)

echo.
echo Application stopped successfully.
pause