@echo off
setlocal
cd /d "%~dp0"

set "REPO_ROOT=%~dp0..\.."
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" goto VENV_READY

set "BOOTSTRAP_PYTHON="
if exist "%USERPROFILE%\anaconda3\envs\sqlquery311\python.exe" (
  set "BOOTSTRAP_PYTHON=%USERPROFILE%\anaconda3\envs\sqlquery311\python.exe"
)
if defined BOOTSTRAP_PYTHON goto CHECK_PYTHON

for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "BOOTSTRAP_PYTHON=%%i"
if defined BOOTSTRAP_PYTHON goto CHECK_PYTHON

for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "BOOTSTRAP_PYTHON=%%i"
if not defined BOOTSTRAP_PYTHON goto PYTHON_MISSING

:CHECK_PYTHON
"%BOOTSTRAP_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto PYTHON_VERSION

echo [1/5] Creating a Python 3.11 virtual environment...
"%BOOTSTRAP_PYTHON%" -m venv ".venv"
if errorlevel 1 goto FAILED

:VENV_READY
echo [2/5] Checking Python dependencies...
"%VENV_PYTHON%" -c "import fastapi, uvicorn, sqlalchemy, cv2, qrcode, ultralytics, mediapipe, torch" >nul 2>&1
if not errorlevel 1 (
  "%VENV_PYTHON%" -m pip check >nul 2>&1
  if not errorlevel 1 goto MODELS
)

echo Installing first-run dependencies. This may take several minutes...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto FAILED
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto FAILED

:MODELS
echo [3/5] Checking Git LFS models...
"%VENV_PYTHON%" verify_models.py >nul 2>&1
if not errorlevel 1 goto SECURITY

where git >nul 2>&1
if errorlevel 1 goto MODEL_FAILED
git -C "%REPO_ROOT%" lfs version >nul 2>&1
if errorlevel 1 goto MODEL_FAILED
git -C "%REPO_ROOT%" lfs pull --include="database/ctpgr-pytorch-master/checkpoints/lstm_yolo11s.pt,database/vehicle-vision-system/backend/app/models/fh02.pth"
if errorlevel 1 goto MODEL_FAILED
"%VENV_PYTHON%" verify_models.py
if errorlevel 1 goto MODEL_FAILED

:SECURITY
echo [4/5] Initializing local security and database...
"%VENV_PYTHON%" setup_security.py
if errorlevel 1 goto FAILED

set "APP_PORT=8001"
if defined PORT set "APP_PORT=%PORT%"
if not defined PORT if exist ".env" (
  for /f "tokens=1,* delims==" %%a in ('findstr /B /C:"PORT=" ".env"') do set "APP_PORT=%%b"
)

set PORT_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%APP_PORT% .*LISTENING"') do set PORT_PID=%%a
if not defined PORT_PID goto START_SERVER

curl.exe -k -sS --max-time 3 https://localhost:%APP_PORT%/ >nul 2>&1
if not errorlevel 1 goto HTTPS_RUNNING
goto PORT_BUSY

:START_SERVER
echo [5/5] Starting https://localhost:%APP_PORT% ...
"%VENV_PYTHON%" run.py
set "SERVER_EXIT=%ERRORLEVEL%"
pause
exit /b %SERVER_EXIT%

:HTTPS_RUNNING
echo Service is already running: https://localhost:%APP_PORT%
echo PID: %PORT_PID%
pause
exit /b 0

:PORT_BUSY
echo Port %APP_PORT% is occupied by another process. It was not terminated.
echo Close PID %PORT_PID% or change PORT in .env, then try again.
pause
exit /b 1

:MODEL_FAILED
echo A required model is missing or failed validation.
echo Install Git LFS, run git lfs pull in the repository root, then try again.
pause
exit /b 1

:PYTHON_MISSING
echo Python 3.11 was not found. Install 64-bit Python 3.11 and add it to PATH.
pause
exit /b 1

:PYTHON_VERSION
echo The selected Python is not version 3.11. Install Python 3.11 and try again.
pause
exit /b 1

:FAILED
echo Initialization failed. Review the error output above.
pause
exit /b 1
