@echo off
setlocal
chcp 65001 >nul
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

echo [1/5] 正在创建 Python 3.11 虚拟环境...
"%BOOTSTRAP_PYTHON%" -m venv ".venv"
if errorlevel 1 goto FAILED

:VENV_READY
echo [2/5] 正在检查 Python 依赖...
"%VENV_PYTHON%" -c "import fastapi, uvicorn, sqlalchemy, cv2, qrcode, ultralytics, mediapipe, torch" >nul 2>&1
if not errorlevel 1 (
  "%VENV_PYTHON%" -m pip check >nul 2>&1
  if not errorlevel 1 goto MODELS
)

echo 首次运行需要安装依赖，可能需要数分钟...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto FAILED
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto FAILED

:MODELS
echo [3/5] 正在检查 Git LFS 模型...
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
echo [4/5] 正在初始化本机安全配置和数据库...
"%VENV_PYTHON%" setup_security.py
if errorlevel 1 goto FAILED

set PORT_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8001 .*LISTENING"') do set PORT_PID=%%a
if not defined PORT_PID goto START_SERVER

curl.exe -k -sS --max-time 3 https://localhost:8001/ >nul 2>&1
if not errorlevel 1 goto HTTPS_RUNNING
goto PORT_BUSY

:START_SERVER
echo [5/5] 正在启动 https://localhost:8001 ...
"%VENV_PYTHON%" run.py
set "SERVER_EXIT=%ERRORLEVEL%"
pause
exit /b %SERVER_EXIT%

:HTTPS_RUNNING
echo 服务已经运行：https://localhost:8001
echo PID: %PORT_PID%
pause
exit /b 0

:PORT_BUSY
echo 端口 8001 已被其他程序占用，未结束该程序。
echo 请关闭 PID %PORT_PID% 或在 .env 中修改 PORT 后重试。
pause
exit /b 1

:MODEL_FAILED
echo 必需模型缺失或校验失败。
echo 请安装 Git LFS，在仓库根目录运行 git lfs pull，然后重试。
pause
exit /b 1

:PYTHON_MISSING
echo 未找到 Python 3.11。请安装 64 位 Python 3.11，并勾选 Add Python to PATH。
pause
exit /b 1

:PYTHON_VERSION
echo 当前 Python 不是 3.11。请安装 Python 3.11 后重试。
pause
exit /b 1

:FAILED
echo 初始化失败，请查看上方错误信息。
pause
exit /b 1
