@echo off
setlocal
title IDP Attendance Camera

set "PROJECT_DIR=C:\Users\chan0\Documents\Codex\idp-attendance-system"
set "UPLOADER_DIR=C:\Users\chan0\Documents\Codex\IDP Uploader"
set "UPLOADER_CONFIG=%UPLOADER_DIR%\idp_uploader_config.json"
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PYTHON_EXE="
set "PYTHON_ARGS="

cd /d "%PROJECT_DIR%" || (
  echo Cannot open project folder:
  echo %PROJECT_DIR%
  pause
  exit /b 1
)

if exist "%VENV_PY%" (
  set "PYTHON_EXE=%VENV_PY%"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
  ) else (
    where python >nul 2>nul
    if not errorlevel 1 (
      set "PYTHON_EXE=python"
    )
  )
)

if "%PYTHON_EXE%"=="" (
  echo Python was not found. Install Python 3.10 or newer, then run this again.
  pause
  exit /b 1
)

:menu
cls
echo ================================================
echo IDP Attendance Camera
echo ================================================
echo Project : %PROJECT_DIR%
echo Uploader: %UPLOADER_DIR%
echo.
echo 1. Setup / install Python packages
echo 2. Enrol student name, ID, and face
echo 3. Open camera detection and upload CSVs
echo 4. Upload attendance CSVs once
echo Q. Quit
echo.
choice /c 1234Q /n /m "Choose an option: "

if errorlevel 5 goto end
if errorlevel 4 goto upload_once
if errorlevel 3 goto camera
if errorlevel 2 goto enrol
if errorlevel 1 goto setup

:setup
cls
echo Creating virtual environment if needed...
if not exist "%VENV_PY%" (
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv "%PROJECT_DIR%\.venv"
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    goto menu
  )
)
echo Installing packages. This can take a while the first time...
set "PYTHON_EXE=%VENV_PY%"
set "PYTHON_ARGS="
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%PROJECT_DIR%\requirements.txt"
echo.
echo Setup finished.
pause
goto menu

:enrol
cls
echo Enrolling a student. Enter name and matrix/student ID when asked.
echo A camera window will open to record the face.
echo.
"%PYTHON_EXE%" %PYTHON_ARGS% "%PROJECT_DIR%\enroll.py"
pause
goto menu

:camera
cls
echo Starting uploader in a second window...
if not exist "%UPLOADER_CONFIG%" (
  echo Uploader config not found:
  echo %UPLOADER_CONFIG%
  pause
  goto menu
)
start "IDP CSV Uploader" /D "%UPLOADER_DIR%" cmd /k ""%PYTHON_EXE%" %PYTHON_ARGS% "%UPLOADER_DIR%\pi_csv_uploader.py" --config "%UPLOADER_CONFIG%""
echo.
echo Opening camera detection. Press q in the camera window to quit.
echo.
"%PYTHON_EXE%" %PYTHON_ARGS% "%PROJECT_DIR%\main.py"
pause
goto menu

:upload_once
cls
echo Uploading new rows from all attendance_logs\attendance_*.csv files...
echo.
pushd "%UPLOADER_DIR%"
"%PYTHON_EXE%" %PYTHON_ARGS% "%UPLOADER_DIR%\pi_csv_uploader.py" --config "%UPLOADER_CONFIG%" --once
popd
pause
goto menu

:end
endlocal
exit /b 0
