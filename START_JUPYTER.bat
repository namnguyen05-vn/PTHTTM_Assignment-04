@echo off
setlocal
cd /d "%~dp0"
if not defined ASSIGNMENT04_PYTHON (
  if exist "E:\PTHTTM\ASG_04_runtime\env\Scripts\python.exe" (
    set "ASSIGNMENT04_PYTHON=E:\PTHTTM\ASG_04_runtime\env\Scripts\python.exe"
  ) else (
    set "ASSIGNMENT04_PYTHON=python"
  )
)
if not defined CNN_DATA_DIR (
  if exist "E:\PTHTTM\ASG_04_data\mnist.npz" set "CNN_DATA_DIR=E:\PTHTTM\ASG_04_data"
)
echo Open notebooks with the kernel: Python (Assignment 04)
"%ASSIGNMENT04_PYTHON%" -m jupyterlab "%~dp0notebooks"
endlocal
