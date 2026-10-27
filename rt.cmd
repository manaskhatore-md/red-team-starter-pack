@echo off
rem The rt command without installing it: .\rt scan from the repo root (PowerShell or cmd).
rem `pip install -e .` once makes plain `rt` work from any folder. See redteam\cli.py.
setlocal
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m redteam %*
exit /b %ERRORLEVEL%
