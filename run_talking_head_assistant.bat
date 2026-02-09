@echo off
REM Initialize conda
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

REM Activate environment
call conda activate talking-head-assistant-280

REM Go to project directory (directory of this .bat file)
cd /d "%~dp0"

REM Run main script
python main.py

REM Keep window open
pause
