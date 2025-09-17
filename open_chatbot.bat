@echo off
echo Opening ML Chatbot Login Page...
start "" "login.html"
timeout /t 2
if not exist "login.html" (
    echo Error: login.html not found!
    echo Please make sure login.html is in the same directory as this batch file.
    pause
    exit /b 1
)
echo If the login page doesn't open automatically, please try these steps:
echo 1. Right-click on login.html
echo 2. Select "Open with"
echo 3. Choose your preferred web browser (Chrome, Firefox, Edge, etc.)
pause 