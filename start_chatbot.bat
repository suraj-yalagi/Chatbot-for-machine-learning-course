@echo off
color 0A
cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                           ML Chatbot with Authentication                     ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.
echo Choose an option:
echo.
echo  ╔════════════════════════════════════════════════════════════════════════════╗
echo  ║  [1] Start the chatbot (opens login page)                                  ║
echo  ║  [2] Run integration test                                                 ║
echo  ║  [3] Exit                                                                  ║
echo  ╚════════════════════════════════════════════════════════════════════════════╝
echo.
echo ═════════════════════════════════════════════════════════════════════════════

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo ╔════════════════════════════════════════════════════════════════════════════╗
    echo ║                           Starting ML Chatbot...                            ║
    echo ║                     Opening login page in your default browser...           ║
    echo ╚════════════════════════════════════════════════════════════════════════════╝
    echo.
    start "" "login.html"
) else if "%choice%"=="2" (
    echo.
    echo ╔════════════════════════════════════════════════════════════════════════════╗
    echo ║                           Running integration test...                        ║
    echo ║                     Opening test page in your default browser...             ║
    echo ╚════════════════════════════════════════════════════════════════════════════╝
    echo.
    start "" "test_integration.html"
) else if "%choice%"=="3" (
    echo.
    echo ╔════════════════════════════════════════════════════════════════════════════╗
    echo ║                           Exiting...                                        ║
    echo ╚════════════════════════════════════════════════════════════════════════════╝
    echo.
    exit
) else (
    echo.
    echo ╔════════════════════════════════════════════════════════════════════════════╗
    echo ║                           Invalid choice. Please try again.                  ║
    echo ╚════════════════════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                           Troubleshooting Guide                               ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.
echo If the page doesn't open automatically, please try these steps:
echo.
echo  1. Right-click on the HTML file
echo  2. Select "Open with"
echo  3. Choose your preferred web browser (Chrome, Firefox, Edge, etc.)
echo.
echo ═════════════════════════════════════════════════════════════════════════════
echo.
pause 