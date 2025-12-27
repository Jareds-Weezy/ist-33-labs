@echo off
echo ==========================================
echo Starting Laba 10 Setup...
echo ==========================================

echo [1/4] Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo Error installing dependencies!
    pause
    exit /b %errorlevel%
)

echo [2/4] Creating icons...
if not exist "public\icons" mkdir "public\icons"
copy "C:\Users\Ruslan\Desktop\web\web10\pwa_qr_icon_1765707403989.png" "public\icons\icon-192x192.png" /Y
copy "C:\Users\Ruslan\Desktop\web\web10\pwa_qr_icon_1765707403989.png" "public\icons\icon-512x512.png" /Y

echo [3/4] Building project...
call npm run build

echo ==========================================
echo Setup Complete!
echo ==========================================
echo To start the app, run: npm run dev
echo Then open the link shown (usually http://localhost:5173)
pause