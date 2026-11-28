@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   DANG KHOI DONG UNG DUNG DOI CHIEU TO KHAI (MANG LAN)
echo ========================================================
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    echo   Dia chi truy cap cho may khac: http://%%a:8501
)
echo   (Giu cua so nay mo trong suot thoi gian su dung)
echo ========================================================
echo.

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.runOnSave false
pause