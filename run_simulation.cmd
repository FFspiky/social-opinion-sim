@echo off
:: ==============================================
:: 一键启动模拟（自动创建/激活虚拟环境）
:: ==============================================

cd /d "%~dp0"

echo [INFO] 当前路径: %cd%

:: 检查 Python
py --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 未检测到 Python，请先安装 Python 3。
    pause
    exit /b
)

:: 若 venv 不存在，则自动创建
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] 未发现 venv，正在创建虚拟环境...
    py -m venv venv
)

:: 激活虚拟环境
echo [INFO] 正在激活虚拟环境...
CALL venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    echo [ERROR] 无法激活虚拟环境，请检查 venv 是否创建成功。
    pause
    exit /b
)

:: 检查 streamlit，不存在则安装
streamlit --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [INFO] 未检测到 Streamlit，正在安装...
    pip install streamlit
)

echo [INFO] 启动 Streamlit 应用...
streamlit run app_streamlit.py

echo [INFO] 程序已退出
pause
