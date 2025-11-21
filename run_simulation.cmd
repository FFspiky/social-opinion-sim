@echo off
:: ======================================================
:: AI 多智能体社会舆情模拟 —— 一键启动脚本
:: 适用于 Windows（VSCode / CMD / 双击运行）
:: ======================================================

echo ----------------------------------------------
echo   正在启动大模型驱动的舆情模拟系统...
echo ----------------------------------------------

:: 切换到当前脚本所在目录
cd /d "%~dp0"

:: 检查 Python
echo 正在检查 Python...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3。
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b
)

:: 激活虚拟环境
echo 正在激活虚拟环境 venv...
CALL venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    echo [错误] 无法激活虚拟环境！
    echo 请确认 venv 文件夹存在。
    pause
    exit /b
)

:: 检查 streamlit
echo 正在检查 Streamlit...
streamlit --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo 未检测到 Streamlit，正在安装...
    pip install streamlit
)

echo ----------------------------------------------
echo 虚拟环境已激活，正在启动前端 (Streamlit)...
echo 浏览器将自动打开 http://localhost:8501
echo ----------------------------------------------

:: 运行 Streamlit
streamlit run app_streamlit.py

echo 程序已退出。
pause
