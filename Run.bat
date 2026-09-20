@echo off
chcp 65001 >nul

echo ========================================
echo    象棋机器人启动脚本
echo ========================================

set "WORK_DIR=%~dp0"
cd /d "%WORK_DIR%"

echo 当前工作目录: %WORK_DIR%

echo.
echo [1/2] 启动 Qt 主控界面...
if exist "%WORK_DIR%gui\QT_SLZ.exe" (
    start "" "%WORK_DIR%gui\QT_SLZ.exe"
) else (
    echo [警告] 未找到 gui\QT_SLZ.exe，跳过
)

echo [2/2] 启动机械臂控制...
if exist "%WORK_DIR%robot_arm\arm_controller.py" (
    start "机械臂控制" python "%WORK_DIR%robot_arm\arm_controller.py"
) else (
    echo [警告] 未找到 robot_arm\arm_controller.py，跳过
)

echo.
echo 所有模块已启动，请在新窗口中查看运行日志。
echo 按任意键退出此窗口...
pause >nul
