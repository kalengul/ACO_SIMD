@echo off
setlocal enabledelayedexpansion

echo === CUDA + OpenMP Hybrid Build (Windows/Clang) ===

:: Проверка компиляторов
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found in PATH!
    pause
    exit /b 1
)

where clang++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: clang++ not found in PATH!
    pause
    exit /b 1
)

:: Очистка
echo Cleaning previous builds...
del *.o *.obj *.exe *.dll *.lib *.exp 2>nul

:: Компиляция CUDA DLL
echo.
echo Step 1: Compiling CUDA DLL...
nvcc -O3 -Xcompiler "/MD" -DBUILD_CUDA_DLL -shared -o cuda_module.dll cuda_module.cu
if %errorlevel% neq 0 (
    echo ERROR: CUDA compilation failed!
    pause
    exit /b %errorlevel%
)

:: Компиляция основного приложения с Clang
echo.
echo Step 2: Compiling main application with Clang...
clang++ -std=c++17 -fopenmp -O3 -mavx -mavx2 -mfma -march=native -o test_hybrid.exe main_omp.cpp -L. -lcuda_module
if %errorlevel% neq 0 (
    echo ERROR: Clang compilation failed!
    pause
    exit /b %errorlevel%
)

:: Проверка наличия DLL
if not exist cuda_module.dll (
    echo ERROR: cuda_module.dll not found!
    pause
    exit /b 1
)

echo.
echo === BUILD SUCCESSFUL ===
echo Files created:
echo   - cuda_module.dll (CUDA module)
echo   - test_hybrid.exe (Main application)
echo.

:: Запуск теста
echo Starting test...
echo.
test_hybrid.exe

pause