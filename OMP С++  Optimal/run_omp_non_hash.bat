@echo off
chcp 65001 >nul
echo ===================================================
echo    OPENMP OPTIMIZED PROGRAM LAUNCHER
echo ===================================================
echo.

:: Проверка наличия компилятора
where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: g++ compiler not found in PATH!
    echo Please install MinGW-w64 or add to PATH
    pause
    exit /b 1
)

echo [1/4] Compiling OpenMP program...
g++ -std=c++17 -fopenmp -O3 -mavx2 -march=native -o omp_optimized_non_hash.exe OMP_C++_non_hash.cpp

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

echo [2/4] Setting thread affinity...
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
echo Using %OMP_NUM_THREADS% threads

echo [3/4] Running optimized program...
echo Start time: %date% %time%
echo.

:: Запуск программы
omp_optimized_non_hash.exe

if %errorlevel% neq 0 (
    echo ERROR: Program execution failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Program completed successfully!
echo End time: %date% %time%

pause