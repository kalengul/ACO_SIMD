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

:: Очистка предыдущих сборок
echo Cleaning previous builds...
del /Q *.o *.obj *.exe *.dll *.lib *.exp 2>nul

:: Компиляция CUDA DLL
echo.
echo Compiling CUDA DLL...
nvcc -O3 -arch=sm_86 -Xcompiler "/MD" -use_fast_math -lcudart -maxrregcount=32 -DBUILD_CUDA_DLL -shared -Xptxas "-O3,-v" -o cuda_module.dll cuda_module.cu --resource-usage >> "cuda_compile.txt" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CUDA compilation failed!
    pause
    exit /b %errorlevel%
)

:: Компиляция основного приложения с Clang
echo.
echo Compiling main application with Clang...
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
echo Using %OMP_NUM_THREADS% threads
clang++ -std=c++17 -fopenmp -O3 -mavx2 -mfma -march=native -o test_hybrid.exe main_omp.cpp -L. -lcuda_module
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